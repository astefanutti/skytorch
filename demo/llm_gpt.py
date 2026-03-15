import asyncio
import logging
import os
import signal
import sys
import time

# Enable async scalar copy before any skytorch import (read at import time)
os.environ.setdefault("SKYTORCH_ASYNC_COPY", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from skytorch.client import Compute, compute, log_event
from skytorch.transformers.cache import SpeculativeCache
from skytorch.transformers.harmony import HarmonyStreamer
from skytorch.transformers.streamer import AsyncStreamer
from util import async_input

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@compute(
    name="gpt",
    image="ghcr.io/astefanutti/skytorch-server",
    resources={"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"},
    volumes=[{"name": "cache", "storage": "80Gi", "path": "/cache"}],
    env={"HF_HOME": "/cache", "TRITON_HOME": "/cache"},
    on_events=log_event,
)
async def chat(node: Compute, max_new_tokens: int = 512):
    device = node.device("cuda")
    model_name = "openai/gpt-oss-120b"

    def load_model(model):
        return AutoModelForCausalLM.from_pretrained(model, device_map="cuda")

    # Load the model weights server-side (stays on GPU, only metadata returned)
    # and the tokenizer locally in parallel
    state_dict, tokenizer = await asyncio.gather(
        node.execute(load_model, model_name, retain_model=True),
        asyncio.to_thread(AutoTokenizer.from_pretrained, model_name),
    )

    # Sync model locally (no weights downloaded)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name),
            attn_implementation="eager",
        )
    state_dict.load_into(model, triton_modules=["model.layers.*.mlp"])
    model.eval()

    streamer = AsyncStreamer(HarmonyStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True))
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    print("\nChat with the model (type 'quit' or 'exit' to stop)")

    # Override asyncio's SIGINT handler which defers the first Ctrl-C
    signal.signal(signal.SIGINT, signal.default_int_handler)

    total_tokens = 0
    total_time = 0.0
    num_turns = 0
    past_key_values = None
    past_length = 0

    with torch.no_grad():
        while True:
            user_input = await async_input("\nYou: ")
            if user_input.strip().lower() in ("quit", "exit"):
                break
            if not user_input.strip():
                continue

            history.append({"role": "user", "content": user_input})
            if past_key_values is None:
                inputs = tokenizer.apply_chat_template(
                    history,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["past_key_values"] = SpeculativeCache(config=model.config)
            else:
                # Tokenize full history to extract new tokens via diff.
                # Re-tokenization is needed because the chat template adds
                # role delimiters that can't be computed without it.
                prev_ids = tokenizer.apply_chat_template(
                    history[:-1], add_generation_prompt=False,
                    return_tensors="pt", return_dict=True,
                )["input_ids"]
                full_ids = tokenizer.apply_chat_template(
                    history, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                )["input_ids"]
                new_ids = full_ids[:, prev_ids.shape[1]:]
                total_len = past_length + new_ids.shape[1]
                input_ids = torch.cat(
                    [torch.zeros(1, past_length, dtype=torch.long), new_ids], dim=1,
                ).to(device)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(
                        1, total_len, dtype=torch.long, device=device,
                    ),
                    "past_key_values": past_key_values,
                    "cache_position": torch.arange(
                        past_length, total_len, dtype=torch.long, device=device,
                    ),
                }

            try:
                t0 = time.perf_counter()
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    streamer=streamer,
                    eos_token_id=streamer.eos_token_id,
                    return_dict_in_generate=True,
                )
                elapsed = time.perf_counter() - t0
            except (KeyboardInterrupt, RuntimeError) as exc:
                # KeyboardInterrupt during C++ dispatch is converted to
                # RuntimeError("KeyboardInterrupt") by TORCH_CHECK
                if isinstance(exc, RuntimeError) and "KeyboardInterrupt" not in str(exc):
                    raise
                print("\033[0m")  # Reset ANSI codes
                streamer.reset()
                history.pop()
                past_key_values = None
                past_length = 0
                continue

            response = streamer.get_final_response()
            history.append({"role": "assistant", "content": response})
            gen_tokens = streamer.generated_tokens
            past_key_values = output.past_key_values
            past_length = output.sequences.shape[1]
            # Crop overshoot from KV cache. Speculation may bypass both
            # EOS and max_new_tokens stopping criteria. The SpeculativeCache
            # allows crop on sliding window layers (base class forbids it).
            valid = streamer._valid_generated or max_new_tokens
            overshoot = max(0, gen_tokens - valid)
            if overshoot > 0:
                past_key_values.crop_overshoot(overshoot)
                past_length -= overshoot
            streamer.reset()
            tokens_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
            print(f"\n[{gen_tokens} tokens in {elapsed:.1f}s — {tokens_per_sec:.1f} token/s]")
            total_tokens += gen_tokens
            total_time += elapsed
            num_turns += 1

    if num_turns > 0:
        avg_tok_s = total_tokens / total_time if total_time > 0 else 0
        print(
            f"\nSession: {num_turns} turns, {total_tokens} tokens, "
            f"{total_time:.1f}s, {avg_tok_s:.1f} avg token/s"
        )


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass
