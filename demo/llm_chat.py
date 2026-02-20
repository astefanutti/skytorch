import asyncio
import logging
import signal
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer

from skytorch.client import Compute, compute, log_event

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@compute(
    name="chat",
    image="ghcr.io/astefanutti/skytorch-server",
    resources={"cpu": "1", "memory": "16Gi", "nvidia.com/gpu": "1"},
    volumes=[{"name": "cache", "storage": "20Gi", "path": "/cache"}],
    env={"HF_HOME": "/cache"},
    on_events=log_event,
)
async def chat(node: Compute):
    device = node.device("cuda")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    def load_model(model):
        return AutoModelForCausalLM.from_pretrained(
            model,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to("cuda")

    # Load the model weights server-side (stays on GPU, only metadata returned)
    # and the tokenizer locally in parallel
    state_dict, tokenizer = await asyncio.gather(
        node.execute(load_model, model_name),
        asyncio.to_thread(AutoTokenizer.from_pretrained, model_name),
    )

    # Sync model locally (no weights downloaded)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name),
            dtype=torch.float32,
            attn_implementation="eager",
        )
    state_dict.load_into(model)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    print("\nChat with the model (type 'quit' or 'exit' to stop)\n")

    # Override asyncio's SIGINT handler which defers the first Ctrl-C
    signal.signal(signal.SIGINT, signal.default_int_handler)

    with torch.no_grad():
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                break
            if not user_input.strip():
                continue

            history.append({"role": "user", "content": user_input})
            inputs = tokenizer.apply_chat_template(
                history,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            print("Assistant: ", end="", flush=True)
            generated = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, streamer=streamer
            )

            response = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass
