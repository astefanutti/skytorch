import asyncio
import logging
import signal
import sys

import torch
from openai_harmony import (
    HarmonyEncodingName,
    Role,
    StreamableParser,
    StreamState,
    load_harmony_encoding,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer

from skytorch.client import Compute, compute, log_event

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class HarmonyStreamer(TextStreamer):
    """Streamer for GPT-OSS Harmony response format.

    Uses openai-harmony's StreamableParser to parse the structured output with
    <|start|>, <|channel|>, <|message|>, <|end|>, and <|return|> special tokens.
    Displays analysis/thinking traces in grey and the final response in normal color.
    """

    _GREY = "\033[90m"
    _RESET = "\033[0m"

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._start_token_id = self._encoding.encode(
            "<|start|>", allowed_special={"<|start|>"}
        )[0]
        self._endoftext_token_id = self._encoding.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]
        self.reset()

    def reset(self):
        """Clear state between turns."""
        self._parser = StreamableParser(self._encoding, Role.ASSISTANT, strict=False)
        self._prev_state = self._parser.state
        self._prev_channel = None
        self._final_text = []
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("HarmonyStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token_id in value.tolist():
            # <|endoftext|> is never valid content — skip it in any parser state.
            # With strict=False the parser would pass it through as a content delta.
            if token_id == self._endoftext_token_id:
                return
            # After <|end|>, the parser moves to EXPECT_START (distinct from the
            # initial HEADER state). Only <|start|> is valid there (next message
            # in a multi-message response) — skip everything else (<|return|>,
            # etc.). Speculative scalar may delay the stopping criterion by one
            # token, so the streamer must tolerate trailing tokens.
            if self._parser.state == StreamState.EXPECT_START and token_id != self._start_token_id:
                return
            self._parser.process(token_id)
            state = self._parser.state
            channel = self._parser.current_channel

            # Transition out of content: reset ANSI for analysis
            if self._prev_state == StreamState.CONTENT and state != StreamState.CONTENT:
                if self._prev_channel == "analysis":
                    print(self._RESET, end="", flush=True)

            # Transition into content: print channel label
            if state == StreamState.CONTENT and self._prev_state != StreamState.CONTENT:
                if channel == "analysis":
                    print(f"\n{self._GREY}Thinking: ", end="", flush=True)
                elif channel == "final":
                    print("\n\nAssistant: ", end="", flush=True)

            # Stream content deltas
            if state == StreamState.CONTENT:
                delta = self._parser.last_content_delta
                if delta:
                    print(delta, end="", flush=True)
                    if channel == "final":
                        self._final_text.append(delta)

            self._prev_state = state
            self._prev_channel = channel

    def end(self):
        self._parser.process_eos()
        if self._prev_state == StreamState.CONTENT:
            if self._prev_channel == "analysis":
                print(self._RESET, end="", flush=True)
        self.next_tokens_are_prompt = True
        print()

    @property
    def eos_token_id(self):
        """Harmony stop tokens for use as eos_token_id in model.generate()."""
        return self._encoding.stop_tokens_for_assistant_actions()

    def get_final_response(self):
        """Return text from only the final channel."""
        if not self._final_text:
            return ""
        return "".join(self._final_text).strip()


@compute(
    name="gpt",
    image="ghcr.io/astefanutti/skytorch-server",
    resources={"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"},
    volumes=[{"name": "cache", "storage": "80Gi", "path": "/cache"}],
    env={"HF_HOME": "/cache", "TRITON_HOME": "/cache"},
    on_events=log_event,
)
async def chat(node: Compute):
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

    streamer = HarmonyStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    print("\nChat with the model (type 'quit' or 'exit' to stop)")

    # Override asyncio's SIGINT handler which defers the first Ctrl-C
    signal.signal(signal.SIGINT, signal.default_int_handler)

    past_key_values = None
    past_length = 0
    with torch.no_grad():
        while True:
            user_input = input("\nYou: ")
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
            else:
                # Diff approach: tokenize history without vs with the new user
                # message to extract only the new tokens, avoiding assumptions
                # about template format.
                prev_ids = tokenizer.apply_chat_template(
                    history[:-1], add_generation_prompt=False, return_tensors="pt",
                    return_dict=True,
                )["input_ids"]
                full_ids = tokenizer.apply_chat_template(
                    history, add_generation_prompt=True, return_tensors="pt",
                    return_dict=True,
                )["input_ids"]
                new_ids = full_ids[:, prev_ids.shape[1]:]
                total_len = past_length + new_ids.shape[1]
                # Pad input_ids so prepare_inputs_for_generation keeps the new tokens
                # instead of stripping them (it strips past_length prefix, keeping K real tokens)
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
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    streamer=streamer,
                    eos_token_id=streamer.eos_token_id,
                    return_dict_in_generate=True,
                )
            except (KeyboardInterrupt, RuntimeError) as exc:
                # KeyboardInterrupt during C++ dispatch is converted to
                # RuntimeError("KeyboardInterrupt") by TORCH_CHECK
                if isinstance(exc, RuntimeError) and "KeyboardInterrupt" not in str(exc):
                    raise
                print("\033[0m")  # Reset ANSI codes
                streamer.reset()
                past_key_values = None
                past_length = 0
                history.pop()
                continue

            response = streamer.get_final_response()
            history.append({"role": "assistant", "content": response})
            streamer.reset()
            past_key_values = output.past_key_values
            past_length = output.sequences.shape[1]


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        pass
