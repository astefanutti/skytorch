import asyncio
import logging
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from skytorch.client import Compute, compute, log_event

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@compute(
    name="llm",
    image="ghcr.io/astefanutti/skytorch-server",
    resources={"cpu": "1", "memory": "8Gi", "nvidia.com/gpu": "1"},
    volumes=[{"name": "cache", "storage": "10Gi", "path": "/cache"}],
    env={"HF_HOME": "/cache"},
    on_events=log_event,
)
async def llm(node: Compute):
    device = node.device("cuda")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

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

    prompts = [
        "What is machine learning in one sentence?",
        "Write a haiku about the cloud.",
    ]

    start_time = time.perf_counter()

    with torch.no_grad():
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            response = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            print(f"Response: {response}\n")

    elapsed = time.perf_counter() - start_time
    print(f"Inference completed in {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(llm())
