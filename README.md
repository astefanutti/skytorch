# SkyTorch

Run PyTorch locally with GPUs in the cloud.

SkyTorch registers a `sky` device backend in PyTorch that virtualizes remote GPUs and transparently streams tensor operations.

## Examples

### MNIST Training

```python
import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skytorch.client import compute

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv2(nn.functional.relu(self.conv1(x)))))
        x = self.fc2(nn.functional.relu(self.fc1(torch.flatten(x, 1))))
        return x

@compute(
    name="mnist",
    image="ghcr.io/astefanutti/skytorch-server",
    resources={"cpu": "1", "memory": "8Gi", "nvidia.com/gpu": "1"},
)
async def train(node, epochs: int = 10):
    device = node.device("cuda")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform),
        batch_size=5000, shuffle=True,
    )

    model = MNISTNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

asyncio.run(train())
```

### LLM Chat

```python
import asyncio
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer
from skytorch.client import compute

@compute(
    name="chat",
    image="ghcr.io/astefanutti/skytorch-server",
    resources={"cpu": "1", "memory": "16Gi", "nvidia.com/gpu": "1"},
    volumes=[{"name": "cache", "storage": "20Gi", "path": "/cache"}],
    env={"HF_HOME": "/cache"},
)
async def chat(node):
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

    with torch.no_grad():
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ("quit", "exit"):
                break

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

try:
    asyncio.run(chat())
except KeyboardInterrupt:
    pass
```

### GRPO Training

```python
import copy
from transformers import AutoModelForCausalLM
from skytorch.client import Compute, Cluster

async with Cluster(
    Compute(
        name="trainer",
        resources={"nvidia.com/gpu": "1"},
    ),
    Compute(
        name="vllm",
        resources={"nvidia.com/gpu": "1"},
    ),
) as (trainer, vllm):
    trainer_device = trainer.device("cuda")
    vllm_device = vllm.device("cuda")

    # Load the policy model on the trainer and copy it to vLLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
    model.to(trainer_device)
    ref_model = copy.deepcopy(model).to(vllm_device)

    for step in range(10):
        # GRPO training step on the trainer device
        # ...

        # Sync weights from trainer to vLLM
        for p, ref_p in zip(model.parameters(), ref_model.parameters()):
            ref_p.data.copy_(p.data)
```

> **Note:** Cross-compute tensor copy is not supported yet. This example illustrates a future capability.

## Getting Started

```bash
pip install torch
pip install --no-build-isolation skytorch
```

`--no-build-isolation` is required because the C++ extension needs PyTorch headers at build time, so PyTorch must be installed first.

SkyTorch requires a Kubernetes cluster with [Gateway API](https://gateway-api.sigs.k8s.io/) support and the SkyTorch operator deployed.
You can install the operator using Kustomize, choosing the overlay that matches your cluster:

```bash
# Vanilla Kubernetes / KinD (includes Contour as the Gateway API controller)
kubectl apply --server-side -k config/e2e

# OpenShift (uses the built-in gateway controller)
kubectl apply --server-side -k config/openshift
```

## Configuration

SkyTorch can be configured via environment variables.

### Client

| Variable                       | Default | Description                                                               |
|--------------------------------|---------|---------------------------------------------------------------------------|
| `SKYTORCH_GRPC_COMPRESSION`    | `gzip`  | gRPC compression (`none`, `deflate`, `gzip`)                              |
| `SKYTORCH_BATCH_THRESHOLD`     | `64`    | Ops buffered before forced flush                                          |
| `SKYTORCH_BATCH_COALESCE_MS`   | `2`     | Delay (ms) to coalesce partial batches                                    |
| `SKYTORCH_STREAMING`           | `1`     | Enable bidirectional gRPC streaming, must be set to `0` for IDE debugging |
| `SKYTORCH_CPP_REQUEST_BUILDER` | `1`     | Use C++ fast path for request serialization                               |
| `SKYTORCH_SPECULATIVE_SCALAR`  | `1`     | Predict `.item()` results to avoid sync                                   |
| `SKYTORCH_PROFILE`             | `0`     | Enable lightweight profiling (~200ns/op overhead)                         |

### Server

| Variable                    | Default   | Description                                                     |
|-----------------------------|-----------|-----------------------------------------------------------------|
| `SKYTORCH_PORT`             | `50051`   | gRPC server port                                                |
| `SKYTORCH_HOST`             | `[::]`    | Server bind address                                             |
| `SKYTORCH_LOG_LEVEL`        | `INFO`    | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `SKYTORCH_GRPC_COMPRESSION` | `none`    | gRPC compression (`none`, `deflate`, `gzip`)                    |
| `SKYTORCH_CHUNK_SIZE`       | `1048576` | Chunk size in bytes for streaming tensors                       |
| `SKYTORCH_METRICS_SOURCES`  | _(empty)_ | Comma-separated metrics sources (e.g., `nvidia-gpu`)            |
