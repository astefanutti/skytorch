# SkyTorch

Run PyTorch locally with the Power of Cloud GPUs.

SkyTorch registers a `"sky"` device backend in PyTorch that transparently streams tensor operations to cloud GPUs managed by Kubernetes.

### MNIST Training

```python
import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skytorch.client import Compute, compute

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
    image="ghcr.io/astefanutti/skytorch-server:latest",
    resources={"nvidia.com/gpu": "1"},
)
async def train(node: Compute, epochs: int = 10):
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

### Multi-GPU GRPO Training

```python
import asyncio
import copy
from transformers import AutoModelForCausalLM
from skytorch.client import Compute, Cluster

async def main():
    async with Cluster(
        Compute(name="trainer"),
        Compute(name="vllm"),
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

asyncio.run(main())
```

> **Note:** Cross-compute tensor copy is not supported yet. This example illustrates a future capability.

## Getting Started

```bash
pip install torch
pip install --no-build-isolation skytorch
```

`--no-build-isolation` is required because the C++ extension needs PyTorch headers at build time, so PyTorch must be installed first.
