import asyncio
import logging
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from skytorch.client import Compute, compute, log_event


torch.set_printoptions(precision=2, threshold=10, edgeitems=2)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


@compute(
    name="mnist",
    image="ghcr.io/astefanutti/skytorch-server",
    resources={"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
    on_events=log_event,
)
async def train(node: Compute, epochs: int = 10):
    device = node.device("cuda")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("../data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=5000, shuffle=False)

    model = MNISTNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    print(f"Training on {device}")

    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", file=sys.stdout)

        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating", unit="batch", file=sys.stdout):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum()
                total += len(target)

        accuracy = (correct.float() / total).item() * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

    elapsed = time.perf_counter() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s")


if __name__ == '__main__':
    asyncio.run(train(epochs=5))
