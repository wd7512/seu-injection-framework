import json
import platform
import time

import psutil
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallConvNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


input_size = (3, 64, 64)
batch_sizes = [1, 8, 32, 128]

model = SmallConvNet().to(device)
model.eval()

# Warm-up original model
for _ in range(5):
    x = torch.randn(batch_sizes[-1], *input_size, device=device)
    with torch.no_grad():
        model(x)
if device.type == "cuda":
    torch.cuda.synchronize()

results = []
for batch in batch_sizes:
    x = torch.randn(batch, *input_size, device=device)
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    avg_time = total_time / 50
    avg_time_per_image = avg_time / batch
    results.append(
        {
            "batch_size": batch,
            "total_time_50_forward_passes": total_time,
            "avg_time_per_forward_pass": avg_time,
            "avg_time_per_image": avg_time_per_image,
        }
    )

# Attempt torch.compile benchmark
compiled_results = []
try:
    compiled_model = torch.compile(model)
    compiled_model.eval()

    # Warm-up compiled model
    for _ in range(5):
        x = torch.randn(batch_sizes[-1], *input_size, device=device)
        with torch.no_grad():
            compiled_model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for batch in batch_sizes:
        x = torch.randn(batch, *input_size, device=device)
        start = time.time()
        for _ in range(50):
            with torch.no_grad():
                compiled_model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        total_time = end - start
        avg_time = total_time / 50
        avg_time_per_image = avg_time / batch
        compiled_results.append(
            {
                "batch_size": batch,
                "total_time_50_forward_passes": total_time,
                "avg_time_per_forward_pass": avg_time,
                "avg_time_per_image": avg_time_per_image,
            }
        )

except Exception as e:
    compiled_results = [{"error": f"torch.compile failed with: {str(e)}"}]

record = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "os": platform.platform(),
    "cpu": platform.processor(),
    "cpu_count": psutil.cpu_count(logical=False),
    "cpu_count_logical": psutil.cpu_count(logical=True),
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU only",
    "device": str(device),
    "model": "SmallConvNet CNN (3x64x64 input)",
    "results": results,
    "compiled_results": compiled_results,
}

print(json.dumps(record, indent=2))

with open("benchmark_results.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
