import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.MaxPool2d(2),  # 4x4

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def main():
    parser = argparse.ArgumentParser(description="GPU training stress test (PyTorch + CIFAR-10)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (increase to stress GPU more).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (recommended on RTX/Ampere+).")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--print-every", type=int, default=50, help="Print stats every N steps.")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        print("Torch:", torch.__version__)
        torch.cuda.reset_peak_memory_stats()

    # Data
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # Model / loss / optim
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    print(f"Dataset size: {len(dataset)} | Steps/epoch: {len(loader)}")
    print(f"Batch size: {args.batch_size} | AMP: {scaler.is_enabled()} | workers: {args.num_workers}")

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        step_t0 = time.time()

        for step, (x, y) in enumerate(loader, start=1):
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % args.print_every == 0 or step == 1:
                dt = time.time() - step_t0
                images_per_sec = (args.print_every if step != 1 else 1) * args.batch_size / max(dt, 1e-9)

                if device.type == "cuda":
                    # synchronize for accurate timing/memory reporting
                    torch.cuda.synchronize()
                    mem_alloc = torch.cuda.memory_allocated()
                    mem_reserved = torch.cuda.memory_reserved()
                    mem_peak = torch.cuda.max_memory_allocated()
                    mem_info = (
                        f"mem_alloc={human_bytes(mem_alloc)} "
                        f"mem_reserved={human_bytes(mem_reserved)} "
                        f"mem_peak={human_bytes(mem_peak)}"
                    )
                else:
                    mem_info = "mem=N/A (CPU)"

                print(
                    f"[Epoch {epoch}/{args.epochs}] step {step:4d}/{len(loader)} "
                    f"loss={loss.item():.4f} | {images_per_sec:8.1f} img/s | {mem_info}"
                )
                step_t0 = time.time()

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / len(loader)
        print(f"==> Epoch {epoch} done | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")

    if device.type == "cuda":
        print("Peak GPU allocated:", human_bytes(torch.cuda.max_memory_allocated()))
        print("Done.")


if __name__ == "__main__":
    main()
