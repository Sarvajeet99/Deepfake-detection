import os
from multiprocessing import freeze_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from model import CLASS_NAMES, get_model, get_train_transform

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if torch.version.cuda is None:
            print("CUDA is not available because this PyTorch build is CPU-only.")
            print(f"Installed torch version: {torch.__version__}")
        else:
            print("CUDA-capable PyTorch is installed, but no GPU was detected.")

    transform = get_train_transform()
    dataset = DeepfakeDataset("data", transform=transform)
    num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=1e-4,
    )

    print(f"Training on {device} with {len(dataset)} images")
    print(f"DataLoader workers: {num_workers}")

    for epoch in range(3):
        model.train()
        running_loss = 0.0

        for step, (imgs, labels) in enumerate(loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 10 == 0 or step == len(loader):
                avg_loss = running_loss / step
                print(f"Epoch {epoch+1} Step {step}/{len(loader)} Loss {avg_loss:.4f}")

        print(f"Epoch {epoch+1} done")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
        },
        "model.pth",
    )


if __name__ == "__main__":
    freeze_support()
    main()
