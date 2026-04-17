# dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, cls)
            for file in os.listdir(folder):
                _, ext = os.path.splitext(file)
                if ext.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    self.samples.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)
