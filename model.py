# model.py
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

try:
    import timm
except ImportError:
    timm = None

CLASS_NAMES = ["real", "fake"]


def get_model(pretrained=True):
    if timm is not None:
        return timm.create_model("xception", pretrained=pretrained, num_classes=2)

    # Fallback so training can run even when timm is not installed.
    # Freeze the backbone to keep CPU training much faster.
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_train_transform():
    if timm is not None:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    weights = ResNet18_Weights.DEFAULT
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
    ])


def get_eval_transform():
    if timm is not None:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    weights = ResNet18_Weights.DEFAULT
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
    ])
