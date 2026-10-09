from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224

# ImageNet statistics (required by the pretrained networks)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        # Light augmentation. NO horizontal flip (it would move the heart side).
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class XrayDataset(Dataset):
    def __init__(self, csv_path, train: bool = False):
        self.df = pd.read_csv(csv_path)
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")  # fixes the L / RGB mix
        label = CLASS_TO_IDX[row["label"]]
        return self.transform(img), label


def compute_class_weights(csv_path):
    """Inverse-frequency weights, in the order of CLASSES."""
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().reindex(CLASSES).values.astype(float)
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_loaders(batch_size=16, num_workers=2, splits_dir="data/splits"):
    splits = Path(splits_dir)
    train_ds = XrayDataset(splits / "train.csv", train=True)
    val_ds = XrayDataset(splits / "val.csv", train=False)
    test_ds = XrayDataset(splits / "test.csv", train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_loader, _, _ = get_loaders(batch_size=16, num_workers=0)
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)   # expected [16, 3, 224, 224]
    print("Batch labels      :", labels.tolist())

    weights = compute_class_weights("data/splits/train.csv")
    print("Class order       :", CLASSES)
    print("Class weights     :", weights.round(decimals=3).tolist())

    # Show the batch (undo normalization for display)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    for ax, img, lab in zip(axes.flat, images, labels):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[lab], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("analysis/batch_preview.png", dpi=120)
    plt.show()