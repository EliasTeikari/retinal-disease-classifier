"""
Shared data utilities for image classification pipelines.
Contains transforms, generic Dataset classes, and class-balancing helpers.
Used by both ODIR (dataset.py) and OCT (oct_dataset.py) modules.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


# ImageNet normalization (required for pretrained ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224):
    """Augmented transforms for fundus image training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_oct_train_transforms(image_size=224):
    """Augmented transforms tuned for OCT B-scans.

    Key differences from fundus:
    - No vertical flip (OCT has fixed top-bottom orientation)
    - No saturation/hue jitter (grayscale images)
    - Smaller rotation range
    - GaussianBlur for scan quality variation
    - RandomErasing for robustness
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size=224):
    """Non-augmented transforms for validation and test sets."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageClassificationDataset(Dataset):
    """Generic PyTorch Dataset for image classification.

    Expects a DataFrame with 'image_path' and 'label' columns.
    Returns (image_tensor, label) tuples.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


class HFImageClassificationDataset(Dataset):
    """Dataset returning dicts compatible with HuggingFace Trainer.

    Expects a DataFrame with 'image_path' and 'label' columns.
    Returns {"pixel_values": tensor, "labels": int} dicts.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}


def compute_class_weights(labels, num_classes):
    """Compute inverse frequency class weights for imbalanced classes."""
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def get_weighted_sampler(labels, num_classes):
    """Create a WeightedRandomSampler for balanced training."""
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    sample_weights = 1.0 / class_counts[labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
