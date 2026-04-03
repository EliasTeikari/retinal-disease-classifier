"""
Dataset module for ODIR-5K retinal disease classification.
Handles loading, preprocessing, augmentation, and class balancing.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch


# ODIR-5K disease labels
DISEASE_CLASSES = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]
DISEASE_CODES = ["N", "D", "G", "C", "A", "H", "M", "O"]
NUM_CLASSES = len(DISEASE_CLASSES)


def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def parse_odir_labels(row):
    """Parse ODIR annotation row into a single class label (0-7)."""
    label_cols = ["N", "D", "G", "C", "A", "H", "M", "O"]
    for i, col in enumerate(label_cols):
        if row[col] == 1:
            return i
    return 0  # Default to Normal


def load_odir_dataset(data_dir):
    """
    Load ODIR-5K dataset from the downloaded Kaggle directory.

    Expected structure:
        data_dir/
            ODIR-5K_Training_Annotations(Updated)_V2.xlsx  (or .csv)
            ODIR-5K_Training_Images/
                0_left.jpg
                0_right.jpg
                ...

    Returns:
        DataFrame with columns: image_path, label, label_name
    """
    # Find annotation file
    anno_file = None
    for f in os.listdir(data_dir):
        if "annotation" in f.lower() or "label" in f.lower():
            anno_file = os.path.join(data_dir, f)
            break
        if f.endswith(".xlsx") or f.endswith(".csv"):
            anno_file = os.path.join(data_dir, f)

    if anno_file is None:
        raise FileNotFoundError(
            f"No annotation file found in {data_dir}. "
            "Expected an .xlsx or .csv file with ODIR labels."
        )

    # Load annotations
    if anno_file.endswith(".xlsx"):
        df = pd.read_excel(anno_file)
    else:
        df = pd.read_csv(anno_file)

    # Find image directory
    img_dir = None
    for d in os.listdir(data_dir):
        full_path = os.path.join(data_dir, d)
        if os.path.isdir(full_path) and "image" in d.lower():
            img_dir = full_path
            break
        if os.path.isdir(full_path) and "training" in d.lower():
            img_dir = full_path
            break

    if img_dir is None:
        # Images might be directly in data_dir
        img_dir = data_dir

    # Build dataset entries (one per eye image)
    records = []
    label_cols = ["N", "D", "G", "C", "A", "H", "M", "O"]

    for _, row in df.iterrows():
        label = parse_odir_labels(row)

        # Left eye
        left_img = row.get("Left-Fundus", row.get("left_fundus", None))
        if left_img and pd.notna(left_img):
            left_path = os.path.join(img_dir, str(left_img))
            if os.path.exists(left_path):
                records.append({
                    "image_path": left_path,
                    "label": label,
                    "label_name": DISEASE_CLASSES[label],
                })

        # Right eye
        right_img = row.get("Right-Fundus", row.get("right_fundus", None))
        if right_img and pd.notna(right_img):
            right_path = os.path.join(img_dir, str(right_img))
            if os.path.exists(right_path):
                records.append({
                    "image_path": right_path,
                    "label": label,
                    "label_name": DISEASE_CLASSES[label],
                })

    result_df = pd.DataFrame(records)
    print(f"Loaded {len(result_df)} images from ODIR dataset")
    print(f"Class distribution:\n{result_df['label_name'].value_counts()}")
    return result_df


class RetinalDiseaseDataset(Dataset):
    """PyTorch Dataset for retinal fundus images."""

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


def compute_class_weights(labels):
    """Compute inverse frequency class weights for imbalanced classes."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    # Avoid division by zero for classes with no samples
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.FloatTensor(weights)


def get_weighted_sampler(labels):
    """Create a WeightedRandomSampler for balanced training."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    sample_weights = 1.0 / class_counts[labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def create_dataloaders(data_dir, batch_size=32, image_size=224, val_split=0.15, test_split=0.1, num_workers=2, seed=42):
    """
    Create train/val/test dataloaders from ODIR dataset.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    df = load_odir_dataset(data_dir)

    # Stratified split
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=val_split + test_split, stratify=df["label"], random_state=seed
    )
    relative_test = test_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df["label"], random_state=seed
    )

    print(f"\nSplit sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create datasets
    train_dataset = RetinalDiseaseDataset(train_df, transform=get_train_transforms(image_size))
    val_dataset = RetinalDiseaseDataset(val_df, transform=get_val_transforms(image_size))
    test_dataset = RetinalDiseaseDataset(test_df, transform=get_val_transforms(image_size))

    # Compute class weights and sampler for training
    train_labels = train_df["label"].values
    class_weights = compute_class_weights(train_labels)
    sampler = get_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
