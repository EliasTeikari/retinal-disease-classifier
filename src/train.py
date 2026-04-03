"""
Training script for retinal disease classification using ViT.
Can be run standalone or imported by the Colab notebook.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import json

from dataset import create_dataloaders, DISEASE_CLASSES, NUM_CLASSES


def create_model(num_classes=NUM_CLASSES, pretrained="google/vit-base-patch16-224", freeze_backbone=False):
    """
    Load a pretrained ViT and replace the classification head.

    Args:
        num_classes: Number of output classes
        pretrained: HuggingFace model ID
        freeze_backbone: If True, freeze all layers except the classifier head
    """
    model = ViTForImageClassification.from_pretrained(
        pretrained,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if freeze_backbone:
        # Freeze everything except the classification head
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {pretrained}")
    print(f"Parameters — Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    return model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc": f"{100.*correct/total:.2f}%",
        })

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train(
    data_dir,
    output_dir="checkpoints",
    pretrained="google/vit-base-patch16-224",
    freeze_backbone=False,
    epochs=10,
    batch_size=32,
    lr=2e-5,
    weight_decay=0.01,
    image_size=224,
    fp16=True,
    patience=5,
):
    """Main training function."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_dir, batch_size=batch_size, image_size=image_size,
    )

    # Model
    model = create_model(NUM_CLASSES, pretrained, freeze_backbone)
    model = model.to(device)

    # Loss with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Mixed precision
    scaler = torch.amp.GradScaler(enabled=fp16 and device.type == "cuda")

    # Training loop
    best_val_acc = 0.0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            model.save_pretrained(os.path.join(output_dir, "best_model"))
            print(f"  -> New best model saved (val_acc: {100*val_acc:.2f}%)")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    # Save training history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save label mapping
    label_map = {i: name for i, name in enumerate(DISEASE_CLASSES)}
    with open(os.path.join(output_dir, "best_model", "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nTraining complete. Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"Model saved to {os.path.join(output_dir, 'best_model')}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retinal disease classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ODIR dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for model")
    parser.add_argument("--pretrained", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ViT backbone, train only classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        fp16=not args.no_fp16,
        patience=args.patience,
    )
