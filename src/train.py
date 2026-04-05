"""
Training script for image classification using ViT.
Uses HuggingFace Trainer for training orchestration.
Supports both ODIR-5K (fundus) and Kermany OCT datasets.
"""

import os
import argparse
import json
import torch
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)

from trainer_utils import WeightedLossTrainer, HistoryCallback, compute_metrics


def create_model(num_classes, class_names, pretrained="google/vit-base-patch16-224", freeze_backbone=False):
    """
    Load a pretrained ViT and replace the classification head.

    Args:
        num_classes: Number of output classes
        class_names: List of class name strings
        pretrained: HuggingFace model ID
        freeze_backbone: If True, freeze all layers except the classifier head
    """
    model = ViTForImageClassification.from_pretrained(
        pretrained,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Set label mapping in model config for push_to_hub compatibility
    model.config.id2label = {i: name for i, name in enumerate(class_names)}
    model.config.label2id = {name: i for i, name in enumerate(class_names)}

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {pretrained}")
    print(f"Parameters — Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    return model


def train(
    data_dir,
    output_dir="checkpoints",
    pretrained="google/vit-base-patch16-224",
    freeze_backbone=False,
    dataset_type="odir",
    epochs=10,
    batch_size=32,
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1,
    image_size=224,
    fp16=True,
    patience=5,
    num_workers=4,
    push_to_hub=False,
    hub_model_id=None,
):
    """Main training function using HuggingFace Trainer."""
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset based on type
    if dataset_type == "oct":
        from oct_dataset import create_oct_hf_datasets, OCT_CLASSES, OCT_NUM_CLASSES
        class_names = OCT_CLASSES
        num_classes = OCT_NUM_CLASSES
        train_dataset, val_dataset, test_dataset, class_weights = create_oct_hf_datasets(
            data_dir, image_size=image_size,
        )
    else:
        from dataset import create_hf_datasets, DISEASE_CLASSES, NUM_CLASSES
        class_names = DISEASE_CLASSES
        num_classes = NUM_CLASSES
        train_dataset, val_dataset, test_dataset, class_weights = create_hf_datasets(
            data_dir, image_size=image_size,
        )

    # Model
    model = create_model(num_classes, class_names, pretrained, freeze_backbone)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=fp16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        report_to="none",
    )

    # Callbacks
    history_callback = HistoryCallback(output_dir)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=patience)

    # Trainer with class-weighted loss
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[history_callback, early_stopping],
    )

    # Train
    trainer.train()

    # Save best model
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)

    # Save label mapping
    label_map = {i: name for i, name in enumerate(class_names)}
    with open(os.path.join(best_model_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Push to Hub
    if push_to_hub:
        trainer.push_to_hub()

    history = history_callback.get_history()
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    print(f"\nTraining complete. Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"Model saved to {best_model_dir}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier (ODIR fundus or Kermany OCT)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for model")
    parser.add_argument("--dataset", type=str, default="odir", choices=["odir", "oct"],
                        help="Dataset type: 'odir' for fundus, 'oct' for Kermany OCT")
    parser.add_argument("--pretrained", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ViT backbone, train only classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HuggingFace Hub model ID")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dataset_type=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        image_size=args.image_size,
        fp16=not args.no_fp16,
        patience=args.patience,
        num_workers=args.num_workers,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
