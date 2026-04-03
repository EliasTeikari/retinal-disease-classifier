"""
Evaluation script — load a trained model and generate metrics, confusion matrix, and classification report.
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from transformers import ViTForImageClassification
from tqdm import tqdm

from dataset import create_dataloaders, DISEASE_CLASSES, NUM_CLASSES


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference on a dataloader, return all predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images).logits
        probs = torch.softmax(outputs, dim=1)

        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Retinal Disease Classification")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_training_history(history_path, save_path=None):
    """Plot training curves from saved history JSON."""
    with open(history_path) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-", label="Train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    plt.show()


def evaluate(model_dir, data_dir, output_dir="results", batch_size=32, image_size=224):
    """Full evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ViTForImageClassification.from_pretrained(model_dir)
    model = model.to(device)

    # Load test data
    _, _, test_loader, _ = create_dataloaders(
        data_dir, batch_size=batch_size, image_size=image_size,
    )

    # Get predictions
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=DISEASE_CLASSES, digits=4)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Per-class AUC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        auc_scores = {}
        for i, name in enumerate(DISEASE_CLASSES):
            if y_true_bin[:, i].sum() > 0:
                auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                auc_scores[name] = auc
                print(f"  AUC {name}: {auc:.4f}")
        mean_auc = np.mean(list(auc_scores.values()))
        print(f"  Mean AUC: {mean_auc:.4f}")
    except Exception as e:
        print(f"Could not compute AUC: {e}")

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, DISEASE_CLASSES,
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Training history (if available)
    history_path = os.path.join(os.path.dirname(model_dir), "history.json")
    if os.path.exists(history_path):
        plot_training_history(
            history_path,
            save_path=os.path.join(output_dir, "training_curves.png"),
        )

    # Overall accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\nOverall Test Accuracy: {100*accuracy:.2f}%")

    # Save results summary
    results = {
        "accuracy": float(accuracy),
        "num_test_samples": len(y_true),
        "per_class_auc": {k: float(v) for k, v in auc_scores.items()} if "auc_scores" in dir() else {},
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retinal disease classifier")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ODIR dataset directory")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    evaluate(args.model_dir, args.data_dir, args.output_dir, args.batch_size)
