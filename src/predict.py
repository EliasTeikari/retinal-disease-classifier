"""
Single-image inference script for image classification.
Supports any trained model (ODIR fundus or Kermany OCT) via label_map.json or model config.
"""

import os
import json
import argparse
import torch
from PIL import Image
from transformers import ViTForImageClassification

from data_utils import get_val_transforms


def load_model(model_dir, device=None):
    """Load a trained model from a directory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = ViTForImageClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    # Load label map: label_map.json > model config > fallback
    label_map_path = os.path.join(model_dir, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
    elif model.config.id2label and len(model.config.id2label) > 0:
        label_map = {int(k): v for k, v in model.config.id2label.items()}
    else:
        from dataset import DISEASE_CLASSES
        label_map = {i: name for i, name in enumerate(DISEASE_CLASSES)}

    return model, label_map, device


@torch.no_grad()
def predict_image(image_path, model, label_map, device, image_size=224):
    """
    Predict class from a single image.

    Returns:
        dict with keys: predicted_class, confidence, all_probabilities
    """
    transform = get_val_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(tensor).logits
    probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    predicted_idx = probs.argmax()
    predicted_class = label_map[int(predicted_idx)]
    confidence = float(probs[predicted_idx])

    all_probs = {label_map[i]: float(p) for i, p in enumerate(probs)}
    # Sort by probability descending
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs,
    }


def format_prediction(result):
    """Pretty-print a prediction result."""
    lines = []
    lines.append(f"Prediction: {result['predicted_class']}")
    lines.append(f"Confidence: {result['confidence']*100:.1f}%")
    lines.append("")
    lines.append("All class probabilities:")
    for name, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 30)
        lines.append(f"  {name:15s} {prob*100:6.2f}% {bar}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict from a single image")
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument("--model_dir", type=str, default="checkpoints/best_model", help="Path to model directory")
    args = parser.parse_args()

    model, label_map, device = load_model(args.model_dir)
    result = predict_image(args.image_path, model, label_map, device)
    print(format_prediction(result))
