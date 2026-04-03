"""
Gradio web app for retinal disease detection.
Upload a fundus image and get disease predictions with confidence scores.

Usage:
    python app/gradio_app.py --model_dir checkpoints/best_model

Deploy to HuggingFace Spaces:
    1. Push this repo to a HF Space
    2. Set the model path or upload the model to the Space
"""

import os
import sys
import json
import argparse

import torch
import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import DISEASE_CLASSES, get_val_transforms


# Disease descriptions for the UI
DISEASE_INFO = {
    "Normal": "No abnormalities detected in the fundus image.",
    "Diabetes": "Signs of diabetic retinopathy — damage to blood vessels in the retina caused by diabetes.",
    "Glaucoma": "Signs of glaucoma — increased pressure damaging the optic nerve.",
    "Cataract": "Signs of cataract — clouding of the eye's natural lens.",
    "AMD": "Signs of age-related macular degeneration — deterioration of the central retina.",
    "Hypertension": "Signs of hypertensive retinopathy — retinal damage from high blood pressure.",
    "Myopia": "Signs of pathological myopia — severe nearsightedness causing retinal changes.",
    "Other": "Other abnormalities detected that don't fit standard categories.",
}

RISK_COLORS = {
    "Normal": "green",
    "Diabetes": "red",
    "Glaucoma": "red",
    "Cataract": "orange",
    "AMD": "red",
    "Hypertension": "orange",
    "Myopia": "orange",
    "Other": "yellow",
}


def load_model(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def predict(image, model, device):
    transform = get_val_transforms(224)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(tensor).logits
    probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    return {DISEASE_CLASSES[i]: float(probs[i]) for i in range(len(DISEASE_CLASSES))}


def create_app(model_dir):
    model, device = load_model(model_dir)

    def classify(image):
        if image is None:
            return {}, "Please upload a fundus image."

        probs = predict(image, model, device)
        top_class = max(probs, key=probs.get)
        confidence = probs[top_class]

        # Build info text
        risk = RISK_COLORS.get(top_class, "yellow")
        info = f"## {top_class} ({confidence*100:.1f}% confidence)\n\n"
        info += f"{DISEASE_INFO.get(top_class, '')}\n\n"

        if top_class == "Normal":
            info += "**Risk Level:** Low\n\n"
        else:
            info += f"**Risk Level:** {'High' if risk == 'red' else 'Medium'}\n\n"

        info += "---\n\n"
        info += "*This is a screening tool only. Always consult a qualified ophthalmologist for diagnosis.*"

        return probs, info

    app = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Upload Fundus Image"),
        outputs=[
            gr.Label(num_top_classes=8, label="Disease Probabilities"),
            gr.Markdown(label="Analysis"),
        ],
        title="Retinal Disease Detector",
        description=(
            "Upload a retinal fundus photograph to screen for 8 eye conditions: "
            "Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, "
            "Pathological Myopia, and Other abnormalities.\n\n"
            "**Model:** Fine-tuned Vision Transformer (ViT-Base) on ODIR-5K dataset."
        ),
        examples=[],
        flagging_mode="never",
    )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="checkpoints/best_model")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = create_app(args.model_dir)
    app.launch(server_port=args.port, share=args.share)
