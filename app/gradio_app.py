"""
Gradio web app for eye disease detection.
Auto-detects model type (fundus or OCT) from model config.

Usage:
    python app/gradio_app.py --model_dir checkpoints/best_model

Deploy to HuggingFace Spaces:
    1. Push this repo to a HF Space
    2. Set the model path or upload the model to the Space
"""

import os
import sys
import argparse

import torch
import gradio as gr
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_utils import get_val_transforms


# Disease descriptions for ODIR fundus model
FUNDUS_DISEASE_INFO = {
    "Normal": "No abnormalities detected in the fundus image.",
    "Diabetes": "Signs of diabetic retinopathy — damage to blood vessels in the retina caused by diabetes.",
    "Glaucoma": "Signs of glaucoma — increased pressure damaging the optic nerve.",
    "Cataract": "Signs of cataract — clouding of the eye's natural lens.",
    "AMD": "Signs of age-related macular degeneration — deterioration of the central retina.",
    "Hypertension": "Signs of hypertensive retinopathy — retinal damage from high blood pressure.",
    "Myopia": "Signs of pathological myopia — severe nearsightedness causing retinal changes.",
    "Other": "Other abnormalities detected that don't fit standard categories.",
}

FUNDUS_RISK_COLORS = {
    "Normal": "green",
    "Diabetes": "red",
    "Glaucoma": "red",
    "Cataract": "orange",
    "AMD": "red",
    "Hypertension": "orange",
    "Myopia": "orange",
    "Other": "yellow",
}

# Disease descriptions for Kermany OCT model
OCT_DISEASE_INFO = {
    "CNV": "Choroidal neovascularization — abnormal blood vessel growth beneath the retina, associated with wet AMD. Requires urgent referral.",
    "DME": "Diabetic macular edema — fluid accumulation in the macula from diabetic retinopathy. Requires treatment to prevent vision loss.",
    "DRUSEN": "Drusen deposits — yellow deposits under the retina, associated with dry age-related macular degeneration. Monitor for progression.",
    "NORMAL": "No abnormalities detected in the OCT scan.",
}

OCT_RISK_COLORS = {
    "CNV": "red",
    "DME": "red",
    "DRUSEN": "orange",
    "NORMAL": "green",
}


def load_model(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    # Detect class names from model config
    if model.config.id2label and len(model.config.id2label) > 0:
        class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    else:
        from dataset import DISEASE_CLASSES
        class_names = DISEASE_CLASSES

    return model, device, class_names


def detect_model_type(class_names):
    """Detect whether this is a fundus or OCT model from class names."""
    oct_classes = {"CNV", "DME", "DRUSEN", "NORMAL"}
    if set(class_names) == oct_classes:
        return "oct"
    return "fundus"


@torch.no_grad()
def predict(image, model, device, class_names):
    transform = get_val_transforms(224)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(tensor).logits
    probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


def create_app(model_dir):
    model, device, class_names = load_model(model_dir)
    model_type = detect_model_type(class_names)

    # Select metadata based on model type
    if model_type == "oct":
        disease_info = OCT_DISEASE_INFO
        risk_colors = OCT_RISK_COLORS
        title = "OCT Disease Classifier"
        description = (
            "Upload an OCT B-scan to classify into 4 categories: "
            "CNV (wet AMD), DME (diabetic macular edema), DRUSEN (dry AMD), or NORMAL.\n\n"
            "**Model:** Fine-tuned Vision Transformer (ViT-Base) on Kermany OCT dataset."
        )
        input_label = "Upload OCT B-Scan"
    else:
        disease_info = FUNDUS_DISEASE_INFO
        risk_colors = FUNDUS_RISK_COLORS
        title = "Retinal Disease Detector"
        description = (
            "Upload a retinal fundus photograph to screen for 8 eye conditions: "
            "Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, "
            "Pathological Myopia, and Other abnormalities.\n\n"
            "**Model:** Fine-tuned Vision Transformer (ViT-Base) on ODIR-5K dataset."
        )
        input_label = "Upload Fundus Image"

    def classify(image):
        if image is None:
            return {}, "Please upload an image."

        probs = predict(image, model, device, class_names)
        top_class = max(probs, key=probs.get)
        confidence = probs[top_class]

        # Build info text
        risk = risk_colors.get(top_class, "yellow")
        info = f"## {top_class} ({confidence*100:.1f}% confidence)\n\n"
        info += f"{disease_info.get(top_class, '')}\n\n"

        if risk == "green":
            info += "**Risk Level:** Low\n\n"
        elif risk == "red":
            info += "**Risk Level:** High\n\n"
        else:
            info += "**Risk Level:** Medium\n\n"

        info += "---\n\n"
        info += "*This is a screening tool only. Always consult a qualified ophthalmologist for diagnosis.*"

        return probs, info

    app = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label=input_label),
        outputs=[
            gr.Label(num_top_classes=len(class_names), label="Disease Probabilities"),
            gr.Markdown(label="Analysis"),
        ],
        title=title,
        description=description,
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
