# Retinal Disease Detection with Vision Transformer

Fine-tuned Vision Transformer (ViT-Base) that detects 8 eye diseases from retinal fundus photographs.

## Diseases Detected

| Code | Disease | Description |
|------|---------|-------------|
| N | Normal | No abnormalities |
| D | Diabetic Retinopathy | Retinal blood vessel damage from diabetes |
| G | Glaucoma | Optic nerve damage from increased pressure |
| C | Cataract | Clouding of the eye's lens |
| A | AMD | Age-related macular degeneration |
| H | Hypertension | Retinal damage from high blood pressure |
| M | Myopia | Pathological nearsightedness |
| O | Other | Other abnormalities |

## Quick Start

### Train on Google Colab (Recommended)

1. Open `notebooks/train_retinal_disease_detector.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Get a Kaggle API key from https://www.kaggle.com/settings
4. Run all cells — training takes ~1 hour on T4

### Train Locally

```bash
pip install -r requirements.txt

# Download dataset first via kagglehub or manually from Kaggle
python src/train.py --data_dir /path/to/odir-5k --epochs 10 --batch_size 32
```

### Run Inference

```bash
python src/predict.py /path/to/fundus_image.jpg --model_dir checkpoints/best_model
```

### Launch Web Demo

```bash
python app/gradio_app.py --model_dir checkpoints/best_model
# Opens at http://localhost:7860
```

Add `--share` for a public link.

## Project Structure

```
├── notebooks/
│   └── train_retinal_disease_detector.ipynb   # Main Colab notebook (start here)
├── src/
│   ├── dataset.py          # Dataset loading & preprocessing
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation & metrics
│   └── predict.py          # Single-image inference
├── app/
│   └── gradio_app.py       # Gradio web demo
├── requirements.txt
└── README.md
```

## Model Details

- **Architecture:** ViT-Base (86M parameters) — `google/vit-base-patch16-224`
- **Dataset:** ODIR-5K (10,000 fundus images, 5,000 patients)
- **Training:** Full fine-tuning with weighted loss for class imbalance
- **Augmentation:** Random flip, rotation, color jitter, affine transforms
- **Expected Performance:** 85-92% accuracy, 95%+ AUC per class

## Deploy to HuggingFace Spaces

1. Push the trained model to HuggingFace Hub (see notebook section 10)
2. Create a new Space on huggingface.co/spaces
3. Upload `app/gradio_app.py` and the model files
4. Free hosted demo!

## Disclaimer

This is a screening/educational tool, not a medical diagnostic device. Always consult a qualified ophthalmologist for diagnosis and treatment decisions.
