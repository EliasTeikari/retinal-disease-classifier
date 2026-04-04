# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Retinal disease classifier using a fine-tuned Vision Transformer (ViT-Base, `google/vit-base-patch16-224`) on the ODIR-5K dataset. Classifies fundus photographs into 8 categories: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (requires ODIR-5K dataset downloaded via kagglehub or Kaggle)
python src/train.py --data_dir /path/to/odir-5k --epochs 10 --batch_size 32

# Evaluate a trained model
python src/evaluate.py --model_dir checkpoints/best_model --data_dir /path/to/odir-5k

# Single-image prediction
python src/predict.py /path/to/fundus_image.jpg --model_dir checkpoints/best_model

# Launch Gradio web demo (http://localhost:7860)
python app/gradio_app.py --model_dir checkpoints/best_model
```

The primary training workflow is the Colab notebook at `notebooks/train_retinal_disease_detector.ipynb` — it handles dataset download, training, evaluation, and HuggingFace Hub upload in one place.

## Architecture

- **`src/dataset.py`** — Core data pipeline. `create_hf_datasets()` is the main entry point for Trainer-based training: returns HF-compatible datasets and class weights. `create_dataloaders()` is kept for backward compatibility with `evaluate.py`. Constants `DISEASE_CLASSES`, `DISEASE_CODES`, `NUM_CLASSES` are defined here and imported everywhere.
- **`src/train.py`** — `train()` function uses HuggingFace `Trainer` with a custom `WeightedLossTrainer` subclass for class-weighted loss. Configured with cosine LR scheduler, early stopping, and `load_best_model_at_end`. Saves best model to `checkpoints/best_model/`. Supports `--push_to_hub` for direct HuggingFace Hub upload.
- **`src/trainer_utils.py`** — Custom Trainer utilities: `WeightedLossTrainer` (class-weighted CrossEntropyLoss), `HistoryCallback` (saves per-epoch metrics to `history.json`), and `compute_metrics` (accuracy).
- **`src/evaluate.py`** — Generates classification report, per-class AUC (one-vs-rest), confusion matrix plot, and training curve plots. Outputs go to `results/`.
- **`src/predict.py`** — Single-image inference. `predict_image()` returns predicted class, confidence, and all probabilities.
- **`app/gradio_app.py`** — Web demo. Adds `src/` to `sys.path` to import from dataset module. Includes disease descriptions and risk levels in the UI.

## Key Design Decisions

- All scripts use relative imports from `src/dataset.py` — the Gradio app achieves this via `sys.path.insert`. When running `src/` scripts, run them from within the `src/` directory or ensure `src/` is on the Python path.
- Model checkpoints, data files, and results are gitignored (see `.gitignore`). Only code and the notebook are tracked.
- The model uses HuggingFace's `ViTForImageClassification` with `ignore_mismatched_sizes=True` to replace the classification head. Models are saved/loaded via `save_pretrained()`/`from_pretrained()`. Label mappings are set in `model.config.id2label`/`label2id` for Hub compatibility.
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is used for all transforms.
