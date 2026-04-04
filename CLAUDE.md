# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Retinal disease classifier using a fine-tuned Vision Transformer (ViT-Base, `google/vit-base-patch16-224`) on the ODIR-5K dataset. Classifies fundus photographs into 8 categories: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (requires ODIR-5K dataset downloaded via kagglehub or Kaggle)
# Must run from src/ directory due to relative imports
cd src && python train.py --data_dir /path/to/odir-5k --epochs 10 --batch_size 32

# Evaluate a trained model
cd src && python evaluate.py --model_dir ../checkpoints/best_model --data_dir /path/to/odir-5k

# Single-image prediction
cd src && python predict.py /path/to/fundus_image.jpg --model_dir ../checkpoints/best_model

# Launch Gradio web demo (http://localhost:7860) — runs from project root
python app/gradio_app.py --model_dir checkpoints/best_model
```

The primary training workflow is the Colab notebook at `notebooks/train_retinal_disease_detector.ipynb` — it handles dataset download, training, evaluation, and HuggingFace Hub upload in one place.

## Architecture

- **`src/dataset.py`** — Core data pipeline and single source of truth for constants (`DISEASE_CLASSES`, `DISEASE_CODES`, `NUM_CLASSES`). Has two dataset APIs:
  - `create_hf_datasets()` → returns HF-compatible datasets + class weights (used by `train.py` via Trainer)
  - `create_dataloaders()` → returns PyTorch DataLoaders (legacy path, used by `evaluate.py`)
- **`src/train.py`** — `train()` function uses HuggingFace `Trainer` with a custom `WeightedLossTrainer` subclass for class-weighted loss. Configured with cosine LR scheduler, early stopping, and `load_best_model_at_end`. Saves best model to `checkpoints/best_model/`. Supports `--push_to_hub` and `--freeze_backbone`.
- **`src/trainer_utils.py`** — Custom Trainer utilities: `WeightedLossTrainer` (class-weighted CrossEntropyLoss), `HistoryCallback` (saves per-epoch metrics to `history.json`), and `compute_metrics` (accuracy).
- **`src/evaluate.py`** — Generates classification report, per-class AUC (one-vs-rest), confusion matrix plot, and training curve plots. Outputs go to `results/`.
- **`src/predict.py`** — Single-image inference. `predict_image()` returns predicted class, confidence, and all probabilities. Loads label mapping from `label_map.json` alongside the model.
- **`app/gradio_app.py`** — Web demo. Adds `src/` to `sys.path` to import from dataset module. Includes disease descriptions and risk levels in the UI.

## Key Design Decisions

- **Import path**: All `src/` scripts use relative imports from `dataset.py` (`from dataset import ...`). Scripts in `src/` must be run from within `src/` or with `src/` on PYTHONPATH. The Gradio app (in `app/`) uses `sys.path.insert(0, "../src")` instead.
- **Class imbalance**: Handled via `WeightedLossTrainer` which applies inverse-frequency class weights to CrossEntropyLoss. An older `WeightedRandomSampler` path exists in `create_dataloaders()` for backward compatibility.
- **Model format**: Uses HuggingFace's `ViTForImageClassification` with `ignore_mismatched_sizes=True` to replace the classification head. Models saved/loaded via `save_pretrained()`/`from_pretrained()`. Label mappings stored in both `model.config.id2label`/`label2id` (for Hub) and `label_map.json` (for local inference).
- **ImageNet normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is used for all transforms.
- Model checkpoints, data files, and results are gitignored. Only code and the notebook are tracked.
