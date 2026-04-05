# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two image classification models using fine-tuned Vision Transformers (ViT-Base, `google/vit-base-patch16-224`):

1. **Retinal Fundus Classifier** — ODIR-5K dataset, 8 classes: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other
2. **OCT Disease Classifier** — Kermany OCT dataset (84K images), 4 classes: CNV, DME, DRUSEN, NORMAL. Target: 99%+ accuracy

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train ODIR fundus classifier
python src/train.py --dataset odir --data_dir /path/to/odir-5k --epochs 10 --batch_size 32

# Train OCT classifier (Kermany dataset)
python src/train.py --dataset oct --data_dir /path/to/kermany2018 --epochs 10 --batch_size 64 --warmup_ratio 0.1 --patience 3

# Evaluate a trained model
python src/evaluate.py --model_dir checkpoints/best_model --data_dir /path/to/dataset --dataset odir
python src/evaluate.py --model_dir checkpoints_oct/best_model --data_dir /path/to/kermany2018 --dataset oct

# Single-image prediction (auto-detects model type from label_map.json)
python src/predict.py /path/to/image.jpg --model_dir checkpoints/best_model

# Launch Gradio web demo (auto-detects fundus vs OCT from model config)
python app/gradio_app.py --model_dir checkpoints/best_model

# Download trained models from HuggingFace Hub
huggingface-cli download eliasteikari/retinal_disease_model --local-dir checkpoints/best_model
```

The primary training workflows are Colab notebooks:
- `notebooks/train_retinal_disease_detector.ipynb` — ODIR-5K fundus classifier
- `notebooks/train_oct_classifier.ipynb` — Kermany OCT classifier (CNV/DME/DRUSEN/NORMAL)

## HuggingFace Hub

- **Model:** [eliasteikari/retinal_disease_model](https://huggingface.co/eliasteikari/retinal_disease_model) — trained ViT weights on HF Hub
- **Space:** [eliasteikari/retina-disease](https://huggingface.co/spaces/eliasteikari/retina-disease) — Gradio demo that loads the model directly from the Hub
- Push updated model: `trainer.push_to_hub("eliasteikari/retina-disease-classifier")` from the notebook

## Architecture

- **`src/data_utils.py`** — Shared utilities: transforms (fundus + OCT-specific), generic Dataset classes (`ImageClassificationDataset`, `HFImageClassificationDataset`), `compute_class_weights()`, `get_weighted_sampler()`. Used by both dataset modules.
- **`src/dataset.py`** — ODIR-5K data pipeline. Constants (`DISEASE_CLASSES`, `DISEASE_CODES`, `NUM_CLASSES`), Excel annotation parsing, left/right eye handling. Imports and re-exports shared utilities from `data_utils.py`.
- **`src/oct_dataset.py`** — Kermany OCT data pipeline. Constants (`OCT_CLASSES`, `OCT_NUM_CLASSES`), folder-based loading. Handles the `train/`/`test/` directory structure and optional `OCT2017/` nesting.
- **`src/train.py`** — `train()` function uses HuggingFace `Trainer` with `WeightedLossTrainer`. Supports `--dataset odir|oct` flag to switch between datasets. Configurable warmup, gradient accumulation, and num_workers. Saves best model to `checkpoints/best_model/`.
- **`src/trainer_utils.py`** — Custom Trainer utilities: `WeightedLossTrainer` (class-weighted CrossEntropyLoss), `HistoryCallback` (saves per-epoch metrics to `history.json`), and `compute_metrics` (accuracy).
- **`src/evaluate.py`** — Generates classification report, per-class AUC (one-vs-rest), confusion matrix plot, and training curve plots. Auto-detects class names from `model.config.id2label`. Supports `--dataset odir|oct` for data loading.
- **`src/predict.py`** — Single-image inference. `predict_image()` returns predicted class, confidence, and all probabilities. Label map fallback chain: `label_map.json` → `model.config.id2label` → `DISEASE_CLASSES`.
- **`app/gradio_app.py`** — Web demo. Auto-detects model type (fundus vs OCT) from `model.config.id2label`. Displays appropriate disease descriptions and risk levels.

## Key Design Decisions

- **Import path**: All `src/` scripts use relative imports (`from data_utils import ...`, `from dataset import ...`). Scripts in `src/` must be run from within `src/` or with `src/` on PYTHONPATH. The Gradio app (in `app/`) uses `sys.path.insert(0, "../src")` instead.
- **Shared utilities**: `data_utils.py` contains all dataset-agnostic code (transforms, Dataset classes, weight computation). `dataset.py` and `oct_dataset.py` import from it and add dataset-specific loading logic.
- **Class imbalance**: Handled via `WeightedLossTrainer` which applies inverse-frequency class weights to CrossEntropyLoss. An older `WeightedRandomSampler` path exists in `create_dataloaders()` for backward compatibility.
- **Model format**: Uses HuggingFace's `ViTForImageClassification` with `ignore_mismatched_sizes=True` to replace the classification head. Models saved/loaded via `save_pretrained()`/`from_pretrained()`. Label mappings stored in both `model.config.id2label`/`label2id` (for Hub) and `label_map.json` (for local inference).
- **Auto-detection**: `evaluate.py`, `predict.py`, and `gradio_app.py` auto-detect the model type (fundus vs OCT) from `model.config.id2label`, so they work with any trained model without manual configuration.
- **OCT-specific augmentation**: No vertical flip (OCT has fixed top-bottom orientation), no hue/saturation jitter (grayscale images), smaller rotation range, GaussianBlur and RandomErasing added.
- **ImageNet normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) is used for all transforms (correct even for grayscale OCT converted to RGB).
- Model checkpoints, data files, and results are gitignored. Only code and the notebooks are tracked.
