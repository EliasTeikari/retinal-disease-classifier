"""
Custom HuggingFace Trainer utilities for retinal disease classification.
Provides weighted loss training, history tracking, and metrics computation.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback


class WeightedLossTrainer(Trainer):
    """Trainer subclass that uses class-weighted CrossEntropyLoss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class HistoryCallback(TrainerCallback):
    """Collects per-epoch metrics into the legacy history.json format."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self._current_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Training loss (logged per epoch with logging_strategy="epoch")
        if "loss" in logs and "eval_loss" not in logs:
            self._current_train_loss = logs["loss"]

        # Evaluation metrics (logged after each eval)
        if "eval_loss" in logs:
            self.history["val_loss"].append(logs["eval_loss"])
            self.history["val_acc"].append(logs.get("eval_accuracy", 0.0))

            # Pair with the most recent train loss
            if self._current_train_loss is not None:
                self.history["train_loss"].append(self._current_train_loss)
                # Trainer doesn't compute train accuracy by default;
                # approximate from train loss is not meaningful, so store 0.0
                # The eval accuracy is the primary metric.
                self.history["train_acc"].append(0.0)
                self._current_train_loss = None

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history(self):
        return self.history


def compute_metrics(eval_pred):
    """Compute accuracy from Trainer's EvalPrediction."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}
