"""Evaluate a trained model on a separate test set and compute common metrics."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_curve,
    auc,
)

from data.dataset import RaceWindowDataset


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def run(self, files: Sequence[Path], **dataset_kwargs):
        ds = RaceWindowDataset(files, randomize=False, **dataset_kwargs)
        loader = torch.utils.data.DataLoader(ds, batch_size=64)

        self.model.eval()
        y_true, y_pred_prob, running_loss = [], [], 0.0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss   = self.loss_fn(logits, y)
                running_loss += loss.item() * X.size(0)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred_prob.append(probs)
                y_true.append(y.cpu().numpy())

        y_pred_prob = np.concatenate(y_pred_prob)
        y_true_cls  = np.concatenate(y_true)
        avg_loss    = running_loss / len(ds)

        # winner accuracy (macro)
        macro_acc = accuracy_score(y_true_cls, np.argmax(y_pred_prob, axis=1))

        # micro‑level metrics (flattened per‑horse)
        n_classes = y_pred_prob.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true_cls]
        y_true_flat   = y_true_onehot.ravel()
        y_pred_flat   = y_pred_prob.ravel()

        micro = {
            "avg_precision": average_precision_score(y_true_flat, y_pred_flat),
            "brier_score":   brier_score_loss(y_true_flat, y_pred_flat),
            "log_loss":      log_loss(y_true_flat, y_pred_flat),
        }
        fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
        precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat)
        micro["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc(fpr, tpr)}
        micro["pr_curve"]  = {"precision": precision.tolist(), "recall": recall.tolist()}

        return {
            "val_loss": avg_loss,
            "winner_accuracy": macro_acc,
            **micro,
        }
