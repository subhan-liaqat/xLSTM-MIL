from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xlstm_mil.training.subsample import stride_subsample_hilbert


def evaluate_model(model, dataset_obj, indices, device: torch.device, max_len: int):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for idx in indices:
            bag = dataset_obj[int(idx)]
            feats = stride_subsample_hilbert(bag["features"], max_len)

            x = feats.unsqueeze(0).to(device)
            y = int(bag["label"].item())
            logit = model(x).squeeze().item()
            y_true.append(y)
            y_prob.append(1.0 / (1.0 + math.exp(-logit)))

    y_true = np.array(y_true, dtype=np.int64)
    y_prob = np.array(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "avg_precision": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    return metrics, y_true, y_prob
