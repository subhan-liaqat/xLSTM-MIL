from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split

from xlstm_mil.config import ModelConfig, TrainConfig
from xlstm_mil.data.dataset import WSIFeatureDataset
from xlstm_mil.model.xlstmmil import build_xlstmmil
from xlstm_mil.training.subsample import prepare_bag_features, stride_subsample_hilbert


@dataclass
class TrainingRun:
    model: nn.Module
    dataset: WSIFeatureDataset
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_losses: list[float]
    val_losses: list[float]
    lr_hist: list[float]
    gpu_peak_mem_gb: list[float]
    feature_dim: int


def _resolve_train_val_indices(
    dataset: WSIFeatureDataset, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(len(dataset), dtype=np.int64)
    all_labels = np.array([int(dataset[i]["label"].item()) for i in all_idx], dtype=np.int64)

    train_idx = dataset.indices_for_split("train")
    val_idx = dataset.indices_for_split("test")

    if len(train_idx) == 0:
        raise RuntimeError("No training samples detected. Check manifest split labels or slide naming.")

    if len(val_idx) == 0:
        if getattr(dataset, "task", "") == "tcga_nsclc":
            groups = np.array(dataset.patient_ids())
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr, va = next(splitter.split(all_idx, all_labels, groups=groups))
            train_idx, val_idx = all_idx[tr], all_idx[va]
        else:
            train_idx, val_idx = train_test_split(
                all_idx,
                test_size=0.2,
                random_state=seed,
                stratify=all_labels,
            )

    print(f"Train slides: {len(train_idx)} | Eval slides: {len(val_idx)}")
    return train_idx, val_idx


def run_fit(
    model: nn.Module,
    dataset: WSIFeatureDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[nn.Module, dict]:
    train_labels = np.array([int(dataset[i]["label"].item()) for i in train_idx], dtype=np.int64)
    num_pos = int((train_labels == 1).sum())
    num_neg = int((train_labels == 0).sum())
    if num_pos == 0:
        raise RuntimeError("No positive training slides found; cannot compute pos_weight.")
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_losses, val_losses, lr_hist, gpu_peak_mem_gb = [], [], [], []
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_train = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        order = np.random.permutation(train_idx)
        for idx in order:
            bag = dataset[int(idx)]
            feats = prepare_bag_features(bag["features"], cfg.max_seq_len)

            x = feats.unsqueeze(0).to(device)
            y = bag["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            epoch_train += float(loss.item())

        epoch_train /= len(train_idx)
        train_losses.append(epoch_train)

        model.eval()
        epoch_val = 0.0
        val_true, val_prob = [], []
        with torch.no_grad():
            for idx in val_idx:
                bag = dataset[int(idx)]
                feats = stride_subsample_hilbert(bag["features"], cfg.max_seq_len)

                x = feats.unsqueeze(0).to(device)
                y = bag["label"].to(device)
                logits = model(x).squeeze(-1)
                epoch_val += float(criterion(logits, y).item())
                val_true.append(float(y.item()))
                val_prob.append(float(torch.sigmoid(logits).item()))

        epoch_val /= len(val_idx)
        val_losses.append(epoch_val)

        val_true = np.array(val_true, dtype=np.float32)
        val_prob = np.array(val_prob, dtype=np.float32)
        val_pred = (val_prob >= 0.5).astype(np.int64)
        val_acc = float((val_pred == val_true.astype(np.int64)).mean())
        if len(np.unique(val_true)) > 1:
            val_roc_auc = float(roc_auc_score(val_true, val_prob))
            val_pr_auc = float(average_precision_score(val_true, val_prob))
        else:
            val_roc_auc = float("nan")
            val_pr_auc = float("nan")

        lr_now = optimizer.param_groups[0]["lr"]
        lr_hist.append(lr_now)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0
        gpu_peak_mem_gb.append(float(peak_mem))

        if epoch_val < best_val - cfg.early_stop_min_delta:
            best_val = epoch_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | LR {lr_now:.2e} | Train {epoch_train:.4f} | "
            f"EvalLoss {epoch_val:.4f} | EvalAcc {val_acc:.4f} | EvalROC-AUC {val_roc_auc:.4f} | "
            f"EvalPR-AUC {val_pr_auc:.4f} | PeakGPU {peak_mem:.3f} GB"
        )

        scheduler.step()

        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}/{cfg.epochs} "
                f"(no eval loss improvement for {cfg.early_stop_patience} epochs; "
                f"best eval loss={best_val:.4f})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model weights based on eval loss.")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lr_hist": lr_hist,
        "gpu_peak_mem_gb": gpu_peak_mem_gb,
    }
    return model, history


def run_training_pipeline(
    dataset: WSIFeatureDataset,
    device: torch.device,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    feature_dim: int,
) -> TrainingRun:
    all_idx = np.arange(len(dataset), dtype=np.int64)
    all_labels = np.array([int(dataset[i]["label"].item()) for i in all_idx], dtype=np.int64)

    train_idx, val_idx = _resolve_train_val_indices(dataset, train_cfg.seed)

    cv_histories = []

    if train_cfg.k_fold >= 2:
        print(f"Running {train_cfg.k_fold}-fold stratified CV on {len(all_idx)} slides.")
        skf = StratifiedKFold(n_splits=train_cfg.k_fold, shuffle=True, random_state=train_cfg.seed)
        cv_fold_rocs = []
        model = None
        for fold, (tr, va) in enumerate(skf.split(all_idx, all_labels)):
            print(f"--- Fold {fold + 1}/{train_cfg.k_fold} ---")
            mdl = build_xlstmmil(feature_dim, model_cfg).to(device)
            mdl, hist = run_fit(mdl, dataset, all_idx[tr], all_idx[va], device, train_cfg)
            cv_histories.append(hist)

            val_true, val_prob = [], []
            mdl.eval()
            with torch.no_grad():
                for idx in all_idx[va]:
                    bag = dataset[int(idx)]
                    feats = stride_subsample_hilbert(bag["features"], train_cfg.max_seq_len)
                    x = feats.unsqueeze(0).to(device)
                    logits = mdl(x).squeeze(-1)
                    val_true.append(float(bag["label"].item()))
                    val_prob.append(float(torch.sigmoid(logits).item()))
            val_true = np.array(val_true, dtype=np.float32)
            val_prob = np.array(val_prob, dtype=np.float32)
            roc = float(roc_auc_score(val_true, val_prob)) if len(np.unique(val_true)) > 1 else float("nan")
            cv_fold_rocs.append(roc)
            print(f"Fold {fold + 1} holdout ROC-AUC: {roc:.4f}")
            model = mdl

        min_len = min(len(h["train_losses"]) for h in cv_histories)
        train_losses = np.mean([h["train_losses"][:min_len] for h in cv_histories], axis=0).tolist()
        val_losses = np.mean([h["val_losses"][:min_len] for h in cv_histories], axis=0).tolist()
        lr_hist = np.mean([h["lr_hist"][:min_len] for h in cv_histories], axis=0).tolist()
        gpu_peak_mem_gb = np.mean([h["gpu_peak_mem_gb"][:min_len] for h in cv_histories], axis=0).tolist()

        cv_results_df = pd.DataFrame({"fold": np.arange(1, train_cfg.k_fold + 1), "roc_auc": cv_fold_rocs})
        print(cv_results_df)
        print(f"CV mean ROC-AUC: {float(np.nanmean(cv_fold_rocs)):.4f}")

        return TrainingRun(
            model=model,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            train_losses=train_losses,
            val_losses=val_losses,
            lr_hist=lr_hist,
            gpu_peak_mem_gb=gpu_peak_mem_gb,
            feature_dim=feature_dim,
        )

    model = build_xlstmmil(feature_dim, model_cfg).to(device)
    print(model.__class__.__name__)
    if len(model.blocks) > 0:
        print("First mLSTM block:")
        print(model.blocks[0])

    model, hist = run_fit(model, dataset, train_idx, val_idx, device, train_cfg)
    return TrainingRun(
        model=model,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        train_losses=hist["train_losses"],
        val_losses=hist["val_losses"],
        lr_hist=hist["lr_hist"],
        gpu_peak_mem_gb=hist["gpu_peak_mem_gb"],
        feature_dim=feature_dim,
    )
