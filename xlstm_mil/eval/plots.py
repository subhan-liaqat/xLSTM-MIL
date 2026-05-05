from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from xlstm_mil.data.dataset import WSIFeatureDataset
from xlstm_mil.training.subsample import subsample_bag_feats_coords


def plot_roc_curve(
    y_true,
    y_prob,
    metrics: dict,
    out_path: Path | None = None,
    show: bool = False,
):
    plt.figure(figsize=(6, 5))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    else:
        plt.plot([0, 1], [0, 1], "k--", label="Insufficient class diversity")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_pr_curve(y_true, y_prob, metrics: dict, out_path: Path | None = None, show: bool = False):
    plt.figure(figsize=(6, 5))
    if len(np.unique(y_true)) > 1:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(rec, prec, linewidth=2, label=f"AP = {metrics['avg_precision']:.3f}")
    else:
        plt.plot([0, 1], [np.mean(y_true), np.mean(y_true)], "k--", label="Insufficient class diversity")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_seq_memory_scaling(mem_df, out_path: Path | None = None, show: bool = False):
    if mem_df.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(mem_df["seq_len"], mem_df["peak_mem_gb"], marker="o", linewidth=2, label="xLSTM-MIL (chunkwise)")
    plt.xlabel("Sequence Length (patches)")
    plt.ylabel("Peak GPU Memory (GB)")
    plt.title("Memory Scaling vs Sequence Length")
    plt.grid(alpha=0.25)
    plt.legend()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    gpu_peak_mem_gb: list[float],
    out_dir: Path | None = None,
    show: bool = False,
):
    epochs_axis = np.arange(1, len(gpu_peak_mem_gb) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs_axis, gpu_peak_mem_gb, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Peak GPU Memory (GB)")
    plt.title("GPU Memory Scaling")
    plt.grid(alpha=0.2)
    if out_dir:
        plt.savefig(out_dir / "gpu_memory_epochs.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs_axis, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs_axis, val_losses, marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    if out_dir:
        plt.savefig(out_dir / "train_val_loss.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def patch_input_grad_saliency(model, feats, device, max_len=None):
    """Gradient of scalar logit w.r.t. each patch embedding (input-space saliency)."""
    import torch

    if max_len is not None:
        feats = subsample_bag_feats_coords(feats, None, max_len)[0]
    model.eval()
    x = feats.unsqueeze(0).to(device).detach().clone().requires_grad_(True)
    logit = model(x).squeeze()
    if logit.ndim != 0:
        logit = logit.reshape(())
    model.zero_grad(set_to_none=True)
    logit.backward()
    g = x.grad.detach()[0]
    importance = g.abs().mean(dim=1).cpu().numpy()
    return importance, float(logit.item())


def plot_slide_saliency_scatter(
    coords_xy,
    importance,
    title="",
    max_points=5000,
    top_k_mark=10,
    s=8,
    cmap="magma",
    seed: int = 42,
    out_path: Path | None = None,
    show: bool = False,
):
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    imp = np.asarray(importance, dtype=np.float64)
    if coords_xy.shape[0] != imp.shape[0]:
        raise ValueError(f"coords {coords_xy.shape[0]} vs importance {imp.shape[0]}")

    rng = np.random.default_rng(seed)
    n = coords_xy.shape[0]
    if n > max_points:
        sub = rng.choice(n, size=max_points, replace=False)
        cx, cy, z = coords_xy[sub, 0], coords_xy[sub, 1], imp[sub]
    else:
        cx, cy, z = coords_xy[:, 0], coords_xy[:, 1], imp

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(cx, cy, c=z, s=s, cmap=cmap, linewidths=0, alpha=0.85)
    plt.colorbar(sc, label="|d logit / d patch_emb| (mean abs dim)")
    if top_k_mark and n >= top_k_mark:
        top = np.argsort(-imp)[:top_k_mark]
        plt.scatter(
            coords_xy[top, 0],
            coords_xy[top, 1],
            s=40,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.2,
            label=f"top-{top_k_mark} patches",
        )
        plt.legend(loc="best")
    plt.gca().invert_yaxis()
    plt.xlabel("x (patch grid)")
    plt.ylabel("y (patch grid)")
    plt.title(title or "Patch saliency (input gradients)")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_saliency_demo(
    model,
    dataset: WSIFeatureDataset,
    val_idx: np.ndarray,
    device,
    max_seq_len: int,
    seed: int,
    out_path: Path | None = None,
    show: bool = False,
):
    demo_idx = int(val_idx[0]) if len(val_idx) > 0 else 0
    bag_demo = dataset.load_bag_with_coords(demo_idx)
    feats_d, coords_d = subsample_bag_feats_coords(bag_demo["features"], bag_demo["coords"], max_seq_len)
    imp, logit0 = patch_input_grad_saliency(model, feats_d, device, max_len=None)
    print(
        f"Saliency demo | slide={bag_demo['slide_id']} | patches={len(imp)} | logit={logit0:.4f} | label={bag_demo['label'].item():.0f}"
    )
    plot_slide_saliency_scatter(
        coords_d.numpy(),
        imp,
        title=f"Saliency | {bag_demo['slide_id']} | Hilbert-ordered subsample (n={len(imp)})",
        max_points=6000,
        top_k_mark=10,
        seed=seed,
        out_path=out_path,
        show=show,
    )
