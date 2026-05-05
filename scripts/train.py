#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_optional_int(raw: str) -> int | None:
    value = raw.strip().lower()
    if value in {"none", "null"}:
        return None
    return int(raw)


def parse_args():
    p = argparse.ArgumentParser(description="Train xLSTM-MIL on WSI patch features.")
    p.add_argument(
        "--task",
        type=str,
        default="camelyon16",
        choices=["camelyon16", "tcga_nsclc"],
        help="Task/dataset mode. tcga_nsclc expects LUAD/LUSC labels from slide_id or CSV metadata columns.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root containing embeddings/ and patches/ (used with local data or downloads).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download CAMELYON16-UNI layout from Hugging Face into --data-root.",
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default="kaczmarj/camelyon16-uni",
        help="HF dataset repo id when --download.",
    )
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token (falls back to HF_TOKEN env var).",
    )
    p.add_argument("--embed-dir", type=Path, default=None, help="Override embeddings directory.")
    p.add_argument("--patch-dir", type=Path, default=None, help="Override patches directory.")
    p.add_argument("--process-csv", type=Path, default=None, help="Override process_list CSV path.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=12_000)
    p.add_argument(
        "--max-seq-eval",
        type=_parse_optional_int,
        default=12_000,
        help='Eval stride cap. Use "none" for full bag eval (higher VRAM).',
    )
    p.add_argument("--k-fold", type=int, default=0, help=">=2 enables stratified K-fold CV on all slides.")
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument(
        "--vision-lstm-root",
        type=Path,
        default=None,
        help="Optional checkout of NX-AI/vision-lstm for alternate mLSTM import paths.",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Save figures here; default headless Agg only if unset (no PNGs).",
    )
    p.add_argument("--show-plots", action="store_true", help="Interactive show() (needs GUI backend).")

    p.add_argument("--benchmark-memory", action="store_true", help="Run sequence-length VRAM scaling.")
    p.add_argument(
        "--saliency-demo",
        action="store_true",
        help="Compute input-gradient saliency on first val slide.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    import matplotlib

    if not args.show_plots:
        matplotlib.use("Agg")

    import numpy as np
    import torch

    from xlstm_mil.benchmark.memory import benchmark_memory_scaling
    from xlstm_mil.config import ModelConfig, TrainConfig
    from xlstm_mil.data.dataset import WSIFeatureDataset, infer_feature_dim
    from xlstm_mil.data.download import download_camelyon16_uni_layout
    from xlstm_mil.env import get_device, set_seed, setup_optional_xlstm_paths
    from xlstm_mil.eval.metrics import evaluate_model
    from xlstm_mil.eval.plots import (
        plot_loss_curves,
        plot_pr_curve,
        plot_roc_curve,
        plot_saliency_demo,
        plot_seq_memory_scaling,
    )
    from xlstm_mil.training.fit import run_training_pipeline

    setup_optional_xlstm_paths(
        vision_lstm_root=args.vision_lstm_root,
    )

    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    plots_dir = args.plots_dir
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)

    embed_dir = args.embed_dir
    patch_dir = args.patch_dir
    process_csv = args.process_csv

    if args.download:
        if args.data_root is None:
            raise SystemExit("--download requires --data-root")
        tok = args.hf_token or os.environ.get("HF_TOKEN")
        _, embed_dir2, patch_dir2, csv2 = download_camelyon16_uni_layout(
            args.data_root, repo_id=args.hf_repo, token=tok
        )
        embed_dir = embed_dir or embed_dir2
        patch_dir = patch_dir or patch_dir2
        process_csv = process_csv or csv2
    else:
        if args.data_root:
            root = Path(args.data_root)
            embed_dir = embed_dir or (root / "embeddings")
            patch_dir = patch_dir or (root / "patches")
            process_csv = process_csv or (patch_dir / "process_list_autogen.csv")

    if embed_dir is None or patch_dir is None or process_csv is None:
        raise SystemExit("Provide --data-root (with embeddings/patches layout) or all of --embed-dir, --patch-dir, --process-csv")

    dataset = WSIFeatureDataset(embed_dir, patch_dir, process_csv, task=args.task)
    feature_dim = infer_feature_dim(dataset)
    print("Detected feature dim:", feature_dim)

    train_cfg = TrainConfig(
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        max_seq_eval=args.max_seq_eval,
        weight_decay=args.weight_decay,
        k_fold=args.k_fold,
        early_stop_patience=args.early_stop_patience,
        epochs=args.epochs,
        lr=args.lr,
    )
    model_cfg = ModelConfig()

    run = run_training_pipeline(dataset, device, train_cfg, model_cfg, feature_dim)

    metrics, y_true, y_prob = evaluate_model(
        run.model,
        run.dataset,
        run.val_idx,
        device=device,
        max_len=train_cfg.max_seq_eval,
    )
    eval_split_name = "test" if len(dataset.indices_for_split("test")) > 0 else "validation"
    print(f"Final {eval_split_name.title()} Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    show = args.show_plots
    roc_path = plots_dir / "roc.png" if plots_dir else None
    pr_path = plots_dir / "pr.png" if plots_dir else None
    plot_roc_curve(y_true, y_prob, metrics, out_path=roc_path, show=show)
    plot_pr_curve(y_true, y_prob, metrics, out_path=pr_path, show=show)
    plot_loss_curves(
        run.train_losses,
        run.val_losses,
        run.gpu_peak_mem_gb,
        out_dir=plots_dir,
        show=show,
    )

    if args.saliency_demo:
        sal_path = plots_dir / "saliency.png" if plots_dir else None
        plot_saliency_demo(
            run.model,
            run.dataset,
            run.val_idx,
            device=device,
            max_seq_len=train_cfg.max_seq_len,
            seed=train_cfg.seed,
            out_path=sal_path,
            show=show,
        )

    if args.benchmark_memory:
        seq_lens = [1000, 2000, 4000, 6000, 8000]
        bench_idx = int(run.train_idx[0]) if len(run.train_idx) > 0 else 0
        mem_df = benchmark_memory_scaling(
            run.model,
            feature_dim,
            seq_lens,
            device=device,
            dataset_obj=dataset,
            index_hint=bench_idx,
            batch_size=1,
            runs_per_len=2,
        )
        if not mem_df.empty:
            mem_plot = plots_dir / "memory_scaling.png" if plots_dir else None
            plot_seq_memory_scaling(mem_df, out_path=mem_plot, show=show)
            x = mem_df["seq_len"].to_numpy(dtype=np.float64)
            y = mem_df["peak_mem_gb"].to_numpy(dtype=np.float64)
            slope, intercept = np.polyfit(x, y, 1)
            r2 = 1.0 - (np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2))
            print(f"Linear fit: mem_gb = {slope:.6f} * seq_len + {intercept:.4f} | R^2={r2:.4f}")
            print(mem_df)


if __name__ == "__main__":
    main()
