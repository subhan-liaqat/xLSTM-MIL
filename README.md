# xLSTM-MIL

Production-style `xLSTM-MIL` training pipeline for whole-slide image (WSI) multiple instance learning using precomputed patch embeddings (CAMELYON16-style layout).

This repository converts the notebook workflow into a modular Python package with a single CLI entrypoint.

## What This Project Includes

- End-to-end training and evaluation for binary WSI classification.
- Hilbert-ordered bag construction from `.pt` features and `.h5` patch coordinates.
- xLSTM-based MIL model (`chunkwise` backend by default).
- Early stopping and optional stratified K-fold cross-validation.
- ROC/PR/loss/memory plots.
- Optional saliency visualization and GPU memory scaling benchmark.
- Optional Hugging Face dataset download helper.

## Repository Structure

- `scripts/train.py` — main CLI runner.
- `xlstm_mil/config.py` — train/model config dataclasses.
- `xlstm_mil/env.py` — seed/device/path setup utilities.
- `xlstm_mil/data/download.py` — HF snapshot download helper.
- `xlstm_mil/data/dataset.py` — dataset pairing, split inference, Hilbert ordering.
- `xlstm_mil/model/xlstmmil.py` — xLSTM-MIL model + block resolver.
- `xlstm_mil/training/subsample.py` — deterministic sequence subsampling.
- `xlstm_mil/training/fit.py` — training loop + CV orchestration.
- `xlstm_mil/eval/metrics.py` — final metric computation.
- `xlstm_mil/eval/plots.py` — plotting and saliency utilities.
- `xlstm_mil/benchmark/memory.py` — sequence-length memory benchmark.
- `NOTEBOOK_SPLIT.md` — notebook-to-module mapping notes.

## Data Layout Expected

Either provide directories explicitly, or a `--data-root` containing:

- `embeddings/*.pt`
- `patches/**/*.h5`
- `patches/process_list_autogen.csv`

The CSV must include `slide_id`. Optional split columns supported: `split`, `set`, `subset`, `partition`.

## Quick Start

## 1) Install dependencies

```bash
pip install -e .
```

If you prefer requirements:

```bash
pip install -r requirements.txt
```

## 2) Run training with local dataset

```bash
py scripts/train.py --data-root "C:\path\to\camelyon16_uni_data"
```

## 3) Run with Hugging Face download

```bash
py scripts/train.py ^
  --download ^
  --data-root "C:\path\to\camelyon16_uni_data" ^
  --hf-repo "kaczmarj/camelyon16-uni" ^
  --hf-token "<YOUR_HF_TOKEN>"
```

## 4) Save plots

```bash
py scripts/train.py --data-root "C:\path\to\camelyon16_uni_data" --plots-dir "outputs\plots"
```

## Useful CLI Options

- `--epochs 20` — number of epochs.
- `--lr 1e-4` — AdamW learning rate.
- `--weight-decay 1e-4` — AdamW weight decay.
- `--max-seq-len 12000` — train/in-epoch eval sequence cap.
- `--max-seq-eval 12000` — final eval sequence cap.
- `--max-seq-eval none` — full-bag final eval (higher VRAM).
- `--k-fold 5` — enable stratified cross-validation.
- `--early-stop-patience 10` — early stop patience (`0` disables).
- `--saliency-demo` — generate saliency view for one validation slide.
- `--benchmark-memory` — run GPU memory scaling benchmark.
- `--vision-lstm-root <path>` — optional local checkout for alternate import paths.

## Outputs

Console logs include:

- split counts and inferred feature dimension,
- per-epoch training/validation losses,
- validation accuracy, ROC-AUC, PR-AUC,
- GPU peak memory per epoch.

If `--plots-dir` is provided, the pipeline writes:

- `roc.png`
- `pr.png`
- `gpu_memory_epochs.png`
- `train_val_loss.png`
- `saliency.png` (if `--saliency-demo`)
- `memory_scaling.png` (if `--benchmark-memory`)

## Notes on Naming

The canonical project/model name is `xLSTM-MIL`.

Primary model symbols:

- `XLSTMMIL`
- `build_xlstmmil`

## Troubleshooting

- `python` not found on Windows: use `py` launcher (as shown above).
- CUDA OOM:
  - reduce `--max-seq-len`,
  - use `--max-seq-eval` with a smaller value (avoid `none`),
  - disable memory benchmark/saliency during initial runs.
- Missing xLSTM modules:
  - install official xLSTM package/checkouts,
  - optionally set `--vision-lstm-root`,
  - or use `XLSTM_EXTRA_PATHS` env var.

## Citation / Upstream Context

Repository base: [subhan-liaqat/xLSTM-MIL](https://github.com/subhan-liaqat/xLSTM-MIL)
