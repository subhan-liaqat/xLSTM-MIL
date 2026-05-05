# xLSTM-MIL

`xLSTM-MIL` is an end-to-end training pipeline for whole-slide image (WSI) multiple instance learning using precomputed patch embeddings.

It currently supports binary classification for:
- `camelyon16`: normal vs tumor
- `tcga_nsclc`: TCGA-LUSC vs TCGA-LUAD

---

## 1) What This Repository Provides

- Structured Python package (not notebook-only workflow).
- Hilbert-ordered bag construction from `.pt` patch embeddings and `.h5` coordinates.
- xLSTM-based MIL model (`chunkwise` backend default).
- Full train/eval loop with early stopping and optional K-fold CV.
- ROC/PR/loss/memory plots.
- Optional saliency demo and sequence-memory benchmark.
- Optional CAMELYON16 download helper from Hugging Face.

---

## 2) Repository Structure

- `scripts/train.py`: main CLI entrypoint.
- `xlstm_mil/config.py`: train/model dataclass configs.
- `xlstm_mil/env.py`: seed, device, optional path setup.
- `xlstm_mil/data/dataset.py`: manifest parsing, label inference, bag loading, Hilbert ordering.
- `xlstm_mil/data/download.py`: CAMELYON16 HF snapshot helper.
- `xlstm_mil/model/xlstmmil.py`: xLSTM-MIL architecture.
- `xlstm_mil/training/subsample.py`: sequence subsampling utilities.
- `xlstm_mil/training/fit.py`: training loop and split/CV orchestration.
- `xlstm_mil/eval/metrics.py`: final metrics.
- `xlstm_mil/eval/plots.py`: plot generation and saliency demo.
- `xlstm_mil/benchmark/memory.py`: sequence-length memory benchmark.

---

## 3) Environment Setup

### Install package (recommended)

```bash
pip install -e .
```

### Or install from requirements

```bash
pip install -r requirements.txt
```

Windows note:
- If `python` is not recognized, use `py` launcher in all commands.

---

## 4) Data Requirements

You can pass explicit paths (`--embed-dir`, `--patch-dir`, `--process-csv`) or pass a `--data-root` with this layout:

- `embeddings/*.pt`
- `patches/**/*.h5`
- `patches/process_list_autogen.csv` (or your own manifest CSV)

The manifest CSV must include:
- `slide_id`

Optional split columns (if present, they are used directly):
- `split`, `set`, `subset`, `partition`

### Pairing Rule

Each sample must have all three components:
- manifest row (`slide_id`)
- embedding file (`.pt`)
- coordinate file (`.h5`)

Pairing is done by normalized filename stem.

---

## 5) Task-Specific Labeling Rules

### `--task camelyon16` (default)

- Label inferred from `slide_id`:
  - contains `tumor` -> `1`
  - contains `normal` -> `0`
- If split column is absent:
  - `test_*` -> `test`
  - `normal_*` / `tumor_*` -> `train`

### `--task tcga_nsclc`

- Binary target:
  - LUAD / adenocarcinoma -> `1`
  - LUSC / squamous -> `0`
- Label inference checks:
  - `slide_id`
  - optional metadata columns: `project_id`, `project`, `primary_diagnosis`, `diagnosis`, `label`, `class`
- If split column is absent:
  - auto patient-level 80/20 split is created in training
  - grouping uses TCGA patient barcode to avoid slide leakage across train/val

---

## 6) End-to-End Quick Start

### A) CAMELYON16 with local files

```bash
py scripts/train.py --task camelyon16 --data-root "C:\path\to\camelyon16_data"
```

### B) CAMELYON16 with Hugging Face download

```bash
py scripts/train.py ^
  --task camelyon16 ^
  --download ^
  --data-root "C:\path\to\camelyon16_data" ^
  --hf-repo "kaczmarj/camelyon16-uni" ^
  --hf-token "<YOUR_HF_TOKEN>"
```

### C) TCGA-NSCLC (LUAD vs LUSC)

```bash
py scripts/train.py ^
  --task tcga_nsclc ^
  --data-root "C:\path\to\tcga_nsclc_data" ^
  --process-csv "C:\path\to\tcga_nsclc_data\manifest.csv"
```

### D) Save plots to disk

```bash
py scripts/train.py ^
  --task tcga_nsclc ^
  --data-root "C:\path\to\tcga_nsclc_data" ^
  --process-csv "C:\path\to\tcga_nsclc_data\manifest.csv" ^
  --plots-dir "outputs\plots"
```

---

## 7) Training Pipeline Behavior

High-level flow:

1. Parse CLI config.
2. Resolve dataset paths and optional download.
3. Load dataset and infer feature dimension.
4. Build model and optimizer/scheduler.
5. Train with sequence subsampling and early stopping.
6. Evaluate on validation/test split.
7. Emit metrics and optional figures.

Split logic:
- Uses explicit `train`/`test` if provided in manifest.
- Falls back to automatic split if needed.
- For TCGA NSCLC fallback split, patient-level grouping is enforced.

---

## 8) Key CLI Options

- `--task camelyon16|tcga_nsclc`: selects labeling and split behavior.
- `--data-root <path>`: root containing embeddings and patches layout.
- `--embed-dir <path>`: explicit embeddings override.
- `--patch-dir <path>`: explicit patches override.
- `--process-csv <path>`: explicit manifest override.
- `--epochs <int>`: number of epochs.
- `--lr <float>`: AdamW learning rate.
- `--weight-decay <float>`: AdamW weight decay.
- `--max-seq-len <int>`: sequence cap for training/in-epoch eval.
- `--max-seq-eval <int|none>`: sequence cap for final eval.
- `--k-fold <int>`: stratified K-fold CV when `>=2`.
- `--early-stop-patience <int>`: early stopping patience (`0` disables).
- `--plots-dir <path>`: write PNG figures.
- `--show-plots`: interactive plotting.
- `--saliency-demo`: generate one saliency visualization.
- `--benchmark-memory`: run sequence memory scaling benchmark.
- `--vision-lstm-root <path>`: optional local alternate xLSTM import root.

---

## 9) Outputs and Metrics

Console output includes:
- dataset/split summary
- inferred feature dimension
- epoch-wise train and eval losses
- eval accuracy, ROC-AUC, PR-AUC
- peak GPU memory per epoch

If `--plots-dir` is provided, generated files may include:
- `roc.png`
- `pr.png`
- `train_val_loss.png`
- `gpu_memory_epochs.png`
- `saliency.png` (when `--saliency-demo`)
- `memory_scaling.png` (when `--benchmark-memory`)

---

## 10) Troubleshooting

- CUDA OOM:
  - reduce `--max-seq-len`
  - set smaller `--max-seq-eval`
  - avoid `--max-seq-eval none` initially
  - disable saliency/memory benchmark during first runs
- Missing xLSTM modules:
  - install required xLSTM package/checkouts
  - pass `--vision-lstm-root`
  - or set `XLSTM_EXTRA_PATHS`
- No matched samples:
  - ensure `.pt` and `.h5` stems align with `slide_id`
  - verify manifest has valid labels for selected `--task`

---

## 11) Naming and Primary Symbols

- Canonical project name: `xLSTM-MIL`
- Main model symbols:
  - `XLSTMMIL`
  - `build_xlstmmil`

---

## 12) Upstream Context

Repository: [subhan-liaqat/xLSTM-MIL](https://github.com/subhan-liaqat/xLSTM-MIL)
