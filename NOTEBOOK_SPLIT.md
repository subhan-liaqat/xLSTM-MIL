# `file6_final.ipynb` to package mapping

This repository is structured so the notebook workflow lives in reusable modules.

- Environment setup and deterministic seed/device logic:
  - `xlstm_mil/env.py`
- Dataset download from Hugging Face snapshot:
  - `xlstm_mil/data/download.py`
- WSI feature dataset loading, split inference, and Hilbert ordering:
  - `xlstm_mil/data/dataset.py`
- xLSTM-MIL model definition and mLSTM block resolution:
  - `xlstm_mil/model/xlstmmil.py`
- Training loop, early stopping, and optional K-fold CV:
  - `xlstm_mil/training/fit.py`
- Stride subsampling utilities used in train/eval:
  - `xlstm_mil/training/subsample.py`
- Final evaluation metrics:
  - `xlstm_mil/eval/metrics.py`
- Plots (ROC, PR, losses, GPU memory, saliency):
  - `xlstm_mil/eval/plots.py`
- Sequence-length memory benchmark:
  - `xlstm_mil/benchmark/memory.py`
- End-to-end executable entrypoint:
  - `scripts/train.py`

## Notebook parity notes

- `--max-seq-eval none` matches notebook behavior for full-bag evaluation (`MAX_SEQ_EVAL = None`).
- Saliency and memory benchmark sections are exposed via:
  - `--saliency-demo`
  - `--benchmark-memory`
