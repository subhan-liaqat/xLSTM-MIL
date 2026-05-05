from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def benchmark_memory_scaling(
    model,
    in_dim: int,
    seq_lens,
    *,
    device: torch.device,
    dataset_obj=None,
    index_hint: int = 0,
    batch_size: int = 1,
    runs_per_len: int = 2,
):
    model.eval()
    rows = []
    if not torch.cuda.is_available():
        print("CUDA not available: skipping GPU memory benchmark.")
        return pd.DataFrame(columns=["seq_len", "peak_mem_gb", "tokens_per_gb"])

    ref = None
    if dataset_obj is not None and len(dataset_obj) > 0:
        ref = dataset_obj[int(index_hint) % len(dataset_obj)]["features"]
        ref = ref.to(device)

    for n in seq_lens:
        peak_vals = []
        for _ in range(runs_per_len):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            if ref is None:
                x = torch.randn(batch_size, n, in_dim, device=device)
            else:
                m = int(ref.shape[0])
                if m >= n:
                    idx = torch.randperm(m, device=ref.device)[:n]
                    base = ref[idx]
                else:
                    reps = int(math.ceil(n / m))
                    base = ref.repeat(reps, 1)[:n]
                base = base + 0.01 * torch.randn_like(base)
                x = base.unsqueeze(0).repeat(batch_size, 1, 1)

            _ = model(x)
            torch.cuda.synchronize(device)
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_vals.append(float(peak_gb))
            del x

        peak_mem_gb = float(np.mean(peak_vals))
        rows.append(
            {
                "seq_len": int(n),
                "peak_mem_gb": peak_mem_gb,
                "tokens_per_gb": (n / peak_mem_gb) if peak_mem_gb > 0 else float("nan"),
            }
        )
        print(f"N={n:6d} | Peak GPU Mem={peak_mem_gb:.3f} GB")

    return pd.DataFrame(rows)
