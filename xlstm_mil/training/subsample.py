from __future__ import annotations

import torch


def stride_subsample_hilbert(feats, max_len=None):
    """Hilbert-ordered stride subsample; max_len None keeps full sequence."""
    feats = feats if isinstance(feats, torch.Tensor) else torch.as_tensor(feats, dtype=torch.float32)
    feats = feats.contiguous()
    n = int(feats.shape[0])
    if n == 0:
        return feats
    if max_len is None:
        return feats
    if n <= max_len:
        return feats
    idx = torch.linspace(0, n - 1, steps=max_len).round().long()
    return feats[idx]


def prepare_bag_features(feats, max_seq_len: int | None):
    return stride_subsample_hilbert(feats, max_seq_len)


def subsample_bag_feats_coords(feats, coords, max_len):
    """Match deterministic eval subsampling on both feats and coords tensors."""
    n = int(feats.shape[0])
    if coords is not None and int(coords.shape[0]) != n:
        raise ValueError("feats/coords length mismatch in subsample_bag_feats_coords")
    if n <= max_len:
        return feats, coords
    idx = torch.linspace(0, n - 1, steps=max_len).round().long()
    if coords is None:
        return feats[idx], None
    return feats[idx], coords[idx]
