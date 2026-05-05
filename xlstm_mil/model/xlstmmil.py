from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm_mil.config import ModelConfig


def _try_set_mlstm_dropout(mlstm_cfg, p=0.1):
    """Apply dropout-related fields when the installed xlstm exposes them (often none)."""
    applied = []
    for name in (
        "dropout",
        "attention_dropout",
        "attn_dropout",
        "proj_dropout",
        "output_dropout",
        "embedding_dropout",
    ):
        if hasattr(mlstm_cfg, name):
            try:
                setattr(mlstm_cfg, name, float(p))
                applied.append(name)
            except Exception:
                pass
    return applied


def resolve_mlstm_block_factory(
    hidden_dim=256,
    context_length=10_000,
    backend="chunkwise",
    num_heads=4,
    mlstm_dropout=0.1,
):
    candidates = [
        ("xlstm.blocks.mlstm.block", "mLSTMBlock", "mLSTMBlockConfig"),
        ("vislstm.modules.xlstm.blocks.mlstm.block", "mLSTMBlock", "mLSTMBlockConfig"),
        ("vision_lstm.vision_lstm2", "mLSTMBlock", "mLSTMBlockConfig"),
    ]
    errors = []
    for mod_name, cls_name, cfg_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, cls_name):
                continue
            mblock_cls = getattr(mod, cls_name)
            cfg_cls = getattr(mod, cfg_name, None)

            if cfg_cls is not None:
                try:
                    cfg = cfg_cls()
                    if hasattr(cfg, "_num_blocks"):
                        cfg._num_blocks = 2
                    if hasattr(cfg, "_block_idx"):
                        cfg._block_idx = 0
                    applied = []
                    if hasattr(cfg, "mlstm"):
                        cfg.mlstm.embedding_dim = int(hidden_dim)
                        cfg.mlstm.context_length = int(context_length)
                        cfg.mlstm.num_heads = int(num_heads)
                        cfg.mlstm.qkv_proj_blocksize = 4
                        cfg.mlstm._num_blocks = getattr(cfg, "_num_blocks", 2)
                        cfg.mlstm.backend = backend
                        applied = _try_set_mlstm_dropout(cfg.mlstm, mlstm_dropout)
                    if hasattr(cfg, "__post_init__"):
                        cfg.__post_init__()
                    _ = mblock_cls(cfg)

                    def factory(dim):
                        c = cfg_cls()
                        if hasattr(c, "_num_blocks"):
                            c._num_blocks = 2
                        if hasattr(c, "_block_idx"):
                            c._block_idx = 0
                        if hasattr(c, "mlstm"):
                            c.mlstm.embedding_dim = int(dim)
                            c.mlstm.context_length = int(context_length)
                            c.mlstm.num_heads = int(num_heads)
                            c.mlstm.qkv_proj_blocksize = 4
                            c.mlstm._num_blocks = getattr(c, "_num_blocks", 2)
                            c.mlstm.backend = backend
                            _try_set_mlstm_dropout(c.mlstm, mlstm_dropout)
                        if hasattr(c, "__post_init__"):
                            c.__post_init__()
                        return mblock_cls(c)

                    print(
                        f"mLSTMBlock resolved: {mod_name} (config constructor, heads={num_heads}, "
                        f"mlstm_dropout_attempt={mlstm_dropout}, applied={applied if applied else 'none'})"
                    )
                    return factory
                except Exception as e:
                    errors.append((mod_name, "config", repr(e)))

            try:
                _ = mblock_cls(hidden_dim)

                def factory(dim):
                    return mblock_cls(dim)

                print(f"mLSTMBlock resolved: {mod_name} (dim constructor; no config dropout)")
                return factory
            except Exception as e:
                errors.append((mod_name, "dim", repr(e)))
        except Exception as e:
            errors.append((mod_name, "import", repr(e)))

    raise RuntimeError(f"Could not construct official mLSTMBlock. errors={errors}")


class XLSTMMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mlstm_block_factory: Callable[[int], nn.Module],
        *,
        hidden_dim: int,
        num_blocks: int,
        mlp_dim: int,
        dropout: float,
        input_dropout: float,
        post_block_dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.post_block_dropout = float(post_block_dropout)
        self.blocks = nn.ModuleList([mlstm_block_factory(hidden_dim) for _ in range(num_blocks)])
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, x):
        x = self.input_dropout(self.input_proj(x))
        for i, b in enumerate(self.blocks):
            if (i + 1) % 2 == 1:
                x = b(x)
            else:
                xr = torch.flip(x, dims=[1])
                xr = b(xr)
                x = torch.flip(xr, dims=[1])
            x = F.dropout(x, p=self.post_block_dropout, training=self.training)
        pooled = x.mean(dim=1)
        return self.cls_head(pooled)


def build_xlstmmil(in_dim: int, mcfg: ModelConfig) -> XLSTMMIL:
    factory = resolve_mlstm_block_factory(
        hidden_dim=mcfg.hidden_dim,
        context_length=mcfg.mlstm_context_length,
        backend=mcfg.mlstm_backend,
        num_heads=mcfg.num_heads,
        mlstm_dropout=mcfg.mlstm_dropout,
    )
    return XLSTMMIL(
        in_dim,
        mlstm_block_factory=factory,
        hidden_dim=mcfg.hidden_dim,
        num_blocks=mcfg.num_mlstm_blocks,
        mlp_dim=mcfg.mlp_dim,
        dropout=mcfg.classifier_dropout,
        input_dropout=mcfg.input_dropout,
        post_block_dropout=mcfg.post_block_dropout,
    )
