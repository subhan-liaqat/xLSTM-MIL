from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def insert_path_if_exists(p: str | Path) -> None:
    s = str(p)
    if os.path.isdir(s) and s not in sys.path:
        sys.path.insert(0, s)


def setup_optional_xlstm_paths(
    vision_lstm_root: Path | None = None,
    extra_paths: list[str | Path] | None = None,
) -> None:
    """Match Colab-style sys.path extensions for vision-lstm / xlstm checkouts."""
    if vision_lstm_root is not None:
        root = Path(vision_lstm_root)
        insert_path_if_exists(root)
        insert_path_if_exists(root / "src")
    if extra_paths:
        for p in extra_paths:
            insert_path_if_exists(p)
    env_extra = os.environ.get("XLSTM_EXTRA_PATHS", "")
    if env_extra:
        for part in env_extra.split(os.pathsep):
            if part.strip():
                insert_path_if_exists(part.strip())
