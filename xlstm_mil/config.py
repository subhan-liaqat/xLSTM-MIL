from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    max_seq_len: int = 12_000
    max_seq_eval: int = 12_000
    weight_decay: float = 1e-4
    k_fold: int = 0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0
    epochs: int = 20
    lr: float = 1e-4


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_mlstm_blocks: int = 2
    num_heads: int = 4
    classifier_dropout: float = 0.5
    input_dropout: float = 0.1
    post_block_dropout: float = 0.1
    mlp_dim: int = 128
    mlstm_context_length: int = 16_000
    mlstm_backend: str = "chunkwise"
    mlstm_dropout: float = 0.1
