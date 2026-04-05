from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    root: Path = Path("real_enhanced_data")
    model_dir: Path = Path("models") / "multimodal_coldstart"
    max_seq_len: int = 3
    candidate_size: int = 200
    top_k: int = 10


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 5000
    tag_vocab_size: int = 256
    text_embed_dim: int = 64
    text_hidden_dim: int = 64
    image_dim: int = 32
    context_dim: int = 48
    tag_dim: int = 32
    fusion_dim: int = 128
    lstm_hidden_dim: int = 64
    label_count: int = 16
    match_dim: int = 5
    dropout: float = 0.1


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    ranking_weight: float = 1.0
    multilabel_weight: float = 0.3
    negative_ratio_train: int = 3
    negative_ratio_valid: int = 6
    hard_negative_ratio: float = 0.7
    early_stopping_patience: int = 3
    cold_simulation_prob: float = 0.35
