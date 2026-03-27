from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ColdStartBatch:
    user_context: np.ndarray
    interaction_seq: np.ndarray
    item_text_tokens: np.ndarray
    item_image_vectors: np.ndarray
    item_tag_ids: np.ndarray
    labels: np.ndarray
    multilabel_targets: np.ndarray


def load_multimodal_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "users": pd.read_csv(data_dir / "users.csv"),
        "items": pd.read_csv(data_dir / "items.csv"),
        "item_tags": pd.read_csv(data_dir / "item_tags.csv"),
        "user_preferences": pd.read_csv(data_dir / "user_preferences.csv"),
        "browsing_logs": pd.read_csv(data_dir / "browsing_logs.csv"),
        "multimodal_items": pd.read_csv(data_dir / "item_multimodal_features.csv"),
    }


def simple_tokenize(text: str, vocab_limit: int = 5000) -> list[int]:
    tokens = []
    for token in str(text).lower().replace("|", " ").replace(",", " ").split():
        token_id = (abs(hash(token)) % (vocab_limit - 1)) + 1
        tokens.append(token_id)
    return tokens[:32]


def parse_image_stub(raw: str) -> np.ndarray:
    return np.array(json.loads(raw), dtype=np.float32)
