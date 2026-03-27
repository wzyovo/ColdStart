from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig


def _long_tail_bucket(rank_pct: float) -> str:
    if rank_pct <= 0.05:
        return "head"
    if rank_pct <= 0.25:
        return "upper_mid"
    if rank_pct <= 0.65:
        return "mid_tail"
    return "long_tail"


def _image_stub_vector(item_id: int) -> np.ndarray:
    rng = np.random.default_rng(item_id)
    return rng.normal(0.0, 1.0, size=32).round(5)


def build_multimodal_item_features(data_dir: Path) -> pd.DataFrame:
    items = pd.read_csv(data_dir / "items.csv")
    item_tags = pd.read_csv(data_dir / "item_tags.csv")

    tag_map = item_tags.groupby(["item_id", "tag_type"])["tag_name"].apply(list).reset_index()
    grouped: dict[int, dict[str, list[str]]] = {}
    for row in tag_map.itertuples(index=False):
        grouped.setdefault(int(row.item_id), {})[str(row.tag_type)] = list(row.tag_name)

    ranked = items["popularity_score"].rank(method="average", pct=True, ascending=False)
    rows: list[dict[str, object]] = []
    for item, pct in zip(items.itertuples(index=False), ranked):
        tags = grouped.get(int(item.item_id), {})
        taste_tags = tags.get("taste", [])[:3]
        scene_tags = tags.get("scene", [])[:2]
        price_tags = tags.get("price", [str(item.price_band)])[:1]
        title = f"{item.primary_cuisine} {item.item_name}"
        description = (
            f"{item.primary_cuisine} item with "
            f"{', '.join(taste_tags or ['balanced'])} taste for "
            f"{', '.join(scene_tags or ['daily_meal'])}."
        )
        rows.append(
            {
                "item_id": int(item.item_id),
                "merchant_id": int(item.merchant_id),
                "title_text": title,
                "description_text": description,
                "taste_tags": "|".join(taste_tags),
                "scene_tags": "|".join(scene_tags),
                "price_tags": "|".join(price_tags),
                "image_token": f"img_token_{int(item.item_id)}",
                "image_vector_stub": json.dumps(_image_stub_vector(int(item.item_id)).tolist()),
                "long_tail_bucket": _long_tail_bucket(float(pct)),
            }
        )
    return pd.DataFrame(rows)


def build_strict_cold_protocol(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cold_users = pd.read_csv(data_dir / "cold_start_users.csv")
    browsing_logs = pd.read_csv(data_dir / "browsing_logs.csv")
    users = pd.read_csv(data_dir / "users.csv")

    cold_user_ids = set(cold_users["user_id"].astype(int).tolist())
    cold_profiles = users[users["user_id"].isin(cold_user_ids)].copy()

    cold_logs = browsing_logs[browsing_logs["user_id"].isin(cold_user_ids)].copy()
    cold_logs = cold_logs.sort_values(["user_id", "event_time", "exposure_rank"])

    one_shot = cold_logs.groupby("user_id").head(1).reset_index(drop=True)
    three_shot = cold_logs.groupby("user_id").head(3).reset_index(drop=True)

    return cold_profiles.reset_index(drop=True), one_shot, three_shot


def prepare_protocol_files(config: DataConfig) -> dict[str, str]:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    multimodal = build_multimodal_item_features(config.root)
    cold_profiles, one_shot, three_shot = build_strict_cold_protocol(config.root)

    multimodal_path = config.root / "item_multimodal_features.csv"
    cold_eval_path = config.root / "cold_start_eval_users.csv"
    one_shot_path = config.root / "cold_interactions_step1.csv"
    three_shot_path = config.root / "cold_interactions_step3.csv"
    manifest_path = config.model_dir / "data_manifest.json"

    multimodal.to_csv(multimodal_path, index=False)
    cold_profiles.to_csv(cold_eval_path, index=False)
    one_shot.to_csv(one_shot_path, index=False)
    three_shot.to_csv(three_shot_path, index=False)

    manifest = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "outputs": {
            "item_multimodal_features": str(multimodal_path),
            "cold_start_eval_users": str(cold_eval_path),
            "cold_interactions_step1": str(one_shot_path),
            "cold_interactions_step3": str(three_shot_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest["outputs"]


if __name__ == "__main__":
    outputs = prepare_protocol_files(DataConfig())
    for name, path in outputs.items():
        print(f"{name}: {path}")
