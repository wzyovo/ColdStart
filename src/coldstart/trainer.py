from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, ModelConfig, TrainConfig
from .data_pipeline import prepare_protocol_files
from .datasets import load_multimodal_tables, parse_image_stub, simple_tokenize

try:  # pragma: no cover
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


MAX_TEXT_LEN = 32
MAX_TAG_LEN = 8
INTERACTION_WIDTH = 4
POSITIVE_EVENT_TYPES = {"click", "collect", "order"}


def _json_ready(payload: dict[str, object]) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _save_pending_plan(data_config: DataConfig, model_config: ModelConfig, train_config: TrainConfig) -> Path:
    data_config.model_dir.mkdir(parents=True, exist_ok=True)
    path = data_config.model_dir / "training_plan.json"
    payload = {
        "status": "pending_runtime_dependencies",
        "reason": "torch is not installed in the current environment",
        "data_config": _json_ready(asdict(data_config)),
        "model_config": _json_ready(asdict(model_config)),
        "train_config": _json_ready(asdict(train_config)),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path


def _choose_device() -> "torch.device":
    preferred = str(os.environ.get("COLDSTART_DEVICE", "auto")).lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("COLDSTART_DEVICE=cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_ints(values: list[int], size: int) -> list[int]:
    trimmed = values[:size]
    return trimmed + [0] * max(0, size - len(trimmed))


def _build_encoders(users: pd.DataFrame) -> dict[str, dict[object, int]]:
    return {
        "device_type": {value: idx for idx, value in enumerate(sorted(users["device_type"].dropna().unique()))},
        "location_id": {value: idx for idx, value in enumerate(sorted(users["location_id"].dropna().unique()))},
        "query_id": {value: idx for idx, value in enumerate(sorted(users["query_id"].dropna().unique()))},
        "circle_id": {value: idx for idx, value in enumerate(sorted(users["circle_id"].dropna().unique()))},
        "budget_level": {value: idx for idx, value in enumerate(sorted(users["budget_level"].dropna().unique()))},
    }


def _user_context_vector(
    user_id: int,
    user_row: pd.Series,
    user_stats: dict[int, dict[str, float]],
    encoders: dict[str, dict[object, int]],
    size: int,
    user_label_target: np.ndarray | None = None,
) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    stats = user_stats.get(int(user_id), {})
    vector[0] = float(user_row["timestamp"]) / 100.0
    vector[1] = float(stats.get("behavior_count", 0.0)) / 50.0
    vector[2] = float(stats.get("click_count", 0.0)) / 20.0
    vector[3] = float(stats.get("collect_count", 0.0)) / 20.0
    vector[4] = float(stats.get("order_count", 0.0)) / 20.0
    vector[5] = float(stats.get("avg_dwell_ms", 0.0)) / 15000.0
    vector[6] = float(stats.get("avg_scroll_depth", 0.0))
    vector[7] = float(stats.get("avg_event_strength", 0.0)) / 4.0

    slots = [
        ("device_type", 8),
        ("location_id", 12),
        ("query_id", 16),
        ("circle_id", 20),
        ("budget_level", 24),
    ]
    for feature, offset in slots:
        mapping = encoders[feature]
        width = 4
        idx = mapping.get(user_row[feature], 0) % width
        vector[offset + idx] = 1.0
    if user_label_target is not None and size > 32:
        label_width = min(len(user_label_target), size - 32)
        vector[32 : 32 + label_width] = user_label_target[:label_width]
    return vector


def _apply_cold_simulation(
    context: np.ndarray,
    sequence: np.ndarray,
    rng: np.random.Generator,
    prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    # 随机去掉行为信号，让模型也能学习严格冷启动场景下的排序。
    if prob <= 0 or rng.random() > prob:
        return context, sequence
    cold_context = context.copy()
    cold_sequence = np.zeros_like(sequence)
    cold_context[1:8] = 0.0
    return cold_context, cold_sequence


def _interaction_row(item_id: int, event_type: str, item_popularity: dict[int, float], exposure_rank: int) -> list[float]:
    event_strength = {"browse": 0.25, "click": 0.5, "collect": 0.75, "order": 1.0}
    return [
        float(event_strength.get(event_type, 0.0)),
        float(item_popularity.get(int(item_id), 0.0)) / 500.0,
        float(exposure_rank) / 20.0,
        float((int(item_id) % 1000) / 1000.0),
    ]


def _interaction_sequence(
    interaction_map: dict[int, list[dict[str, object]]],
    user_id: int,
    max_seq_len: int,
    item_popularity: dict[int, float],
) -> np.ndarray:
    rows = interaction_map.get(int(user_id), [])[-max_seq_len:]
    sequence = np.zeros((max_seq_len, INTERACTION_WIDTH), dtype=np.float32)
    start = max_seq_len - len(rows)
    for idx, row in enumerate(rows, start=start):
        sequence[idx] = _interaction_row(
            item_id=int(row["item_id"]),
            event_type=str(row["event_type"]),
            item_popularity=item_popularity,
            exposure_rank=int(row["exposure_rank"]),
        )
    return sequence


def _build_item_feature_maps(
    multimodal_items: pd.DataFrame,
    item_tags: pd.DataFrame,
    vocab_size: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    text_map: dict[int, np.ndarray] = {}
    image_map: dict[int, np.ndarray] = {}
    tag_map: dict[int, np.ndarray] = {}

    grouped_tags = item_tags.groupby("item_id")["tag_name"].apply(list).to_dict()
    lookup = multimodal_items.set_index("item_id")
    for item_id in lookup.index.astype(int).tolist():
        row = lookup.loc[item_id]
        text = f"{row['title_text']} {row['description_text']}"
        text_tokens = _pad_ints(simple_tokenize(text, vocab_size), MAX_TEXT_LEN)
        raw_tags = grouped_tags.get(item_id, [])
        tag_ids = _pad_ints([(abs(hash(tag)) % 255) + 1 for tag in raw_tags], MAX_TAG_LEN)
        text_map[item_id] = np.array(text_tokens, dtype=np.int64)
        image_map[item_id] = parse_image_stub(str(row["image_vector_stub"]))
        tag_map[item_id] = np.array(tag_ids, dtype=np.int64)
    return text_map, image_map, tag_map


def _build_user_stats(browsing_logs: pd.DataFrame) -> dict[int, dict[str, float]]:
    event_strength = {"browse": 1.0, "click": 2.0, "collect": 3.0, "order": 4.0}
    logs = browsing_logs.copy()
    logs["event_strength"] = logs["event_type"].map(event_strength)
    stats = (
        logs.groupby("user_id")
        .agg(
            behavior_count=("item_id", "size"),
            order_count=("event_type", lambda s: float((s == "order").sum())),
            collect_count=("event_type", lambda s: float((s == "collect").sum())),
            click_count=("event_type", lambda s: float((s == "click").sum())),
            avg_dwell_ms=("dwell_ms", "mean"),
            avg_scroll_depth=("scroll_depth", "mean"),
            avg_event_strength=("event_strength", "mean"),
        )
        .fillna(0.0)
        .to_dict(orient="index")
    )
    return {int(user_id): value for user_id, value in stats.items()}


def _build_label_vocab(user_preferences: pd.DataFrame, label_count: int) -> dict[str, int]:
    tags = user_preferences["tag_name"].value_counts().head(label_count).index.tolist()
    return {tag: idx for idx, tag in enumerate(tags)}


def _build_multilabel_targets(
    user_preferences: pd.DataFrame,
    label_vocab: dict[str, int],
    label_count: int,
) -> dict[int, np.ndarray]:
    targets: dict[int, np.ndarray] = {}
    for user_id, frame in user_preferences.groupby("user_id"):
        vector = np.zeros(label_count, dtype=np.float32)
        for tag in frame["tag_name"].tolist():
            if tag in label_vocab:
                vector[label_vocab[tag]] = 1.0
        targets[int(user_id)] = vector
    return targets


def _build_interaction_map(browsing_logs: pd.DataFrame) -> dict[int, list[dict[str, object]]]:
    logs = browsing_logs.sort_values(["user_id", "event_time", "exposure_rank"])
    grouped: dict[int, list[dict[str, object]]] = {}
    for user_id, frame in logs.groupby("user_id"):
        grouped[int(user_id)] = frame[["item_id", "event_type", "exposure_rank"]].to_dict(orient="records")
    return grouped


def _positive_item_map(browsing_logs: pd.DataFrame, user_ids: set[int]) -> dict[int, set[int]]:
    logs = browsing_logs[browsing_logs["user_id"].isin(user_ids)].copy()
    logs = logs[logs["event_type"].isin(POSITIVE_EVENT_TYPES)]
    grouped = logs.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(int).tolist())).to_dict()
    return {int(user_id): value for user_id, value in grouped.items()}


def _split_warm_users(users: pd.DataFrame) -> tuple[set[int], set[int]]:
    warm_users = users[users["user_type"] == "warm"].sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    split_index = max(1, int(len(warm_users) * 0.8))
    train_ids = set(warm_users.iloc[:split_index]["user_id"].astype(int).tolist())
    valid_ids = set(warm_users.iloc[split_index:]["user_id"].astype(int).tolist())
    return train_ids, valid_ids


def _sample_negatives(
    candidate_items: np.ndarray,
    positive_items: set[int],
    rng: np.random.Generator,
    sample_size: int,
    item_popularity: dict[int, float],
    hard_negative_ratio: float,
) -> list[int]:
    available = np.array([item for item in candidate_items if int(item) not in positive_items], dtype=np.int64)
    if len(available) == 0:
        return []
    size = min(sample_size, len(available))
    hard_size = min(size, int(round(size * hard_negative_ratio)))
    random_size = max(0, size - hard_size)

    if hard_size > 0:
        scored = sorted(available.tolist(), key=lambda item_id: item_popularity.get(int(item_id), 0.0), reverse=True)
        hard_pool = np.array(scored[: max(hard_size * 5, hard_size)], dtype=np.int64)
        chosen_hard = rng.choice(hard_pool, size=min(hard_size, len(hard_pool)), replace=False).astype(int).tolist()
    else:
        chosen_hard = []

    remaining = np.array([item for item in available.tolist() if int(item) not in set(chosen_hard)], dtype=np.int64)
    if random_size > 0 and len(remaining) > 0:
        chosen_random = rng.choice(remaining, size=min(random_size, len(remaining)), replace=False).astype(int).tolist()
    else:
        chosen_random = []
    return chosen_hard + chosen_random


def _build_samples(
    users: pd.DataFrame,
    positive_map: dict[int, set[int]],
    candidate_items: np.ndarray,
    user_stats: dict[int, dict[str, float]],
    interaction_map: dict[int, list[dict[str, object]]],
    item_popularity: dict[int, float],
    text_map: dict[int, np.ndarray],
    image_map: dict[int, np.ndarray],
    tag_map: dict[int, np.ndarray],
    multilabel_targets: dict[int, np.ndarray],
    encoders: dict[str, dict[object, int]],
    max_seq_len: int,
    context_dim: int,
    negative_ratio: int,
    seed: int,
    hard_negative_ratio: float,
    cold_simulation_prob: float = 0.0,
) -> list[dict[str, np.ndarray | float | int]]:
    rng = np.random.default_rng(seed)
    user_lookup = users.set_index("user_id")
    rows: list[dict[str, np.ndarray | float | int]] = []

    for user_id, positive_items in positive_map.items():
        if user_id not in user_lookup.index:
            continue
        user_row = user_lookup.loc[user_id]
        negatives = _sample_negatives(
            candidate_items=candidate_items,
            positive_items=positive_items,
            rng=rng,
            sample_size=max(len(positive_items) * negative_ratio, negative_ratio),
            item_popularity=item_popularity,
            hard_negative_ratio=hard_negative_ratio,
        )
        target = multilabel_targets.get(user_id, np.zeros(len(next(iter(multilabel_targets.values()))), dtype=np.float32))
        context = _user_context_vector(
            user_id,
            user_row,
            user_stats,
            encoders,
            context_dim,
            user_label_target=target,
        )
        sequence = _interaction_sequence(interaction_map, user_id, max_seq_len, item_popularity)

        for item_id in list(positive_items) + negatives:
            if int(item_id) not in text_map:
                continue
            row_context = context
            row_sequence = sequence
            if cold_simulation_prob > 0.0:
                row_context, row_sequence = _apply_cold_simulation(context, sequence, rng, cold_simulation_prob)
            rows.append(
                {
                    "user_id": int(user_id),
                    "item_id": int(item_id),
                    "user_context": row_context,
                    "interaction_seq": row_sequence,
                    "item_text_tokens": text_map[int(item_id)],
                    "item_image_vectors": image_map[int(item_id)],
                    "item_tag_ids": tag_map[int(item_id)],
                    "label": float(int(item_id) in positive_items),
                    "multilabel_target": target,
                }
            )
    return rows


def _candidate_pool(items: pd.DataFrame, size: int) -> np.ndarray:
    ranked = items.sort_values(["popularity_score", "user_coverage", "item_id"], ascending=[False, False, True])
    return ranked["item_id"].astype(int).head(size).to_numpy()


def _build_candidate_eval_rows(
    users: pd.DataFrame,
    target_user_ids: set[int],
    truth_map: dict[int, set[int]],
    candidate_items: np.ndarray,
    user_stats: dict[int, dict[str, float]],
    interaction_map: dict[int, list[dict[str, object]]],
    item_popularity: dict[int, float],
    text_map: dict[int, np.ndarray],
    image_map: dict[int, np.ndarray],
    tag_map: dict[int, np.ndarray],
    multilabel_targets: dict[int, np.ndarray],
    encoders: dict[str, dict[object, int]],
    context_dim: int,
    max_seq_len: int,
) -> list[dict[str, np.ndarray | float | int]]:
    # 为每个验证用户构造固定候选集，让训练阶段的验证更接近真实评估。
    rows: list[dict[str, np.ndarray | float | int]] = []
    user_lookup = users.set_index("user_id")
    label_size = len(next(iter(multilabel_targets.values()))) if multilabel_targets else 16

    for user_id in sorted(target_user_ids):
        if user_id not in user_lookup.index:
            continue
        user_row = user_lookup.loc[user_id]
        target = multilabel_targets.get(user_id, np.zeros(label_size, dtype=np.float32))
        context = _user_context_vector(
            user_id,
            user_row,
            user_stats,
            encoders,
            context_dim,
            user_label_target=target,
        )
        sequence = _interaction_sequence(interaction_map, user_id, max_seq_len, item_popularity)
        stage_candidates = list(dict.fromkeys(candidate_items.tolist() + sorted(truth_map.get(user_id, set()))))

        for item_id in stage_candidates:
            item_id = int(item_id)
            if item_id not in text_map:
                continue
            rows.append(
                {
                    "user_id": int(user_id),
                    "item_id": item_id,
                    "user_context": context,
                    "interaction_seq": sequence,
                    "item_text_tokens": text_map[item_id],
                    "item_image_vectors": image_map[item_id],
                    "item_tag_ids": tag_map[item_id],
                    "label": float(item_id in truth_map.get(user_id, set())),
                    "multilabel_target": target,
                }
            )
    return rows


class WarmRankingDataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: list[dict[str, np.ndarray | float | int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "user_id": int(row["user_id"]),
            "item_id": int(row["item_id"]),
            "user_context": torch.tensor(row["user_context"], dtype=torch.float32),
            "interaction_seq": torch.tensor(row["interaction_seq"], dtype=torch.float32),
            "item_text_tokens": torch.tensor(row["item_text_tokens"], dtype=torch.long),
            "item_image_vectors": torch.tensor(row["item_image_vectors"], dtype=torch.float32),
            "item_tag_ids": torch.tensor(row["item_tag_ids"], dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "multilabel_target": torch.tensor(row["multilabel_target"], dtype=torch.float32),
        }


def _collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "user_id": torch.tensor([row["user_id"] for row in batch], dtype=torch.long),
        "item_id": torch.tensor([row["item_id"] for row in batch], dtype=torch.long),
        "user_context": torch.stack([row["user_context"] for row in batch]),
        "interaction_seq": torch.stack([row["interaction_seq"] for row in batch]),
        "item_text_tokens": torch.stack([row["item_text_tokens"] for row in batch]),
        "item_image_vectors": torch.stack([row["item_image_vectors"] for row in batch]),
        "item_tag_ids": torch.stack([row["item_tag_ids"] for row in batch]),
        "label": torch.stack([row["label"] for row in batch]),
        "multilabel_target": torch.stack([row["multilabel_target"] for row in batch]),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    ranking_loss_fn: nn.Module,
    multilabel_loss_fn: nn.Module,
    device: torch.device,
    train_config: TrainConfig,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        outputs = model(
            user_context=batch["user_context"].to(device),
            interaction_seq=batch["interaction_seq"].to(device),
            item_text_tokens=batch["item_text_tokens"].to(device),
            item_image_vectors=batch["item_image_vectors"].to(device),
            item_tag_ids=batch["item_tag_ids"].to(device),
        )
        ranking_loss = ranking_loss_fn(outputs["ranking_score"], batch["label"].to(device))
        multilabel_loss = multilabel_loss_fn(outputs["multilabel_logits"], batch["multilabel_target"].to(device))
        loss = train_config.ranking_weight * ranking_loss + train_config.multilabel_weight * multilabel_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        if batch_idx % 200 == 0:
            print(
                f"  批次 {batch_idx}/{len(loader)} "
                f"总损失={loss.item():.4f} "
                f"排序损失={ranking_loss.item():.4f} "
                f"多标签损失={multilabel_loss.item():.4f} "
                f"学习率={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    train_loss = total_loss / max(total_batches, 1)
    scheduler.step(train_loss)
    return {"train_loss": train_loss}


def _score_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                user_context=batch["user_context"].to(device),
                interaction_seq=batch["interaction_seq"].to(device),
                item_text_tokens=batch["item_text_tokens"].to(device),
                item_image_vectors=batch["item_image_vectors"].to(device),
                item_tag_ids=batch["item_tag_ids"].to(device),
            )
            scores = torch.sigmoid(outputs["ranking_score"]).cpu().numpy()
            for user_id, item_id, score, label in zip(
                batch["user_id"].tolist(),
                batch["item_id"].tolist(),
                scores.tolist(),
                batch["label"].tolist(),
            ):
                rows.append(
                    {
                        "user_id": int(user_id),
                        "item_id": int(item_id),
                        "score": float(score),
                        "label": float(label),
                    }
                )
    return pd.DataFrame(rows)


def _ranking_metrics(scored_df: pd.DataFrame, top_k: int) -> dict[str, float]:
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    hit_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for _, group in scored_df.groupby("user_id"):
        ranked = group.sort_values("score", ascending=False)
        top = ranked.head(top_k)
        true_items = set(group[group["label"] > 0]["item_id"].astype(int).tolist())
        ranked_items = top["item_id"].astype(int).tolist()
        hits = [1 if item_id in true_items else 0 for item_id in ranked_items]
        hit_count = sum(hits)

        precision_scores.append(hit_count / float(top_k))
        recall_scores.append(hit_count / float(len(true_items)) if true_items else 0.0)
        hit_scores.append(1.0 if hit_count else 0.0)

        reciprocal_rank = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal_rank = 1.0 / rank
                break
        mrr_scores.append(reciprocal_rank)

        dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(true_items), top_k) + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision_at_10": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_at_10": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "hit_rate_at_10": float(np.mean(hit_scores)) if hit_scores else 0.0,
        "mrr_at_10": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "evaluated_user_count": int(scored_df["user_id"].nunique()) if not scored_df.empty else 0,
    }


def _recommendation_metrics(scored_df: pd.DataFrame, truth_map: dict[int, set[int]], top_k: int) -> dict[str, float]:
    # 基于完整候选集排序结果计算 Top-K 指标，而不是基于采样后的二分类样本。
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    hit_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for user_id, group in scored_df.groupby("user_id"):
        ranked_items = group.sort_values("score", ascending=False)["item_id"].astype(int).head(top_k).tolist()
        true_items = truth_map.get(int(user_id), set())
        hits = [1 if item_id in true_items else 0 for item_id in ranked_items]
        hit_count = sum(hits)

        precision_scores.append(hit_count / float(top_k))
        recall_scores.append(hit_count / float(len(true_items)) if true_items else 0.0)
        hit_scores.append(1.0 if hit_count else 0.0)

        reciprocal_rank = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal_rank = 1.0 / rank
                break
        mrr_scores.append(reciprocal_rank)

        dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(true_items), top_k) + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision_at_10": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_at_10": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "hit_rate_at_10": float(np.mean(hit_scores)) if hit_scores else 0.0,
        "mrr_at_10": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "evaluated_user_count": int(len(truth_map)),
    }


def train_model(data_config: DataConfig, model_config: ModelConfig, train_config: TrainConfig) -> Path:
    if torch is None:  # pragma: no cover
        return _save_pending_plan(data_config, model_config, train_config)

    from .model import MultiModalColdStartModel

    _set_seed(train_config.seed)
    prepare_protocol_files(data_config)
    tables = load_multimodal_tables(data_config.root)

    users = tables["users"]
    item_tags = tables["item_tags"]
    user_preferences = tables["user_preferences"]
    browsing_logs = tables["browsing_logs"]
    multimodal_items = tables["multimodal_items"]

    encoders = _build_encoders(users)
    user_stats = _build_user_stats(browsing_logs)
    label_vocab = _build_label_vocab(user_preferences, model_config.label_count)
    multilabel_targets = _build_multilabel_targets(user_preferences, label_vocab, model_config.label_count)
    interaction_map = _build_interaction_map(browsing_logs)
    item_popularity = (
        multimodal_items.set_index("item_id").join(tables["items"].set_index("item_id")[["popularity_score"]], how="left")[
            "popularity_score"
        ].fillna(0.0)
        .to_dict()
    )
    text_map, image_map, tag_map = _build_item_feature_maps(multimodal_items, item_tags, model_config.vocab_size)
    train_user_ids, valid_user_ids = _split_warm_users(users)
    train_positive = _positive_item_map(browsing_logs, train_user_ids)
    valid_positive = _positive_item_map(browsing_logs, valid_user_ids)
    candidate_items = multimodal_items["item_id"].astype(int).to_numpy()
    eval_candidate_items = _candidate_pool(tables["items"], data_config.candidate_size)

    train_rows = _build_samples(
        users=users,
        positive_map=train_positive,
        candidate_items=candidate_items,
        user_stats=user_stats,
        interaction_map=interaction_map,
        item_popularity=item_popularity,
        text_map=text_map,
        image_map=image_map,
        tag_map=tag_map,
        multilabel_targets=multilabel_targets,
        encoders=encoders,
        max_seq_len=data_config.max_seq_len,
        context_dim=model_config.context_dim,
        negative_ratio=train_config.negative_ratio_train,
        seed=train_config.seed,
        hard_negative_ratio=train_config.hard_negative_ratio,
        cold_simulation_prob=train_config.cold_simulation_prob,
    )
    # 验证阶段改用候选集排序样本，而不是随机负样本，保证指标更接近实际推荐场景。
    valid_rows = _build_candidate_eval_rows(
        users=users,
        target_user_ids=valid_user_ids,
        truth_map=valid_positive,
        candidate_items=eval_candidate_items,
        user_stats=user_stats,
        interaction_map=interaction_map,
        item_popularity=item_popularity,
        text_map=text_map,
        image_map=image_map,
        tag_map=tag_map,
        multilabel_targets=multilabel_targets,
        encoders=encoders,
        max_seq_len=data_config.max_seq_len,
        context_dim=model_config.context_dim,
    )

    if not train_rows:
        raise RuntimeError("No training samples were built. Check the input data and positive interaction map.")
    if not valid_rows:
        raise RuntimeError("No validation samples were built. Check the input data and validation split.")

    train_dataset = WarmRankingDataset(train_rows)
    valid_dataset = WarmRankingDataset(valid_rows)
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=train_config.batch_size, shuffle=False, collate_fn=_collate_fn)

    device = _choose_device()
    model = MultiModalColdStartModel(
        vocab_size=model_config.vocab_size,
        text_embed_dim=model_config.text_embed_dim,
        text_hidden_dim=model_config.text_hidden_dim,
        image_dim=model_config.image_dim,
        context_dim=model_config.context_dim,
        tag_dim=model_config.tag_dim,
        lstm_hidden_dim=model_config.lstm_hidden_dim,
        fusion_dim=model_config.fusion_dim,
        label_count=model_config.label_count,
        dropout=model_config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    positive_count = sum(1 for row in train_rows if float(row["label"]) > 0.0)
    negative_count = max(1, len(train_rows) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1)], device=device, dtype=torch.float32)
    ranking_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    multilabel_loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] | None = None
    best_path = data_config.model_dir / "best_model.pt"
    data_config.model_dir.mkdir(parents=True, exist_ok=True)
    patience_counter = 0

    print(f"设备: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"训练样本数: {len(train_rows)}", flush=True)
    print(f"验证样本数: {len(valid_rows)}", flush=True)
    print(f"验证候选集大小: {len(eval_candidate_items)}", flush=True)
    print(f"训练用户数: {len(train_user_ids)}", flush=True)
    print(f"验证用户数: {len(valid_user_ids)}", flush=True)
    print(f"正样本权重: {float(pos_weight.item()):.4f}", flush=True)

    for epoch in range(1, train_config.epochs + 1):
        print(f"第 {epoch}/{train_config.epochs} 轮", flush=True)
        epoch_info = {"epoch": float(epoch)}
        epoch_info.update(
            _run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                ranking_loss_fn=ranking_loss_fn,
                multilabel_loss_fn=multilabel_loss_fn,
                device=device,
                train_config=train_config,
            )
        )
        valid_scored = _score_loader(model, valid_loader, device)
        metrics = _recommendation_metrics(valid_scored, valid_positive, data_config.top_k)
        epoch_info.update(metrics)
        history.append(epoch_info)

        print(
            "  "
            f"训练损失={epoch_info['train_loss']:.4f} "
            f"Precision@10={metrics['precision_at_10']:.4f} "
            f"Recall@10={metrics['recall_at_10']:.4f} "
            f"HitRate@10={metrics['hit_rate_at_10']:.4f} "
            f"NDCG@10={metrics['ndcg_at_10']:.4f}",
            flush=True,
        )

        if best_metrics is None or metrics["precision_at_10"] > best_metrics["precision_at_10"]:
            best_metrics = metrics
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "data_config": _json_ready(asdict(data_config)),
                    "model_config": _json_ready(asdict(model_config)),
                    "train_config": _json_ready(asdict(train_config)),
                    "label_vocab": label_vocab,
                    "encoders": {key: {str(k): int(v) for k, v in value.items()} for key, value in encoders.items()},
                },
                best_path,
            )
            print(f"  已保存最佳模型 -> {best_path}", flush=True)
        else:
            patience_counter += 1
            print(f"  本轮未提升，耐心计数={patience_counter}/{train_config.early_stopping_patience}", flush=True)
            if patience_counter >= train_config.early_stopping_patience:
                print("  触发提前停止", flush=True)
                break

    summary_path = data_config.model_dir / "training_summary.json"
    summary = {
        "device": str(device),
        "train_sample_count": len(train_rows),
        "valid_sample_count": len(valid_rows),
        "history": history,
        "best_metrics": best_metrics or {},
        "artifacts": {
            "best_model": str(best_path),
        },
        "optimizer": {
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "pos_weight": float(pos_weight.item()),
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"训练结果已写入 -> {summary_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    output = train_model(DataConfig(), ModelConfig(), TrainConfig())
    print(output)
