from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
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


def _log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _json_ready(payload: dict[str, object]) -> dict[str, object]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in payload.items()}


def _save_pending_plan(data_config: DataConfig, model_config: ModelConfig, train_config: TrainConfig) -> Path:
    data_config.model_dir.mkdir(parents=True, exist_ok=True)
    path = data_config.model_dir / "training_plan.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "pending_runtime_dependencies",
                "reason": "torch is not installed in the current environment",
                "data_config": _json_ready(asdict(data_config)),
                "model_config": _json_ready(asdict(model_config)),
                "train_config": _json_ready(asdict(train_config)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_ints(values: list[int], size: int) -> list[int]:
    return values[:size] + [0] * max(0, size - len(values))


def _build_encoders(users: pd.DataFrame) -> dict[str, dict[object, int]]:
    return {
        k: {v: i for i, v in enumerate(sorted(users[k].dropna().unique()))}
        for k in ["device_type", "location_id", "query_id", "circle_id", "budget_level"]
    }


def _build_label_vocab(user_preferences: pd.DataFrame, label_count: int) -> dict[str, int]:
    tags = user_preferences["tag_name"].value_counts().head(label_count).index.tolist()
    return {tag: idx for idx, tag in enumerate(tags)}


def _build_multilabel_targets(user_preferences: pd.DataFrame, label_vocab: dict[str, int], label_count: int) -> dict[int, np.ndarray]:
    targets: dict[int, np.ndarray] = {}
    for user_id, frame in user_preferences.groupby("user_id"):
        vec = np.zeros(label_count, dtype=np.float32)
        for tag in frame["tag_name"].tolist():
            if tag in label_vocab:
                vec[label_vocab[tag]] = 1.0
        targets[int(user_id)] = vec
    return targets


def _build_user_preference_map(user_preferences: pd.DataFrame) -> dict[int, dict[str, dict[str, float]]]:
    grouped = user_preferences.groupby(["user_id", "tag_type", "tag_name"])["preference_weight"].mean().reset_index()
    out: dict[int, dict[str, dict[str, float]]] = {}
    for r in grouped.itertuples(index=False):
        out.setdefault(int(r.user_id), {}).setdefault(str(r.tag_type), {})[str(r.tag_name)] = float(r.preference_weight)
    return out


def _build_item_tag_weight_map(item_tags: pd.DataFrame) -> dict[int, dict[str, dict[str, float]]]:
    grouped = item_tags.groupby(["item_id", "tag_type", "tag_name"])["tag_weight"].mean().reset_index()
    out: dict[int, dict[str, dict[str, float]]] = {}
    for r in grouped.itertuples(index=False):
        out.setdefault(int(r.item_id), {}).setdefault(str(r.tag_type), {})[str(r.tag_name)] = float(r.tag_weight)
    return out


def _weighted_overlap(user_tags: dict[str, float], item_tags: dict[str, float]) -> float:
    if not user_tags or not item_tags:
        return 0.0
    num = sum(min(float(w), float(item_tags.get(tag, 0.0))) for tag, w in user_tags.items())
    den = sum(float(v) for v in user_tags.values())
    return num / den if den > 0 else 0.0


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
    return {int(k): v for k, v in stats.items()}


def _user_context_vector(
    user_id: int, user_row: pd.Series, user_stats: dict[int, dict[str, float]], encoders: dict[str, dict[object, int]], size: int, user_label_target: np.ndarray | None = None
) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    s = user_stats.get(int(user_id), {})
    v[0] = float(user_row["timestamp"]) / 100.0
    v[1] = float(s.get("behavior_count", 0.0)) / 50.0
    v[2] = float(s.get("click_count", 0.0)) / 20.0
    v[3] = float(s.get("collect_count", 0.0)) / 20.0
    v[4] = float(s.get("order_count", 0.0)) / 20.0
    v[5] = float(s.get("avg_dwell_ms", 0.0)) / 15000.0
    v[6] = float(s.get("avg_scroll_depth", 0.0))
    v[7] = float(s.get("avg_event_strength", 0.0)) / 4.0
    for feat, offset in [("device_type", 8), ("location_id", 12), ("query_id", 16), ("circle_id", 20), ("budget_level", 24)]:
        v[offset + (encoders[feat].get(user_row[feat], 0) % 4)] = 1.0
    if user_label_target is not None and size > 32:
        width = min(len(user_label_target), size - 32)
        v[32 : 32 + width] = user_label_target[:width]
    return v


def _interaction_row(item_id: int, event_type: str, item_popularity: dict[int, float], exposure_rank: int) -> list[float]:
    strength = {"browse": 0.25, "click": 0.5, "collect": 0.75, "order": 1.0}
    return [float(strength.get(event_type, 0.0)), float(item_popularity.get(int(item_id), 0.0)) / 500.0, float(exposure_rank) / 20.0, float((int(item_id) % 1000) / 1000.0)]


def _interaction_sequence(interaction_map: dict[int, list[dict[str, object]]], user_id: int, max_seq_len: int, item_popularity: dict[int, float]) -> np.ndarray:
    rows = interaction_map.get(int(user_id), [])[-max_seq_len:]
    seq = np.zeros((max_seq_len, INTERACTION_WIDTH), dtype=np.float32)
    start = max_seq_len - len(rows)
    for i, r in enumerate(rows, start=start):
        seq[i] = _interaction_row(int(r["item_id"]), str(r["event_type"]), item_popularity, int(r["exposure_rank"]))
    return seq


def _build_item_feature_maps(multimodal_items: pd.DataFrame, item_tags: pd.DataFrame, vocab_size: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    text_map: dict[int, np.ndarray] = {}
    image_map: dict[int, np.ndarray] = {}
    tag_map: dict[int, np.ndarray] = {}
    grouped = item_tags.groupby("item_id")["tag_name"].apply(list).to_dict()
    lookup = multimodal_items.set_index("item_id")
    for item_id in lookup.index.astype(int).tolist():
        row = lookup.loc[item_id]
        text_map[item_id] = np.array(_pad_ints(simple_tokenize(f"{row['title_text']} {row['description_text']}", vocab_size), MAX_TEXT_LEN), dtype=np.int64)
        image_map[item_id] = parse_image_stub(str(row["image_vector_stub"]))
        tag_map[item_id] = np.array(_pad_ints([(abs(hash(t)) % 255) + 1 for t in grouped.get(item_id, [])], MAX_TAG_LEN), dtype=np.int64)
    return text_map, image_map, tag_map


def _build_interaction_map(browsing_logs: pd.DataFrame) -> dict[int, list[dict[str, object]]]:
    logs = browsing_logs.sort_values(["user_id", "event_time", "exposure_rank"])
    return {int(uid): f[["item_id", "event_type", "exposure_rank"]].to_dict(orient="records") for uid, f in logs.groupby("user_id")}


def _positive_item_map(browsing_logs: pd.DataFrame, user_ids: set[int]) -> dict[int, set[int]]:
    logs = browsing_logs[browsing_logs["user_id"].isin(user_ids)]
    logs = logs[logs["event_type"].isin(POSITIVE_EVENT_TYPES)]
    return {int(uid): set(s.astype(int).tolist()) for uid, s in logs.groupby("user_id")["item_id"]}


def _split_warm_users(users: pd.DataFrame) -> tuple[set[int], set[int]]:
    warm = users[users["user_type"] == "warm"].sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    split = max(1, int(len(warm) * 0.8))
    return set(warm.iloc[:split]["user_id"].astype(int)), set(warm.iloc[split:]["user_id"].astype(int))


def _sample_negatives(candidate_items: np.ndarray, positive_items: set[int], rng: np.random.Generator, sample_size: int, item_popularity: dict[int, float], hard_negative_ratio: float) -> list[int]:
    available = np.array([i for i in candidate_items if int(i) not in positive_items], dtype=np.int64)
    if len(available) == 0:
        return []
    size = min(sample_size, len(available))
    hard_size = min(size, int(round(size * hard_negative_ratio)))
    rand_size = max(0, size - hard_size)
    hard: list[int] = []
    if hard_size > 0:
        ranked = sorted(available.tolist(), key=lambda x: item_popularity.get(int(x), 0.0), reverse=True)
        pool = np.array(ranked[: max(hard_size * 5, hard_size)], dtype=np.int64)
        hard = rng.choice(pool, size=min(hard_size, len(pool)), replace=False).astype(int).tolist()
    rest = np.array([i for i in available.tolist() if int(i) not in set(hard)], dtype=np.int64)
    rand = rng.choice(rest, size=min(rand_size, len(rest)), replace=False).astype(int).tolist() if rand_size > 0 and len(rest) > 0 else []
    return hard + rand


def _compute_match_features(user_id: int, user_row: pd.Series, item_id: int, item_popularity: dict[int, float], user_pref_map: dict[int, dict[str, dict[str, float]]], item_tag_map: dict[int, dict[str, dict[str, float]]]) -> np.ndarray:
    prefs = user_pref_map.get(int(user_id), {})
    tags = item_tag_map.get(int(item_id), {})
    taste = _weighted_overlap(prefs.get("taste", {}), tags.get("taste", {}))
    scene = _weighted_overlap(prefs.get("scene", {}), tags.get("scene", {}))
    price = _weighted_overlap(prefs.get("price", {}), tags.get("price", {}))
    budget = float(tags.get("price", {}).get(str(user_row.get("budget_level", "")), 0.0))
    pop = min(float(item_popularity.get(int(item_id), 0.0)) / 500.0, 1.0)
    return np.array([taste, scene, price, budget, pop], dtype=np.float32)


def _apply_cold_simulation(context: np.ndarray, sequence: np.ndarray, rng: np.random.Generator, prob: float) -> tuple[np.ndarray, np.ndarray]:
    if prob <= 0 or rng.random() > prob:
        return context, sequence
    c = context.copy()
    c[1:8] = 0.0
    return c, np.zeros_like(sequence)


def _build_samples(
    users: pd.DataFrame, positive_map: dict[int, set[int]], candidate_items: np.ndarray, user_stats: dict[int, dict[str, float]], interaction_map: dict[int, list[dict[str, object]]],
    item_popularity: dict[int, float], text_map: dict[int, np.ndarray], image_map: dict[int, np.ndarray], tag_map: dict[int, np.ndarray], multilabel_targets: dict[int, np.ndarray],
    user_preference_map: dict[int, dict[str, dict[str, float]]], item_tag_weight_map: dict[int, dict[str, dict[str, float]]], encoders: dict[str, dict[object, int]], max_seq_len: int,
    context_dim: int, negative_ratio: int, seed: int, hard_negative_ratio: float, cold_simulation_prob: float = 0.0
) -> list[dict[str, np.ndarray | float | int]]:
    rng = np.random.default_rng(seed)
    lookup = users.set_index("user_id")
    rows: list[dict[str, np.ndarray | float | int]] = []
    label_size = len(next(iter(multilabel_targets.values()))) if multilabel_targets else 16
    for user_id, pos in positive_map.items():
        if user_id not in lookup.index:
            continue
        user_row = lookup.loc[user_id]
        neg = _sample_negatives(candidate_items, pos, rng, max(len(pos) * negative_ratio, negative_ratio), item_popularity, hard_negative_ratio)
        target = multilabel_targets.get(user_id, np.zeros(label_size, dtype=np.float32))
        context = _user_context_vector(user_id, user_row, user_stats, encoders, context_dim, user_label_target=target)
        seq = _interaction_sequence(interaction_map, user_id, max_seq_len, item_popularity)
        for item_id in list(pos) + neg:
            item_id = int(item_id)
            if item_id not in text_map:
                continue
            rc, rs = _apply_cold_simulation(context, seq, rng, cold_simulation_prob) if cold_simulation_prob > 0 else (context, seq)
            rows.append({
                "user_id": int(user_id), "item_id": item_id, "user_context": rc, "interaction_seq": rs, "item_text_tokens": text_map[item_id], "item_image_vectors": image_map[item_id],
                "item_tag_ids": tag_map[item_id], "item_match_features": _compute_match_features(user_id, user_row, item_id, item_popularity, user_preference_map, item_tag_weight_map),
                "label": float(item_id in pos), "multilabel_target": target,
            })
    return rows


def _candidate_pool(items: pd.DataFrame, size: int) -> np.ndarray:
    ranked = items.sort_values(["popularity_score", "user_coverage", "item_id"], ascending=[False, False, True])
    return ranked["item_id"].astype(int).head(size).to_numpy()


def _build_candidate_eval_rows(
    users: pd.DataFrame, target_user_ids: set[int], truth_map: dict[int, set[int]], candidate_items: np.ndarray, user_stats: dict[int, dict[str, float]], interaction_map: dict[int, list[dict[str, object]]],
    item_popularity: dict[int, float], text_map: dict[int, np.ndarray], image_map: dict[int, np.ndarray], tag_map: dict[int, np.ndarray], multilabel_targets: dict[int, np.ndarray],
    user_preference_map: dict[int, dict[str, dict[str, float]]], item_tag_weight_map: dict[int, dict[str, dict[str, float]]], encoders: dict[str, dict[object, int]], context_dim: int, max_seq_len: int
) -> list[dict[str, np.ndarray | float | int]]:
    rows: list[dict[str, np.ndarray | float | int]] = []
    lookup = users.set_index("user_id")
    label_size = len(next(iter(multilabel_targets.values()))) if multilabel_targets else 16
    for user_id in sorted(target_user_ids):
        if user_id not in lookup.index:
            continue
        user_row = lookup.loc[user_id]
        target = multilabel_targets.get(user_id, np.zeros(label_size, dtype=np.float32))
        context = _user_context_vector(user_id, user_row, user_stats, encoders, context_dim, user_label_target=target)
        seq = _interaction_sequence(interaction_map, user_id, max_seq_len, item_popularity)
        for item_id in list(dict.fromkeys(candidate_items.tolist() + sorted(truth_map.get(user_id, set())))):
            item_id = int(item_id)
            if item_id not in text_map:
                continue
            rows.append({
                "user_id": int(user_id), "item_id": item_id, "user_context": context, "interaction_seq": seq, "item_text_tokens": text_map[item_id], "item_image_vectors": image_map[item_id],
                "item_tag_ids": tag_map[item_id], "item_match_features": _compute_match_features(user_id, user_row, item_id, item_popularity, user_preference_map, item_tag_weight_map),
                "label": float(item_id in truth_map.get(user_id, set())), "multilabel_target": target,
            })
    return rows


class WarmRankingDataset(Dataset):  # type: ignore[misc]
    def __init__(self, rows: list[dict[str, np.ndarray | float | int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        r = self.rows[idx]
        return {
            "user_id": int(r["user_id"]), "item_id": int(r["item_id"]),
            "user_context": torch.tensor(r["user_context"], dtype=torch.float32),
            "interaction_seq": torch.tensor(r["interaction_seq"], dtype=torch.float32),
            "item_text_tokens": torch.tensor(r["item_text_tokens"], dtype=torch.long),
            "item_image_vectors": torch.tensor(r["item_image_vectors"], dtype=torch.float32),
            "item_tag_ids": torch.tensor(r["item_tag_ids"], dtype=torch.long),
            "item_match_features": torch.tensor(r["item_match_features"], dtype=torch.float32),
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "multilabel_target": torch.tensor(r["multilabel_target"], dtype=torch.float32),
        }


def _collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {k: torch.stack([r[k] for r in batch]) if k not in {"user_id", "item_id"} else torch.tensor([r[k] for r in batch], dtype=torch.long) for k in batch[0].keys()}


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, ranking_loss_fn: nn.Module, multilabel_loss_fn: nn.Module, device: torch.device, train_config: TrainConfig) -> dict[str, float]:
    model.train()
    total = 0.0
    for i, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        out = model(
            user_context=batch["user_context"].to(device), interaction_seq=batch["interaction_seq"].to(device), item_text_tokens=batch["item_text_tokens"].to(device),
            item_image_vectors=batch["item_image_vectors"].to(device), item_tag_ids=batch["item_tag_ids"].to(device), item_match_features=batch["item_match_features"].to(device),
        )
        rank = ranking_loss_fn(out["ranking_score"], batch["label"].to(device))
        ml = multilabel_loss_fn(out["multilabel_logits"], batch["multilabel_target"].to(device))
        loss = train_config.ranking_weight * rank + train_config.multilabel_weight * ml
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        if i % 200 == 0:
            print(f"  batch {i}/{len(loader)} loss={loss.item():.4f} rank={rank.item():.4f} ml={ml.item():.4f} lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)
    train_loss = total / max(len(loader), 1)
    scheduler.step(train_loss)
    return {"train_loss": train_loss}


def _score_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                user_context=batch["user_context"].to(device), interaction_seq=batch["interaction_seq"].to(device), item_text_tokens=batch["item_text_tokens"].to(device),
                item_image_vectors=batch["item_image_vectors"].to(device), item_tag_ids=batch["item_tag_ids"].to(device), item_match_features=batch["item_match_features"].to(device),
            )
            scores = torch.sigmoid(out["ranking_score"]).cpu().numpy()
            for uid, iid, s, l in zip(batch["user_id"].tolist(), batch["item_id"].tolist(), scores.tolist(), batch["label"].tolist()):
                rows.append({"user_id": int(uid), "item_id": int(iid), "score": float(s), "label": float(l)})
    return pd.DataFrame(rows)


def _recommendation_metrics(scored_df: pd.DataFrame, truth_map: dict[int, set[int]], top_k: int) -> dict[str, float]:
    p, r, h, m, n = [], [], [], [], []
    for uid, g in scored_df.groupby("user_id"):
        ranked = g.sort_values("score", ascending=False)["item_id"].astype(int).head(top_k).tolist()
        truth = truth_map.get(int(uid), set())
        hits = [1 if i in truth else 0 for i in ranked]
        hit_count = sum(hits)
        p.append(hit_count / float(top_k))
        r.append(hit_count / float(len(truth)) if truth else 0.0)
        h.append(1.0 if hit_count else 0.0)
        rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / rank
                break
        m.append(rr)
        dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(truth), top_k) + 1))
        n.append(dcg / idcg if idcg > 0 else 0.0)
    return {
        "precision_at_10": float(np.mean(p)) if p else 0.0, "recall_at_10": float(np.mean(r)) if r else 0.0, "hit_rate_at_10": float(np.mean(h)) if h else 0.0,
        "mrr_at_10": float(np.mean(m)) if m else 0.0, "ndcg_at_10": float(np.mean(n)) if n else 0.0, "evaluated_user_count": int(len(truth_map)),
    }


def train_model(data_config: DataConfig, model_config: ModelConfig, train_config: TrainConfig) -> Path:
    if torch is None:  # pragma: no cover
        return _save_pending_plan(data_config, model_config, train_config)
    from .model import MultiModalColdStartModel

    _log("开始训练流程")
    _set_seed(train_config.seed)
    _log(f"随机种子: {train_config.seed}")
    prepare_protocol_files(data_config)
    _log(f"数据目录: {data_config.root}")
    t = load_multimodal_tables(data_config.root)
    users, item_tags, user_preferences, browsing_logs, mm_items = t["users"], t["item_tags"], t["user_preferences"], t["browsing_logs"], t["multimodal_items"]
    enc = _build_encoders(users)
    u_stats = _build_user_stats(browsing_logs)
    label_vocab = _build_label_vocab(user_preferences, model_config.label_count)
    targets = _build_multilabel_targets(user_preferences, label_vocab, model_config.label_count)
    u_pref = _build_user_preference_map(user_preferences)
    i_tag = _build_item_tag_weight_map(item_tags)
    inter = _build_interaction_map(browsing_logs)
    pop = mm_items.set_index("item_id").join(t["items"].set_index("item_id")[["popularity_score"]], how="left")["popularity_score"].fillna(0.0).to_dict()
    text_map, image_map, tag_map = _build_item_feature_maps(mm_items, item_tags, model_config.vocab_size)
    train_u, valid_u = _split_warm_users(users)
    train_pos, valid_pos = _positive_item_map(browsing_logs, train_u), _positive_item_map(browsing_logs, valid_u)
    all_items = mm_items["item_id"].astype(int).to_numpy()
    eval_items = _candidate_pool(t["items"], data_config.candidate_size)

    _log("开始构建训练/验证样本")
    build_start = time.perf_counter()
    train_rows = _build_samples(users, train_pos, all_items, u_stats, inter, pop, text_map, image_map, tag_map, targets, u_pref, i_tag, enc, data_config.max_seq_len, model_config.context_dim, train_config.negative_ratio_train, train_config.seed, train_config.hard_negative_ratio, train_config.cold_simulation_prob)
    valid_rows = _build_candidate_eval_rows(users, valid_u, valid_pos, eval_items, u_stats, inter, pop, text_map, image_map, tag_map, targets, u_pref, i_tag, enc, model_config.context_dim, data_config.max_seq_len)
    _log(f"样本构建完成，用时 {time.perf_counter() - build_start:.2f}s")
    if not train_rows or not valid_rows:
        raise RuntimeError("训练或验证样本为空，请检查数据。")

    train_loader = DataLoader(WarmRankingDataset(train_rows), batch_size=train_config.batch_size, shuffle=True, collate_fn=_collate_fn)
    valid_loader = DataLoader(WarmRankingDataset(valid_rows), batch_size=train_config.batch_size, shuffle=False, collate_fn=_collate_fn)
    device = _choose_device()
    model = MultiModalColdStartModel(
        vocab_size=model_config.vocab_size, text_embed_dim=model_config.text_embed_dim, text_hidden_dim=model_config.text_hidden_dim, image_dim=model_config.image_dim,
        context_dim=model_config.context_dim, tag_dim=model_config.tag_dim, lstm_hidden_dim=model_config.lstm_hidden_dim, fusion_dim=model_config.fusion_dim,
        label_count=model_config.label_count, match_dim=model_config.match_dim, dropout=model_config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    pos_count = sum(1 for r in train_rows if float(r["label"]) > 0.0)
    neg_count = max(1, len(train_rows) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device, dtype=torch.float32)
    rank_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ml_loss = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_path = data_config.model_dir / "best_model.pt"
    data_config.model_dir.mkdir(parents=True, exist_ok=True)
    hist: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    patience = 0

    _log(f"设备: {device}")
    if torch.cuda.is_available():
        _log(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    _log(f"训练样本数: {len(train_rows)}")
    _log(f"验证样本数: {len(valid_rows)}")
    _log(f"训练用户数: {len(train_u)}")
    _log(f"验证用户数: {len(valid_u)}")
    _log(f"正样本权重: {float(pos_weight.item()):.4f}")

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.perf_counter()
        _log(f"epoch {epoch}/{train_config.epochs} 开始")
        info = {"epoch": float(epoch)}
        info.update(_run_epoch(model, train_loader, optimizer, scheduler, rank_loss, ml_loss, device, train_config))
        scored = _score_loader(model, valid_loader, device)
        metrics = _recommendation_metrics(scored, valid_pos, data_config.top_k)
        info.update(metrics)
        hist.append(info)
        _log(
            f"epoch {epoch} 完成 "
            f"train_loss={info['train_loss']:.4f} "
            f"precision@10={metrics['precision_at_10']:.4f} "
            f"recall@10={metrics['recall_at_10']:.4f} "
            f"hit@10={metrics['hit_rate_at_10']:.4f} "
            f"ndcg@10={metrics['ndcg_at_10']:.4f} "
            f"epoch耗时={time.perf_counter() - epoch_start:.2f}s"
        )
        if best is None or metrics["precision_at_10"] > best["precision_at_10"]:
            best = metrics
            patience = 0
            torch.save({"model_state_dict": model.state_dict(), "data_config": _json_ready(asdict(data_config)), "model_config": _json_ready(asdict(model_config)), "train_config": _json_ready(asdict(train_config)), "label_vocab": label_vocab, "encoders": {k: {str(kk): int(vv) for kk, vv in m.items()} for k, m in enc.items()}}, best_path)
            _log(f"已保存最优模型 -> {best_path}")
        else:
            patience += 1
            _log(f"本轮无提升，耐心计数={patience}/{train_config.early_stopping_patience}")
            if patience >= train_config.early_stopping_patience:
                _log("触发提前停止")
                break

    summary_path = data_config.model_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"device": str(device), "train_sample_count": len(train_rows), "valid_sample_count": len(valid_rows), "history": hist, "best_metrics": best or {}, "artifacts": {"best_model": str(best_path)}, "optimizer": {"final_learning_rate": float(optimizer.param_groups[0]["lr"]), "pos_weight": float(pos_weight.item())}}, f, ensure_ascii=False, indent=2)
    _log(f"训练结果已写入 -> {summary_path}")
    return summary_path


if __name__ == "__main__":
    print(train_model(DataConfig(), ModelConfig(), TrainConfig()))
