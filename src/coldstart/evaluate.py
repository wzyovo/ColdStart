from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import DataConfig, ModelConfig, TrainConfig
from .data_pipeline import prepare_protocol_files
from .datasets import load_multimodal_tables
from .trainer import (
    WarmRankingDataset,
    _build_encoders,
    _build_interaction_map,
    _build_item_feature_maps,
    _build_item_tag_weight_map,
    _build_label_vocab,
    _build_multilabel_targets,
    _build_samples,
    _build_user_preference_map,
    _build_user_stats,
    _choose_device,
    _collate_fn,
    _compute_match_features,
    _interaction_sequence,
    _json_ready,
    _positive_item_map,
    _score_loader,
    _set_seed,
    _split_warm_users,
    _user_context_vector,
)

try:  # pragma: no cover
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = None


def _log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def ranking_metrics(recommendations: dict[int, list[int]], truth_map: dict[int, set[int]], top_k: int) -> dict[str, float]:
    p, r, h, m, n = [], [], [], [], []
    for user_id, ranked_items in recommendations.items():
        truth = truth_map.get(int(user_id), set())
        hits = [1 if item in truth else 0 for item in ranked_items[:top_k]]
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
        "precision_at_10": float(np.mean(p)) if p else 0.0,
        "recall_at_10": float(np.mean(r)) if r else 0.0,
        "hit_rate_at_10": float(np.mean(h)) if h else 0.0,
        "mrr_at_10": float(np.mean(m)) if m else 0.0,
        "ndcg_at_10": float(np.mean(n)) if n else 0.0,
        "evaluated_user_count": int(len(recommendations)),
    }


def build_truth_map(data_dir: Path, stage_file: str) -> dict[int, set[int]]:
    df = pd.read_csv(data_dir / stage_file)
    return df.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(int).tolist())).to_dict()


def _recommendation_map(scored_df: pd.DataFrame, top_k: int) -> dict[int, list[int]]:
    return {
        int(user_id): group.sort_values("score", ascending=False)["item_id"].astype(int).head(top_k).tolist()
        for user_id, group in scored_df.groupby("user_id")
    }


def _build_item_popularity(tables: dict[str, pd.DataFrame]) -> dict[int, float]:
    return (
        tables["multimodal_items"]
        .set_index("item_id")
        .join(tables["items"].set_index("item_id")[["popularity_score"]], how="left")["popularity_score"]
        .fillna(0.0)
        .to_dict()
    )


def _score_candidate_rows(model, rows: list[dict[str, np.ndarray | float | int]], batch_size: int, device) -> pd.DataFrame:
    loader = DataLoader(WarmRankingDataset(rows), batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    return _score_loader(model, loader, device)


def _build_cold_rows(
    cold_users: pd.DataFrame,
    candidate_items: np.ndarray,
    truth_map: dict[int, set[int]],
    user_stats: dict[int, dict[str, float]],
    interaction_map: dict[int, list[dict[str, object]]],
    item_popularity: dict[int, float],
    text_map: dict[int, np.ndarray],
    image_map: dict[int, np.ndarray],
    tag_map: dict[int, np.ndarray],
    multilabel_targets: dict[int, np.ndarray],
    user_preference_map: dict[int, dict[str, dict[str, float]]],
    item_tag_weight_map: dict[int, dict[str, dict[str, float]]],
    encoders: dict[str, dict[object, int]],
    context_dim: int,
    max_seq_len: int,
) -> list[dict[str, np.ndarray | float | int]]:
    rows: list[dict[str, np.ndarray | float | int]] = []
    label_size = len(next(iter(multilabel_targets.values()))) if multilabel_targets else 16
    for row in cold_users.itertuples(index=False):
        user_id = int(row.user_id)
        user_series = pd.Series(row._asdict())
        target = multilabel_targets.get(user_id, np.zeros(label_size, dtype=np.float32))
        context = _user_context_vector(user_id, user_series, user_stats, encoders, context_dim, user_label_target=target)
        seq = _interaction_sequence(interaction_map, user_id, max_seq_len, item_popularity)
        stage_candidates = list(dict.fromkeys(candidate_items.tolist() + sorted(truth_map.get(user_id, set()))))
        for item_id in stage_candidates:
            item_id = int(item_id)
            if item_id not in text_map:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "user_context": context,
                    "interaction_seq": seq,
                    "item_text_tokens": text_map[item_id],
                    "item_image_vectors": image_map[item_id],
                    "item_tag_ids": tag_map[item_id],
                    "item_match_features": _compute_match_features(user_id, user_series, item_id, item_popularity, user_preference_map, item_tag_weight_map),
                    "label": float(item_id in truth_map.get(user_id, set())),
                    "multilabel_target": target,
                }
            )
    return rows


def _build_stage_user_frame(base_users: pd.DataFrame, stage_logs: pd.DataFrame) -> pd.DataFrame:
    stage_user_ids = set(stage_logs["user_id"].astype(int).tolist())
    return base_users[base_users["user_id"].isin(stage_user_ids)].copy()


def _candidate_pool(items: pd.DataFrame, size: int) -> np.ndarray:
    ranked = items.sort_values(["popularity_score", "user_coverage", "item_id"], ascending=[False, False, True])
    return ranked["item_id"].astype(int).head(size).to_numpy()


def _benchmark_model_latency(model, sample_rows: list[dict[str, np.ndarray | float | int]], batch_size: int, device) -> dict[str, float]:
    if not sample_rows:
        return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0}
    loader = DataLoader(WarmRankingDataset(sample_rows[: min(len(sample_rows), 256)]), batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    times = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            start = perf_counter()
            model(
                user_context=batch["user_context"].to(device),
                interaction_seq=batch["interaction_seq"].to(device),
                item_text_tokens=batch["item_text_tokens"].to(device),
                item_image_vectors=batch["item_image_vectors"].to(device),
                item_tag_ids=batch["item_tag_ids"].to(device),
                item_match_features=batch["item_match_features"].to(device),
            )
            if str(device) == "cuda":
                torch.cuda.synchronize()
            times.append((perf_counter() - start) * 1000.0)
    return {
        "p50_latency_ms": float(np.percentile(times, 50)) if times else 0.0,
        "p95_latency_ms": float(np.percentile(times, 95)) if times else 0.0,
        "p99_latency_ms": float(np.percentile(times, 99)) if times else 0.0,
    }


def evaluate_model(data_config: DataConfig = DataConfig(), model_config: ModelConfig = ModelConfig(), train_config: TrainConfig = TrainConfig()) -> Path:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run evaluation.")
    from .model import MultiModalColdStartModel

    _log("开始评估流程")
    _set_seed(train_config.seed)
    prepare_protocol_files(data_config)
    checkpoint_path = data_config.model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    _log(f"检查点: {checkpoint_path}")

    t = load_multimodal_tables(data_config.root)
    users, item_tags, user_preferences, browsing_logs, items, mm_items = t["users"], t["item_tags"], t["user_preferences"], t["browsing_logs"], t["items"], t["multimodal_items"]
    enc = _build_encoders(users)
    user_stats = _build_user_stats(browsing_logs)
    label_vocab = _build_label_vocab(user_preferences, model_config.label_count)
    targets = _build_multilabel_targets(user_preferences, label_vocab, model_config.label_count)
    user_pref_map = _build_user_preference_map(user_preferences)
    item_tag_map = _build_item_tag_weight_map(item_tags)
    inter = _build_interaction_map(browsing_logs)
    pop = _build_item_popularity(t)
    text_map, image_map, tag_map = _build_item_feature_maps(mm_items, item_tags, model_config.vocab_size)
    candidate_items = _candidate_pool(items, data_config.candidate_size)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = _choose_device()
    _log(f"设备: {device}")
    if torch.cuda.is_available():
        _log(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    _log(f"候选集大小: {len(candidate_items)}")
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
        match_dim=model_config.match_dim,
        dropout=model_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_u, valid_u = _split_warm_users(users)
    _log(f"Warm 训练用户数: {len(train_u)}")
    _log(f"Warm 验证用户数: {len(valid_u)}")
    valid_pos = _positive_item_map(browsing_logs, valid_u)
    valid_rows = _build_samples(
        users=users,
        positive_map=valid_pos,
        candidate_items=candidate_items,
        user_stats=user_stats,
        interaction_map=inter,
        item_popularity=pop,
        text_map=text_map,
        image_map=image_map,
        tag_map=tag_map,
        multilabel_targets=targets,
        user_preference_map=user_pref_map,
        item_tag_weight_map=item_tag_map,
        encoders=enc,
        max_seq_len=data_config.max_seq_len,
        context_dim=model_config.context_dim,
        negative_ratio=train_config.negative_ratio_valid,
        seed=train_config.seed + 7,
        hard_negative_ratio=train_config.hard_negative_ratio,
    )
    valid_scored = _score_candidate_rows(model, valid_rows, train_config.batch_size, device)
    warm_metrics = ranking_metrics(_recommendation_map(valid_scored, data_config.top_k), valid_pos, data_config.top_k)
    _log(
        f"Warm 验证 precision@10={warm_metrics['precision_at_10']:.4f} "
        f"recall@10={warm_metrics['recall_at_10']:.4f} hit@10={warm_metrics['hit_rate_at_10']:.4f} ndcg@10={warm_metrics['ndcg_at_10']:.4f}",
    )

    cold_eval_users = pd.read_csv(data_config.root / "cold_start_eval_users.csv")
    cold_step1 = pd.read_csv(data_config.root / "cold_interactions_step1.csv")
    cold_step3 = pd.read_csv(data_config.root / "cold_interactions_step3.csv")
    _log(f"cold_zero_shot_users: {cold_eval_users['user_id'].nunique()}")
    _log(f"cold_one_shot_users: {cold_step1['user_id'].nunique()}")
    _log(f"cold_three_shot_users: {cold_step3['user_id'].nunique()}")

    truth_zero = build_truth_map(data_config.root, "cold_interactions_step1.csv")
    truth_one = build_truth_map(data_config.root, "cold_interactions_step3.csv")
    truth_three = build_truth_map(data_config.root, "cold_interactions_step3.csv")
    zero_rows = _build_cold_rows(cold_eval_users, candidate_items, truth_zero, user_stats, {}, pop, text_map, image_map, tag_map, targets, user_pref_map, item_tag_map, enc, model_config.context_dim, data_config.max_seq_len)
    one_rows = _build_cold_rows(_build_stage_user_frame(cold_eval_users, cold_step1), candidate_items, truth_one, user_stats, _build_interaction_map(cold_step1), pop, text_map, image_map, tag_map, targets, user_pref_map, item_tag_map, enc, model_config.context_dim, data_config.max_seq_len)
    three_rows = _build_cold_rows(_build_stage_user_frame(cold_eval_users, cold_step3), candidate_items, truth_three, user_stats, _build_interaction_map(cold_step3), pop, text_map, image_map, tag_map, targets, user_pref_map, item_tag_map, enc, model_config.context_dim, data_config.max_seq_len)
    _log(f"zero_shot_rows: {len(zero_rows)}")
    _log(f"one_shot_rows: {len(one_rows)}")
    _log(f"three_shot_rows: {len(three_rows)}")

    zero_scored = _score_candidate_rows(model, zero_rows, train_config.batch_size, device)
    one_scored = _score_candidate_rows(model, one_rows, train_config.batch_size, device)
    three_scored = _score_candidate_rows(model, three_rows, train_config.batch_size, device)
    zero_metrics = ranking_metrics(_recommendation_map(zero_scored, data_config.top_k), truth_zero, data_config.top_k)
    one_metrics = ranking_metrics(_recommendation_map(one_scored, data_config.top_k), truth_one, data_config.top_k)
    three_metrics = ranking_metrics(_recommendation_map(three_scored, data_config.top_k), truth_three, data_config.top_k)
    _log(
        f"cold_zero_shot precision@10={zero_metrics['precision_at_10']:.4f} recall@10={zero_metrics['recall_at_10']:.4f} hit@10={zero_metrics['hit_rate_at_10']:.4f}",
    )
    _log(
        f"cold_one_shot precision@10={one_metrics['precision_at_10']:.4f} recall@10={one_metrics['recall_at_10']:.4f} hit@10={one_metrics['hit_rate_at_10']:.4f}",
    )
    _log(
        f"cold_three_shot precision@10={three_metrics['precision_at_10']:.4f} recall@10={three_metrics['recall_at_10']:.4f} hit@10={three_metrics['hit_rate_at_10']:.4f}",
    )

    top10_path = data_config.model_dir / "cold_user_top10_multistage.csv"
    three_scored.sort_values(["user_id", "score"], ascending=[True, False]).groupby("user_id").head(data_config.top_k).to_csv(top10_path, index=False)
    latency = _benchmark_model_latency(model, zero_rows, train_config.batch_size, device)
    _log(f"latency_ms p50={latency['p50_latency_ms']:.4f} p95={latency['p95_latency_ms']:.4f} p99={latency['p99_latency_ms']:.4f}")

    output = {
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "warm_valid_metrics": warm_metrics,
        "cold_zero_shot_metrics": zero_metrics,
        "cold_one_shot_metrics": one_metrics,
        "cold_three_shot_metrics": three_metrics,
        "latency_ms": latency,
        "artifacts": {"cold_user_top10_multistage": str(top10_path)},
        "config": {
            "data_config": _json_ready(asdict(data_config)),
            "model_config": _json_ready(asdict(model_config)),
            "train_config": _json_ready(asdict(train_config)),
        },
    }
    output_path = data_config.model_dir / "evaluation.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    _log(f"评估结果已写入 -> {output_path}")
    return output_path


if __name__ == "__main__":
    print(evaluate_model())
