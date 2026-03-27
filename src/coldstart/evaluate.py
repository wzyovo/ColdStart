from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import DataConfig, ModelConfig, TrainConfig
from .data_pipeline import prepare_protocol_files
from .datasets import load_multimodal_tables
from .trainer import (
    _build_encoders,
    _build_interaction_map,
    _build_item_feature_maps,
    _build_label_vocab,
    _build_multilabel_targets,
    _build_samples,
    _build_user_stats,
    _choose_device,
    _collate_fn,
    _interaction_sequence,
    _json_ready,
    _positive_item_map,
    _score_loader,
    _set_seed,
    _split_warm_users,
    _user_context_vector,
    WarmRankingDataset,
)

try:  # pragma: no cover
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = None


def ranking_metrics(recommendations: dict[int, list[int]], truth_map: dict[int, set[int]], top_k: int) -> dict[str, float]:
    precision_scores = []
    recall_scores = []
    hit_scores = []
    mrr_scores = []
    ndcg_scores = []

    for user_id, ranked_items in recommendations.items():
        true_items = truth_map.get(int(user_id), set())
        hits = [1 if item_id in true_items else 0 for item_id in ranked_items[:top_k]]
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
        "evaluated_user_count": int(len(recommendations)),
    }


def build_truth_map(data_dir: Path, stage_file: str) -> dict[int, set[int]]:
    df = pd.read_csv(data_dir / stage_file)
    return df.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(int).tolist())).to_dict()


def _recommendation_map(scored_df: pd.DataFrame, top_k: int) -> dict[int, list[int]]:
    recommendations: dict[int, list[int]] = {}
    for user_id, group in scored_df.groupby("user_id"):
        ranked = group.sort_values("score", ascending=False)["item_id"].astype(int).head(top_k).tolist()
        recommendations[int(user_id)] = ranked
    return recommendations


def _build_item_popularity(tables: dict[str, pd.DataFrame]) -> dict[int, float]:
    multimodal_items = tables["multimodal_items"]
    return (
        multimodal_items.set_index("item_id").join(tables["items"].set_index("item_id")[["popularity_score"]], how="left")[
            "popularity_score"
        ]
        .fillna(0.0)
        .to_dict()
    )


def _score_candidate_rows(model, rows: list[dict[str, np.ndarray | float | int]], batch_size: int, device) -> pd.DataFrame:
    dataset = WarmRankingDataset(rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
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
    encoders: dict[str, dict[object, int]],
    context_dim: int,
    max_seq_len: int,
) -> list[dict[str, np.ndarray | float | int]]:
    # 对每个冷启动用户，评估时使用共享候选集并补入该用户的真实目标商品。
    rows: list[dict[str, np.ndarray | float | int]] = []
    label_size = len(next(iter(multilabel_targets.values()))) if multilabel_targets else 16

    for row in cold_users.itertuples(index=False):
        user_id = int(row.user_id)
        user_series = pd.Series(row._asdict())
        target = multilabel_targets.get(user_id, np.zeros(label_size, dtype=np.float32))
        context = _user_context_vector(
            user_id,
            user_series,
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
                    "user_id": user_id,
                    "item_id": item_id,
                    "user_context": context,
                    "interaction_seq": sequence,
                    "item_text_tokens": text_map[item_id],
                    "item_image_vectors": image_map[item_id],
                    "item_tag_ids": tag_map[item_id],
                    "label": 0.0,
                    "multilabel_target": target,
                }
            )
    return rows


def _build_stage_user_frame(base_users: pd.DataFrame, stage_logs: pd.DataFrame, browsing_logs: pd.DataFrame) -> pd.DataFrame:
    stage_user_ids = set(stage_logs["user_id"].astype(int).tolist())
    return base_users[base_users["user_id"].isin(stage_user_ids)].copy()


def _candidate_pool(items: pd.DataFrame, size: int) -> np.ndarray:
    ranked = items.sort_values(["popularity_score", "user_coverage", "item_id"], ascending=[False, False, True])
    return ranked["item_id"].astype(int).head(size).to_numpy()


def _benchmark_model_latency(model, sample_rows: list[dict[str, np.ndarray | float | int]], batch_size: int, device) -> dict[str, float]:
    if not sample_rows:
        return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0}

    dataset = WarmRankingDataset(sample_rows[: min(len(sample_rows), 256)])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    measurements = []
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
            )
            if str(device) == "cuda":
                torch.cuda.synchronize()
            measurements.append((perf_counter() - start) * 1000.0)
    return {
        "p50_latency_ms": float(np.percentile(measurements, 50)) if measurements else 0.0,
        "p95_latency_ms": float(np.percentile(measurements, 95)) if measurements else 0.0,
        "p99_latency_ms": float(np.percentile(measurements, 99)) if measurements else 0.0,
    }


def evaluate_model(
    data_config: DataConfig = DataConfig(),
    model_config: ModelConfig = ModelConfig(),
    train_config: TrainConfig = TrainConfig(),
) -> Path:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run evaluation.")

    from .model import MultiModalColdStartModel

    _set_seed(train_config.seed)
    prepare_protocol_files(data_config)

    checkpoint_path = data_config.model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"检查点: {checkpoint_path}", flush=True)

    tables = load_multimodal_tables(data_config.root)
    users = tables["users"]
    item_tags = tables["item_tags"]
    user_preferences = tables["user_preferences"]
    browsing_logs = tables["browsing_logs"]
    items = tables["items"]
    multimodal_items = tables["multimodal_items"]

    encoders = _build_encoders(users)
    user_stats = _build_user_stats(browsing_logs)
    label_vocab = _build_label_vocab(user_preferences, model_config.label_count)
    multilabel_targets = _build_multilabel_targets(user_preferences, label_vocab, model_config.label_count)
    interaction_map = _build_interaction_map(browsing_logs)
    item_popularity = _build_item_popularity(tables)
    text_map, image_map, tag_map = _build_item_feature_maps(multimodal_items, item_tags, model_config.vocab_size)
    candidate_items = _candidate_pool(items, data_config.candidate_size)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = _choose_device()
    print(f"设备: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"候选集大小: {len(candidate_items)}", flush=True)
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
    model.load_state_dict(checkpoint["model_state_dict"])

    train_user_ids, valid_user_ids = _split_warm_users(users)
    print(f"Warm 训练用户数: {len(train_user_ids)}", flush=True)
    print(f"Warm 验证用户数: {len(valid_user_ids)}", flush=True)
    valid_positive = _positive_item_map(browsing_logs, valid_user_ids)
    valid_rows = _build_samples(
        users=users,
        positive_map=valid_positive,
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
        negative_ratio=train_config.negative_ratio_valid,
        seed=train_config.seed + 7,
        hard_negative_ratio=train_config.hard_negative_ratio,
    )
    valid_scored = _score_candidate_rows(model, valid_rows, train_config.batch_size, device)
    valid_recommendations = _recommendation_map(valid_scored, data_config.top_k)
    warm_valid_metrics = ranking_metrics(valid_recommendations, valid_positive, data_config.top_k)
    print(
        "Warm 验证集 "
        f"Precision@10={warm_valid_metrics['precision_at_10']:.4f} "
        f"Recall@10={warm_valid_metrics['recall_at_10']:.4f} "
        f"HitRate@10={warm_valid_metrics['hit_rate_at_10']:.4f} "
        f"NDCG@10={warm_valid_metrics['ndcg_at_10']:.4f}",
        flush=True,
    )

    cold_eval_users = pd.read_csv(data_config.root / "cold_start_eval_users.csv")
    cold_step1 = pd.read_csv(data_config.root / "cold_interactions_step1.csv")
    cold_step3 = pd.read_csv(data_config.root / "cold_interactions_step3.csv")
    print(f"Zero-shot 冷启动用户数: {cold_eval_users['user_id'].nunique()}", flush=True)
    print(f"One-shot 冷启动用户数: {cold_step1['user_id'].nunique()}", flush=True)
    print(f"Three-shot 冷启动用户数: {cold_step3['user_id'].nunique()}", flush=True)

    # Zero-shot 用首次交互商品作为真值，后续阶段则在更多交互后进行评估。
    cold_truth_zero = build_truth_map(data_config.root, "cold_interactions_step1.csv")
    cold_truth_one = build_truth_map(data_config.root, "cold_interactions_step3.csv")
    cold_truth_three = build_truth_map(data_config.root, "cold_interactions_step3.csv")

    zero_rows = _build_cold_rows(
        cold_users=cold_eval_users,
        candidate_items=candidate_items,
        truth_map=cold_truth_zero,
        user_stats=user_stats,
        interaction_map={},
        item_popularity=item_popularity,
        text_map=text_map,
        image_map=image_map,
        tag_map=tag_map,
        multilabel_targets=multilabel_targets,
        encoders=encoders,
        context_dim=model_config.context_dim,
        max_seq_len=data_config.max_seq_len,
    )
    print(f"Zero-shot 评估样本数: {len(zero_rows)}", flush=True)
    one_rows = _build_cold_rows(
        cold_users=_build_stage_user_frame(cold_eval_users, cold_step1, browsing_logs),
        candidate_items=candidate_items,
        truth_map=cold_truth_one,
        user_stats=user_stats,
        interaction_map=_build_interaction_map(cold_step1),
        item_popularity=item_popularity,
        text_map=text_map,
        image_map=image_map,
        tag_map=tag_map,
        multilabel_targets=multilabel_targets,
        encoders=encoders,
        context_dim=model_config.context_dim,
        max_seq_len=data_config.max_seq_len,
    )
    print(f"One-shot 评估样本数: {len(one_rows)}", flush=True)
    three_rows = _build_cold_rows(
        cold_users=_build_stage_user_frame(cold_eval_users, cold_step3, browsing_logs),
        candidate_items=candidate_items,
        truth_map=cold_truth_three,
        user_stats=user_stats,
        interaction_map=_build_interaction_map(cold_step3),
        item_popularity=item_popularity,
        text_map=text_map,
        image_map=image_map,
        tag_map=tag_map,
        multilabel_targets=multilabel_targets,
        encoders=encoders,
        context_dim=model_config.context_dim,
        max_seq_len=data_config.max_seq_len,
    )
    print(f"Three-shot 评估样本数: {len(three_rows)}", flush=True)

    print("开始评估 Zero-shot...", flush=True)
    zero_scored = _score_candidate_rows(model, zero_rows, train_config.batch_size, device)
    print("开始评估 One-shot...", flush=True)
    one_scored = _score_candidate_rows(model, one_rows, train_config.batch_size, device)
    print("开始评估 Three-shot...", flush=True)
    three_scored = _score_candidate_rows(model, three_rows, train_config.batch_size, device)

    zero_recommendations = _recommendation_map(zero_scored, data_config.top_k)
    one_recommendations = _recommendation_map(one_scored, data_config.top_k)
    three_recommendations = _recommendation_map(three_scored, data_config.top_k)

    cold_zero_metrics = ranking_metrics(zero_recommendations, cold_truth_zero, data_config.top_k)
    cold_one_metrics = ranking_metrics(one_recommendations, cold_truth_one, data_config.top_k)
    cold_three_metrics = ranking_metrics(three_recommendations, cold_truth_three, data_config.top_k)
    print(
        "Cold Zero-shot "
        f"Precision@10={cold_zero_metrics['precision_at_10']:.4f} "
        f"Recall@10={cold_zero_metrics['recall_at_10']:.4f} "
        f"HitRate@10={cold_zero_metrics['hit_rate_at_10']:.4f}",
        flush=True,
    )
    print(
        "Cold One-shot "
        f"Precision@10={cold_one_metrics['precision_at_10']:.4f} "
        f"Recall@10={cold_one_metrics['recall_at_10']:.4f} "
        f"HitRate@10={cold_one_metrics['hit_rate_at_10']:.4f}",
        flush=True,
    )
    print(
        "Cold Three-shot "
        f"Precision@10={cold_three_metrics['precision_at_10']:.4f} "
        f"Recall@10={cold_three_metrics['recall_at_10']:.4f} "
        f"HitRate@10={cold_three_metrics['hit_rate_at_10']:.4f}",
        flush=True,
    )

    cold_top10 = three_scored.sort_values(["user_id", "score"], ascending=[True, False]).groupby("user_id").head(data_config.top_k)
    cold_top10_path = data_config.model_dir / "cold_user_top10_multistage.csv"
    cold_top10.to_csv(cold_top10_path, index=False)

    latency = _benchmark_model_latency(model, zero_rows, train_config.batch_size, device)
    print(
        "延迟(ms) "
        f"p50={latency['p50_latency_ms']:.4f} "
        f"p95={latency['p95_latency_ms']:.4f} "
        f"p99={latency['p99_latency_ms']:.4f}",
        flush=True,
    )

    output = {
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "warm_valid_metrics": warm_valid_metrics,
        "cold_zero_shot_metrics": cold_zero_metrics,
        "cold_one_shot_metrics": cold_one_metrics,
        "cold_three_shot_metrics": cold_three_metrics,
        "latency_ms": latency,
        "artifacts": {
            "cold_user_top10_multistage": str(cold_top10_path),
        },
        "config": {
            "data_config": _json_ready(asdict(data_config)),
            "model_config": _json_ready(asdict(model_config)),
            "train_config": _json_ready(asdict(train_config)),
        },
    }
    output_path = data_config.model_dir / "evaluation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(f"评估结果已写入 -> {output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    print(evaluate_model())
