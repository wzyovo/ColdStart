from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("real_enhanced_data")
OUTPUT_DIR = Path("models") / "popularity_baseline"
TOP_K = 10


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(DATA_DIR / "users.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")
    browsing_logs = pd.read_csv(DATA_DIR / "browsing_logs.csv")
    return users, items, browsing_logs


def split_warm_users(users: pd.DataFrame) -> tuple[set[int], set[int]]:
    warm_users = users[users["user_type"] == "warm"].sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    split_index = max(1, int(len(warm_users) * 0.8))
    train_users = set(warm_users.iloc[:split_index]["user_id"].astype(int).tolist())
    valid_users = set(warm_users.iloc[split_index:]["user_id"].astype(int).tolist())
    return train_users, valid_users


def positive_interaction_map(browsing_logs: pd.DataFrame, user_ids: set[int]) -> dict[int, set[int]]:
    positive_types = {"click", "collect", "order"}
    logs = browsing_logs[browsing_logs["user_id"].isin(user_ids)].copy()
    logs = logs[logs["event_type"].isin(positive_types)]
    return logs.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(int).tolist())).to_dict()


def global_topk(items: pd.DataFrame, train_positive: dict[int, set[int]], top_k: int) -> list[int]:
    train_item_counts: dict[int, int] = {}
    for item_ids in train_positive.values():
        for item_id in item_ids:
            train_item_counts[item_id] = train_item_counts.get(item_id, 0) + 1

    ranked = items.copy()
    ranked["train_positive_count"] = ranked["item_id"].map(lambda item_id: train_item_counts.get(int(item_id), 0))
    ranked = ranked.sort_values(
        ["train_positive_count", "popularity_score", "user_coverage", "item_id"],
        ascending=[False, False, False, True],
    )
    return ranked["item_id"].astype(int).head(top_k).tolist()


def ranking_metrics(recommended_items: list[int], user_positive_map: dict[int, set[int]], top_k: int) -> dict[str, float]:
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    hit_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for true_items in user_positive_map.values():
        hits = [1 if item_id in true_items else 0 for item_id in recommended_items[:top_k]]
        hit_count = sum(hits)

        precision_scores.append(hit_count / float(top_k))
        recall_scores.append(hit_count / float(len(true_items)) if true_items else 0.0)
        hit_scores.append(1.0 if hit_count > 0 else 0.0)

        reciprocal_rank = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal_rank = 1.0 / rank
                break
        mrr_scores.append(reciprocal_rank)

        dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        ideal_hits = min(len(true_items), top_k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "evaluated_user_count": int(len(user_positive_map)),
        "precision_at_10": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_at_10": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "hit_rate_at_10": float(np.mean(hit_scores)) if hit_scores else 0.0,
        "mrr_at_10": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
    }


def save_outputs(top_items: list[int], metrics: dict[str, float]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rank": range(1, len(top_items) + 1), "item_id": top_items}).to_csv(
        OUTPUT_DIR / "top10_items.csv", index=False
    )
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"top10_items": top_items, "metrics": metrics}, file, ensure_ascii=False, indent=2)


def main() -> None:
    users, items, browsing_logs = load_tables()
    train_users, valid_users = split_warm_users(users)
    train_positive = positive_interaction_map(browsing_logs, train_users)
    valid_positive = positive_interaction_map(browsing_logs, valid_users)

    top_items = global_topk(items, train_positive, TOP_K)
    metrics = ranking_metrics(top_items, valid_positive, TOP_K)
    save_outputs(top_items, metrics)

    print(f"全局热门 Top-{TOP_K}: {top_items}")
    print("========== Baseline 评测结果 ==========")
    print(f"评测用户数: {metrics['evaluated_user_count']}")
    print(f"Precision@10: {metrics['precision_at_10']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"HitRate@10: {metrics['hit_rate_at_10']:.4f}")
    print(f"MRR@10: {metrics['mrr_at_10']:.4f}")
    print(f"NDCG@10: {metrics['ndcg_at_10']:.4f}")
    print("=======================================")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
