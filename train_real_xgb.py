from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


DATA_DIR = Path("real_enhanced_data")
OUTPUT_DIR = Path("models") / "real_xgb_ranker"
RANDOM_SEED = 42
NEGATIVE_PER_POSITIVE = 4
RANKING_NEGATIVE_COUNT = 200
TOP_K = 10


def load_tables() -> tuple[pd.DataFrame, ...]:
    users = pd.read_csv(DATA_DIR / "users.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")
    item_tags = pd.read_csv(DATA_DIR / "item_tags.csv")
    user_preferences = pd.read_csv(DATA_DIR / "user_preferences.csv")
    browsing_logs = pd.read_csv(DATA_DIR / "browsing_logs.csv")
    cold_core = pd.read_csv(DATA_DIR / "cold_start_users.csv")
    return users, items, item_tags, user_preferences, browsing_logs, cold_core


def build_encoders(users: pd.DataFrame, items: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "circle_id": {value: idx for idx, value in enumerate(sorted(users["circle_id"].dropna().unique()))},
        "budget_level": {value: idx for idx, value in enumerate(sorted(users["budget_level"].dropna().unique()))},
        "primary_cuisine": {value: idx for idx, value in enumerate(sorted(items["primary_cuisine"].dropna().unique()))},
        "price_band": {value: idx for idx, value in enumerate(sorted(items["price_band"].dropna().unique()))},
    }


def attach_user_stats(users: pd.DataFrame, browsing_logs: pd.DataFrame) -> pd.DataFrame:
    event_strength = {"browse": 1.0, "click": 2.0, "collect": 3.0, "order": 4.0}
    logs = browsing_logs.copy()
    logs["event_strength"] = logs["event_type"].map(event_strength)

    stats = (
        logs.groupby("user_id")
        .agg(
            behavior_count=("item_id", "size"),
            order_count=("event_type", lambda s: int((s == "order").sum())),
            collect_count=("event_type", lambda s: int((s == "collect").sum())),
            click_count=("event_type", lambda s: int((s == "click").sum())),
            avg_dwell_ms=("dwell_ms", "mean"),
            avg_scroll_depth=("scroll_depth", "mean"),
            avg_event_strength=("event_strength", "mean"),
        )
        .reset_index()
    )

    users = users.merge(stats, on="user_id", how="left")
    fill_zero_columns = [
        "behavior_count",
        "order_count",
        "collect_count",
        "click_count",
        "avg_dwell_ms",
        "avg_scroll_depth",
        "avg_event_strength",
    ]
    users[fill_zero_columns] = users[fill_zero_columns].fillna(0)
    return users


def attach_item_stats(items: pd.DataFrame, item_tags: pd.DataFrame) -> pd.DataFrame:
    tag_stats = (
        item_tags.groupby("item_id")
        .agg(
            tag_count=("tag_name", "size"),
            avg_tag_weight=("tag_weight", "mean"),
        )
        .reset_index()
    )
    items = items.merge(tag_stats, on="item_id", how="left")
    items[["tag_count", "avg_tag_weight"]] = items[["tag_count", "avg_tag_weight"]].fillna(0)
    return items


def build_tag_maps(
    user_preferences: pd.DataFrame,
    item_tags: pd.DataFrame,
) -> tuple[dict[int, set[str]], dict[int, dict[str, float]], dict[int, set[str]], dict[int, str]]:
    user_tag_map = user_preferences.groupby("user_id")["tag_name"].apply(set).to_dict()
    user_pref_weight_map = (
        user_preferences.groupby(["user_id", "tag_name"])["preference_weight"].mean().reset_index()
        .groupby("user_id")
        .apply(lambda df: dict(zip(df["tag_name"], df["preference_weight"])))
        .to_dict()
    )
    item_tag_map = item_tags.groupby("item_id")["tag_name"].apply(set).to_dict()
    item_price_map = (
        item_tags[item_tags["tag_type"] == "price"]
        .drop_duplicates(subset=["item_id"])
        .set_index("item_id")["tag_name"]
        .to_dict()
    )
    return user_tag_map, user_pref_weight_map, item_tag_map, item_price_map


def split_warm_users(users: pd.DataFrame) -> tuple[set[int], set[int]]:
    warm_users = users[users["user_type"] == "warm"].sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    split_index = max(1, int(len(warm_users) * 0.8))
    train_users = set(warm_users.iloc[:split_index]["user_id"].astype(int).tolist())
    valid_users = set(warm_users.iloc[split_index:]["user_id"].astype(int).tolist())
    return train_users, valid_users


def positive_interaction_map(browsing_logs: pd.DataFrame, user_ids: set[int]) -> dict[int, set[int]]:
    logs = browsing_logs[browsing_logs["user_id"].isin(user_ids)].copy()
    positive_types = {"click", "collect", "order"}
    logs = logs[logs["event_type"].isin(positive_types)]
    return logs.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(int).tolist())).to_dict()


def sample_negative_items(
    candidate_items: np.ndarray,
    positive_items: set[int],
    sample_size: int,
    rng: np.random.Generator,
) -> list[int]:
    available = np.array([item for item in candidate_items if item not in positive_items], dtype=np.int64)
    if len(available) == 0:
        return []
    actual_size = min(sample_size, len(available))
    sampled = rng.choice(available, size=actual_size, replace=False)
    return sampled.astype(int).tolist()


def make_feature_row(
    user_row: pd.Series,
    item_row: pd.Series,
    user_tags: set[str],
    user_pref_weights: dict[str, float],
    item_tags: set[str],
    item_price_tag: str | None,
    encoders: dict[str, dict[str, int]],
) -> dict[str, float | int]:
    overlap_tags = user_tags.intersection(item_tags)
    weighted_overlap = float(sum(user_pref_weights.get(tag, 0.0) for tag in overlap_tags))
    price_match = int(item_price_tag == user_row["budget_level"])

    return {
        "timestamp": int(user_row["timestamp"]),
        "device_type": int(user_row["device_type"]),
        "location_id": int(user_row["location_id"]),
        "query_id": int(user_row["query_id"]),
        "circle_id_enc": encoders["circle_id"][user_row["circle_id"]],
        "budget_level_enc": encoders["budget_level"][user_row["budget_level"]],
        "behavior_count": float(user_row["behavior_count"]),
        "order_count": float(user_row["order_count"]),
        "collect_count": float(user_row["collect_count"]),
        "click_count": float(user_row["click_count"]),
        "avg_dwell_ms": float(user_row["avg_dwell_ms"]),
        "avg_scroll_depth": float(user_row["avg_scroll_depth"]),
        "avg_event_strength": float(user_row["avg_event_strength"]),
        "merchant_id": int(item_row["merchant_id"]),
        "primary_cuisine_enc": encoders["primary_cuisine"][item_row["primary_cuisine"]],
        "price_band_enc": encoders["price_band"][item_row["price_band"]],
        "popularity_score": float(item_row["popularity_score"]),
        "user_coverage": float(item_row["user_coverage"]),
        "avg_position": float(item_row["avg_position"]),
        "tag_count": float(item_row["tag_count"]),
        "avg_tag_weight": float(item_row["avg_tag_weight"]),
        "tag_overlap_count": float(len(overlap_tags)),
        "weighted_tag_overlap": weighted_overlap,
        "price_match": price_match,
    }


def build_samples(
    users: pd.DataFrame,
    items: pd.DataFrame,
    user_positive_map: dict[int, set[int]],
    user_tag_map: dict[int, set[str]],
    user_pref_weight_map: dict[int, dict[str, float]],
    item_tag_map: dict[int, set[str]],
    item_price_map: dict[int, str],
    encoders: dict[str, dict[str, int]],
    negative_per_positive: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    user_lookup = users.set_index("user_id")
    item_lookup = items.set_index("item_id")
    candidate_items = items["item_id"].astype(int).to_numpy()

    rows: list[dict[str, float | int]] = []
    for user_id, positive_items in user_positive_map.items():
        if user_id not in user_lookup.index:
            continue
        user_row = user_lookup.loc[user_id]
        user_tags = user_tag_map.get(user_id, set())
        user_pref_weights = user_pref_weight_map.get(user_id, {})

        negatives = sample_negative_items(
            candidate_items=candidate_items,
            positive_items=positive_items,
            sample_size=max(len(positive_items) * negative_per_positive, negative_per_positive),
            rng=rng,
        )

        for item_id in positive_items:
            item_row = item_lookup.loc[item_id]
            feature_row = make_feature_row(
                user_row,
                item_row,
                user_tags,
                user_pref_weights,
                item_tag_map.get(item_id, set()),
                item_price_map.get(item_id),
                encoders,
            )
            feature_row.update({"user_id": int(user_id), "item_id": int(item_id), "label": 1})
            rows.append(feature_row)

        for item_id in negatives:
            item_row = item_lookup.loc[item_id]
            feature_row = make_feature_row(
                user_row,
                item_row,
                user_tags,
                user_pref_weights,
                item_tag_map.get(item_id, set()),
                item_price_map.get(item_id),
                encoders,
            )
            feature_row.update({"user_id": int(user_id), "item_id": int(item_id), "label": 0})
            rows.append(feature_row)

    return pd.DataFrame(rows)


def feature_columns() -> list[str]:
    return [
        "timestamp",
        "device_type",
        "location_id",
        "query_id",
        "circle_id_enc",
        "budget_level_enc",
        "behavior_count",
        "order_count",
        "collect_count",
        "click_count",
        "avg_dwell_ms",
        "avg_scroll_depth",
        "avg_event_strength",
        "merchant_id",
        "primary_cuisine_enc",
        "price_band_enc",
        "popularity_score",
        "user_coverage",
        "avg_position",
        "tag_count",
        "avg_tag_weight",
        "tag_overlap_count",
        "weighted_tag_overlap",
        "price_match",
    ]


def train_xgb(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[xgb.Booster, dict[str, dict[str, list[float]]], str]:
    dtrain = xgb.DMatrix(train_df[feature_columns()], label=train_df["label"], feature_names=feature_columns())
    dvalid = xgb.DMatrix(valid_df[feature_columns()], label=valid_df["label"], feature_names=feature_columns())

    def params(device: str) -> dict[str, object]:
        return {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
            "device": device,
            "eta": 0.06,
            "max_depth": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "seed": RANDOM_SEED,
        }

    for device in ["cuda", "cpu"]:
        evals_result: dict[str, dict[str, list[float]]] = {}
        try:
            booster = xgb.train(
                params(device),
                dtrain,
                num_boost_round=400,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=30,
                evals_result=evals_result,
                verbose_eval=False,
            )
            return booster, evals_result, device
        except xgb.core.XGBoostError:
            continue
    raise RuntimeError("XGBoost training failed on both GPU and CPU.")


def ranking_metrics(scored_df: pd.DataFrame, user_positive_map: dict[int, set[int]], top_k: int = TOP_K) -> dict[str, float]:
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    hit_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for user_id, group in scored_df.groupby("user_id"):
        ranked_items = group.sort_values("score", ascending=False)["item_id"].astype(int).head(top_k).tolist()
        true_items = user_positive_map[int(user_id)]
        hits = [1 if item in true_items else 0 for item in ranked_items]
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
        "precision_at_10": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_at_10": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "hit_rate_at_10": float(np.mean(hit_scores)) if hit_scores else 0.0,
        "mrr_at_10": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "evaluated_user_count": int(len(scored_df["user_id"].unique())) if not scored_df.empty else 0,
    }


def build_ranking_candidates(
    users: pd.DataFrame,
    items: pd.DataFrame,
    user_positive_map: dict[int, set[int]],
    user_tag_map: dict[int, set[str]],
    user_pref_weight_map: dict[int, dict[str, float]],
    item_tag_map: dict[int, set[str]],
    item_price_map: dict[int, str],
    encoders: dict[str, dict[str, int]],
    negative_count: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    item_lookup = items.set_index("item_id")
    user_lookup = users.set_index("user_id")
    candidate_items = items["item_id"].astype(int).to_numpy()
    rows: list[dict[str, float | int]] = []

    for user_id, positive_items in user_positive_map.items():
        if user_id not in user_lookup.index:
            continue
        negatives = sample_negative_items(candidate_items, positive_items, negative_count, rng)
        user_row = user_lookup.loc[user_id]
        user_tags = user_tag_map.get(user_id, set())
        user_pref_weights = user_pref_weight_map.get(user_id, {})

        for item_id in list(positive_items) + negatives:
            item_row = item_lookup.loc[item_id]
            feature_row = make_feature_row(
                user_row,
                item_row,
                user_tags,
                user_pref_weights,
                item_tag_map.get(item_id, set()),
                item_price_map.get(item_id),
                encoders,
            )
            feature_row.update({"user_id": int(user_id), "item_id": int(item_id), "label": int(item_id in positive_items)})
            rows.append(feature_row)

    return pd.DataFrame(rows)


def build_cold_candidates(
    users: pd.DataFrame,
    items: pd.DataFrame,
    cold_core: pd.DataFrame,
    user_tag_map: dict[int, set[str]],
    user_pref_weight_map: dict[int, dict[str, float]],
    item_tag_map: dict[int, set[str]],
    item_price_map: dict[int, str],
    encoders: dict[str, dict[str, int]],
) -> pd.DataFrame:
    candidate_items = items.sort_values(["popularity_score", "user_coverage"], ascending=[False, False]).head(300)
    item_lookup = candidate_items.set_index("item_id")
    user_lookup = users.set_index("user_id")

    rows: list[dict[str, float | int]] = []
    for user_id in cold_core["user_id"].astype(int).tolist():
        if user_id not in user_lookup.index:
            continue
        user_row = user_lookup.loc[user_id]
        user_tags = user_tag_map.get(user_id, set())
        user_pref_weights = user_pref_weight_map.get(user_id, {})

        for item_id in candidate_items["item_id"].astype(int).tolist():
            item_row = item_lookup.loc[item_id]
            feature_row = make_feature_row(
                user_row,
                item_row,
                user_tags,
                user_pref_weights,
                item_tag_map.get(item_id, set()),
                item_price_map.get(item_id),
                encoders,
            )
            feature_row.update({"user_id": int(user_id), "item_id": int(item_id)})
            rows.append(feature_row)

    return pd.DataFrame(rows)


def save_outputs(
    booster: xgb.Booster,
    evals_result: dict[str, dict[str, list[float]]],
    ranking_result: dict[str, float],
    cold_topn: pd.DataFrame,
    used_device: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUTPUT_DIR / "model.json"))
    cold_topn.to_csv(OUTPUT_DIR / "cold_user_top10.csv", index=False)

    metrics = {
        "device": used_device,
        "best_iteration": int(booster.best_iteration),
        "best_score": float(booster.best_score),
        "evals_result": evals_result,
        "ranking_metrics": ranking_result,
        "feature_importance_gain": booster.get_score(importance_type="gain"),
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def main() -> None:
    users, items, item_tags, user_preferences, browsing_logs, cold_core = load_tables()
    users = attach_user_stats(users, browsing_logs)
    items = attach_item_stats(items, item_tags)
    encoders = build_encoders(users, items)
    user_tag_map, user_pref_weight_map, item_tag_map, item_price_map = build_tag_maps(user_preferences, item_tags)

    train_users, valid_users = split_warm_users(users)
    train_positive = positive_interaction_map(browsing_logs, train_users)
    valid_positive = positive_interaction_map(browsing_logs, valid_users)

    train_df = build_samples(
        users=users,
        items=items,
        user_positive_map=train_positive,
        user_tag_map=user_tag_map,
        user_pref_weight_map=user_pref_weight_map,
        item_tag_map=item_tag_map,
        item_price_map=item_price_map,
        encoders=encoders,
        negative_per_positive=NEGATIVE_PER_POSITIVE,
    )
    valid_df = build_samples(
        users=users,
        items=items,
        user_positive_map=valid_positive,
        user_tag_map=user_tag_map,
        user_pref_weight_map=user_pref_weight_map,
        item_tag_map=item_tag_map,
        item_price_map=item_price_map,
        encoders=encoders,
        negative_per_positive=NEGATIVE_PER_POSITIVE,
    )

    booster, evals_result, used_device = train_xgb(train_df, valid_df)

    ranking_df = build_ranking_candidates(
        users=users,
        items=items,
        user_positive_map=valid_positive,
        user_tag_map=user_tag_map,
        user_pref_weight_map=user_pref_weight_map,
        item_tag_map=item_tag_map,
        item_price_map=item_price_map,
        encoders=encoders,
        negative_count=RANKING_NEGATIVE_COUNT,
    )
    drank = xgb.DMatrix(ranking_df[feature_columns()], feature_names=feature_columns())
    ranking_df["score"] = booster.predict(drank)
    ranking_result = ranking_metrics(ranking_df, valid_positive, top_k=TOP_K)

    cold_df = build_cold_candidates(
        users=users,
        items=items,
        cold_core=cold_core,
        user_tag_map=user_tag_map,
        user_pref_weight_map=user_pref_weight_map,
        item_tag_map=item_tag_map,
        item_price_map=item_price_map,
        encoders=encoders,
    )
    dcold = xgb.DMatrix(cold_df[feature_columns()], feature_names=feature_columns())
    cold_df["score"] = booster.predict(dcold)
    cold_topn = (
        cold_df.sort_values(["user_id", "score"], ascending=[True, False])
        .groupby("user_id")
        .head(TOP_K)[["user_id", "item_id", "score"]]
        .reset_index(drop=True)
    )

    save_outputs(booster, evals_result, ranking_result, cold_topn, used_device)

    print(f"训练样本数: {len(train_df)}")
    print(f"验证样本数: {len(valid_df)}")
    print(f"训练设备: {used_device}")
    print(f"最佳轮数: {booster.best_iteration}")
    print(f"验证集 AUC: {evals_result['valid']['auc'][booster.best_iteration]:.4f}")
    print(f"验证集 LogLoss: {evals_result['valid']['logloss'][booster.best_iteration]:.4f}")
    print("========== 排序评测结果 ==========")
    print(f"评测用户数: {ranking_result['evaluated_user_count']}")
    print(f"Precision@10: {ranking_result['precision_at_10']:.4f}")
    print(f"Recall@10: {ranking_result['recall_at_10']:.4f}")
    print(f"HitRate@10: {ranking_result['hit_rate_at_10']:.4f}")
    print(f"MRR@10: {ranking_result['mrr_at_10']:.4f}")
    print(f"NDCG@10: {ranking_result['ndcg_at_10']:.4f}")
    print("==================================")
    print(f"模型输出目录: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
