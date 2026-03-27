from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("tianchi_public_data_train_new.txt")
OUTPUT_DIR = Path("real_enhanced_data")

RANDOM_SEED = 42
WARM_USER_COUNT = 3200
COLD_USER_COUNT = 100
MERCHANT_COUNT = 520
ITEM_POOL_COUNT = 15000
SEQUENCE_COLUMNS = ["item_id_seq", "aux_seq_1", "aux_seq_2"]
INVALID_CONTEXT_VALUES = {0, -1}

TASTE_TAGS = ["spicy", "sweet", "savory", "crispy", "umami", "fresh", "mild", "rich"]
SCENE_TAGS = ["lunch", "dinner", "late_night", "afternoon_tea", "solo_meal", "group_order"]
PRICE_TAGS = ["budget", "midrange", "premium", "luxury"]
CIRCLE_TAGS = [
    "office_white_collar",
    "campus_students",
    "night_owl_gamers",
    "fitness_group",
    "family_group",
    "budget_life",
    "foodie_hunters",
]
CUISINES = ["sichuan", "cantonese", "hotpot", "barbecue", "milk_tea", "dessert", "burger", "noodles", "salad"]


def build_column_names(column_count: int) -> list[str]:
    base_count = column_count - len(SEQUENCE_COLUMNS)
    base_columns = [f"feature_{idx}" for idx in range(base_count)]
    return base_columns + SEQUENCE_COLUMNS


def split_sequence(value: object) -> list[int]:
    if pd.isna(value):
        return []

    values: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed != 0:
            values.append(parsed)
    return values


def load_cleaned_users() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, sep="\t", header=None)
    df.columns = build_column_names(df.shape[1])
    df = df.rename(
        columns={
            "feature_1": "user_id",
            "feature_2": "timestamp",
            "feature_3": "device_type",
            "feature_4": "location_id",
            "feature_5": "query_id",
        }
    )

    for column in ["timestamp", "device_type", "location_id"]:
        df = df[df[column].notna()]
        df = df[~df[column].isin(INVALID_CONTEXT_VALUES)]

    first_rows = (
        df.sort_values(["user_id", "timestamp"])
        .drop_duplicates(subset=["user_id"], keep="first")
        .reset_index(drop=True)
    )
    return first_rows


def select_real_users(first_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    actual_cold = min(COLD_USER_COUNT, len(first_rows))
    cold_users = first_rows.sample(n=actual_cold, random_state=RANDOM_SEED).reset_index(drop=True)

    remaining = first_rows[~first_rows["user_id"].isin(set(cold_users["user_id"]))].reset_index(drop=True)
    actual_warm = min(WARM_USER_COUNT, len(remaining))
    warm_indices = rng.choice(len(remaining), size=actual_warm, replace=False)
    warm_users = remaining.iloc[warm_indices].sort_values("user_id").reset_index(drop=True)
    return warm_users, cold_users


def build_merchants() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for merchant_id in range(1, MERCHANT_COUNT + 1):
        rows.append(
            {
                "merchant_id": merchant_id,
                "merchant_name": f"merchant_{merchant_id:04d}",
                "primary_cuisine": CUISINES[(merchant_id - 1) % len(CUISINES)],
                "city_zone": f"zone_{(merchant_id - 1) % 20 + 1:02d}",
                "brand_score": round(0.45 + ((merchant_id * 17) % 45) / 100, 3),
                "service_score": round(0.5 + ((merchant_id * 13) % 40) / 100, 3),
            }
        )
    return pd.DataFrame(rows)


def derive_item_stats(users: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, int]] = []

    for _, row in users.iterrows():
        item_ids = split_sequence(row["item_id_seq"])
        aux_1 = split_sequence(row["aux_seq_1"])
        aux_2 = split_sequence(row["aux_seq_2"])
        shared_len = min(len(item_ids), len(aux_1), len(aux_2))

        for position in range(shared_len):
            records.append(
                {
                    "user_id": int(row["user_id"]),
                    "item_id": int(item_ids[position]),
                    "position": position,
                    "aux_1": int(aux_1[position]),
                    "aux_2": int(aux_2[position]),
                }
            )

    interaction_df = pd.DataFrame(records)
    item_stats = (
        interaction_df.groupby("item_id")
        .agg(
            exposure_count=("item_id", "size"),
            user_count=("user_id", "nunique"),
            avg_position=("position", "mean"),
            dominant_aux_1=("aux_1", lambda s: int(s.mode().iat[0])),
            dominant_aux_2=("aux_2", lambda s: int(s.mode().iat[0])),
        )
        .reset_index()
        .sort_values(["exposure_count", "user_count", "item_id"], ascending=[False, False, True])
        .head(ITEM_POOL_COUNT)
        .reset_index(drop=True)
    )
    return item_stats


def build_items(item_stats: pd.DataFrame, merchants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_df = item_stats.copy()
    item_df["merchant_id"] = item_df["item_id"].map(lambda item_id: (item_id % MERCHANT_COUNT) + 1)
    percentile = item_df["exposure_count"].rank(method="average", pct=True)
    item_df["price_band"] = percentile.map(
        lambda value: "budget"
        if value <= 0.25
        else "midrange"
        if value <= 0.5
        else "premium"
        if value <= 0.75
        else "luxury"
    )
    merchant_lookup = merchants.set_index("merchant_id")
    item_rows: list[dict[str, object]] = []
    tag_rows: list[dict[str, object]] = []

    for row in item_df.itertuples(index=False):
        merchant = merchant_lookup.loc[row.merchant_id]
        primary_scene = SCENE_TAGS[int(row.avg_position) % len(SCENE_TAGS)]
        secondary_scene = SCENE_TAGS[(int(row.dominant_aux_2) + row.item_id) % len(SCENE_TAGS)]
        taste_1 = TASTE_TAGS[(int(row.dominant_aux_1) + row.item_id) % len(TASTE_TAGS)]
        taste_2 = TASTE_TAGS[(int(row.dominant_aux_2) + row.item_id) % len(TASTE_TAGS)]
        price_tag = str(row.price_band)
        tags = [taste_1, taste_2, primary_scene, secondary_scene, price_tag]

        item_rows.append(
            {
                "item_id": row.item_id,
                "merchant_id": row.merchant_id,
                "item_name": f"real_item_{row.item_id}",
                "primary_cuisine": merchant["primary_cuisine"],
                "price_band": price_tag,
                "popularity_score": float(row.exposure_count),
                "user_coverage": int(row.user_count),
                "avg_position": float(round(row.avg_position, 3)),
                "description_text": " | ".join(dict.fromkeys(tags)),
            }
        )

        for tag in dict.fromkeys(tags):
            if tag in TASTE_TAGS:
                tag_type = "taste"
            elif tag in SCENE_TAGS:
                tag_type = "scene"
            else:
                tag_type = "price"
            tag_rows.append(
                {
                    "item_id": row.item_id,
                    "tag_type": tag_type,
                    "tag_name": tag,
                    "tag_weight": round(0.45 + ((row.item_id + len(tag)) % 45) / 100, 3),
                }
            )

    items = pd.DataFrame(item_rows)
    item_tags = pd.DataFrame(tag_rows)
    return items, item_tags


def infer_user_circle(row: pd.Series) -> str:
    score = int(row["device_type"]) + int(row["location_id"]) + int(row["query_id"]) + int(row["timestamp"])
    return CIRCLE_TAGS[score % len(CIRCLE_TAGS)]


def build_user_profiles(
    users: pd.DataFrame,
    cold_users: pd.DataFrame,
    item_tags: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tag_map = item_tags.groupby("item_id")["tag_name"].apply(list).to_dict()

    def collect_preferences(row: pd.Series) -> list[str]:
        item_ids = split_sequence(row["item_id_seq"])
        tags: list[str] = []
        for item_id in item_ids[:12]:
            tags.extend(tag_map.get(item_id, []))
        if not tags:
            return ["budget", "lunch", "savory"]
        counts = pd.Series(tags).value_counts().head(5).index.tolist()
        return counts

    def build_profile_table(source_df: pd.DataFrame, user_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        user_rows: list[dict[str, object]] = []
        pref_rows: list[dict[str, object]] = []

        for _, row in source_df.iterrows():
            preferences = collect_preferences(row)
            circle = infer_user_circle(row)
            price_tag = next((tag for tag in preferences if tag in PRICE_TAGS), "midrange")

            user_rows.append(
                {
                    "user_id": int(row["user_id"]),
                    "user_type": user_type,
                    "timestamp": int(row["timestamp"]),
                    "device_type": int(row["device_type"]),
                    "location_id": int(row["location_id"]),
                    "query_id": int(row["query_id"]),
                    "circle_id": circle,
                    "budget_level": price_tag,
                }
            )

            for rank, tag in enumerate(preferences, start=1):
                if tag in TASTE_TAGS:
                    tag_type = "taste"
                elif tag in SCENE_TAGS:
                    tag_type = "scene"
                elif tag in PRICE_TAGS:
                    tag_type = "price"
                else:
                    tag_type = "other"
                pref_rows.append(
                    {
                        "user_id": int(row["user_id"]),
                        "tag_type": tag_type,
                        "tag_name": tag,
                        "preference_weight": round(max(0.3, 0.95 - rank * 0.1), 3),
                    }
                )

        return pd.DataFrame(user_rows), pd.DataFrame(pref_rows)

    warm_profiles, warm_preferences = build_profile_table(users, "warm")
    cold_profiles, cold_preferences = build_profile_table(cold_users, "cold")
    all_users = pd.concat([warm_profiles, cold_profiles], ignore_index=True)
    all_preferences = pd.concat([warm_preferences, cold_preferences], ignore_index=True)
    user_circles = all_users[["user_id", "circle_id"]].copy()
    return all_users, all_preferences, user_circles


def explode_behavior_logs(users: pd.DataFrame, allowed_items: set[int]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    base_time = pd.Timestamp("2026-01-01 08:00:00")

    for _, row in users.iterrows():
        item_ids = split_sequence(row["item_id_seq"])
        aux_1 = split_sequence(row["aux_seq_1"])
        aux_2 = split_sequence(row["aux_seq_2"])
        shared_len = min(len(item_ids), len(aux_1), len(aux_2))

        for position in range(shared_len):
            item_id = item_ids[position]
            if item_id not in allowed_items:
                continue

            event_score = aux_1[position] % 10
            event_type = "browse"
            if event_score >= 8:
                event_type = "order"
            elif event_score >= 5:
                event_type = "collect"
            elif event_score >= 3:
                event_type = "click"

            records.append(
                {
                    "user_id": int(row["user_id"]),
                    "item_id": int(item_id),
                    "event_time": base_time + pd.Timedelta(minutes=len(records)),
                    "session_id": f"real_session_{int(row['user_id'])}_{position + 1:02d}",
                    "event_type": event_type,
                    "dwell_ms": int(1500 + aux_2[position] % 12000),
                    "scroll_depth": round(min(1.0, 0.2 + (position + 1) / max(shared_len, 1)), 3),
                    "exposure_rank": position + 1,
                }
            )

    return pd.DataFrame(records)


def build_reviews(behavior_logs: pd.DataFrame) -> pd.DataFrame:
    review_source = behavior_logs[behavior_logs["event_type"].isin(["order", "collect"])].copy()
    review_rows: list[dict[str, object]] = []

    for idx, row in enumerate(review_source.itertuples(index=False), start=1):
        rating = 5 if row.event_type == "order" else 4
        sentiment = 0.92 if row.event_type == "order" else 0.78
        review_rows.append(
            {
                "review_id": idx,
                "user_id": int(row.user_id),
                "item_id": int(row.item_id),
                "rating": rating,
                "sentiment_score": round(sentiment - (row.exposure_rank % 3) * 0.04, 3),
                "review_text": f"real_behavior_review_user_{int(row.user_id)}_item_{int(row.item_id)}",
            }
        )

    return pd.DataFrame(review_rows)


def build_cold_core_subset(cold_users: pd.DataFrame) -> pd.DataFrame:
    return cold_users[["user_id", "timestamp", "device_type", "location_id", "query_id"]].copy()


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def save_metadata(tables: dict[str, pd.DataFrame]) -> None:
    payload = {name: int(len(table)) for name, table in tables.items()}
    payload["notes"] = [
        "Users and item IDs come from the real Tianchi-derived dataset.",
        "Merchant attributes, tags, circles, and review text are augmented business fields.",
        "This dataset mixes real behavior with synthetic semantic attributes for cold-start modeling.",
    ]
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    first_rows = load_cleaned_users()
    warm_users, cold_users = select_real_users(first_rows)
    merchants = build_merchants()
    behavior_source = pd.concat([warm_users, cold_users], ignore_index=True)
    item_stats = derive_item_stats(behavior_source)
    items, item_tags = build_items(item_stats, merchants)
    all_users, user_preferences, user_circles = build_user_profiles(warm_users, cold_users, item_tags)

    allowed_items = set(items["item_id"].astype(int).tolist())
    browsing_logs = explode_behavior_logs(behavior_source, allowed_items)
    reviews = build_reviews(browsing_logs)
    cold_core = build_cold_core_subset(cold_users)

    tables = {
        "merchants": merchants,
        "items": items,
        "item_tags": item_tags,
        "users": all_users,
        "user_preferences": user_preferences,
        "user_circles": user_circles,
        "browsing_logs": browsing_logs,
        "reviews": reviews,
        "cold_start_users": cold_core,
    }

    save_tables(tables)
    save_metadata(tables)

    print(f"真实增强用户数: {len(all_users)}")
    print(f"其中暖启动用户数: {len(warm_users)}")
    print(f"其中冷启动用户数: {len(cold_core)}")
    print(f"真实商品数: {len(items)}")
    print(f"增强浏览记录数: {len(browsing_logs)}")
    print(f"增强评论记录数: {len(reviews)}")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
