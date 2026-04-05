from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, ModelConfig, TrainConfig
from .data_pipeline import prepare_protocol_files
from .datasets import load_multimodal_tables
from .model import MultiModalColdStartModel
from .trainer import (
    WarmRankingDataset,
    _build_encoders,
    _build_interaction_map,
    _build_item_feature_maps,
    _build_item_tag_weight_map,
    _build_label_vocab,
    _build_multilabel_targets,
    _build_user_preference_map,
    _build_user_stats,
    _choose_device,
    _collate_fn,
    _compute_match_features,
    _interaction_sequence,
    _set_seed,
    _user_context_vector,
)

try:  # pragma: no cover
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = None


MAPPING_FILE = "label_zh_mapping.json"
ITEM_NAME_FILE = "item_name_zh_mapping.csv"


@dataclass(slots=True)
class RecommendationItem:
    rank: int
    item_id: int
    item_name: str
    merchant_id: int
    primary_cuisine: str
    price_band: str
    score: float
    explanation: dict[str, float]
    reason: str


def _load_label_mapping(data_dir: Path) -> dict[str, dict[str, object]]:
    path = data_dir / MAPPING_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"cuisine": {}, "query_synonyms": {}}


def _load_item_name_mapping(data_dir: Path) -> dict[int, str]:
    path = data_dir / ITEM_NAME_FILE
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["item_id", "item_name_zh"])
    return {int(r.item_id): str(r.item_name_zh) for r in df.itertuples(index=False)}


def _map_value(mapping: dict[str, dict[str, object]], mapping_type: str, value: str) -> str:
    return str(mapping.get(mapping_type, {}).get(str(value), str(value)))


def _hard_form_filter(intent: str, item_name: str) -> bool:
    if not intent:
        return True
    text = str(item_name).lower()
    kw: dict[str, tuple[str, ...]] = {
        "burger": ("\u6c49\u5821", "\u5821", "burger"),
        "western": ("\u725b\u6392", "\u610f\u9762", "\u610f\u5927\u5229\u9762", "\u62ab\u8428", "\u6bd4\u8428", "\u897f\u9910", "pizza", "pasta", "steak"),
        "hotpot": ("\u706b\u9505", "\u9505", "hotpot"),
        "drink": ("\u5976\u8336", "\u5496\u5561", "\u996e\u54c1", "\u679c\u8336", "milk tea", "coffee", "latte"),
        "dessert": ("\u751c\u54c1", "\u86cb\u7cd5", "\u6155\u65af", "\u5e03\u4e01", "dessert"),
        "salad": ("\u6c99\u62c9", "\u8f7b\u98df", "salad"),
        "noodles": ("\u9762", "\u7c89", "\u7c73\u7ebf", "noodle"),
        "rice": ("\u996d", "\u76d6\u996d", "\u7172\u4ed4\u996d", "\u7092\u996d", "\u4fbf\u5f53"),
    }
    return any(token in text for token in kw.get(intent, ()))


def _normalize_meal_scene_name(item_name: str, primary_cuisine: str) -> str:
    text = str(item_name)
    meal_cuisines = {"burger", "noodles", "sichuan", "cantonese", "hotpot", "barbecue", "western"}
    if str(primary_cuisine) in meal_cuisines and "下午茶" in text:
        text = text.replace("下午茶", "午餐")
    return text


def _build_new_user_row(users: pd.DataFrame) -> pd.Series:
    # 新用户默认画像：无行为 + 中档预算 + 众数设备/地域配置。
    mode_device = users["device_type"].mode().iloc[0]
    mode_location = users["location_id"].mode().iloc[0]
    mode_query = users["query_id"].mode().iloc[0]
    mode_circle = users["circle_id"].mode().iloc[0]
    row = {
        "timestamp": int(users["timestamp"].median()),
        "device_type": int(mode_device),
        "location_id": int(mode_location),
        "query_id": int(mode_query),
        "circle_id": str(mode_circle),
        "budget_level": "midrange",
    }
    return pd.Series(row)


def _extract_query_terms(query: str, label_mapping: dict[str, dict[str, object]]) -> set[str]:
    q = str(query).strip().lower()
    if not q:
        return set()

    terms: set[str] = {q}
    parts = [p for p in re.split(r"[\s,，]+", q) if p]
    terms.update(parts)

    # 汉字串拆分为 2~4 gram，增强“红烧牛肉面”这类短语匹配。
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", q)
    for chunk in chinese_chunks:
        n = len(chunk)
        for size in range(2, min(4, n) + 1):
            for i in range(0, n - size + 1):
                terms.add(chunk[i : i + size])
        if n == 1:
            terms.add(chunk)
        if "面" in chunk:
            terms.update({"面", "noodles"})
        if "汉堡" in chunk:
            terms.update({"汉堡", "burger"})
        if "奶茶" in chunk:
            terms.update({"奶茶", "milk_tea"})
        if "咖啡" in chunk:
            terms.update({"咖啡", "coffee", "milk_tea", "latte"})
        if "披萨" in chunk or "比萨" in chunk:
            terms.update({"披萨", "pizza", "western"})
        if "意面" in chunk or "意大利面" in chunk:
            terms.update({"意面", "pasta", "western"})
        if "牛排" in chunk:
            terms.update({"牛排", "steak", "western"})
        if "甜品" in chunk:
            terms.update({"甜品", "dessert"})

    # 外部同义词映射（来自数据目录配置）
    synonyms = label_mapping.get("query_synonyms", {})
    for part in list(terms):
        for key, mapped_terms in synonyms.items():
            if str(key).lower() in part:
                for mt in mapped_terms:
                    terms.add(str(mt).lower())
    return terms


def _cuisine_intent_scores(query: str) -> dict[str, float]:
    q = str(query).lower()
    rule_map = {
        "burger": ["汉堡", "burger", "鸡腿堡", "牛肉堡"],
        "noodles": ["面", "粉", "拉面", "米线", "noodles"],
        "sichuan": ["川菜", "麻辣", "水煮", "sichuan"],
        "cantonese": ["粤", "煲仔", "叉烧", "cantonese"],
        "hotpot": ["火锅", "hotpot", "锅"],
        "barbecue": ["烧烤", "烤", "barbecue"],
        "salad": ["轻食", "沙拉", "salad"],
        "milk_tea": ["奶茶", "果茶", "milk_tea", "咖啡", "拿铁", "美式", "摩卡", "厚乳"],
        "dessert": ["甜品", "蛋糕", "慕斯", "dessert"],
    }
    out = {k: 0.0 for k in rule_map}
    for cuisine, kws in rule_map.items():
        for kw in kws:
            if kw in q:
                out[cuisine] = 1.0
                break
    return out


def _price_intent_from_query(query: str) -> tuple[str, dict[str, float]]:
    q = str(query).lower()
    budget_kw = ["便宜", "平价", "实惠", "省钱", "学生", "优惠", "低价", "budget"]
    luxury_kw = ["高端", "奢", "豪华", "精品", "高档", "贵", "luxury"]
    mid_kw = ["中档", "均价", "普通", "大众", "midrange"]

    if any(k in q for k in budget_kw):
        inferred = "budget"
    elif any(k in q for k in luxury_kw):
        inferred = "luxury"
    elif any(k in q for k in mid_kw):
        inferred = "midrange"
    else:
        inferred = "midrange"

    score_map = {"budget": 0.0, "midrange": 0.0, "luxury": 0.0}
    score_map[inferred] = 1.0
    return inferred, score_map


def _query_form_intent(query: str) -> str:
    q = str(query)
    if "牛排" in q or "意面" in q or "意大利面" in q or "披萨" in q or "比萨" in q or "西餐" in q:
        return "western"
    if "汉堡" in q or "堡" in q:
        return "burger"
    if "火锅" in q or "锅" in q:
        return "hotpot"
    if "奶茶" in q or "咖啡" in q or "果茶" in q:
        return "drink"
    if "甜品" in q or "蛋糕" in q or "慕斯" in q:
        return "dessert"
    if "沙拉" in q or "轻食" in q:
        return "salad"
    if "面" in q or "粉" in q or "米线" in q:
        return "noodles"
    if "饭" in q or "盖饭" in q or "便当" in q:
        return "rice"
    return ""


def _form_match(intent: str, item_name: str, primary_cuisine: str) -> float:
    name = str(item_name)
    cuisine = str(primary_cuisine)
    if not intent:
        return 0.0
    if intent == "burger":
        return 1.0 if ("堡" in name or cuisine == "burger") else 0.0
    if intent == "western":
        return 1.0 if (("牛排" in name) or ("意面" in name) or ("披萨" in name) or ("比萨" in name)) else 0.0
    if intent == "hotpot":
        return 1.0 if ("锅" in name or cuisine == "hotpot") else 0.0
    if intent == "drink":
        return 1.0 if (("奶茶" in name) or ("咖啡" in name) or cuisine == "milk_tea") else 0.0
    if intent == "dessert":
        return 1.0 if (("甜" in name) or ("蛋糕" in name) or cuisine == "dessert") else 0.0
    if intent == "salad":
        return 1.0 if (("沙拉" in name) or ("轻食" in name) or cuisine == "salad") else 0.0
    if intent == "noodles":
        return 1.0 if (("面" in name) or ("粉" in name) or cuisine == "noodles") else 0.0
    if intent == "rice":
        if cuisine in {"burger", "milk_tea", "dessert", "salad"}:
            return 0.0
        return 1.0 if (("饭" in name) or ("盖饭" in name) or ("煲仔饭" in name) or ("炒饭" in name) or cuisine in {"sichuan", "cantonese", "barbecue"}) else 0.0
    return 0.0


def _query_relevance(query_terms: set[str], item_row: pd.Series, item_tags: list[str], display_name: str) -> float:
    if not query_terms:
        return 0.0

    name_text = str(display_name).lower()
    cuisine_text = str(item_row["primary_cuisine"]).lower()
    desc_text = str(item_row.get("description_text", "")).lower()
    tags_text = " ".join(str(t).lower() for t in item_tags)
    searchable = " ".join([cuisine_text, desc_text, tags_text, name_text])

    # 1) 基础覆盖率：查询词是否被检索字段命中
    weighted_hit = 0.0
    total = 0.0
    for term in query_terms:
        if not term:
            continue
        w = min(len(term), 4) / 4.0
        total += w
        if term in searchable:
            weighted_hit += w
    coverage = weighted_hit / total if total > 0 else 0.0

    # 2) 名称字符相似度：抑制“都命中但都=1.0”的情况
    raw_query = max(query_terms, key=len)
    q_chars = set(re.sub(r"\s+", "", raw_query.lower()))
    n_chars = set(re.sub(r"\s+", "", name_text))
    if q_chars and n_chars:
        overlap = len(q_chars & n_chars)
        char_dice = (2.0 * overlap) / (len(q_chars) + len(n_chars))
    else:
        char_dice = 0.0

    # 3) 短语命中：完整短语在名称中出现给额外加分
    phrase_bonus = 1.0 if raw_query and raw_query.lower() in name_text else 0.0

    score = 0.50 * coverage + 0.35 * char_dice + 0.15 * phrase_bonus
    return float(max(0.0, min(1.0, score)))


def _reason(row: pd.Series) -> str:
    return (
        f"相关度{row['query_relevance']:.2f}，品类意图{row['cuisine_intent']:.2f}，"
        f"形态匹配{row['form_intent_match']:.2f}，价格匹配{row['price_intent_match']:.2f}，模型分{row['model_score']:.2f}"
    )


def _reason(row: pd.Series) -> str:
    return f"相关度{row['query_relevance']:.2f}，模型分{row['model_score']:.2f}"


def run_query_recommendation(
    query: str,
    user_id: int | None = None,
    top_k: int = 10,
    candidate_size: int = 300,
    new_user: bool = True,
    data_config: DataConfig = DataConfig(),
    model_config: ModelConfig = ModelConfig(),
    train_config: TrainConfig = TrainConfig(),
) -> dict[str, object]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run inference.")

    _set_seed(train_config.seed)
    prepare_protocol_files(data_config)
    tables = load_multimodal_tables(data_config.root)
    users = tables["users"]
    items = tables["items"]
    item_tags = tables["item_tags"]
    user_preferences = tables["user_preferences"]
    browsing_logs = tables["browsing_logs"]
    multimodal_items = tables["multimodal_items"]

    label_mapping = _load_label_mapping(data_config.root)
    item_name_mapping = _load_item_name_mapping(data_config.root)

    checkpoint_path = data_config.model_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    encoders = _build_encoders(users)
    user_stats = _build_user_stats(browsing_logs)
    label_vocab = _build_label_vocab(user_preferences, model_config.label_count)
    targets = _build_multilabel_targets(user_preferences, label_vocab, model_config.label_count)
    user_pref_map = _build_user_preference_map(user_preferences)
    item_tag_weight_map = _build_item_tag_weight_map(item_tags)
    interaction_map = _build_interaction_map(browsing_logs)
    item_popularity = (
        multimodal_items.set_index("item_id")
        .join(items.set_index("item_id")[["popularity_score"]], how="left")["popularity_score"]
        .fillna(0.0)
        .to_dict()
    )
    text_map, image_map, tag_map = _build_item_feature_maps(multimodal_items, item_tags, model_config.vocab_size)

    is_new_user = bool(new_user or user_id is None)
    if is_new_user:
        effective_user_id = -1
        user_row = _build_new_user_row(users)
        target = np.zeros(model_config.label_count, dtype=np.float32)
        user_context = _user_context_vector(
            user_id=effective_user_id,
            user_row=user_row,
            user_stats={},
            encoders=encoders,
            size=model_config.context_dim,
            user_label_target=target,
        )
        user_seq = np.zeros((data_config.max_seq_len, 4), dtype=np.float32)
    else:
        user_row_df = users[users["user_id"] == int(user_id)]
        if user_row_df.empty:
            raise ValueError(f"user_id={user_id} not found in users.csv")
        effective_user_id = int(user_id)
        user_row = user_row_df.iloc[0]
        target = targets.get(effective_user_id, np.zeros(model_config.label_count, dtype=np.float32))
        user_context = _user_context_vector(
            effective_user_id, user_row, user_stats, encoders, model_config.context_dim, user_label_target=target
        )
        user_seq = _interaction_sequence(interaction_map, effective_user_id, data_config.max_seq_len, item_popularity)

    tag_lookup = item_tags.groupby("item_id")["tag_name"].apply(list).to_dict()
    item_lookup = items.set_index("item_id")
    query_terms = _extract_query_terms(query, label_mapping)
    cuisine_intent = _cuisine_intent_scores(query)
    inferred_price, price_intent_scores = _price_intent_from_query(query)
    form_intent = _query_form_intent(query)

    candidates: list[tuple[int, float, float, float, float, str]] = []
    for item_id in items["item_id"].astype(int).tolist():
        if item_id not in text_map:
            continue
        item_row = item_lookup.loc[item_id]
        item_tags_list = tag_lookup.get(item_id, [])
        raw_display_name = item_name_mapping.get(item_id, str(item_row["item_name"]))
        display_name = _normalize_meal_scene_name(raw_display_name, str(item_row["primary_cuisine"]))
        qrel = _query_relevance(query_terms, item_row, item_tags_list, display_name)
        c_intent = cuisine_intent.get(str(item_row["primary_cuisine"]), 0.0)
        f_intent = _form_match(form_intent, display_name, str(item_row["primary_cuisine"]))
        pop = float(item_row["popularity_score"])
        candidates.append((item_id, qrel, c_intent, f_intent, pop, display_name))

    # 强约束：有明确查询时优先相关候选。
    candidates.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    picked = [c for c in candidates if c[1] > 0.05 or c[2] > 0 or c[3] > 0]
    if form_intent:
        hard_filtered = [c for c in picked if _hard_form_filter(form_intent, c[5])]
        if hard_filtered:
            picked = hard_filtered
    if form_intent:
        strict = [c for c in picked if c[3] > 0]
        if len(strict) >= min(100, candidate_size // 2):
            picked = strict
    if not picked:
        picked = candidates
    picked = picked[:candidate_size]

    rows: list[dict[str, np.ndarray | float | int]] = []
    meta: dict[int, dict[str, float | str]] = {}
    for item_id, qrel, cint, fint, _, display_name in picked:
        meta[item_id] = {
            "query_relevance": float(qrel),
            "cuisine_intent": float(cint),
            "form_intent_match": float(fint),
            "display_name": display_name,
        }
        rows.append(
            {
                "user_id": int(effective_user_id),
                "item_id": int(item_id),
                "user_context": user_context,
                "interaction_seq": user_seq,
                "item_text_tokens": text_map[item_id],
                "item_image_vectors": image_map[item_id],
                "item_tag_ids": tag_map[item_id],
                "item_match_features": _compute_match_features(
                    user_id=int(effective_user_id),
                    user_row=user_row,
                    item_id=int(item_id),
                    item_popularity=item_popularity,
                    user_pref_map=user_pref_map,
                    item_tag_map=item_tag_weight_map,
                ),
                "label": 0.0,
                "multilabel_target": target,
            }
        )

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
        match_dim=model_config.match_dim,
        dropout=model_config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = DataLoader(WarmRankingDataset(rows), batch_size=train_config.batch_size, shuffle=False, collate_fn=_collate_fn)
    scored_rows: list[dict[str, float | int]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(
                user_context=batch["user_context"].to(device),
                interaction_seq=batch["interaction_seq"].to(device),
                item_text_tokens=batch["item_text_tokens"].to(device),
                item_image_vectors=batch["item_image_vectors"].to(device),
                item_tag_ids=batch["item_tag_ids"].to(device),
                item_match_features=batch["item_match_features"].to(device),
            )
            mscore = torch.sigmoid(out["ranking_score"]).cpu().numpy()
            mfeat = batch["item_match_features"].cpu().numpy()
            for uid, iid, s, feat in zip(batch["user_id"].tolist(), batch["item_id"].tolist(), mscore.tolist(), mfeat.tolist()):
                qrel = float(meta[int(iid)]["query_relevance"])
                cint = float(meta[int(iid)]["cuisine_intent"])
                fint = float(meta[int(iid)]["form_intent_match"])
                tag_match = float(np.mean(feat[:3]))
                budget_match = float(feat[3])
                popularity = float(feat[4])
                item_price_band = str(item_lookup.loc[int(iid)]["price_band"])
                price_intent_match = float(price_intent_scores.get(item_price_band, 0.0))
                if is_new_user:
                    final = 0.12 * float(s) + 0.35 * qrel + 0.18 * cint + 0.20 * fint + 0.10 * price_intent_match + 0.05 * popularity
                else:
                    final = 0.30 * float(s) + 0.25 * qrel + 0.10 * cint + 0.10 * fint + 0.10 * tag_match + 0.10 * budget_match + 0.05 * price_intent_match
                scored_rows.append(
                    {
                        "user_id": int(uid),
                        "item_id": int(iid),
                        "model_score": float(s),
                        "query_relevance": qrel,
                        "cuisine_intent": cint,
                        "form_intent_match": fint,
                        "tag_match": tag_match,
                        "budget_match": budget_match,
                        "price_intent_match": price_intent_match,
                        "popularity": popularity,
                        "score": float(final),
                    }
                )

    scored = pd.DataFrame(scored_rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    results: list[RecommendationItem] = []
    for rank, row in enumerate(scored.itertuples(index=False), start=1):
        item_row = item_lookup.loc[int(row.item_id)]
        row_series = pd.Series(row._asdict())
        display_name = str(meta[int(row.item_id)]["display_name"])
        display_name_with_id = f"{display_name} (item_id:{int(row.item_id)})"
        results.append(
            RecommendationItem(
                rank=rank,
                item_id=int(row.item_id),
                item_name=display_name_with_id,
                merchant_id=int(item_row["merchant_id"]),
                primary_cuisine=_map_value(label_mapping, "cuisine", str(item_row["primary_cuisine"])),
                price_band=str(item_row["price_band"]),
                score=float(row.score),
                explanation={
                    "query_relevance": float(row.query_relevance),
                    "model_score": float(row.model_score),
                    "popularity": float(row.popularity),
                },
                reason=_reason(row_series),
            )
        )

    return {
        "query": query,
        "user_id": int(effective_user_id),
        "mode": "new_user" if is_new_user else "existing_user",
        "device": str(device),
        "top_k": top_k,
        "mapping_file": str(data_config.root / MAPPING_FILE),
        "item_name_file": str(data_config.root / ITEM_NAME_FILE),
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search-aware recommendation inference")
    parser.add_argument("--query", type=str, default="", help="query text")
    parser.add_argument("--user-id", type=int, default=None, help="existing user id")
    parser.add_argument("--new-user", action="store_true", help="force new user mode")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-size", type=int, default=300)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    def run_once(q: str) -> str:
        payload = run_query_recommendation(
            query=q,
            user_id=args.user_id,
            top_k=args.top_k,
            candidate_size=args.candidate_size,
            new_user=args.new_user or args.user_id is None,
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(text)
        return text

    interactive = bool(args.interactive or not args.query.strip())
    if interactive:
        print("interactive mode: input query, type exit to quit", flush=True)
        while True:
            try:
                q = input("query> ").strip()
            except EOFError:
                print("input stream ended, exit interactive mode", flush=True)
                break
            if not q:
                print("empty query, try again", flush=True)
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("bye", flush=True)
                break
            text = run_once(q)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    else:
        text = run_once(args.query)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
