from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExplanationOutput:
    item_id: int
    score: float
    label_contributions: dict[str, float]
    stage: str


def normalize_contributions(raw_scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in raw_scores.values()) or 1.0
    return {key: round(max(value, 0.0) / total, 4) for key, value in raw_scores.items()}


def explain_stub(item_id: int, stage: str) -> ExplanationOutput:
    raw = {
        "sichuan": 0.6 if stage == "zero_shot" else 0.35,
        "late_night": 0.2 if stage == "zero_shot" else 0.15,
        "budget_match": 0.2 if stage == "zero_shot" else 0.1,
        "recent_click_similarity": 0.0 if stage == "zero_shot" else 0.4,
    }
    return ExplanationOutput(
        item_id=item_id,
        score=0.82 if stage == "zero_shot" else 0.91,
        label_contributions=normalize_contributions(raw),
        stage=stage,
    )
