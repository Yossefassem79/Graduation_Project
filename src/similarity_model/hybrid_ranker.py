# src/hybrid_ranker.py

import logging
from typing import List, Dict, Any

import pandas as pd

from src.similarity_model import (
    compute_feature_similarity,
    load_feature_model,
    safe_feature_list
)

# =====================================================
# Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# Columns
# =====================================================
TITLE_COL = "project_title"
FEATURE_COL = "features"

# =====================================================
# Weights / Thresholds
# =====================================================
DEFAULT_SEMANTIC_WEIGHT = 0.65
DEFAULT_FEATURE_WEIGHT = 0.35

HIGH_FEATURE_WEIGHT = 0.45
LOW_FEATURE_WEIGHT = 0.20

BONUS_WEIGHT = 0.05
MIN_HYBRID_SCORE = 0.35

# =====================================================
# Helpers
# =====================================================
def clamp(value: float) -> float:
    """
    Keep score inside [0,1]
    """
    return max(0.0, min(1.0, float(value)))


def get_dynamic_weights(
    feature_count: int,
    coverage: float
):
    """
    Adaptive weights depending on
    feature richness.
    """

    semantic_w = DEFAULT_SEMANTIC_WEIGHT
    feature_w = DEFAULT_FEATURE_WEIGHT

    # many strong features
    if feature_count >= 5 and coverage >= 0.60:
        semantic_w = 0.30
        feature_w = HIGH_FEATURE_WEIGHT

    # weak features -> trust semantic more
    elif feature_count <= 2:
        semantic_w = 0.80
        feature_w = LOW_FEATURE_WEIGHT

    return semantic_w, feature_w


# =====================================================
# Score Engines
# =====================================================
def compute_hybrid_score(
    semantic_score: float,
    feature_score: float,
    coverage: float,
    feature_count: int
) -> float:

    semantic_score = clamp(semantic_score)
    feature_score = clamp(feature_score)
    coverage = clamp(coverage)

    # ==========================================
    # Strong feature overlap case
    # ==========================================
    if coverage >= 0.90 and feature_score >= 0.65:
        return round(
            clamp(
                0.75 +
                (0.15 * feature_score) +
                (0.10 * semantic_score)
            ),
            4
        )

    # ==========================================
    # Normal scoring
    # ==========================================
    score = (
        0.25 * semantic_score +
        0.55 * feature_score +
        0.20 * coverage
    )

    return round(clamp(score), 4)


def compute_originality(
    hybrid_score: float,
    unique_query_features: int,
    total_query_features: int
) -> float:
    """
    Higher similarity => lower originality
    More unique features => higher originality
    """

    hybrid_score = clamp(hybrid_score)

    inverse_similarity = 1.0 - hybrid_score

    uniqueness_ratio = (
        unique_query_features / total_query_features
        if total_query_features > 0 else 0.0
    )

    originality = 1 - hybrid_score

    return round(clamp(originality), 4)


def compute_confidence(
    semantic_score: float,
    feature_score: float,
    coverage: float
) -> float:
    """
    Trust score for ranking result.
    """

    confidence = (
        0.40 * clamp(semantic_score) +
        0.40 * clamp(feature_score) +
        0.20 * clamp(coverage)
    )

    return round(clamp(confidence), 4)


def risk_label(score: float) -> str:
    """
    Duplicate risk label.
    """

    if score >= 0.85:
        return "Very High"

    if score >= 0.70:
        return "High"

    if score >= 0.55:
        return "Medium"

    if score >= 0.40:
        return "Low"

    return "Very Low"


# =====================================================
# Core Comparison
# =====================================================
def compare_single_candidate(
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    semantic_score: float,
    model=None
) -> Dict[str, Any]:

    if model is None:
        model = load_feature_model()

    query_features = safe_feature_list(
        query_row.get(FEATURE_COL, [])
    )

    candidate_features = safe_feature_list(
        candidate_row.get(FEATURE_COL, [])
    )

    feature_result = compute_feature_similarity(
        query_features,
        candidate_features,
        model=model
    )

    feature_score = feature_result["score"]
    coverage = feature_result["coverage"]

    total_query_features = len(query_features)
    unique_query_count = len(
        feature_result["unique_a"]
    )

    hybrid_score = compute_hybrid_score(
        semantic_score=semantic_score,
        feature_score=feature_score,
        coverage=coverage,
        feature_count=total_query_features
    )

    originality_score = compute_originality(
        hybrid_score=hybrid_score,
        unique_query_features=unique_query_count,
        total_query_features=total_query_features
    )

    confidence_score = compute_confidence(
        semantic_score=semantic_score,
        feature_score=feature_score,
        coverage=coverage
    )

    return {
        "project_title":
            candidate_row.get(TITLE_COL, ""),

        "semantic_score":
            round(float(semantic_score), 4),

        "feature_score":
            feature_score,

        "coverage":
            coverage,

        "hybrid_score":
            hybrid_score,

        "originality_score":
            originality_score,

        "confidence_score":
            confidence_score,

        "duplicate_risk":
            risk_label(hybrid_score),

        "shared_features_count":
            feature_result["shared_count"],

        "matches":
            feature_result["matches"],

        "unique_query_features":
            feature_result["unique_a"],

        "unique_candidate_features":
            feature_result["unique_b"]
    }


# =====================================================
# Rank Candidates
# =====================================================
def rank_candidates(
    query_row,
    candidate_rows,
    semantic_scores: List[float],
    model=None,
    min_score: float = MIN_HYBRID_SCORE
) -> pd.DataFrame:
    """
    Re-rank semantic retrieval results
    using hybrid logic.
    """

    if model is None:
        model = load_feature_model()

    rows = []

    for candidate_row, sem_score in zip(
        candidate_rows,
        semantic_scores
    ):

        result = compare_single_candidate(
            query_row=query_row,
            candidate_row=candidate_row,
            semantic_score=float(sem_score),
            model=model
        )

        if result["hybrid_score"] >= min_score:
            rows.append(result)

    if not rows:
        return pd.DataFrame([{
            "message": "No strong similar projects found."
        }])

    results = pd.DataFrame(rows)

    results = results.sort_values(
        by=[
            "hybrid_score",
            "confidence_score",
            "semantic_score"
        ],
        ascending=False
    ).reset_index(drop=True)

    return results


# =====================================================
# Example Run
# =====================================================
if __name__ == "__main__":

    query_project = {
        "project_title": "Clinic AI System",
        "features": [
            "appointment booking",
            "ai chatbot",
            "patient records",
            "doctor dashboard"
        ]
    }

    candidate_projects = [
        {
            "project_title":
                "Hospital Reservation Platform",

            "features": [
                "online booking",
                "chatbot assistant",
                "medical records",
                "analytics dashboard"
            ]
        },
        {
            "project_title":
                "Smart Farming Monitor",

            "features": [
                "crop analysis",
                "soil sensors"
            ]
        }
    ]

    semantic_scores = [0.84, 0.33]

    ranked = rank_candidates(
        query_row=query_project,
        candidate_rows=candidate_projects,
        semantic_scores=semantic_scores
    )

    print(ranked)