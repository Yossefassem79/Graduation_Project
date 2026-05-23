# src/recommendation_engine/novelty_checker.py

import logging
import re

from typing import List
from functools import lru_cache

from src.similarity_model import (
    compare_two_ideas,
    find_similar_projects
)

from src.recommendation_engine.config import (
    IDEA_DUPLICATE_THRESHOLD,
    FEATURE_DUPLICATE_THRESHOLD
)

logger = logging.getLogger(__name__)


# =====================================================
# Generic weak patterns
# =====================================================
GENERIC_PATTERNS = [
    "dashboard",
    "platform",
    "system",
    "application",
    "website",
    "ai module",
    "analytics module",
    "smart system",
    "management system"
]


# =====================================================
# Text normalization
# =====================================================
def normalize(text: str) -> str:

    text = str(text).lower().strip()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# Generic detector
# =====================================================
def is_generic(text: str) -> bool:

    low = normalize(text)

    for pattern in GENERIC_PATTERNS:

        if pattern in low:
            return True

    return False


# =====================================================
# Token overlap similarity
# =====================================================
def token_overlap_score(a: str, b: str) -> float:

    a_tokens = set(normalize(a).split())

    b_tokens = set(normalize(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    overlap = len(a_tokens & b_tokens)

    union = len(a_tokens | b_tokens)

    return overlap / union


# =====================================================
# Feature Novelty
# =====================================================
def is_feature_novel(
    feature: str,
    existing_features: List[str]
) -> bool:

    feature = normalize(feature)

    if not feature:
        return False

    # =========================================
    # Reject generic weak features
    # =========================================
    if is_generic(feature):

        logger.debug(f"[GENERIC FEATURE] {feature}")

        return False

    # =========================================
    # Exact duplicate
    # =========================================
    existing_norm = [
        normalize(f)
        for f in existing_features
    ]

    if feature in existing_norm:

        logger.debug(f"[EXACT DUP] {feature}")

        return False

    # =========================================
    # Semantic comparison
    # =========================================
    for old in existing_norm:

        if not old:
            continue

        # embedding similarity
        semantic_score = compare_two_ideas(
            feature,
            old
        )

        # token overlap
        overlap_score = token_overlap_score(
            feature,
            old
        )

        # combined reasoning
        final_score = max(
            semantic_score,
            overlap_score
        )

        logger.debug(
            f"[COMPARE] {feature} ~ {old} "
            f"(semantic={semantic_score:.2f}, "
            f"overlap={overlap_score:.2f})"
        )

        # stricter threshold
        if final_score >= (
            FEATURE_DUPLICATE_THRESHOLD + 0.08
        ):

            logger.debug(
                f"[FEATURE DUPLICATE] "
                f"{feature} ~ {old}"
            )

            return False

    return True


# =====================================================
# Duplicate Feature Filter
# =====================================================
def filter_duplicate_features(
    generated_features: List[str],
    existing_features: List[str]
) -> List[str]:

    final = []

    seen = set()

    for feat in generated_features:

        clean = str(feat).strip()

        norm = normalize(clean)

        if not clean:
            continue

        if norm in seen:
            continue

        if not is_feature_novel(
            norm,
            existing_features + final
        ):
            continue

        seen.add(norm)

        final.append(clean)

    return final


# =====================================================
# Cached DB originality check
# =====================================================
@lru_cache(maxsize=256)
def _cached_db_check(idea: str) -> float:

    try:

        results = find_similar_projects(
            title=idea,
            description=idea,
            top_k=3
        )

        if (
            hasattr(results, "iloc")
            and len(results) > 0
        ):

            scores = []

            for _, row in results.iterrows():

                score = float(
                    row.get("hybrid_score", 0)
                )

                scores.append(score)

            if scores:

                # strongest similarity
                return max(scores)

    except Exception as e:

        logger.warning(f"[DB ERROR] {e}")

    # safe fallback
    return 0.0


# =====================================================
# Idea Novelty
# =====================================================
def is_idea_novel(idea_title: str) -> bool:

    idea_title = normalize(idea_title)

    if not idea_title:
        return False

    if is_generic(idea_title):

        logger.info(
            f"[GENERIC IDEA REJECTED] "
            f"{idea_title}"
        )

        return False

    score = _cached_db_check(idea_title)

    logger.info(f"[DB CHECK] {idea_title}")
    logger.info(f"[SIMILARITY SCORE] {score:.4f}")

    return score < IDEA_DUPLICATE_THRESHOLD


# =====================================================
# Optional novelty score
# =====================================================
def score_feature_novelty(
    feature: str,
    existing_features: List[str]
) -> float:

    feature = normalize(feature)

    if not existing_features:
        return 1.0

    scores = []

    for old in existing_features:

        old_norm = normalize(old)

        semantic = compare_two_ideas(
            feature,
            old_norm
        )

        overlap = token_overlap_score(
            feature,
            old_norm
        )

        scores.append(
            max(semantic, overlap)
        )

    if not scores:
        return 1.0

    return round(
        1.0 - max(scores),
        4
    )