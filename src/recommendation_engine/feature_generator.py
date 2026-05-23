# src/recommendation_engine/feature_generator.py

import logging
from typing import List, Dict, Any, Set

from src.recommendation_engine.context_builder import build_project_context
from src.recommendation_engine.prompt_builder import build_feature_prompt
from src.recommendation_engine.llm_client import generate_text
from src.recommendation_engine.validator import validate_generated_list
from src.recommendation_engine.novelty_checker import is_feature_novel

from src.similarity_model import compare_two_ideas

from src.recommendation_engine.config import (
    DEFAULT_FEATURE_COUNT,
    GENERATION_BATCH_SIZE
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 5

# 🔥 stronger duplicate control
SIMILARITY_THRESHOLD_LOCAL = 0.82


# =====================================================
# Helpers
# =====================================================
def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


# =====================================================
# Generic feature detector
# =====================================================
GENERIC_PATTERNS = [
    "dashboard",
    "login",
    "signup",
    "authentication",
    "analytics module",
    "ai module",
    "admin panel",
    "settings page",
    "reports system",
    "user management"
]


def is_generic_feature(text: str) -> bool:

    low = normalize(text)

    if len(low.split()) < 2:
        return True

    for bad in GENERIC_PATTERNS:
        if bad in low:
            return True

    return False


# =====================================================
# Feature cleaner
# =====================================================
def clean_features(features: List[str]) -> List[str]:

    final = []

    for f in features:

        clean = str(f).strip()

        if not clean:
            continue

        words = clean.split()

        # 🔥 ideal feature size
        if len(words) < 3 or len(words) > 10:
            continue

        if is_generic_feature(clean):
            continue

        final.append(clean)

    return final


# =====================================================
# Semantic duplicate checker
# =====================================================
def is_duplicate_local(feature: str, existing: List[str]) -> bool:

    for old in existing:

        score = compare_two_ideas(feature, old)

        if score >= SIMILARITY_THRESHOLD_LOCAL:
            logger.info(f"[LOCAL DUPLICATE] {feature} ~ {old} ({score:.2f})")
            return True

    return False


# =====================================================
# Better contextual fallback
# =====================================================
def fallback_features(title: str) -> List[str]:

    title = (title or "").lower()

    # =========================================
    # Healthcare
    # =========================================
    if any(k in title for k in ["health", "hospital", "medical", "clinic"]):

        return [
            "Real-time patient monitoring",
            "Emergency alert notification system",
            "AI-assisted diagnosis support",
            "Medical data visualization dashboard",
            "Predictive patient risk analysis"
        ]

    # =========================================
    # Education
    # =========================================
    if any(k in title for k in ["education", "learning", "student", "school"]):

        return [
            "Adaptive learning recommendation engine",
            "Student performance prediction system",
            "Automated assignment evaluation",
            "Gamified engagement tracking",
            "Personalized study path generation"
        ]

    # =========================================
    # Security
    # =========================================
    if any(k in title for k in ["security", "cyber", "threat"]):

        return [
            "Real-time threat detection engine",
            "Behavior anomaly monitoring",
            "Automated attack alert system",
            "Security event visualization",
            "Risk prediction analytics"
        ]

    # =========================================
    # Default
    # =========================================
    return [
        "Real-time intelligent monitoring",
        "Predictive analytics engine",
        "Smart recommendation system",
        "Automated decision support",
        "Dynamic performance optimization"
    ]


# =====================================================
# Main Generator
# =====================================================
def generate_features(
    title: str,
    description: str,
    abstract: str = "",
    features: List[str] = None,
    previous_generated_features: List[str] = None,
    top_k: int = DEFAULT_FEATURE_COUNT
) -> Dict[str, Any]:

    features = features or []
    previous_generated_features = previous_generated_features or []

    top_k = max(1, min(top_k, 20))

    logger.info(f"Starting feature generation | title={title}")

    # =========================================
    # Build Context
    # =========================================
    context = build_project_context(
        title=title,
        description=description,
        abstract=abstract,
        features=features
    )

    final_features: List[str] = []
    final_norm_set: Set[str] = set()

    existing_features = context.get("features", [])

    existing_norm = set(
        normalize(f)
        for f in existing_features
    )

    previous_norm = set(
        normalize(f)
        for f in previous_generated_features
    )

    attempts = 0

    # =========================================
    # Generation Loop
    # =========================================
    while len(final_features) < top_k and attempts < MAX_RETRIES:

        attempts += 1

        logger.info(f"Generation attempt #{attempts}")

        generation_count = max(
            top_k * 4,
            GENERATION_BATCH_SIZE
        )

        # =====================================
        # Build Prompt
        # =====================================
        prompt = build_feature_prompt(
            context=context,
            count=generation_count,
            previous_features=previous_generated_features
        )

        # =====================================
        # LLM Generation
        # =====================================
        raw_text = generate_text(
            prompt,
            task="feature"
        )

        if not raw_text:
            logger.warning("Empty feature response")
            continue

        # =====================================
        # Validation
        # =====================================
        generated = validate_generated_list(
            text=raw_text,
            top_k=generation_count
        )

        generated = clean_features(generated)

        logger.info(f"Generated {len(generated)} candidate features")

        # =====================================
        # Filtering
        # =====================================
        for feat in generated:

            norm = normalize(feat)

            if not norm:
                continue

            # exact duplicate
            if (
                norm in final_norm_set
                or norm in existing_norm
                or norm in previous_norm
            ):
                continue

            # semantic novelty
            if not is_feature_novel(feat, existing_features):
                continue

            # local semantic duplicate
            if is_duplicate_local(feat, final_features):
                continue

            final_features.append(feat)
            final_norm_set.add(norm)

            logger.info(f"[NEW FEATURE] {feat}")

            if len(final_features) >= top_k:
                break

    # =========================================
    # Fallback
    # =========================================
    if len(final_features) < top_k:

        logger.warning("Using fallback features")

        fallback = fallback_features(title)

        for feat in fallback:

            norm = normalize(feat)

            if (
                norm not in final_norm_set
                and norm not in existing_norm
            ):

                final_features.append(feat)
                final_norm_set.add(norm)

            if len(final_features) >= top_k:
                break

    logger.info(f"Final generated features: {final_features}")

    return {
        "project_title": context.get("project_title", title),
        "current_features": existing_features,
        "recommended_features": final_features,
        "originality_score": context.get("originality_score", 1.0),
        "similar_projects": context.get("similar_titles", [])
    }