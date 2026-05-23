# src/recommendation_engine/context_builder.py

import logging
import re

from collections import Counter
from typing import Dict, Any, List
from functools import lru_cache
from difflib import get_close_matches

import pandas as pd

from src.similarity_model import (
    find_similar_projects,
    extract_features
)

from src.recommendation_engine.config import (
    SIMILARITY_TOP_K,
    MAX_FEATURES
)

logger = logging.getLogger(__name__)


# =====================================================
# DOMAIN DEFINITIONS
# =====================================================
DOMAIN_KEYWORDS = {
    "artificial intelligence": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "deep learning",
        "neural network",
        "nlp",
        "computer vision"
    ],

    "healthcare": [
        "hospital",
        "health",
        "medical",
        "healthcare",
        "clinic",
        "patient"
    ],

    "fintech": [
        "fintech",
        "finance",
        "bank",
        "payment",
        "crypto",
        "blockchain"
    ],

    "education": [
        "education",
        "school",
        "learning",
        "edtech",
        "student",
        "university"
    ],

    "ecommerce": [
        "ecommerce",
        "shopping",
        "retail",
        "store",
        "marketplace"
    ],

    "agriculture": [
        "agriculture",
        "farming",
        "crop",
        "livestock",
        "smart farming"
    ],

    "security": [
        "security",
        "cyber",
        "cybersecurity",
        "threat",
        "attack",
        "malware"
    ],

    "general": [
        "general",
        "random",
        "anything",
        "any",
        "whatever",
        "surprise me",
        "mixed",
        "all",
        "open",
        "everything"
    ]
}


# =====================================================
# Helpers
# =====================================================
def normalize(text: str) -> str:

    text = str(text).lower().strip()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_list(
    items: List[str],
    limit: int = 20
) -> List[str]:

    final = []

    seen = set()

    for item in items:

        val = normalize(item)

        if not val:
            continue

        if val not in seen:

            seen.add(val)

            final.append(val)

    return final[:limit]


# =====================================================
# Multi-domain detection
# =====================================================
def detect_domains(text: str) -> List[str]:

    text = normalize(text)

    detected = []

    words_in_text = set(text.split())

    for domain, words in DOMAIN_KEYWORDS.items():

        for w in words:

            # phrase
            if " " in w:

                if w in text:

                    detected.append(domain)
                    break

            # word
            else:

                if w in words_in_text:

                    detected.append(domain)
                    break

    return clean_list(detected, limit=3)


# =====================================================
# Main domain extractor
# =====================================================
def extract_domain(text: str) -> str:

    if not text:
        return ""

    text = normalize(text)

    # hard normalization
    if text in ["ai", "ml"]:
        return "artificial intelligence"

    # direct domain match
    if text in DOMAIN_KEYWORDS:
        return text

    # multi-domain detection
    domains = detect_domains(text)

    if domains:

        # prioritize non-general
        for d in domains:

            if d != "general":
                return d

        return domains[0]

    # fuzzy typo matching
    all_words = []

    word_map = {}

    for domain, words in DOMAIN_KEYWORDS.items():

        for w in words:

            all_words.append(w)

            word_map[w] = domain

    match = get_close_matches(
        text,
        all_words,
        n=1,
        cutoff=0.75
    )

    if match:
        return word_map[match[0]]

    # partial match
    for domain, words in DOMAIN_KEYWORDS.items():

        for w in words:

            if text in w or w.startswith(text):

                return domain

    return ""


# =====================================================
# Similarity cache
# =====================================================
@lru_cache(maxsize=100)
def cached_similarity(
    title: str,
    description: str
):

    return find_similar_projects(
        title=title,
        description=description,
        top_k=SIMILARITY_TOP_K
    )


# =====================================================
# Common feature extraction
# =====================================================
def extract_common_features(
    results: pd.DataFrame
) -> List[str]:

    counter = Counter()

    if not isinstance(results, pd.DataFrame):
        return []

    for _, row in results.iterrows():

        matches = row.get(
            "matched_features",
            []
        )

        for item in matches:

            if isinstance(item, dict):

                feat = item.get(
                    "feature_b",
                    ""
                )

                feat = normalize(feat)

                if feat:

                    counter[feat] += 1

    return [
        feat
        for feat, _
        in counter.most_common(12)
    ]


# =====================================================
# Similar titles
# =====================================================
def extract_titles(
    results: pd.DataFrame
) -> List[str]:

    if not isinstance(results, pd.DataFrame):
        return []

    titles = [
        str(row.get("project_title", "")).strip()
        for _, row in results.iterrows()
        if row.get("project_title")
    ]

    return clean_list(titles, limit=10)


# =====================================================
# Architecture hints
# =====================================================
def build_architecture_hints(
    domains: List[str]
) -> List[str]:

    hints = []

    if "artificial intelligence" in domains:
        hints.extend([
            "AI inference pipeline",
            "Model prediction workflow",
            "Data preprocessing module"
        ])

    if "healthcare" in domains:
        hints.extend([
            "Emergency handling workflow",
            "Patient monitoring logic",
            "Medical alert system"
        ])

    if "security" in domains:
        hints.extend([
            "Threat detection pipeline",
            "Behavior anomaly analysis",
            "Risk monitoring engine"
        ])

    if "education" in domains:
        hints.extend([
            "Adaptive learning workflow",
            "Student performance analytics",
            "Recommendation engine"
        ])

    return clean_list(hints, limit=10)


# =====================================================
# Project Context
# =====================================================
def build_project_context(
    title: str,
    description: str,
    abstract: str = "",
    features: List[str] = None
) -> Dict[str, Any]:

    features = features or []

    logger.info("Building project context")

    full_text = (
        f"{title}. "
        f"{abstract}. "
        f"{description}"
    )

    # =========================================
    # Domains
    # =========================================
    domains = detect_domains(full_text)

    main_domain = (
        domains[0]
        if domains
        else "general"
    )

    # =========================================
    # Auto features
    # =========================================
    auto_features = extract_features(
        full_text
    )

    user_features = clean_list(
        features + auto_features,
        MAX_FEATURES
    )

    # =========================================
    # Similarity search
    # =========================================
    try:

        results = cached_similarity(
            title,
            description
        )

    except Exception as e:

        logger.warning(
            f"Similarity failed: {e}"
        )

        results = None

    # =========================================
    # Empty fallback
    # =========================================
    if (
        not isinstance(results, pd.DataFrame)
        or len(results) == 0
        or "message" in results.columns
    ):

        return {
            "project_title": title,
            "domain": main_domain,
            "domains": domains,
            "features": user_features,
            "similar_titles": [],
            "common_features": [],
            "unique_features": user_features,
            "architecture_hints": build_architecture_hints(domains),
            "originality_score": 1.0,
            "context_strength": 0.0
        }

    # =========================================
    # Rich context
    # =========================================
    similar_titles = extract_titles(results)

    common_features = extract_common_features(
        results
    )

    unique_features = [
        f
        for f in user_features
        if f not in common_features
    ]

    originality = float(
        results.get(
            "originality_score",
            pd.Series([1])
        ).mean()
    )

    hybrid_scores = results.get(
        "hybrid_score",
        pd.Series([0])
    )

    context_strength = float(
        hybrid_scores.mean()
    )

    return {
        "project_title": title,
        "domain": main_domain,
        "domains": domains,
        "features": user_features,
        "similar_titles": similar_titles,
        "common_features": common_features,
        "unique_features": unique_features,
        "architecture_hints": build_architecture_hints(domains),
        "originality_score": round(originality, 4),
        "context_strength": round(context_strength, 4)
    }


# =====================================================
# Domain Context
# =====================================================
def build_domain_context(
    domain: str
) -> Dict[str, Any]:

    extracted = extract_domain(domain)

    if extracted:
        domain_clean = extracted
    else:
        logger.warning(
            f"[DOMAIN WARNING] Unknown domain: {domain}"
        )

        domain_clean = normalize(domain)

    logger.info(
        f"Building domain context: {domain_clean}"
    )

    try:

        results = cached_similarity(
            domain_clean,
            domain_clean
        )

    except Exception as e:

        logger.warning(
            f"Domain similarity failed: {e}"
        )

        results = None

    if (
        not isinstance(results, pd.DataFrame)
        or len(results) == 0
        or "message" in results.columns
    ):

        return {
            "domain": domain_clean,
            "existing_titles": [],
            "common_features": [],
            "architecture_hints": build_architecture_hints([domain_clean]),
            "context_strength": 0.0
        }

    hybrid_scores = results.get(
        "hybrid_score",
        pd.Series([0])
    )

    return {
        "domain": domain_clean,
        "existing_titles": extract_titles(results),
        "common_features": extract_common_features(results),
        "architecture_hints": build_architecture_hints([domain_clean]),
        "context_strength": round(
            float(hybrid_scores.mean()),
            4
        )
    }