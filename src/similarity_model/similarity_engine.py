# src/similarity_engine.py
# FINAL VERSION
# Full Similarity Engine with Auto Feature Extraction + Hybrid Ranking

import logging
from typing import Dict, Any, List, Optional

import pandas as pd

from src.similarity_model import (
    normalize_text,
    extract_features,
    load_model,
    load_faiss_index,
    load_metadata,
    search_by_text,
    load_feature_model,
    safe_feature_list,
    compute_feature_similarity,
    compute_hybrid_score,
    compute_originality,
    compute_confidence,
    risk_label
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
# Config
# =====================================================
TITLE_COL = "project_title"
FEATURE_COL = "features"

DEFAULT_TOP_K = 5
DEFAULT_SEARCH_POOL = 20
DEFAULT_MIN_SEMANTIC_SCORE = 0.30
MAX_QUERY_FEATURES = 12

# =====================================================
# Query Builders
# =====================================================
def build_raw_text(
    title: str = "",
    abstract: str = "",
    description: str = ""
) -> str:
    """
    Merge available text fields.
    """
    parts = [
        str(title).strip(),
        str(title).strip(),   # title weighted
        str(abstract).strip(),
        str(description).strip()
    ]

    return ". ".join(
        [p for p in parts if p]
    ).strip()


def merge_features(
    title: str = "",
    abstract: str = "",
    description: str = "",
    features: Optional[List[str]] = None
) -> List[str]:
    """
    Use same extractor from preprocessing.py
    + merge manual features
    + remove duplicates
    """

    if features is None:
     features = []

    manual_features = safe_feature_list(features)

    raw_text = build_raw_text(
        title=title,
        abstract=abstract,
        description=description
    )

    auto_features = extract_features(
        normalize_text(raw_text)
    )

    final = []
    seen = set()

    # manual first
    for feat in manual_features + auto_features:

        feat = str(feat).strip().lower()

        if feat and feat not in seen:
            seen.add(feat)
            final.append(feat)

    return final[:MAX_QUERY_FEATURES]


def build_query_project(
    title: str,
    abstract: str,
    description: str,
    features: List[str]
) -> Dict[str, Any]:

    return {
        TITLE_COL: str(title).strip(),
        "abstract": str(abstract).strip(),
        "description": str(description).strip(),
        FEATURE_COL: features
    }


def build_query_text(
    title: str,
    abstract: str,
    description: str,
    features: List[str]
) -> str:
    """
    Build semantic query text.
    """

    raw_text = build_raw_text(
        title=title,
        abstract=abstract,
        description=description
    )

    feature_text = " ".join(features)

    text = f"{raw_text}. {feature_text}"

    return normalize_text(text)


# =====================================================
# Compare Candidate
# =====================================================
def compare_candidate(
    query_project: Dict[str, Any],
    candidate_row,
    semantic_score: float,
    feature_model
) -> Dict[str, Any]:

    feature_result = compute_feature_similarity(
        query_project[FEATURE_COL],
        candidate_row[FEATURE_COL],
        model=feature_model
    )

    feature_score = feature_result["score"]
    coverage = feature_result["coverage"]

    query_feature_count = len(
        query_project[FEATURE_COL]
    )

    hybrid_score = compute_hybrid_score(
        semantic_score=semantic_score,
        feature_score=feature_score,
        coverage=coverage,
        feature_count=query_feature_count
    )

    originality_score = compute_originality(
        hybrid_score=hybrid_score,
        unique_query_features=len(
            feature_result["unique_a"]
        ),
        total_query_features=query_feature_count
    )

    confidence_score = compute_confidence(
        semantic_score=semantic_score,
        feature_score=feature_score,
        coverage=coverage
    )

    return {
        "project_id": int(candidate_row.name),
        "project_title": candidate_row[TITLE_COL],

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

        "matched_features":
            feature_result["matches"],

        "unique_query_features":
            feature_result["unique_a"],

        "unique_candidate_features":
            feature_result["unique_b"]
    }


# =====================================================
# Main Engine
# =====================================================
def find_similar_projects(
    title: str = "",
    abstract: str = "",
    description: str = "",
    features: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    search_pool: int = DEFAULT_SEARCH_POOL
) -> pd.DataFrame:
    """
    Final Smart Pipeline

    Supports:
    - title only
    - title + description
    - title + abstract
    - title + abstract + description
    - optional features
    """

    logger.info(
        "Loading models and artifacts..."
    )

    load_model()
    load_faiss_index()

    feature_model = load_feature_model()
    df = load_metadata()

    # =============================================
    # Query preprocessing
    # =============================================
    logger.info(
        "Preparing query..."
    )

    final_features = merge_features(
        title=title,
        abstract=abstract,
        description=description,
        features=features
    )

    query_project = build_query_project(
        title=title,
        abstract=abstract,
        description=description,
        features=final_features
    )

    query_text = build_query_text(
        title=title,
        abstract=abstract,
        description=description,
        features=final_features
    )

    # =============================================
    # Stage 1 Semantic Retrieval
    # =============================================
    logger.info(
        "Running semantic retrieval..."
    )

    semantic_results = search_by_text(
        query_text=query_text,
        k=search_pool,
        min_score=DEFAULT_MIN_SEMANTIC_SCORE
    )

    if "message" in semantic_results.columns:
        return semantic_results

    # =============================================
    # Stage 2 Hybrid Ranking
    # =============================================
    logger.info(
        "Running hybrid ranking..."
    )

    rows = []

    for _, row in semantic_results.iterrows():

        candidate_id = int(
            row["project_id"]
        )

        semantic_score = float(
            row["score"]
        )

        candidate_row = df.loc[
            candidate_id
        ]

        result = compare_candidate(
            query_project=query_project,
            candidate_row=candidate_row,
            semantic_score=semantic_score,
            feature_model=feature_model
        )

        rows.append(result)

    if not rows:
        return pd.DataFrame([{
            "message":
            "No strong similar projects found."
        }])

    final_df = pd.DataFrame(rows)

    final_df = final_df.sort_values(
        by=[
            "hybrid_score",
            "confidence_score",
            "semantic_score"
        ],
        ascending=False
    ).head(top_k).reset_index(drop=True)

    # add debug info
    final_df["query_features_used"] = [
        final_features
    ] * len(final_df)

    final_df["query_clean_text"] = [
        query_text
    ] * len(final_df)

    return final_df


# =====================================================
# Example Run
# =====================================================
if __name__ == "__main__":

    results = find_similar_projects(
        title="Smart Library",
        abstract="""
        AI based digital library for students.
        """,
        description="""
        Includes chatbot,
        recommendation system,
        qr code scanner,
        mobile application.
        """,
        features=["library"],
        top_k=5
    )

    print(results)