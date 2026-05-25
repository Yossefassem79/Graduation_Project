# src/feature_similarity.py

import logging
import ast
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

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
MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_THRESHOLD = 0.65

SIMILARITY_WEIGHT = 0.70
COVERAGE_WEIGHT = 0.30

# =====================================================
# Model Loader
# =====================================================
@lru_cache(maxsize=1)
def load_feature_model():
    """
    Load feature embedding model once.
    """
    logger.info(f"Loading feature model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


# =====================================================
# Helpers
# =====================================================
def safe_feature_list(features):
    """
    Convert any feature input into clean List[str]
    Supports:
    list, tuple, numpy array, string, NaN
    """

    import numpy as np

    # None
    if features is None:
        return []

    # numpy nan scalar only
    if isinstance(features, float):
        if pd.isna(features):
            return []

    # numpy array
    if isinstance(features, np.ndarray):
        features = features.tolist()

    # tuple
    if isinstance(features, tuple):
        features = list(features)

    # string
    if isinstance(features, str):
        features = [features]

    # list
    if isinstance(features, list):

        cleaned = []

        for item in features:
            val = str(item).strip().lower()

            if val and val != "nan":
                cleaned.append(val)

        return list(dict.fromkeys(cleaned))

    return []


def remove_redundant_features(features):

    cleaned = []
    seen_words = []

    for feat in features:

        feat_words = set(feat.split())

        redundant = False

        for existing in seen_words:

            overlap = len(
                feat_words & existing
            ) / max(len(feat_words), 1)

            if overlap >= 0.90:
                redundant = True
                break

        if not redundant:
            cleaned.append(feat)
            seen_words.append(feat_words)

    return cleaned



def empty_result(
    unique_a=None,
    unique_b=None
) -> Dict[str, Any]:

    return {
        "score": 0.0,
        "coverage": 0.0,
        "shared_count": 0,
        "matches": [],
        "unique_a": unique_a or [],
        "unique_b": unique_b or []
    }


def encode_features(
    features: List[str],
    model
) -> np.ndarray:
    """
    Encode feature phrases into normalized vectors.
    """

    if not features:
        return np.array([])

    vectors = model.encode(
        features,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return vectors.astype("float32")


# =====================================================
# Core Similarity Engine
# =====================================================
def compute_feature_similarity(
    features_a,
    features_b,
    model=None,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, Any]:
    """
    Compare two feature lists using:

    1. Sentence embeddings
    2. Cosine similarity matrix
    3. Hungarian optimal matching
    4. Coverage-aware final score
    """

    if model is None:
        model = load_feature_model()

    fa = remove_redundant_features(
    safe_feature_list(features_a)
    )

    fb = remove_redundant_features(
        safe_feature_list(features_b)
    )

    # empty cases
    if not fa or not fb:
        return empty_result(
            unique_a=fa,
            unique_b=fb
        )

    # -------------------------------------------------
    # Encode features
    # -------------------------------------------------
    emb_a = encode_features(fa, model)
    emb_b = encode_features(fb, model)

    # -------------------------------------------------
    # Similarity matrix
    # -------------------------------------------------
    sim_matrix = cosine_similarity(
        emb_a,
        emb_b
    )

    # -------------------------------------------------
    # Hungarian Algorithm
    # maximize similarity => minimize negative matrix
    # -------------------------------------------------
    row_idx, col_idx = linear_sum_assignment(
        -sim_matrix
    )

    matches = []

    matched_a = set()
    matched_b = set()

    for i, j in zip(row_idx, col_idx):

        sim = float(sim_matrix[i, j])

        if sim >= threshold:

            matches.append({
                "feature_a": fa[i],
                "feature_b": fb[j],
                "score": round(sim, 3)
            })

            matched_a.add(i)
            matched_b.add(j)

    # -------------------------------------------------
    # Final Metrics
    # -------------------------------------------------
    shared_scores = [
        m["score"] for m in matches
    ]

    mean_similarity = (
        float(np.mean(shared_scores))
        if shared_scores else 0.0
    )

    max_len = max(len(fa), len(fb))

    coverage = (
        len(matches) / max_len
        if max_len > 0 else 0.0
    )

    final_score = (
        (SIMILARITY_WEIGHT * mean_similarity)
        +
        (COVERAGE_WEIGHT * coverage)
    )

    final_score = min(final_score, 1.0)

    matched_text_a = " ".join(
    [
        m["feature_a"]
        for m in matches
    ]
    ).lower()

    matched_text_b = " ".join(
        [
            m["feature_b"]
            for m in matches
        ]
    ).lower()


    def is_semantically_redundant(
        feature,
        matched_text
    ):
        words = set(feature.lower().split())

        overlap = sum(
            1 for w in words
            if w in matched_text
        )

        ratio = overlap / max(len(words), 1)

        return ratio >= 0.5


    unique_a = [
        fa[i]
        for i in range(len(fa))
        if i not in matched_a
        and not is_semantically_redundant(
            fa[i],
            matched_text_a
        )
    ]

    unique_b = [
        fb[j]
        for j in range(len(fb))
        if j not in matched_b
        and not is_semantically_redundant(
            fb[j],
            matched_text_b
        )
    ]

    return {
        "score": round(final_score, 4),
        "coverage": round(coverage, 4),
        "shared_count": len(matches),
        "matches": matches,
        "unique_a": unique_a,
        "unique_b": unique_b
    }


# =====================================================
# Compare Two Rows From DataFrame
# =====================================================
def compare_projects(
    df: pd.DataFrame,
    idx1: int,
    idx2: int,
    model=None
) -> Dict[str, Any]:
    """
    Compare two projects from dataset.
    """

    if model is None:
        model = load_feature_model()

    f1 = df.loc[idx1, "features"]
    f2 = df.loc[idx2, "features"]

    result = compute_feature_similarity(
        f1,
        f2,
        model=model
    )

    result["project_a_id"] = int(idx1)
    result["project_b_id"] = int(idx2)

    return result


# =====================================================
# Compare One Against Many
# =====================================================
def compare_project_against_many(
    query_features,
    candidate_feature_lists,
    model=None,
    threshold: float = DEFAULT_THRESHOLD
):
    """
    Compare one project against many candidates.
    """

    if model is None:
        model = load_feature_model()

    results = []

    for idx, candidate in enumerate(
        candidate_feature_lists
    ):

        result = compute_feature_similarity(
            query_features,
            candidate,
            model=model,
            threshold=threshold
        )

        result["candidate_id"] = idx

        results.append(result)

    return results


# =====================================================
# Example Run
# =====================================================
if __name__ == "__main__":

    project_a = [
        "online reservation",
        "ai chatbot",
        "patient records",
        "doctor dashboard"
    ]

    project_b = [
        "appointment booking",
        "chatbot assistant",
        "medical records",
        "analytics dashboard"
    ]

    result = compute_feature_similarity(
        project_a,
        project_b
    )

    print(result)