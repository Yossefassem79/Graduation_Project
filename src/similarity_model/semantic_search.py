# src/semantic_search.py

import re
import ast
import logging
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from Data.database.sql_connector import (
    load_preprocessed_projects
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
DEFAULT_MODEL = "all-MiniLM-L6-v2"

TITLE_COL = "project_title"
TECH_COL = "technologies"

INDEX_PATH = "models/faiss_index.bin"
META_PATH = "models/metadata.parquet"
EMBED_PATH = "models/project_embeddings.npy"

TOP_K = 10
MIN_SCORE = 0.35

# =====================================================
# Text Helpers
# =====================================================
def normalize_text(text: str) -> str:
    """
    Normalize user query to match preprocessing style.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\#\./\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> set:
    """
    Tokenize normalized text.
    """
    return set(normalize_text(text).split())


# =====================================================
# Cached Loaders
# =====================================================
@lru_cache(maxsize=1)
def load_model():
    logger.info(f"Loading model: {DEFAULT_MODEL}")
    return SentenceTransformer(DEFAULT_MODEL)


@lru_cache(maxsize=1)
def load_faiss_index():
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError("FAISS index not found.")

    logger.info("Loading FAISS index...")
    return faiss.read_index(INDEX_PATH)


@lru_cache(maxsize=1)
def load_metadata():

    logger.info(
        "Loading metadata from Azure SQL..."
    )

    df = load_preprocessed_projects()

    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_embeddings():
    if not Path(EMBED_PATH).exists():
        raise FileNotFoundError("Embeddings not found.")

    logger.info("Loading embeddings...")
    return np.load(EMBED_PATH)


# =====================================================
# Core Result Builder
# =====================================================
def build_results(
    df: pd.DataFrame,
    ids,
    scores,
    query_text: str = "",
    min_score: float = MIN_SCORE
) -> pd.DataFrame:

    rows = []

    query_words = tokenize(query_text)

    for idx, score in zip(ids, scores):

        if idx == -1:
            continue

        row = df.loc[idx]

        final_score = float(score)

        # ---------------------------------------------
        # Keyword Boosting
        # ---------------------------------------------
        if query_words:

            title_words = tokenize(row[TITLE_COL])

            tech_words = tokenize(
                row.get(TECH_COL, "")
            )

            overlap = len(query_words & title_words)
            overlap += len(query_words & tech_words)

            if overlap > 0:
                final_score += 0.02 * overlap

        # cap score
        final_score = min(final_score, 1.0)

        if final_score < min_score:
            continue

        rows.append({
            "project_id": int(idx),
            "project_title": row[TITLE_COL],
            "technologies": row.get(TECH_COL, ""),
            "score": round(final_score, 4)
        })

    if not rows:
        return pd.DataFrame([{
            "message": "No similar projects found.",
            "score": 0
        }])

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


# =====================================================
# Search by Free Text
# =====================================================
def search_by_text(
    query_text: str,
    k: int = TOP_K,
    min_score: float = MIN_SCORE
) -> pd.DataFrame:

    model = load_model()
    index = load_faiss_index()
    df = load_metadata()

    query_clean = normalize_text(query_text)

    query_vec = model.encode(
        [query_clean],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(query_vec, k)

    return build_results(
        df=df,
        ids=ids[0],
        scores=scores[0],
        query_text=query_clean,
        min_score=min_score
    )


# =====================================================
# Search Similar to Existing Project
# =====================================================
def search_by_project_id(
    project_id: int,
    k: int = TOP_K,
    min_score: float = MIN_SCORE,
    exclude_self: bool = True
) -> pd.DataFrame:

    df = load_metadata()
    index = load_faiss_index()
    embeddings = load_embeddings()

    if project_id < 0 or project_id >= len(df):
        raise IndexError("Project ID out of range.")

    query_vec = embeddings[
        project_id
    ].reshape(1, -1).astype("float32")

    extra_k = k + 1 if exclude_self else k

    scores, ids = index.search(
        query_vec,
        extra_k
    )

    rows = []

    for idx, score in zip(ids[0], scores[0]):

        if idx == -1:
            continue

        if exclude_self and idx == project_id:
            continue

        final_score = min(float(score), 1.0)

        if final_score < min_score:
            continue

        row = df.loc[idx]

        rows.append({
            "project_id": int(idx),
            "project_title": row[TITLE_COL],
            "technologies": row.get(TECH_COL, ""),
            "score": round(final_score, 4)
        })

        if len(rows) == k:
            break

    if not rows:
        return pd.DataFrame([{
            "message": "No similar projects found.",
            "score": 0
        }])

    return pd.DataFrame(rows)


# =====================================================
# Compare Two Ideas
# =====================================================
def compare_two_ideas(
    text_a: str,
    text_b: str
) -> float:

    model = load_model()

    vecs = model.encode(
        [
            normalize_text(text_a),
            normalize_text(text_b)
        ],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    score = float(np.dot(vecs[0], vecs[1]))

    return round(score, 4)


# =====================================================
# Example Run
# =====================================================
if __name__ == "__main__":

    print("\n=== Search by Text ===")
    print(
        search_by_text(
            "mobile app for expense tracking using flutter",
            k=5
        )
    )

    print("\n=== Similar Projects ===")
    print(
        search_by_project_id(
            project_id=0,
            k=5
        )
    )

    print("\n=== Compare Two Ideas ===")
    print(
        compare_two_ideas(
            "AI Medical Chatbot",
            "Healthcare Assistant with AI"
        )
    )