# src/embedding_engine.py

import re
import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
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

TEXT_COL = "clean_text"
TITLE_COL = "project_title"
TECH_COL = "technologies"

MODEL_DIR = Path("models")
INDEX_PATH = MODEL_DIR / "faiss_index.bin"
META_PATH = MODEL_DIR / "metadata.parquet"

TOP_K_DEFAULT = 10
MIN_SCORE_THRESHOLD = 0.35

# =====================================================
# Helpers
# =====================================================
def normalize_text(text: str) -> str:
    """
    Same cleaning logic used in preprocessing.
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
    Simple tokenization for keyword boosting.
    """
    text = normalize_text(text)
    return set(text.split())


# =====================================================
# Core Engine
# =====================================================
class ProjectEmbedder:

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = None

    # -------------------------------------------------
    # Embeddings
    # -------------------------------------------------
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 64
    ) -> np.ndarray:

        logger.info(f"Generating embeddings for {len(texts)} projects...")

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return vectors.astype("float32")

    # -------------------------------------------------
    # Build Index
    # -------------------------------------------------
    def build_index(self, df: pd.DataFrame):
        """
        Build FAISS cosine index.
        """

        self.metadata = df.copy()

        # preserve ids
        self.metadata = self.metadata.reset_index(drop=True)

        # ensure needed columns exist
        for col in [TITLE_COL, TEXT_COL]:
            if col not in self.metadata.columns:
                self.metadata[col] = ""

        if TECH_COL not in self.metadata.columns:
            self.metadata[TECH_COL] = ""

        # weighted content:
        # title repeated twice
        rich_texts = (
            self.metadata[TITLE_COL].fillna("").astype(str) + " " +
            self.metadata[TITLE_COL].fillna("").astype(str) + " " +
            self.metadata[TEXT_COL].fillna("").astype(str)
        ).tolist()

        embeddings = self.generate_embeddings(rich_texts)

        dim = embeddings.shape[1]

        base_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base_index)

        ids = np.arange(len(self.metadata)).astype("int64")

        self.index.add_with_ids(embeddings, ids)

        logger.info(
            f"FAISS index built successfully with {self.index.ntotal} vectors."
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    def save_artifacts(self, folder: str = "models"):

        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "faiss_index.bin")
        )

        self.metadata.to_parquet(
            path / "metadata.parquet",
            index=False
        )

        logger.info(f"Artifacts saved to {folder}")

    # -------------------------------------------------
    # Load
    # -------------------------------------------------
    def load_artifacts(self, folder: str = "models"):

        path = Path(folder)

        self.index = faiss.read_index(
            str(path / "faiss_index.bin")
        )

        self.metadata = pd.read_parquet(
            path / "metadata.parquet"
        )

        logger.info("Artifacts loaded successfully.")

    # -------------------------------------------------
    # Search
    # -------------------------------------------------
    def search(
        self,
        query: str,
        k: int = TOP_K_DEFAULT,
        threshold: float = MIN_SCORE_THRESHOLD
    ) -> pd.DataFrame:

        if self.index is None or self.metadata is None:
            raise ValueError("Index or metadata not loaded.")

        # normalize query
        query_clean = normalize_text(query)

        query_vec = self.model.encode(
            [query_clean],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, ids = self.index.search(query_vec, k)

        query_words = tokenize(query_clean)

        results = []

        for idx, score in zip(ids[0], scores[0]):

            if idx == -1:
                continue

            row = self.metadata.loc[idx]

            final_score = float(score)

            # keyword boost
            title_words = tokenize(row[TITLE_COL])
            tech_words = tokenize(row[TECH_COL])

            overlap = len(query_words & title_words)
            overlap += len(query_words & tech_words)

            if overlap > 0:
                final_score += 0.02 * overlap

            # cap score
            final_score = min(final_score, 1.0)

            # threshold
            if final_score < threshold:
                continue

            results.append({
                "project_id": int(idx),
                "title": row[TITLE_COL],
                "technologies": row[TECH_COL],
                "similarity_score": round(final_score, 4)
            })

        if not results:
            return pd.DataFrame([{
                "message": "No similar projects found."
            }])

        return pd.DataFrame(results).sort_values(
            by="similarity_score",
            ascending=False
        ).reset_index(drop=True)

# =====================================================
# Full Training Pipeline
# =====================================================
def train_embedding_engine():

    logger.info(
        "Loading processed dataset from Azure SQL..."
    )

    df = load_preprocessed_projects()

    engine = ProjectEmbedder()

    engine.build_index(df)

    engine.save_artifacts()

    logger.info(
        "Embedding engine completed successfully."
    )

    return engine


# =====================================================
# Example Run
# =====================================================
if __name__ == "__main__":

    engine = train_embedding_engine()

    query = "Build a mobile app for expense tracking using flutter and firebase"

    print(f"\nQuery: {query}\n")

    results = engine.search(query, k=5)

    print(results)