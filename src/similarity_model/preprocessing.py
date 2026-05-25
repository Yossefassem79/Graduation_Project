# src/preprocessing.py
# FINAL POLISHED VERSION
# Best Practical Feature Extraction for Graduation Project System

import re
import logging
from pathlib import Path

import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# =====================================================
# Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# Models
# =====================================================
MODEL_NAME = "all-MiniLM-L6-v2"

embed_model = SentenceTransformer(
    MODEL_NAME
)

kw_model = KeyBERT(
    model=embed_model
)

# =====================================================
# Config
# =====================================================
MIN_WORDS = 8
MAX_WORDS = 4000
MAX_FEATURES = 10

TECH_TERMS = {
    "python", "java", "c++", "c#", "flutter",
    "react", "node.js", "firebase",
    "sql", "mysql", "mongodb",
    "tensorflow", "pytorch",
    "opencv", "arduino",
    "chatbot", "cnn", "nlp",
    "qr code", "api"
}

BAD_WORDS = {
    "government",
    "facility",
    "facilities",
    "nationalization",
    "content",
    "keyword",
    "keywords",
    "presence",
    "addition",
    "appropriate",
    "developed",
    "system",
    "application",
    "project"
}

# =====================================================
# Helpers
# =====================================================
def normalize_text(text):
    """
    Clean raw text
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # remove urls/emails
    text = re.sub(
        r"http\S+|www\S+|\S+@\S+",
        " ",
        text
    )

    # keep useful chars
    text = re.sub(
        r"[^a-z0-9\+\#\./\- ]",
        " ",
        text
    )

    # remove spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_phrase(text):
    """
    Clean extracted phrase
    """

    text = normalize_text(text)

    # remove numbering
    text = re.sub(
        r"^\d+\-?\s*",
        "",
        text
    )

    # remove articles
    text = re.sub(
        r"^(a|an|the)\s+",
        "",
        text
    )

    return text.strip()


def is_valid_phrase(text):
    """
    Validate phrase
    """

    if not text:
        return False

    if len(text.split()) < 2:
        return False

    if len(text.split()) > 4:
        return False

    words = text.split()

    # reject if contains junk words
    if any(word in BAD_WORDS for word in words):
        return False

    return True


def detect_tech_terms(text):
    """
    Detect important exact technical terms
    """

    found = []

    for term in TECH_TERMS:
        if term in text:
            found.append(term)

    return found


# =====================================================
# Main Feature Extraction
# =====================================================
def extract_features(text):
    """
    Final high-quality extractor
    """

    try:

        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=12,
            use_mmr=True,
            diversity=0.5
        )

        final = []
        seen = set()

        # =============================================
        # Keep KeyBERT original ranking
        # =============================================
        for phrase, score in keywords:

            phrase = clean_phrase(phrase)

            if not is_valid_phrase(phrase):
                continue

            if phrase not in seen:
                seen.add(phrase)
                final.append(phrase)

        # =============================================
        # Add missing exact tech terms
        # =============================================
        for term in detect_tech_terms(text):

            if term not in seen:
                seen.add(term)
                final.append(term)

        return final[:MAX_FEATURES]

    except Exception as e:

        logger.warning(
            f"Feature extraction failed: {e}"
        )

        return []


# =====================================================
# Main Pipeline
# =====================================================
def preprocess_dataset(df):
    """
    Full preprocessing pipeline
    """

    logger.info(
        "Starting preprocessing..."
    )

    df = df.copy()

    # clean columns
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(
            r"\W+",
            "_",
            regex=True
        )
    )

    # =============================================
    # Column Mapping Fix
    # =============================================
    column_mapping = {
        "title": "project_title",
        "ai_summary": "ai_summary",
        "technologies": "technologies",
        "keywords": "keywords",
        "abstract": "abstract",
        "description": "description",
        "problem_statement": "problem_statement",
        "proposed_solution": "proposed_solution",
        "objectives": "objectives",
        "category": "category"
    }

    df = df.rename(columns=column_mapping)

    # ensure needed columns
    for col in [
        "project_title",
        "abstract",
        "description"
    ]:

        if col not in df.columns:
            df[col] = ""

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
        )

    # =============================================
    # Smart weighted merge
    # =============================================
    df["full_content"] = (
        df["project_title"] + ". " +
        df["project_title"] + ". " +
        df["abstract"] + ". " +
        df["description"]
    )

    # normalize
    df["clean_text"] = (
        df["full_content"]
        .apply(normalize_text)
    )

    # remove duplicates
    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "project_title",
            "clean_text"
        ]
    ).copy()

    logger.info(
        f"Removed duplicates: {before-len(df)}"
    )

    # word count filter
    df["word_count"] = (
        df["clean_text"]
        .str.split()
        .str.len()
    )

    df = df[
        df["word_count"].between(
            MIN_WORDS,
            MAX_WORDS
        )
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    # =============================================
    # Feature Extraction
    # =============================================
    logger.info(
        "Extracting features..."
    )

    df["features"] = (
        df["clean_text"]
        .apply(extract_features)
    )

    # remove empty rows
    df = df[
        df["features"]
        .apply(len) > 0
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    logger.info(
        f"Final rows: {len(df)}"
    )

    return df


# =====================================================
# Save
# =====================================================
def save_processed_data(
    df,
    output_dir="Data/processed"
):

    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        path / "projects_clean.parquet",
        index=False
    )

    df.to_csv(
        path / "projects_clean.csv",
        index=False
    )

    logger.info(
        f"Saved to {path}"
    )


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":

    file_path = "Data/raw/projects.xlsx"

    if file_path.endswith(".csv"):
        raw_df = pd.read_csv(file_path)
    else:
        raw_df = pd.read_excel(file_path)

    clean_df = preprocess_dataset(raw_df)

    save_processed_data(clean_df)