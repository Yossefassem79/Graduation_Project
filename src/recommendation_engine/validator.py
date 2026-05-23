# src/recommendation_engine/validator.py

import re
from typing import List


# =====================================================
# Generic / weak patterns
# =====================================================
GENERIC_PATTERNS = [
    "dashboard",
    "login",
    "signup",
    "authentication",
    "admin panel",
    "analytics system",
    "analytics platform",
    "management system",
    "tracking system",
    "monitoring system",
    "ai module",
    "smart system",
    "web platform",
    "mobile app",
    "website",
    "reports page",
    "user management"
]


BAD_STARTS = [
    "here are",
    "below are",
    "these are",
    "the following",
    "project ideas",
    "features include"
]


LOW_VALUE_WORDS = [
    "system",
    "platform",
    "application",
    "website",
    "solution"
]


# =====================================================
# CLEAN TEXT
# =====================================================
def clean_text(text: str) -> str:

    if not text:
        return ""

    text = str(text).strip()

    # remove numbering
    text = re.sub(r"^\d+[\)\.\-\s]+", "", text)

    # remove bullets
    text = re.sub(r"^[\-\*\•\→\▪\s]+", "", text)

    # remove markdown
    text = text.replace("**", "")

    # remove quotes
    text = text.replace('"', "").replace("'", "")

    # remove brackets
    text = re.sub(r"\(.*?\)", "", text)

    # remove long explanations after colon
    if ":" in text and len(text.split()) > 6:
        text = text.split(":")[0]

    # remove assistant prefixes
    text = re.sub(r"^(assistant|bot)\s*[:\-]\s*", "", text, flags=re.I)

    # remove trailing punctuation
    text = re.sub(r"[.,\-:;]+$", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# NORMALIZATION
# =====================================================
def normalize_key(text: str) -> str:

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# Generic detector
# =====================================================
def is_generic(text: str) -> bool:

    low = normalize_key(text)

    for pattern in GENERIC_PATTERNS:
        if pattern in low:
            return True

    return False


# =====================================================
# Weak phrase detector
# =====================================================
def is_low_quality(text: str) -> bool:

    low = normalize_key(text)

    words = low.split()

    # too short
    if len(words) < 3:
        return True

    # too long
    if len(words) > 12:
        return True

    # starts badly
    if any(low.startswith(x) for x in BAD_STARTS):
        return True

    # mostly weak words
    weak_count = sum(
        1 for w in words
        if w in LOW_VALUE_WORDS
    )

    if weak_count >= len(words) / 2:
        return True

    return False


# =====================================================
# VALIDATION RULES
# =====================================================
def is_valid_item(text: str) -> bool:

    if not text:
        return False

    # reject generic garbage
    if is_generic(text):
        return False

    # reject low-quality phrases
    if is_low_quality(text):
        return False

    return True


# =====================================================
# FILTER ITEMS
# =====================================================
def filter_items(items: List[str]) -> List[str]:

    final = []

    seen = set()

    for item in items:

        text = clean_text(item)

        if not text:
            continue

        if not is_valid_item(text):
            continue

        key = normalize_key(text)

        # exact duplicate
        if key in seen:
            continue

        # semantic-ish duplicate
        duplicate = False

        for old in seen:

            # overlap similarity
            overlap = set(key.split()) & set(old.split())

            if len(overlap) >= max(2, min(len(key.split()), len(old.split())) - 1):
                duplicate = True
                break

        if duplicate:
            continue

        seen.add(key)

        final.append(text)

    return final


# =====================================================
# SMART SPLITTER
# =====================================================
def smart_split(text: str) -> List[str]:

    if not text:
        return []

    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        # split inline numbering
        parts = re.split(r"\d+[\.\)]\s*", line)

        for p in parts:

            p = p.strip()

            if not p:
                continue

            # split bullets inline
            subparts = re.split(r"\s*[-•▪]\s*", p)

            for sp in subparts:

                sp = sp.strip()

                if sp:
                    lines.append(sp)

    return lines


# =====================================================
# MAIN VALIDATOR
# =====================================================
def validate_generated_list(
    text: str,
    top_k: int = 10
) -> List[str]:

    if not text:
        return []

    raw_items = smart_split(text)

    cleaned = filter_items(raw_items)

    return cleaned[:top_k]