# src/recommendation_engine/intent_classifier.py

import logging
from typing import Dict

from src.recommendation_engine.llm_client import generate_text

logger = logging.getLogger(__name__)


# =========================================
# VALID INTENTS
# =========================================
VALID_INTENTS = {
    "idea",
    "feature",
    "full_project",
    "chat"
}


# =========================================
# SAFE NORMALIZER
# =========================================
def normalize_intent(text: str) -> str:

    if not text:
        return "chat"

    text = text.lower().strip().split()[0]

    if text in VALID_INTENTS:
        return text

    return "chat"


# =========================================
# LLM INTENT CLASSIFIER
# =========================================
def classify_with_llm(user_input: str, state: Dict) -> str:

    prompt = f"""
You are an intent classifier for a graduation project assistant.

Return ONLY ONE word from:
idea
feature
description
full_project
chat

Context:
Project Title: {state.get("project_title") or "None"}
Has Features: {"yes" if state.get("features") else "no"}
Has Description: {"yes" if state.get("description") else "no"}

User:
"{user_input}"

Rules:
- Asking for ideas → idea
- Asking for another idea → idea
- Giving a project idea → feature
- Asking for features → feature
- Asking for description → description
- Asking for full project → full_project
- Otherwise → chat
"""

    try:
        result = generate_text(prompt, task="intent")
        return normalize_intent(result)

    except Exception as e:
        logger.warning(f"[INTENT ERROR] {e}")
        return "chat"


# =========================================
# HYBRID DETECTION (FINAL 🔥)
# =========================================
def detect_intent(text: str, state: dict = None) -> str:

    if state is None:
        state = {}

    text_clean = text.lower().strip()

    has_project = bool(state.get("project_title"))
    has_features = bool(state.get("features"))

    # =========================================
    # HARD RULES (NO FLOW FORCING ❌)
    # =========================================

    # explicit idea requests
    if any(x in text_clean for x in [
        "idea", "project idea", "new idea", "another idea", "suggest"
    ]):
        return "idea"

    # explicit feature requests
    if "feature" in text_clean:
        return "feature"


    # full project
    if any(x in text_clean for x in [
        "full project", "complete", "all details"
    ]):
        return "full_project"

    # =========================================
    # USER PROVIDED IDEA (🔥 important)
    # =========================================
    # If user writes sentence like a project idea
    if not has_project and len(text_clean.split()) >= 3:
        return "feature"

    # =========================================
    # LLM fallback
    # =========================================
    intent = classify_with_llm(text, state)

    logger.info(f"[INTENT] {intent}")

    # =========================================
    # SAFETY FIXES
    # =========================================
    if intent == "feature" and not has_project:
        return "feature"  # user likely gave idea

    if intent == "description" and not has_project:
        return "idea"

    return intent