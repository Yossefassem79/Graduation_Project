# src/recommendation_engine/llm_client.py

import time
import logging
from typing import List

from google import genai

from src.recommendation_engine.config import (
    GEMINI_API_KEY,
    MODEL_CANDIDATES,
    IDEA_TEMPERATURE,
    FEATURE_TEMPERATURE,
    CHAT_TEMPERATURE,
    INTENT_TEMPERATURE,
    IDEA_MAX_TOKENS,
    FEATURE_MAX_TOKENS,
    CHAT_MAX_TOKENS,
    INTENT_MAX_TOKENS,
    TOP_P,
    TOP_K,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    ENABLE_LOGGING
)

from src.recommendation_engine.validator import validate_generated_list

# =========================================
# Logging
# =========================================
logger = logging.getLogger(__name__)

if ENABLE_LOGGING:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

# =========================================
# Init Gemini
# =========================================
client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================
# Extract Text (Robust)
# =========================================
def extract_text(response) -> str:

    if not response:
        return ""

    # direct
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    # fallback
    try:
        candidates = getattr(response, "candidates", [])
        if candidates:
            parts = candidates[0].content.parts
            return " ".join(
                p.text for p in parts if hasattr(p, "text")
            ).strip()
    except Exception:
        pass

    return ""


# =========================================
# Temperature Selector
# =========================================
def get_temperature(task: str) -> float:

    return {
        "idea": IDEA_TEMPERATURE,
        "feature": FEATURE_TEMPERATURE,
        "intent": INTENT_TEMPERATURE,
    }.get(task, CHAT_TEMPERATURE)


# =========================================
# Max Tokens Selector
# =========================================
def get_max_tokens(task: str) -> int:

    return {
        "idea": IDEA_MAX_TOKENS,
        "feature": FEATURE_MAX_TOKENS,
        "intent": INTENT_MAX_TOKENS,
    }.get(task, CHAT_MAX_TOKENS)


# =========================================
# Prompt Safety
# =========================================
def safe_prompt(prompt: str, max_chars: int = 12000) -> str:
    return prompt[-max_chars:]


# =========================================
# Response Quality Check (🔥 NEW)
# =========================================
def is_bad_response(text: str) -> bool:

    if not text:
        return True

    text = text.strip()

    # ❌ only reject VERY bad outputs
    if len(text) < 3:
        return True

    # ❌ reject only pure assistant chatter
    bad_phrases = [
        "as an ai",
        "i can help you",
        "let me know"
    ]

    lower = text.lower()

    if all(p in lower for p in bad_phrases):
        return True

    return False


# =========================================
# Main Generate Function
# =========================================
def generate_text(
    prompt: str,
    task: str = "chat"
) -> str:

    prompt = safe_prompt(prompt)

    temperature = get_temperature(task)
    max_tokens = get_max_tokens(task)

    for model_name in MODEL_CANDIDATES:

        for attempt in range(MAX_RETRIES):

            try:
                logger.info(
                    f"[LLM] model={model_name} | task={task} | attempt={attempt+1}"
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "max_output_tokens": max_tokens
                    }
                )

                text = extract_text(response)

                # 🔥 skip garbage responses
                if is_bad_response(text):
                    logger.warning("[LLM] Weak response, using anyway")
                    return text  # 🔥 DO NOT RETRY

                return text

            except Exception as e:
                logger.warning(f"[LLM ERROR] {e}")

                # exponential backoff
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        logger.info(f"[LLM] switching model...")

    logger.error("All LLM models failed")

    return ""


# =========================================
# Generate List (🔥 FIXED)
# =========================================
def generate_list(prompt: str, task="chat") -> List[str]:

    text = generate_text(prompt, task=task)

    return validate_generated_list(text)