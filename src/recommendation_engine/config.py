# src/recommendation_engine/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# =====================================================
# Load Environment Variables
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# =====================================================
# App Mode
# =====================================================
ENV = os.getenv("ENV", "development")
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# =====================================================
# Gemini API Config
# =====================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MODEL_CANDIDATES = [
    os.getenv("PRIMARY_MODEL", "gemini-3.1-flash-lite-preview"),
    os.getenv("FAST_MODEL", "gemini-2.5-flash-lite"),
    os.getenv("BALANCED_MODEL", "gemini-2.5-flash"),
    os.getenv("QUALITY_MODEL", "gemini-2.5-pro"),
]

# remove duplicates safely
_seen = set()
MODEL_CANDIDATES = [
    m for m in MODEL_CANDIDATES
    if m and not (m in _seen or _seen.add(m))
]

# =====================================================
# Generation Settings
# =====================================================
IDEA_TEMPERATURE = float(os.getenv("IDEA_TEMPERATURE", 0.9))
FEATURE_TEMPERATURE = float(os.getenv("FEATURE_TEMPERATURE", 0.6))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.7))
INTENT_TEMPERATURE = float(os.getenv("INTENT_TEMPERATURE", 0.0))

# Max tokens
IDEA_MAX_TOKENS = int(os.getenv("IDEA_MAX_TOKENS", 800))
FEATURE_MAX_TOKENS = int(os.getenv("FEATURE_MAX_TOKENS", 600))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", 700))
INTENT_MAX_TOKENS = int(os.getenv("INTENT_MAX_TOKENS", 20))

TOP_P = float(os.getenv("TOP_P", 0.95))
TOP_K = int(os.getenv("TOP_K", 40))

# =====================================================
# Retry / Timeout
# =====================================================
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", 2))

# =====================================================
# Recommendation Defaults
# =====================================================
DEFAULT_FEATURE_COUNT = int(os.getenv("DEFAULT_FEATURE_COUNT", 5))
DEFAULT_IDEA_COUNT = int(os.getenv("DEFAULT_IDEA_COUNT", 5))
GENERATION_BATCH_SIZE = int(os.getenv("GENERATION_BATCH_SIZE", 10))

# =====================================================
# Duplicate Detection
# =====================================================
IDEA_DUPLICATE_THRESHOLD = float(
    os.getenv("IDEA_DUPLICATE_THRESHOLD", 0.82)
)

FEATURE_DUPLICATE_THRESHOLD = float(
    os.getenv("FEATURE_DUPLICATE_THRESHOLD", 0.75)
)

# =====================================================
# Idea Generator Control
# =====================================================
MAX_IDEA_RETRIES = int(os.getenv("MAX_IDEA_RETRIES", 3))
MAX_IDEA_LIMIT = int(os.getenv("MAX_IDEA_LIMIT", 20))

# =====================================================
# Similarity Engine
# =====================================================
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))

# =====================================================
# State / Memory Control
# =====================================================
MAX_HISTORY = int(os.getenv("MAX_HISTORY", 20))
MAX_FEATURES = int(os.getenv("MAX_FEATURES", 20))
MAX_IDEAS = int(os.getenv("MAX_IDEAS", 10))

# =====================================================
# Semantic Routing
# =====================================================
ENABLE_SEMANTIC_INTENT = os.getenv(
    "ENABLE_SEMANTIC_INTENT", "true"
).lower() == "true"

# =====================================================
# Logging
# =====================================================
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =====================================================
# Safety Check (🔥 IMPORTANT)
# =====================================================
if not GEMINI_API_KEY:
    raise ValueError(
        "❌ GEMINI_API_KEY is missing. Please set it in .env file."
    )

# =====================================================
# Debug Print
# =====================================================
if DEBUG_MODE and ENV == "development":
    print("\n CONFIG LOADED:")
    print(f"ENV: {ENV}")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"MODELS: {MODEL_CANDIDATES}")
    print(f"MAX_RETRIES: {MAX_RETRIES}")
    print(f"IDEA_TEMP: {IDEA_TEMPERATURE}")
    print("=================================\n")