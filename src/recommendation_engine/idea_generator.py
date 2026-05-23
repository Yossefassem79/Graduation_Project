# src/recommendation_engine/idea_generator.py

import logging
from typing import Dict, Any, List, Set

from src.recommendation_engine.context_builder import build_domain_context
from src.recommendation_engine.prompt_builder import build_idea_prompt
from src.recommendation_engine.llm_client import generate_text
from src.recommendation_engine.validator import validate_generated_list

from src.similarity_model import compare_two_ideas, load_metadata
from src.recommendation_engine.config import DEFAULT_IDEA_COUNT

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
SIMILARITY_THRESHOLD_LOCAL = 0.60


# =========================================
# 🔥 LOAD DATASET TITLES (ONCE)
# =========================================
DATA_TITLES = set()

def normalize_idea(text: str) -> str:
    return " ".join(text.lower().strip().split())


def init_dataset_titles():
    global DATA_TITLES

    df = load_metadata()

    if df is not None and "project_title" in df.columns:
        DATA_TITLES = set(
            normalize_idea(t)
            for t in df["project_title"].dropna()
        )

init_dataset_titles()


# =========================================
# HELPERS
# =========================================
def clean_ideas(ideas: List[str]) -> List[str]:
    return [i.strip() for i in ideas if 2 <= len(i.split()) <= 8]


def is_duplicate_local(idea: str, existing: Set[str]) -> bool:
    for old in existing:
        if compare_two_ideas(idea, old) >= SIMILARITY_THRESHOLD_LOCAL:
            return True
    return False


# =========================================
# FALLBACK
# =========================================
def fallback_by_domain(domain: str) -> List[str]:

    domain = (domain or "general").lower()

    fallback_map = {
        "education": [
            "AI adaptive learning system",
            "Student performance prediction platform",
            "Gamified learning mobile application",
            "Automated grading system",
            "Virtual classroom engagement analyzer"
        ],
        "healthcare": [
            "AI disease prediction system",
            "Smart patient monitoring system",
            "Medical diagnosis assistant",
            "IoT health tracking device",
            "Hospital resource optimization system"
        ],
        "fintech": [
            "Fraud detection AI system",
            "Smart expense tracking app",
            "Blockchain payment system",
            "Credit risk prediction model",
            "AI investment advisor"
        ]
    }

    return fallback_map.get(domain, [
        "AI recommendation system",
        "Smart automation platform",
        "Data analytics dashboard",
        "Intelligent decision support system",
        "Predictive analytics engine"
    ])


# =========================================
# MAIN GENERATOR
# =========================================
def generate_ideas(
    domain: str,
    top_k: int = DEFAULT_IDEA_COUNT,
    previous_generated_ideas: List[str] = None
) -> Dict[str, Any]:

    if previous_generated_ideas is None:
        previous_generated_ideas = []

    top_k = max(1, min(top_k, 20))
    domain = domain or "general"

    logger.info(f"Starting idea generation | domain={domain} | top_k={top_k}")

    context = build_domain_context(domain)

    final_ideas: List[str] = []
    final_set: Set[str] = set()
    previous_norm = set(normalize_idea(i) for i in previous_generated_ideas)

    all_generated: List[str] = []
    attempts = 0

    # =========================================
    # GENERATION LOOP
    # =========================================
    while len(final_ideas) < top_k and attempts < MAX_RETRIES:

        attempts += 1
        logger.info(f"Attempt #{attempts}")

        generation_count = max(top_k * 3, 12)

        prompt = build_idea_prompt(
            context=context,
            count=generation_count,
            previous_ideas=previous_generated_ideas
        )

        raw_text = generate_text(prompt, task="idea")

        if not raw_text:
            logger.warning("Empty LLM response")
            continue

        generated = validate_generated_list(
            text=raw_text,
            top_k=generation_count
        )

        generated = clean_ideas(generated)

        logger.info(f"Generated {len(generated)} ideas")
        all_generated.extend(generated)

        # =========================================
        # FILTERING
        # =========================================
        for idea in generated:

            normalized = normalize_idea(idea)

            if not normalized:
                continue

            # ❌ dataset duplicate
            # 🔥 dataset exact match
            if normalized in DATA_TITLES:
                logger.info(f"[SKIP DATASET EXACT] {idea}")
                continue

            # 🔥 dataset semantic match
            skip_dataset_similar = False
            for data_title in list(DATA_TITLES)[:500]:  # limit for speed
                if compare_two_ideas(idea, data_title) >= 0.85:
                    logger.info(f"[SKIP DATASET SIMILAR] {idea}")
                    skip_dataset_similar = True
                    break

            if skip_dataset_similar:
                continue

            # ❌ duplicate inside current batch
            if normalized in final_set:
                continue

            # ❌ duplicate from previous session
            if normalized in previous_norm:
                logger.info(f"[SKIP PREVIOUS] {idea}")
                continue


            # 🔥 semantic check with previous ideas
            skip_similar_prev = False
            for old in previous_generated_ideas:
                if compare_two_ideas(idea, old) >= 0.85:
                    logger.info(f"[SKIP SIMILAR PREVIOUS] {idea}")
                    skip_similar_prev = True
                    break

            if skip_similar_prev:
                continue

            # ❌ semantic duplicate
            if is_duplicate_local(idea, final_set):
                logger.info(f"[SKIP SIMILAR] {idea}")
                continue

            # ✅ accept idea
            logger.info(f"[NEW IDEA] {idea}")

            final_ideas.append(idea)
            final_set.add(normalized)

            if len(final_ideas) >= top_k:
                break

    # =========================================
    # FALLBACK
    # =========================================
    if len(final_ideas) < top_k:

        logger.warning("Using fallback ideas")

        fallback = fallback_by_domain(domain)

        for f in fallback:

            normalized = normalize_idea(f)

            if normalized not in final_set:
                final_ideas.append(f)
                final_set.add(normalized)

            if len(final_ideas) >= top_k:
                break

    logger.info(f"Final ideas: {final_ideas}")

    return {
        "domain": domain,
        "generated_ideas": all_generated,
        "final_ideas": final_ideas
    }