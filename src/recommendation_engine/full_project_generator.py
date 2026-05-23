from src.recommendation_engine.llm_client import generate_text
from src.recommendation_engine.prompt_builder import (
    build_full_project_prompt
)

import re


# =====================================================
# SECTION EXTRACTOR
# =====================================================
def extract_section(text, section_name):

    text = text.strip()

    marker = section_name + ":"

    if marker not in text:
        return ""

    start = text.find(marker) + len(marker)

    sections = [
        "CATEGORY:",
        "ABSTRACT:",
        "DESCRIPTION:",
        "TECHNOLOGIES:",
        "KEYWORDS:",
        "PROBLEM_STATEMENT:",
        "PROPOSED_SOLUTION:",
        "OBJECTIVES:",
        "AI_SUMMARY:",
        "FUTURE_WORK:",
        "METHODOLOGY:"
    ]

    end = len(text)

    for s in sections:

        if s == marker:
            continue

        pos = text.find(s, start)

        if pos != -1 and pos < end:
            end = pos

    return text[start:end].strip()


# =====================================================
# BULLET PARSER
# =====================================================
def parse_bullets(text):

    lines = text.splitlines()

    final = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^[-•*0-9.\)\s]+",
            "",
            line
        )

        if line:
            final.append(line)

    return final


# =====================================================
# FULL PROJECT GENERATOR
# =====================================================
def generate_full_project(
    title,
    features,
    description="",
    abstract=""
):

    context = {
        "project_title": title,
        "features": features,
        "description": description,
        "abstract": abstract
    }

    prompt = build_full_project_prompt(context)

    raw = generate_text(
        prompt,
        task="chat"
    )

    result = {

        # =============================================
        # BASIC
        # =============================================
        "project_title": title,

        "category":
            extract_section(raw, "CATEGORY"),

        "abstract":
            extract_section(raw, "ABSTRACT"),

        "description":
            extract_section(raw, "DESCRIPTION"),

        # =============================================
        # LISTS
        # =============================================
        "technologies":
            parse_bullets(
                extract_section(raw, "TECHNOLOGIES")
            ),

        "keywords":
            parse_bullets(
                extract_section(raw, "KEYWORDS")
            ),

        "objectives":
            parse_bullets(
                extract_section(raw, "OBJECTIVES")
            ),

        "future_work":
            parse_bullets(
                extract_section(raw, "FUTURE_WORK")
            ),

        # =============================================
        # TEXT FIELDS
        # =============================================
        "problem_statement":
            extract_section(
                raw,
                "PROBLEM_STATEMENT"
            ),

        "proposed_solution":
            extract_section(
                raw,
                "PROPOSED_SOLUTION"
            ),

        "methodology":
            extract_section(
                raw,
                "METHODOLOGY"
            ),

        "ai_summary":
            extract_section(
                raw,
                "AI_SUMMARY"
            )
    }

    # =================================================
    # FALLBACKS
    # =================================================

    if not result.get("category"):
        result["category"] = "General AI System"

    if not result.get("keywords"):
        result["keywords"] = [
            "Artificial Intelligence",
            "Automation",
            "Smart System"
        ]

    if not result.get("problem_statement"):
        result["problem_statement"] = (
            "Current traditional systems suffer from "
            "limited automation, inefficiency, and "
            "lack of intelligent decision-making."
        )

    if not result.get("proposed_solution"):
        result["proposed_solution"] = (
            "The proposed system uses AI-driven "
            "automation and intelligent analytics "
            "to improve operational efficiency."
        )

    if not result.get("objectives"):
        result["objectives"] = [
            "Improve automation efficiency",
            "Enhance system accuracy",
            "Reduce operational costs"
        ]

    if not result.get("methodology"):
        result["methodology"] = (
            "The system will be developed using "
            "data collection, preprocessing, "
            "AI model training, testing, and deployment."
        )

    if not result.get("future_work"):
        result["future_work"] = [
            "Cloud integration",
            "Mobile application support",
            "Advanced AI optimization"
        ]

    if not result.get("ai_summary"):
        result["ai_summary"] = (
            f"{title} is an intelligent AI-powered "
            f"graduation project designed to provide "
            f"automation, monitoring, and predictive analysis."
        )

    return result