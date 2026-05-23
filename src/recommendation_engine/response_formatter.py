# src/recommendation_engine/response_formatter.py

from typing import Dict, Any, List


# =====================================================
# Helpers
# =====================================================
def unique_items(items: List[str]) -> List[str]:

    seen = set()

    final = []

    for item in items:

        key = str(item).strip().lower()

        if not key:
            continue

        if key not in seen:
            seen.add(key)
            final.append(str(item).strip())

    return final


def format_list(
    items: List[str],
    prefix: str = "- "
) -> str:

    items = unique_items(items)

    if not items:
        return "None"

    return "\n".join(
        f"{prefix}{item}"
        for item in items
    )


def clean_text(text: str) -> str:

    if not text:
        return ""

    text = str(text).strip()

    # remove duplicated assistant prefixes
    bad_prefixes = [
        "assistant:",
        "🤖 assistant:",
        "bot:"
    ]

    lower = text.lower()

    for prefix in bad_prefixes:

        if lower.startswith(prefix):

            text = text[len(prefix):].strip()

    return text


def divider() -> str:
    return "━━━━━━━━━━━━━━━━━━━━━━"


# =====================================================
# Dynamic Next Step
# =====================================================
def next_step(state: Dict[str, Any]) -> str:

    # =========================================
    # No project yet
    # =========================================
    if not state.get("project_title"):

        return (
            "👉 Ask for project ideas "
            "or specify a domain."
        )

    # =========================================
    # No features yet
    # =========================================
    if not state.get("features"):

        return (
            "👉 Next options:\n"
            "1️⃣ Generate features\n"
            "2️⃣ Generate another idea\n"
        )

    # =========================================
    # No description yet
    # =========================================
    
    # =========================================
    # Full project exists
    # =========================================
    


# =====================================================
# Full Project Overview
# =====================================================
def format_full_project(state: Dict[str, Any]) -> str:

    title = state.get("project_title", "Not defined")

    features = unique_items(
        state.get("features", [])
    )

    description = clean_text(
        state.get("description", "Not generated")
    )

    technologies = state.get("technologies", [])

    originality = state.get(
        "originality_score",
        "Unknown"
    )

    context_strength = state.get(
        "context_strength",
        "Unknown"
    )

    return f"""
📦 Full Project Overview

📌 Project Title:
{title}

⚙️ Smart Features:
{format_list(features)}

📄 Description:
{description}

🧠 Suggested Technologies:
{", ".join(technologies) if technologies else "Not defined"}

⭐ Originality Score:
{originality}

📊 Similarity Context:
{context_strength}
""".strip()


# =====================================================
# Main Formatter
# =====================================================
def format_response(
    intent: str,
    raw_response: str,
    state: Dict[str, Any]
) -> str:

    title = state.get("project_title", "")

    ideas = unique_items(
        state.get("ideas", [])
    )

    features = unique_items(
        state.get("features", [])
    )

    description = clean_text(
        state.get("description", "")
    )

    # =================================================
    # IDEA RESPONSE
    # =================================================
    if intent == "idea":

        if not ideas:
            return "⚠️ No project ideas generated."

        return f"""
💡 Suggested Project Ideas

{format_list(ideas)}

{divider()}
{next_step(state)}
""".strip()

    # =================================================
    # FEATURE RESPONSE
    # =================================================
    if intent == "feature":

        if not features:
            return "⚠️ No features generated."

        return f"""
⚙️ Smart Features for:
{title or "Selected Project"}

{format_list(features)}

{divider()}
{next_step(state)}
""".strip()

    # =================================================
    # DESCRIPTION RESPONSE
    # =================================================
    if intent == "description":

        return f"""
📄 Project Description

📌 {title or "Your Project"}

{description}

{divider()}
{next_step(state)}
""".strip()

    # =================================================
    # FULL PROJECT
    # =================================================
    if intent == "full_project":

        return f"""
{format_full_project(state)}

{divider()}
👉 You can continue improving the project.
""".strip()

    # =================================================
    # CHAT RESPONSE
    # =================================================
    cleaned = clean_text(raw_response)

    return f"""
💬 Project Assistant

{cleaned}

{divider()}
📌 Current Project:
{title or "Not defined"}

{divider()}
{next_step(state)}
""".strip()