from typing import Dict, Any, List


# =====================================================
# HELPERS
# =====================================================
def normalize(text: str) -> str:
    return str(text).strip().lower()


def merge_lists(old: List[str], new: List[str]) -> List[str]:

    seen = set()

    result = []

    for item in (old or []) + (new or []):

        if not item:
            continue

        key = normalize(item)

        if key not in seen:

            seen.add(key)

            result.append(str(item).strip())

    return result


# =====================================================
# SMART UPDATE STATE
# =====================================================
def update_state(
    state: Dict[str, Any],
    new_data: Dict[str, Any],
    mode: str = "merge"
) -> Dict[str, Any]:

    for key, value in new_data.items():

        if key in ["user_input", "intent"]:
            continue

        if value is None or value == "":
            continue

        # =========================================
        # ALWAYS REPLACE
        # =========================================
        if key in [
            "ideas",
            "project_title",
            "description",
            "abstract",
            "category",
            "problem_statement",
            "proposed_solution",
            "methodology",
            "ai_summary"
        ]:

            state[key] = value
            continue

        # =========================================
        # FEATURES
        # =========================================
        if key == "features":

            if mode == "replace":
                state[key] = value
            else:
                state[key] = merge_lists(
                    state.get(key, []),
                    value
                )

            continue

        # =========================================
        # LIST FIELDS
        # =========================================
        if isinstance(value, list):

            if mode == "replace":

                state[key] = value

            else:

                state[key] = merge_lists(
                    state.get(key, []),
                    value
                )

            continue

        # =========================================
        # STRING FIELDS
        # =========================================
        if isinstance(value, str):

            clean_val = value.strip()

            old_val = str(
                state.get(key, "")
            ).strip()

            if (
                mode == "replace"
                or clean_val not in old_val
            ):
                state[key] = (
                    f"{old_val}\n{clean_val}"
                ).strip()

            continue

        # =========================================
        # OTHER TYPES
        # =========================================
        state[key] = value

    return state


# =====================================================
# RESET HELPERS
# =====================================================
def reset_for_new_idea(
    state: Dict[str, Any]
) -> Dict[str, Any]:

    return {

        "domain": state.get("domain"),

        "ideas": [],

        "project_title": "",

        "features": [],

        "description": "",

        "abstract": "",

        "technologies": [],

        "keywords": [],

        "category": "",

        "problem_statement": "",

        "proposed_solution": "",

        "objectives": [],

        "future_work": [],

        "methodology": "",

        "ai_summary": "",

        "originality_score": None,

        "context_strength": None
    }


def reset_features_only(
    state: Dict[str, Any]
) -> Dict[str, Any]:

    state["features"] = []

    state["description"] = ""

    return state