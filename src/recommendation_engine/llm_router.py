import re
from src.recommendation_engine.context_builder import extract_domain


def word_to_number(text: str):
    mapping = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    text = text.lower()

    for word, num in mapping.items():
        if word in text:
            return num

    nums = re.findall(r'\d+', text)
    if nums:
        return int(nums[0])

    return None


def analyze_user_input(user_input: str, state: dict) -> dict:

    text = user_input.lower()

    # =============================
    # INTENT DETECTION (RULE BASED)
    # =============================
    if any(w in text for w in ["idea", "project", "suggest", "recommend"]):
        intent = "idea"

    elif "feature" in text:
        intent = "feature"

    elif "describe" in text or "description" in text:
        intent = "description"

    else:
        intent = "chat"

    # =============================
    # DOMAIN DETECTION
    # =============================
    domain = extract_domain(text)

    # =============================
    # NUMBER DETECTION (🔥 NEW)
    # =============================
    number = word_to_number(text)

    # =============================
    # PROJECT TITLE DETECTION
    # =============================
    project_title = None

    if intent == "chat" and len(text.split()) > 4:
        project_title = user_input

    return {
        "intent": intent,
        "domain": domain,
        "project_title": project_title,
        "number": number
    }