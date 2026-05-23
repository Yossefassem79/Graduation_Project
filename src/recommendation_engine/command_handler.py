# src/recommendation_engine/command_handler.py

import re


# =====================================================
# Command Detection (Improved)
# =====================================================
def is_command(text: str) -> bool:
    """
    Detect if input is a system/terminal command
    """

    if not text:
        return False

    text = text.strip().lower()

    # =========================================
    # Exact commands
    # =========================================
    if text in {"exit", "quit", "clear"}:
        return True

    # =========================================
    # Command-like patterns (regex)
    # =========================================
    command_patterns = [
        r"^python\s+\S+\.py",          # python file.py
        r"^pip\s+install",            # pip install
        r"^npm\s+install",            # npm install
        r"^node\s+\S+",               # node file.js
        r"^cd\s+.+",                  # cd folder
        r"^ls\b",                     # ls
        r"^dir\b",                    # dir
        r"^git\s+.+",                 # git command
        r"^sudo\s+.+",                # sudo command
        r".+\.py\s*$",                # ends with file.py (ONLY if standalone)
    ]

    for pattern in command_patterns:
        if re.match(pattern, text):
            return True

    # =========================================
    # Inline execution attempts
    # =========================================
    if "run" in text and re.search(r"\.py\b", text):
        return True

    return False


# =====================================================
# Command Handling
# =====================================================
def handle_command(text: str) -> str:
    """
    Return safe response for command-like inputs
    """

    text = text.lower().strip()

    # exit commands
    if text in {"exit", "quit"}:
        return "👋 Session ended. Start a new chat anytime."

    # python execution
    if re.search(r"python\s+\S+\.py", text):
        return (
            "⚠️ This looks like a code execution command.\n"
            "I only help with graduation project ideas and development."
        )

    # package installation
    if "pip install" in text or "npm install" in text:
        return (
            "⚠️ Installation commands are outside my scope.\n"
            "I can help you design your graduation project instead."
        )

    # generic fallback
    return (
        "⚠️ This looks like a system command.\n"
        "Please ask about graduation projects (ideas, features, or system design)."
    )