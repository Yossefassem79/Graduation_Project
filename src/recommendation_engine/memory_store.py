# src/recommendation_engine/memory_store.py

import time
import uuid
from copy import deepcopy

MEMORY = {}

TTL = 86400  # 24 hours
MAX_HISTORY = 20
MAX_CHATS = 10


# =====================================================
# Default State
# =====================================================
def default_state():
    return {
        "project_title": "",
        "ideas": [],
        "features": [],
        "description": "",
        "abstract": "",
        "technologies": [],
        "originality_score": None,
        "context_strength": None,
        "problem_statement": "",
        "proposed_solution": "",
        "keywords": [],
        "ai_summary": "",
        "category": ""
    }


# =====================================================
# Create New Chat
# =====================================================
def create_chat(user_id: str):

    user = MEMORY.setdefault(user_id, {
        "chats": {},
        "active_chat_id": None,
        "timestamp": time.time()
    })

    if len(user["chats"]) >= MAX_CHATS:
        oldest = sorted(
            user["chats"].items(),
            key=lambda x: x[1]["last_updated"]
        )[0][0]
        del user["chats"][oldest]

    chat_id = str(uuid.uuid4())

    user["chats"][chat_id] = {
        "history": [],
        "state": default_state(),
        "last_updated": time.time()
    }

    user["active_chat_id"] = chat_id

    return chat_id


# =====================================================
# Get Active Chat
# =====================================================
def get_user_memory(user_id: str):

    user = MEMORY.get(user_id)

    if not user:
        create_chat(user_id)
        user = MEMORY[user_id]

    # TTL reset
    if time.time() - user["timestamp"] > TTL:
        MEMORY[user_id] = {}
        create_chat(user_id)
        user = MEMORY[user_id]

    chat_id = user.get("active_chat_id")

    if not chat_id or chat_id not in user["chats"]:
        chat_id = create_chat(user_id)

    return user["chats"][chat_id]


# =====================================================
# Merge State (FIXED 🔥)
# =====================================================
def merge_state(old: dict, new: dict):

    merged = deepcopy(old)

    for key, value in new.items():

        if value is None:
            continue

        # 🔥 overwrite critical fields
        if key in ["project_title", "description"]:
            merged[key] = value
            continue

        # 🔥 lists → replace if empty OR extend uniquely
        if isinstance(value, list):

            if not value:
                merged[key] = []
                continue

            existing = merged.get(key, [])

            combined = []
            seen = set()

            for item in existing + value:
                norm = str(item).lower().strip()
                if norm and norm not in seen:
                    seen.add(norm)
                    combined.append(item)

            # limit growth
            merged[key] = combined[:20]

        else:
            merged[key] = value

    return merged


# =====================================================
# Save Chat
# =====================================================
def save_user_memory(user_id: str, data: dict):

    user = MEMORY.get(user_id)

    if not user:
        create_chat(user_id)
        user = MEMORY[user_id]

    chat_id = user["active_chat_id"]
    chat = user["chats"][chat_id]

    history = data.get("history", [])
    new_state = data.get("state", {})

    history = history[-MAX_HISTORY:]

    chat["state"] = merge_state(chat["state"], new_state)
    chat["history"] = history
    chat["last_updated"] = time.time()

    user["timestamp"] = time.time()


# =====================================================
# Get All Chats
# =====================================================
def get_all_chats(user_id: str):

    user = MEMORY.get(user_id, {})
    chats = user.get("chats", {})

    return [
        {
            "chat_id": chat_id,
            "title": chat["state"].get("project_title") or "New Chat",
            "last_updated": chat["last_updated"]
        }
        for chat_id, chat in chats.items()
    ]


# =====================================================
# Switch Chat
# =====================================================
def switch_chat(user_id: str, chat_id: str):

    user = MEMORY.get(user_id)

    if not user:
        return

    if chat_id in user.get("chats", {}):
        user["active_chat_id"] = chat_id


# =====================================================
# Clear User
# =====================================================
def clear_user_memory(user_id: str):

    if user_id in MEMORY:
        del MEMORY[user_id]