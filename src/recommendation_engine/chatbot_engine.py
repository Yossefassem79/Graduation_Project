# src.recommendation_engine.chatbot_engine.py

from src.recommendation_engine.memory_store import (
    get_user_memory,
    save_user_memory,
    default_state
)

from src.recommendation_engine.llm_router import analyze_user_input

from src.recommendation_engine.command_handler import (
    is_command,
    handle_command
)

from src.recommendation_engine.idea_generator import generate_ideas
from src.recommendation_engine.feature_generator import generate_features

from src.recommendation_engine.llm_client import generate_text
from src.recommendation_engine.prompt_builder import build_chat_prompt
from src.recommendation_engine.response_formatter import format_response
from src.recommendation_engine.state_manager import update_state
from src.recommendation_engine.context_builder import extract_domain

from src.recommendation_engine.full_project_generator import (
    generate_full_project
)

import re


# =====================================================
# Helpers
# =====================================================
def extract_number(text: str, default=5):

    nums = re.findall(r"\d+", text)

    return min(int(nums[0]), 20) if nums else default


def is_weak_project_title(title: str) -> bool:

    if not title:
        return True

    title = title.strip()

    words = title.split()

    # too short
    if len(words) < 4:
        return True

    weak_words = {
        "system",
        "platform",
        "app",
        "website",
        "application",
        "project",
        "ai",
        "smart",
        "tool"
    }

    meaningful = [
        w.lower()
        for w in words
        if w.lower() not in weak_words
    ]

    return len(words) < 3


def is_generic_project_reference(text: str) -> bool:

    text = text.strip().lower()

    generic_titles = {
        "my project",
        "this project",
        "the project",
        "my system",
        "this system",
        "my app",
        "my application",
        "my idea",
        "project",
        "system",
        "app",
        "idea"
    }

    return text in generic_titles

def looks_like_real_project_title(title: str) -> bool:

    if not title:
        return False

    title = title.strip()

    words = title.split()

    # minimum words
    if len(words) < 2:
        return False

    # too repetitive
    unique_ratio = len(set(words)) / len(words)

    if unique_ratio < 0.5:
        return False

    # gibberish detection
    nonsense_patterns = [
        "asd",
        "qwe",
        "zxc",
        "testtest",
        "aaaa",
        "xxxxx"
    ]

    lowered = title.lower()


     # =========================
    # QUESTION / NORMAL CHAT FILTER
    # =========================
    question_starts = (
        "how ",
        "what ",
        "why ",
        "when ",
        "where ",
        "can ",
        "could ",
        "should ",
        "is ",
        "are ",
        "do ",
        "does "
    )

    

    for p in nonsense_patterns:
        if p in lowered:
            return False

    # at least one meaningful keyword
    keywords = {

    # AI / Systems
    "management",
    "analysis",
    "detection",
    "tracking",
    "recognition",
    "monitoring",
    "security",
    "attendance",
    "automation",
    "prediction",
    "dashboard",
    "diagnosis",
    "learning",
    "recommendation",
    "classification",
    "authentication",
    "optimization",

    # domains
    "healthcare",
    "fintech",
    "education",
    "library",
    "hospital",
    "school",
    "medical",
    "industrial",
    "agriculture",
    "transport",

    # technologies
    "ai",
    "iot",
    "blockchain",
    "cloud",
    "robotics",
    "vision",
    "embedded",

    # project words
    "system",
    "platform",
    "application",
    "app"
}

    if not any(
        k in lowered
        for k in keywords
    ):
        return False

    return True


FOLLOWUP_WORDS = [
    "another",
    "more",
    "again",
    "other ideas",
    "more ideas",
    "more features",
    "another features"
]

# =====================================================
# RESPONSE FINALIZER
# =====================================================
def finalize_response(
    user_input,
    response,
    history,
    state,
    user_id
):

    history.append({
        "role": "user",
        "content": user_input
    })

    history.append({
        "role": "assistant",
        "content": response
    })

    history = history[-20:]

    save_user_memory(user_id, {
        "history": history,
        "state": state
    })

    return response


def is_gibberish_text(text: str) -> bool:

    text = text.strip().lower()

    # allow menu choices
    if text in {"1", "2", "3"}:
        return False

    # very short non-numeric random text
    if len(text) < 3:

        # allow short meaningful commands
        allowed_short = {
            "hi",
            "hey",
            "hello",
            "ai",
            "ml",
            "ui",
            "ux",
            "vr",
            "ar",
            "iot"
        }

        if text in allowed_short:
            return False

        return True

    gibberish_patterns = [
        "asd",
        "qwe",
        "zxc",
        "aaa",
        "bbb",
        "ccc",
        "xxx",
        "testtest"
    ]

    for p in gibberish_patterns:
        if p in text:
            return True

    words = text.split()

    # repeated same words
    if len(words) >= 3:

        unique_ratio = len(set(words)) / len(words)

        if unique_ratio < 0.5:
            return True

    return False

def is_project_related(text: str) -> bool:

    text = text.lower().strip()

    keywords = [

        # project words
        "project",
        "system",
        "platform",
        "application",
        "app",
        "website",
        "dashboard",
        "management",

        # technical
        "ai",
        "ml",
        "machine learning",
        "deep learning",
        "computer vision",
        "blockchain",
        "iot",
        "web",
        "mobile",
        "cloud",
        "security",
        "database",
        "api",

        # actions
        "generate",
        "feature",
        "features",
        "idea",
        "ideas",
        "improve",
        "description",
        "technologies",
        "architecture",

        # domains
        "healthcare",
        "education",
        "fintech",
        "smart",
        "attendance",
        "monitoring",
        "tracking",
        "analysis",
        "recognition"
    ]

    return any(
        keyword in text
        for keyword in keywords
    )

# =====================================================
# MAIN CHATBOT
# =====================================================
def chatbot(user_id: str, user_input: str):

    text = user_input.lower().strip()


    # =========================================
    # EXPLICIT IDEA REQUESTS
    # =========================================
    explicit_idea_requests = [
        "idea",
        "new idea",
        "another idea",
        "project idea",
        "give me idea",
        "i want idea",
        "suggest idea",
        "generate idea",
        "generate ideas",
        "give me ideas",
        "i need idea",
        "ideas"
    ]

    if any(x in text for x in explicit_idea_requests):

        state = get_user_memory(user_id).get("state") or default_state()

        state["waiting_for_domain"] = True

        save_user_memory(user_id, {
            "history": get_user_memory(user_id).get("history", []),
            "state": state
        })

        return (
            "🎯 Specify a domain (AI, healthcare, fintech...)"
        )

    # =================================================
    # ALLOW MENU NUMBERS
    # =================================================
    if text in {"1", "2", "3"}:
        pass
    # =================================================
    # GIBBERISH DETECTION
    # =================================================
    if is_gibberish_text(text):

        return (
            "⚠️ I could not understand your request.\n\n"
            "Try something meaningful like:\n"
            "- generate AI project ideas\n"
            "- generate features for smart hospital system\n"
            "- improve my project\n"
            "- suggest technologies for fintech app"
        )
    # =================================================
    # COMMANDS
    # =================================================
    if is_command(user_input):
        return handle_command(user_input)

    # =================================================
    # MEMORY
    # =================================================
    memory = get_user_memory(user_id)

    history = memory.get("history", [])

    state = memory.get("state") or default_state()

    # ensure keys
    state.setdefault("menu_mode", False)
    state.setdefault("selected_option", None)
    state.setdefault("waiting_for_domain", False)
    state.setdefault("waiting_for_project_action", False)
    state.setdefault("project_chat_mode", False)
    state.setdefault("last_action", None)
    state.setdefault("domain", None)
    state.setdefault("weak_title_candidate", None)
    state.setdefault("waiting_for_feature_title", False)
    state.setdefault("waiting_for_full_project_domain", False)
    state.setdefault("waiting_for_full_project_selection", False)

    # =================================================
    # GREETING
    # =================================================
    if text in ["","hi", "hello", "hey"]:

        save_user_memory(user_id, {
            "history": history,
            "state": state
        })

        return (
            "👋 Welcome!\n\n"
            "I can help you with:\n"
            "• Graduation project ideas\n"
            "• Smart feature generation\n"
            "• Project improvement\n"
            "• Technology suggestions\n"
            "• Project descriptions\n\n"
            "Try saying things like:\n"
            "- give me AI project ideas\n"
            "- generate features for smart hospital system\n"
            "- improve my graduation project\n"
            "- suggest technologies for fintech app"
        )
    # =================================================
    # FULL PROJECT STARTER
    # =================================================
    if any(x in text for x in [

        "generate full project",
        "final project",
        "complete project",
        "full graduation project",
        "all fields",
        "fields"

    ]):

        # ============================================
        # IF PROJECT TITLE ALREADY EXISTS
        # ============================================
        if state.get("project_title"):

            # Generate features automatically if missing
            if not state.get("features"):

                feature_result = generate_features(

                    title=state.get("project_title"),

                    description=state.get("description", ""),

                    features=[],

                    previous_generated_features=[],

                    top_k=8
                )

                state["features"] = feature_result.get(
                    "recommended_features",
                    []
                )

            # Generate full project
            result = generate_full_project(

                title=state.get("project_title"),

                features=state.get("features", []),

                description=state.get("description", ""),

                abstract=state.get("abstract", "")
            )

            state = update_state(
                state,
                result,
                mode="merge"
            )

            response = f"""
📦 Full Project Generated

📌 Title:
{state.get("project_title")}

📄 Abstract:
{state.get("abstract")}

📄 Description:
{state.get("description")}

⚙️ Features:
{chr(10).join("- " + x for x in state.get("features", []))}

🛠 Technologies:
{chr(10).join("- " + x for x in state.get("technologies", []))}

🎯 Objectives:
{chr(10).join("- " + x for x in state.get("objectives", []))}

⚡ Methodology:
{state.get("methodology")}

🚀 Future Work:
{chr(10).join("- " + x for x in state.get("future_work", []))}

📂 Category:
{state.get("category")}

🏷 Keywords:
{", ".join(state.get("keywords", []))}

❗ Problem Statement:
{state.get("problem_statement")}

💡 Proposed Solution:
{state.get("proposed_solution")}

🤖 AI Summary:
{state.get("ai_summary")}
    """

            return finalize_response(
                user_input,
                response,
                history,
                state,
                user_id
            )

        # ============================================
        # NO PROJECT TITLE → ASK FOR DOMAIN
        # ============================================
        state["waiting_for_full_project_domain"] = True

        save_user_memory(user_id, {
            "history": history,
            "state": state
        })

        return (
            "🎯 What domain do you want?\n\n"
            "Examples:\n"
            "- AI\n"
            "- healthcare\n"
            "- education\n"
            "- fintech"
        )
    # =================================================
    # MENU HANDLER
    # =================================================
    if state.get("menu_mode"):

        if text in ["1", "2", "3"]:

            state["menu_mode"] = False
            state["selected_option"] = text

            if text == "1":

                state["waiting_for_domain"] = True

                save_user_memory(user_id, {
                    "history": history,
                    "state": state
                })

                return "🎯 Enter a domain (AI, healthcare, fintech...)"

            elif text == "2":

                if not state.get("project_title"):

                    state["waiting_for_feature_title"] = True

                    save_user_memory(user_id, {
                        "history": history,
                        "state": state
                    })

                    return (
                        "⚠️ Please provide the project title first.\n\n"
                        "Example:\n"
                        "- AI-powered smart library recommendation system\n\n"
                        "👉 You can also ask me to generate an idea first."
                    )

                save_user_memory(user_id, {
                    "history": history,
                    "state": state
                })

                return chatbot(user_id, "generate features")

            elif text == "3":

                if not state.get("project_title"):

                    return (
                        "⚠️ Please generate or provide "
                        "a project idea first."
                    )

                state["project_chat_mode"] = True

                save_user_memory(user_id, {
                    "history": history,
                    "state": state
                })

                return (
                    f"💬 Project discussion mode enabled for:\n"
                    f"{state.get('project_title')}"
                )

        state["menu_mode"] = False

    # =================================================
    # PROJECT ACTION HANDLER
    # =================================================
    if state.get("waiting_for_project_action"):

        feature_words = [
            "1",
            "feature",
            "features",
            "generate features",
            "add features",
            "create features"
        ]

        chat_words = [
            "2",
            "chat",
            "talk",
            "discussion",
            "discuss",
            "improve",
            "improve idea",
            "talk about project"
        ]

        full_project_words = [
    "3",
    "full project",
    "generate full project",
    "complete project",
    "final project"
]

        # =========================================
        # FEATURE GENERATION
        # =========================================
        if any(w in text for w in feature_words):

            state["waiting_for_project_action"] = False

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return chatbot(user_id, "generate features")

        # =========================================
        # PROJECT DISCUSSION
        # =========================================
        elif any(w in text for w in chat_words):

            state["waiting_for_project_action"] = False

            state["project_chat_mode"] = True

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return (
                f"💬 Chat mode enabled for:\n"
                f"{state.get('project_title')}\n\n"
                "You can now discuss or improve the project."
            )

        # =========================================
        # FEATURE GENERATION
        # =========================================
        if any(w in text for w in feature_words):

            state["waiting_for_project_action"] = False

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return chatbot(user_id, "generate features")


        # =========================================
        # PROJECT DISCUSSION
        # =========================================
        elif any(w in text for w in chat_words):

            state["waiting_for_project_action"] = False

            state["project_chat_mode"] = True

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return (
                f"💬 Chat mode enabled for:\n"
                f"{state.get('project_title')}\n\n"
                "You can now discuss or improve the project."
            )


        # =========================================
        # FULL PROJECT GENERATION
        # =========================================
        elif any(w in text for w in full_project_words):

            state["waiting_for_project_action"] = False

            # Generate features first if missing
            if not state.get("features"):

                feature_result = generate_features(

                    title=state.get("project_title"),

                    description=state.get("description", ""),

                    features=[],

                    previous_generated_features=[],

                    top_k=8
                )

                state["features"] = feature_result.get(
                    "recommended_features",
                    []
                )

            # Generate full project
            result = generate_full_project(

                title=state.get("project_title"),

                features=state.get("features", []),

                description=state.get("description", ""),

                abstract=state.get("abstract", "")
            )

            state = update_state(
                state,
                result,
                mode="merge"
            )

            response = f"""
📦 Full Project Generated

📌 Title:
{state.get("project_title")}

📄 Abstract:
{state.get("abstract")}

📄 Description:
{state.get("description")}

⚙️ Features:
{chr(10).join("- " + x for x in state.get("features", []))}

🛠 Technologies:
{chr(10).join("- " + x for x in state.get("technologies", []))}

🎯 Objectives:
{chr(10).join("- " + x for x in state.get("objectives", []))}

⚡ Methodology:
{state.get("methodology")}

🚀 Future Work:
{chr(10).join("- " + x for x in state.get("future_work", []))}

📂 Category:
{state.get("category")}

🏷 Keywords:
{", ".join(state.get("keywords", []))}

❗ Problem Statement:
{state.get("problem_statement")}

💡 Proposed Solution:
{state.get("proposed_solution")}

🤖 AI Summary:
{state.get("ai_summary")}
"""

            return finalize_response(
                user_input,
                response,
                history,
                state,
                user_id
            )
        
    # =================================================
    # WEAK TITLE IMPROVEMENT HANDLER
    # =================================================
    if state.get("weak_title_candidate"):

        improve_words = [
            "1",
            "make it descriptive",
            "improve title",
            "rewrite title",
            "make better"
        ]

        if any(w in text for w in improve_words):

            weak_title = state.get("weak_title_candidate")

            prompt = f"""
                    You are a senior software architect.

                    Convert this weak graduation project title
                    into a professional and descriptive title.

                    Weak title:
                    {weak_title}

                    Rules:
                    - Keep original meaning
                    - Make it specific
                    - Make it professional
                    - Keep it concise
                    - Return ONLY the improved title
                    """

            improved_title = generate_text(
                prompt,
                task="chat"
            ).strip()

            state["weak_title_candidate"] = None
            state["project_title"] = improved_title
            state["waiting_for_project_action"] = True

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return (
                f"📌 Improved Project Title:\n"
                f"{improved_title}\n\n"
                "Choose what you want next:\n"
                "1️⃣ Generate features\n"
                "2️⃣ Talk with chatbot about the idea\n\n"
                "👉 You can also say:\n"
                "- generate features\n"
                "- discuss project"
            )
        

    # =================================================
    # WAITING FOR FEATURE TITLE
    # =================================================
    if state.get("waiting_for_feature_title"):

        state["waiting_for_feature_title"] = False

        possible_title = user_input.strip()

        # weak title
        if is_weak_project_title(possible_title):

            state["weak_title_candidate"] = possible_title

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return (
                "⚠️ Your project title is too short or unclear.\n\n"
                "Examples:\n"
                "- AI-powered smart library recommendation system\n"
                "- Smart hospital emergency response platform\n\n"
                "Choose:\n"
                "1️⃣ Make the title more descriptive\n"
                "2️⃣ Enter another title"
            )

        # valid title
        state["project_title"] = possible_title

        save_user_memory(user_id, {
            "history": history,
            "state": state
        })

        return chatbot(user_id, "generate features")
    
    # =================================================
    # DIRECT FEATURE TITLE EXTRACTION
    # =================================================
    feature_patterns = [
        # standard
        "generate features for",
        "generate feature for",
        "features for",
        "feature for",
        "create features for",
        "suggest features for",

        # casual
        "give me features for",
        "show features for",
        "i need features for",
        "can you generate features for",
        "can you suggest features for",

        # project related
        "add features to",
        "improve features for",
        "feature ideas for",
        "smart features for",

        # short forms
        "features:",
        "project:",
        "idea:",

        # typo tolerance
        "gen features for",
        "generate faetures for",
        "fetures for",

        # direct requests
        "make features for",
        "build features for",
        "what features for",
        "best features for",

        # ai chatbot style
        "i have project",
        "my project is",
        "project title is",
    ]

    analysis = None

    for pattern in feature_patterns:

        if text.startswith(pattern + " "):

            extracted_title = user_input.split(
                pattern, 1
            )[1].strip()

            if (
                extracted_title
                and not is_generic_project_reference(extracted_title)
            ):

                analysis = {
                    "intent": "feature",
                    "project_title": extracted_title
                }

                break

    # =================================================
    # SIMPLE PROJECT TITLE DETECTION (BEFORE LLM)
    # =================================================
    if analysis is None:

        blocked_starts = (
            "how ",
            "what ",
            "why ",
            "when ",
            "where ",
            "can ",
            "could ",
            "should ",
            "is ",
            "are ",
            "do ",
            "does ",
            "help ",
        )
        action_words = {
    "generate",
    "create",
    "suggest",
    "give",
    "show",
    "make",
    "build",
    "improve",
    "discuss",
    "help",
    "recommend",
    "need"
}

        command_phrases = {

    # idea commands
    "new idea",
    "another idea",
    "generate idea",
    "generate ideas",
    "project ideas",
    "more ideas",
    "idea",

    # feature commands
    "generate features",
    "generate feature",
    "more features",
    "features",

    # full project
    "generate full project",
    "full project",
    "complete project",

    # generic commands
    "another",
    "again",
    "more",
    "chat",
    "help",
    "menu",
    "discuss",
    "discuss project"
}

        clean_text = text.strip()

        # detect short titles directly
        words = clean_text.lower().split()

        if (
            2 <= len(words) <= 8
            and not clean_text.endswith("?")
            and not clean_text.startswith(blocked_starts)
            and clean_text.lower() not in command_phrases
            and not any(
                w in action_words
                for w in words
            )
        ):

            analysis = {
                "intent": "project_title",
                "project_title": user_input.strip()
            }

        else:
            analysis = analyze_user_input(user_input, state)

    # =================================================
    # IDEA SELECTION DETECTION
    # =================================================
    existing_ideas = state.get("ideas", [])

    normalized_input = text.strip().lower()

    for idea in existing_ideas:

        idea_norm = idea.strip().lower()

        input_words = set(normalized_input.split())

        # ignore extremely short inputs
        if len(input_words) < 2:
            continue

        idea_words = set(idea_norm.split())

        overlap = input_words & idea_words

        overlap_ratio = (
            len(overlap) / max(1, len(input_words))
        )

        if (
            normalized_input == idea_norm
            or normalized_input in idea_norm
            or overlap_ratio >= 0.6
        ):

            state["project_title"] = idea

            state["waiting_for_project_action"] = True

            state["project_chat_mode"] = False

            state["waiting_for_full_project_selection"] = False

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return finalize_response(

                user_input,

                f"""📌 Project Selected:

{idea}

Choose what you want:

1️⃣ Generate Features
2️⃣ Discuss Project
3️⃣ Generate Full Project

👉 You can also type:
- features
- discuss
- full project
""",

                history,
                state,
                user_id
            )


            # =====================================
            # NORMAL IDEA SELECTION
            # =====================================
            state["project_title"] = idea
            state["waiting_for_project_action"] = True
            state["project_chat_mode"] = False

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })
            return (
                f"📌 Project selected:\n"
                f"{idea}\n\n"
                "Choose what you want next:\n"
                "1️⃣ Generate features\n"
                "2️⃣ Talk with chatbot about the idea\n\n"
                "👉 You can also say:\n"
                "- generate features\n"
                "- discuss project"
            )

    # =================================================
    # FOLLOW-UP CONTINUATION
    # =================================================
    if any(w in text for w in FOLLOWUP_WORDS):

        # continue ideas
        if state.get("last_action") == "idea":

            analysis["intent"] = "idea"
            analysis["domain"] = state.get("domain")

        # continue features
        elif state.get("last_action") == "feature":

            analysis["intent"] = "feature"

    # =================================================
    # DOMAIN FOLLOW-UP
    # =================================================
    if state.get("waiting_for_domain"):

        detected = extract_domain(user_input)

        if detected:

            state["waiting_for_domain"] = False

            analysis["domain"] = detected
            analysis["intent"] = "idea"

        else:

            return (
                "⚠️ Please enter a valid domain "
                "(AI, healthcare, fintech...)"
            )

    intent = analysis.get("intent", "chat")

    domain = analysis.get("domain")

    user_project = analysis.get("project_title")

    # =================================================
    # FULL PROJECT DOMAIN HANDLER
    # =================================================
    if state.get("waiting_for_full_project_domain"):

        domain = extract_domain(user_input)

        if not domain:

            return (
                "⚠️ Please enter valid domain.\n\n"
                "Example:\n"
                "- AI\n"
                "- healthcare\n"
                "- education"
            )

        state["waiting_for_full_project_domain"] = False

        result = generate_ideas(
            domain=domain,
            top_k=5
        )

        ideas = result.get("final_ideas", [])

        state["ideas"] = ideas

        state["waiting_for_full_project_selection"] = True

        save_user_memory(user_id, {
            "history": history,
            "state": state
        })

        return (
            "💡 Choose one project title:\n\n"
            + "\n".join(
                f"- {idea}"
                for idea in ideas
            )
        )
    # =================================================
    # FALLBACK DOMAIN
    # =================================================
    if not domain:
        domain = extract_domain(text)

    # =================================================
    # DOMAIN-ONLY INPUT DETECTION
    # =================================================
    full_project_commands = [
    "fill all",
    "fill all columns",
    "fill all fields",
    "full project",
    "final project",
    "complete project",
    "generate full project",
    "generate final project",
    "generate all details",
    "project specification",
    "complete details"
    ]

    if (
        domain
        and intent == "chat"
        and len(text.split()) <= 3
        and not any(cmd in text for cmd in full_project_commands)
    ):
        intent = "idea"
    # =================================================
    # FALLBACK PROJECT TITLE DETECTION
    # =================================================
    if (
        not user_project
        and intent == "chat"
        and 3 <= len(text.split()) <= 12
    ):
        idea_request_patterns = [

    "new idea",
    "generate idea",
    "generate ideas",
    "project ideas",
    "idea for",
    "ideas for",
    "i need idea",
    "i need ideas",
    "need idea",
    "need ideas",
    "another idea",
    "more ideas",

]
        # reject obvious commands/questions
        blocked_starts = (
            "how ",
            "what ",
            "why ",
            "when ",
            "where ",
            "can ",
            "could ",
            "should ",
            "is ",
            "are ",
            "do ",
            "does ",
            "help ",
        )

        if text.endswith("?"):
            pass

        elif text.startswith(blocked_starts):
            pass

        else:

            command_phrases = {
                "generate features",
                "generate feature",
                "generate ideas",
                "generate idea",
                "more",
                "another",
                "again",
                "discuss project",
                "chat",
                "help",
                "menu"
            }

            if (
                text not in command_phrases
                and not any(
                    p in text
                    for p in idea_request_patterns
                )
            ):

                # treat short non-question text as project title
                user_project = user_input.strip()
    # =================================================
    # USER PROVIDED PROJECT
    # =================================================
    if user_project:

        # =============================================
        # WEAK TITLE VALIDATION
        # =============================================
        if is_weak_project_title(user_project):

            state["weak_title_candidate"] = user_project

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return (
                "⚠️ Your project title is too short or unclear.\n\n"
                "Examples:\n"
                "- AI-powered smart library recommendation system\n"
                "- Smart hospital emergency response platform\n"
                "- Blockchain-secured academic certificate verification system\n\n"
                "Choose:\n"
                "1️⃣ Make the title more descriptive\n"
                "2️⃣ Enter another title\n\n"
                "👉 You can also say:\n"
                "- make it descriptive"
            )

        state = default_state()

        state["project_title"] = user_project
        state["waiting_for_project_action"] = True

        save_user_memory(user_id, {
            "history": history,
            "state": state
        })

        return (
    f"📌 Project detected:\n"
    f"{user_project}\n\n"
    "Choose what you want next:\n"
    "1️⃣ Generate features\n"
    "2️⃣ Talk with chatbot about the idea\n"
    "3️⃣ Generate full project\n\n"
    "👉 You can also say:\n"
    "- generate features\n"
    "- discuss project\n"
    "- generate full project"
)
    # =================================================
    # FEATURE REQUEST WITHOUT TITLE
    # =================================================
    feature_request_phrases = [
        "generate features",
        "generate feature",
        "i need features",
        "feature ideas",
        "add features",
        "create features",
        "suggest features",
        "improve my project",
        "improve project",
        "improve my idea",
        "enhance my project",
        "discuss my project",
        "talk about my project",
        "help my project",
        "improve"
    ]

    if (
        any(p in text for p in feature_request_phrases)
        and not state.get("project_title")
        and not analysis.get("project_title")
    ):

        return (
            "Please give me the project title first.\n\n"
            "Example:\n"
            "- AI-powered smart library recommendation system\n"
            "- Smart hospital emergency response platform"
        )
    # =================================================
    # IDEA GENERATION
    # =================================================
    if intent == "idea":

        if not domain:

            state["waiting_for_domain"] = True

            save_user_memory(user_id, {
                "history": history,
                "state": state
            })

            return "🎯 Specify a domain (AI, healthcare, fintech...)"

        top_k = analysis.get("number") or extract_number(
            user_input,
            5
        )

        result = generate_ideas(
            domain=domain,
            top_k=top_k,
            previous_generated_ideas=state.get("ideas", [])
        )

        ideas = result.get("final_ideas", [])

        state["ideas"] = ideas
        state["domain"] = domain
        state["last_action"] = "idea"

        response = format_response("idea", "", state)
        
        return finalize_response(
            user_input,
            response,
            history,
            state,
            user_id
        )

    # =================================================
    # FEATURE GENERATION
    # =================================================
    elif intent == "feature":

        # =========================================
        # DIRECT PROJECT TITLE FROM USER INPUT
        # =========================================
        if analysis.get("project_title"):

            state["project_title"] = (
                analysis["project_title"]
            )

        if not state.get("project_title"):

            return (
                "Please give me the project title first."
            )

        top_k = analysis.get("number") or extract_number(
            user_input,
            5
        )

        result = generate_features(
            title=state.get("project_title"),
            description=state.get("description", ""),
            features=[],
            previous_generated_features=[],
            top_k=top_k
        )

        features = result.get(
            "recommended_features",
            []
        )

        # IMPORTANT:
        # Replace old features completely
        state["features"] = features

        state["last_action"] = "feature"

        response = format_response("feature", "", state)
        return finalize_response(
            user_input,
            response,
            history,
            state,
            user_id
        )


        
    # =================================================
    # FULL PROJECT GENERATION
    # =================================================
    elif any(x in text for x in [

    "fill all",
    "fill all columns",
    "fill all fields",
    "full project",
    "final project",
    "complete project",
    "generate full project",
    "generate final project",
    "generate all details",
    "project specification",
    "complete details"

]):

        if not state.get("project_title"):

            return (
                "Please generate project idea first."
            )

        if not state.get("features"):

            return (
                "Please generate features first."
            )

        result = generate_full_project(

            title=state.get("project_title"),

            features=state.get("features", []),

            description=state.get("description", ""),

            abstract=state.get("abstract", "")
        )

        state = update_state(
            state,
            result,
            mode="merge"
        )

        response = f"""
📦 Full Project Generated

📌 Title:
{state.get("project_title")}

📄 Abstract:
{state.get("abstract")}

📄 Description:
{state.get("description")}

🛠 Technologies:
{chr(10).join("- " + x for x in state.get("technologies", []))}

🎯 Objectives:
{chr(10).join("- " + x for x in state.get("objectives", []))}

⚡ Methodology:
{state.get("methodology")}

🚀 Future Work:
{chr(10).join("- " + x for x in state.get("future_work", []))}

📂 Category:
{state.get("category")}

🏷 Keywords:
{", ".join(state.get("keywords", []))}

❗ Problem Statement:
{state.get("problem_statement")}

💡 Proposed Solution:
{state.get("proposed_solution")}

🤖 AI Summary:
{state.get("ai_summary")}
"""

        return finalize_response(
            user_input,
            response,
            history,
            state,
            user_id
        )

    # =================================================
    # POSSIBLE NEW PROJECT DETECTION
    # =================================================
    elif (
        state.get("project_title")
        and len(text.split()) >= 3
        and not any(w in text for w in FOLLOWUP_WORDS)
    ):

        current_title = (
            state.get("project_title", "")
            .lower()
            .split()
        )

        new_input_words = text.split()

        overlap = set(current_title) & set(new_input_words)

        overlap_ratio = (
            len(overlap)
            / max(1, len(new_input_words))
        )

        # low overlap may indicate new project/topic
        if (
            overlap_ratio < 0.2
            and analysis.get("intent") == "chat"
        ):

            return (
                "⚠️ This seems unrelated to the current project.\n\n"
                f"Current project:\n"
                f"{state.get('project_title')}\n\n"
                "Do you want to:\n"
                "1️⃣ Start a new project\n"
                "2️⃣ Continue discussing the current project"
            )

        system_prompt = build_chat_prompt(state)

        formatted_history = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in history[-6:]
        ])

        full_prompt = f"""
{system_prompt}

Conversation:
{formatted_history}

User:
{user_input}
"""

        raw = generate_text(
            full_prompt,
            task="chat"
        )

        response = format_response(
            "chat",
            raw,
            state
        )
        return finalize_response(
            user_input,
            response,
            history,
            state,
            user_id
        )
    # =================================================
    # NON PROJECT FILTER
    # =================================================
    if not is_project_related(text):

        return (
            "⚠️ This chatbot is specialized for graduation projects only.\n\n"
            "Try things like:\n"
            "- generate AI project ideas\n"
            "- generate features for smart hospital system\n"
            "- suggest technologies for fintech app\n"
            "- improve my graduation project"
        )
    # =================================================
    # GENERAL CHAT
    # =================================================
    else:

        system_prompt = build_chat_prompt(state)

        formatted_history = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in history[-6:]
        ])

        full_prompt = f"""
{system_prompt}

Conversation:
{formatted_history}

User:
{user_input}
"""

        raw = generate_text(
            full_prompt,
            task="chat"
        )

        response = format_response(
            "chat",
            raw,
            state
        )
        return finalize_response(
            user_input,
            response,
            history,
            state,
            user_id
        )

   