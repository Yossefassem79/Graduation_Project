# src/recommendation_engine/prompt_builder.py

from typing import Dict, Any, List


# =====================================================
# Helpers
# =====================================================
def list_to_text(items: List[str], max_items: int = 10) -> str:

    if not items:
        return "None"

    seen = set()
    cleaned = []

    for item in items[:max_items]:

        val = str(item).strip()

        if not val:
            continue

        key = val.lower()

        if key not in seen:
            seen.add(key)
            cleaned.append(val)

    return ", ".join(cleaned)


# =====================================================
# Feature Prompt
# =====================================================
def build_feature_prompt(
    context: Dict[str, Any],
    count: int = 10,
    previous_features: List[str] = None
) -> str:

    previous_features = previous_features or []

    return f"""
You are a senior software architect and AI systems designer.

TASK:
Generate {count} intelligent and realistic system features
for the following graduation project.

PROJECT TITLE:
{context.get("project_title")}

CURRENT FEATURES:
{list_to_text(context.get("features", []), 15)}

COMMON OVERUSED FEATURES TO AVOID:
{list_to_text(context.get("common_features", []), 15)}

PREVIOUSLY GENERATED FEATURES:
{list_to_text(previous_features, 20)}

PROJECT CONTEXT:
- Originality Score: {context.get("originality_score", 1.0)}
- Context Strength: {context.get("context_strength", 0.0)}

IMPORTANT REQUIREMENTS:
- Features must belong to ONE coherent system
- Features must solve REAL problems
- Prefer intelligent automation and AI logic
- Focus on implementable engineering features
- Avoid vague or generic functionality
- Avoid repeating concepts with different wording
- Avoid simple CRUD/dashboard/login features
- Generate diverse feature types

FEATURE TYPES TO MIX:
- Core system features
- AI/smart features
- Automation features
- Analytics/monitoring features
- Reliability/safety features
- User experience improvements

STRICT RULES:
- One feature per line
- No numbering
- No explanations
- No repeated concepts
- Each feature should be concise
- Each feature should sound like a real product capability
- Prefer 3–10 words per feature

GOOD FEATURE EXAMPLES:
- Real-time gesture recognition
- Predictive patient risk analysis
- AI-assisted diagnosis support
- Emergency response prioritization
- Adaptive learning recommendation engine

BAD FEATURE EXAMPLES:
- Smart dashboard
- AI module
- Login system
- Reports page
- User management

OUTPUT:
""".strip()


# =====================================================
# Idea Prompt
# =====================================================
def build_idea_prompt(
    context: Dict[str, Any],
    count: int = 10,
    previous_ideas: List[str] = None
) -> str:

    previous_ideas = previous_ideas or []

    domain = context.get("domain", "general")

    return f"""
You are a senior AI innovation consultant.

TASK:
Generate {count} unique and high-quality graduation project ideas.

DOMAIN:
{domain}

PREVIOUS IDEAS:
{list_to_text(previous_ideas, 20)}

IMPORTANT REQUIREMENTS:
- Ideas must solve REAL problems
- Ideas must be practical and implementable
- Prefer AI, automation, or intelligent systems
- Avoid generic software projects
- Avoid repeated themes or workflows
- Each idea must represent a DIFFERENT concept
- Encourage domain diversity and creativity
- Prefer modern technologies and impactful use-cases

STRICT RULES:
- No repeated concepts
- No semantic duplicates
- No slight rewording
- Avoid overused ideas like:
  prediction systems,
  recommendation systems,
  management systems,
  analytics dashboards

FORMAT RULES:
- One idea per line
- No numbering
- No explanations
- Keep each idea concise
- Prefer 4–12 words

GOOD IDEA EXAMPLES:
- Smart traffic congestion prediction using drones
- AI-powered emergency sign language translator
- Blockchain-secured medical image sharing
- Computer vision livestock health monitoring
- Adaptive learning assistant for dyslexic students

BAD IDEA EXAMPLES:
- AI management system
- Smart dashboard platform
- Recommendation application
- Analytics website

OUTPUT:
""".strip()


# =====================================================
# Description Prompt
# =====================================================
def build_description_prompt(context: Dict[str, Any]) -> str:

    return f"""
You are a senior technical writer and software architect.

TASK:
Write a professional graduation project description.

PROJECT TITLE:
{context.get("project_title")}

FEATURES:
{list_to_text(context.get("features", []), 20)}

REQUIREMENTS:
- Explain the real-world problem
- Explain the proposed solution
- Connect all features logically
- Explain system intelligence and AI usage
- Keep the description realistic and implementable
- Write clearly and professionally
- Avoid unnecessary marketing language

STRUCTURE:
1. Problem
2. Solution
3. System capabilities
4. Expected impact

OUTPUT:
Professional structured paragraph
""".strip()


# =====================================================
# Chat Prompt
# =====================================================
def build_chat_prompt(context: Dict[str, Any]) -> str:

    return f"""
You are a senior software architect and graduation project consultant.

PROJECT TITLE:
{context.get("project_title", "None")}

PROJECT DESCRIPTION:
{context.get("description", "None")}

PROJECT FEATURES:
{list_to_text(context.get("features", []), 20)}

TECHNOLOGIES:
{list_to_text(context.get("technologies", []), 20)}

YOUR ROLE:
- Help improve the graduation project
- Suggest technical improvements
- Answer project-related questions
- Keep responses professional and practical
- Focus on software engineering and AI systems
- Avoid unrelated discussions

RULES:
- Be concise
- Be realistic
- Be technically accurate
- Keep all answers related to the current project
""".strip()


# =====================================================
# Full Project Prompt
# =====================================================
def build_full_project_prompt(context):

    title = context.get("project_title", "")

    features = context.get("features", [])

    description = context.get("description", "")

    abstract = context.get("abstract", "")

    features_text = "\n".join(
        f"- {f}"
        for f in features
    )

    return f"""
You are a senior software architect and academic researcher.

Generate a COMPLETE graduation project specification.

====================================================

PROJECT TITLE:
{title}

CURRENT ABSTRACT:
{abstract}

CURRENT DESCRIPTION:
{description}

FEATURES:
{features_text}

====================================================

STRICT RULES:

1. Return ALL sections
2. NEVER skip any section
3. NEVER leave any field empty
4. If information is missing, intelligently generate it
5. Use professional academic style
6. Technologies and lists MUST use bullet points
7. Use EXACT section names
8. No explanations outside sections
9. Every section must contain meaningful content
10. OBJECTIVES must contain at least 5 objectives
11. FUTURE_WORK must contain at least 4 items
12. KEYWORDS must contain at least 5 keywords
13. TECHNOLOGIES must contain at least 5 technologies
14. METHODOLOGY must be detailed and multi-step
15. AI_SUMMARY must never be empty

====================================================

CATEGORY:
(short category)

ABSTRACT:
(full academic abstract)

DESCRIPTION:
(full detailed description)

TECHNOLOGIES:
- item
- item
- item

KEYWORDS:
- keyword
- keyword
- keyword

PROBLEM_STATEMENT:
(real-world problem explanation)

PROPOSED_SOLUTION:
(system solution explanation)

OBJECTIVES:
- objective
- objective
- objective

AI_SUMMARY:
(short AI-generated summary)

FUTURE_WORK:
- future enhancement
- future enhancement

METHODOLOGY:
(step-by-step implementation process)
""".strip()