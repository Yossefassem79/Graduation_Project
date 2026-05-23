# # api/services.py

# import pandas as pd

# from src.similarity_engine import find_similar_projects
# from src.preprocessing import extract_features


# # =====================================================
# # Main Analyze Service
# # =====================================================
# def analyze_project(
#     title: str,
#     description: str,
#     abstract: str = "",
#     features=None,
#     top_k: int = 5
# ):
#     """
#     Full project analysis service
#     """

#     if features is None:
#         features = []

#     # ---------------------------------------------
#     # Build text for auto feature extraction
#     # ---------------------------------------------
#     full_text = f"{title}. {abstract}. {description}"

#     auto_features = extract_features(full_text)

#     # merge manual + auto
#     merged = []
#     seen = set()

#     for item in features + auto_features:
#         val = str(item).strip().lower()

#         if val and val not in seen:
#             seen.add(val)
#             merged.append(val)

#     # ---------------------------------------------
#     # Run similarity engine
#     # ---------------------------------------------
#     results = find_similar_projects(
#         title=title,
#         description=f"{abstract} {description}",
#         features=merged,
#         top_k=top_k
#     )

#     # ---------------------------------------------
#     # Convert dataframe to json records
#     # ---------------------------------------------
#     if isinstance(results, pd.DataFrame):
#         rows = results.to_dict(orient="records")
#     else:
#         rows = []

#     return {
#         "extracted_features": merged,
#         "results": rows
#     }


# api/services.py

# import pandas as pd

# from src.similarity_model import find_similar_projects
# from src.similarity_model import extract_features


# # =====================================================
# # Main Analyze Service
# # =====================================================
# def analyze_project(
#     title: str,
#     description: str,
#     abstract: str = "",
#     features=None,
#     top_k: int = 5
# ):
#     """
#     Final clean API response
#     Returns only user-needed metrics
#     """

#     if features is None:
#         features = []

#     # -------------------------------------------------
#     # Build full text for automatic feature extraction
#     # -------------------------------------------------
#     full_text = f"{title}. {abstract}. {description}"

#     auto_features = extract_features(full_text)

#     # -------------------------------------------------
#     # Merge manual + extracted features
#     # -------------------------------------------------
#     merged = []
#     seen = set()

#     for item in features + auto_features:
#         val = str(item).strip().lower()

#         if val and val not in seen:
#             seen.add(val)
#             merged.append(val)

#     # -------------------------------------------------
#     # Run similarity model
#     # -------------------------------------------------
#     results = find_similar_projects(
#         title=title,
#         description=f"{abstract} {description}",
#         features=merged,
#         top_k=top_k
#     )

#     # -------------------------------------------------
#     # No results found
#     # -------------------------------------------------
#     if not isinstance(results, pd.DataFrame) or len(results) == 0:
#         return {
#             "message": "No similar projects found",
#             "extracted_features": merged
#         }

#     # -------------------------------------------------
#     # Take top matched project only
#     # -------------------------------------------------
#     top = results.iloc[0]

#     return {
#         "extracted_features": merged,

#         "matched_features": top.get(
#             "matched_features", []
#         ),

#         "unique_features": top.get(
#             "unique_query_features", []
#         ),

#         "hybrid_similarity": round(
#             float(top.get("hybrid_score", 0)),
#             4
#         ),

#         "final_originality_score": round(
#             float(top.get("originality_score", 0)),
#             4
#         )
#     }

import pandas as pd

from src.similarity_model import find_similar_projects
from src.similarity_model import extract_features


def analyze_project(
    title: str,
    description: str,
    abstract: str = "",
    features=None,
    top_k: int = 5
):

    if features is None:
        features = []

    full_text = f"{title}. {abstract}. {description}"

    auto_features = extract_features(full_text)

    merged = []
    seen = set()

    for item in features + auto_features:
        val = str(item).strip().lower()

        if val and val not in seen:
            seen.add(val)
            merged.append(val)

    results = find_similar_projects(
        title=title,
        description=f"{abstract} {description}",
        features=merged,
        top_k=top_k
    )

    if not isinstance(results, pd.DataFrame) or len(results) == 0:
        return {
            "message": "No similar projects found",
            "extracted_features": merged
        }

    # -----------------------------------
    # رجع Top K كله
    # -----------------------------------
    top_projects = []

    for _, row in results.iterrows():
        top_projects.append({
            "project_title": row.get("project_title", ""),
            "matched_features": row.get("matched_features", []),
            "unique_features": row.get("unique_query_features", []),
            "similarity_score": round(float(row.get("hybrid_score", 0)), 4),
            "final_originality_score": round(float(row.get("originality_score", 0)), 4)
        })

    return {
        "extracted_features": merged,
        "top_similar_projects": top_projects
    }