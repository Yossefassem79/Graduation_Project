# src/evaluation.py
# FINAL PROFESSIONAL EVALUATION VERSION
# Self Test + Real Query Test + Overall Judgment

import logging
import pandas as pd

from src.similarity_model import find_similar_projects
from src.similarity_model import load_metadata

# =====================================================
# Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

TOP_K = 5
SELF_TEST_SAMPLES = 20

# =====================================================
# SELF RETRIEVAL TEST
# =====================================================
def run_self_test():

    df = load_metadata()

    total = min(len(df), SELF_TEST_SAMPLES)

    success = 0

    for i in range(total):

        row = df.loc[i]

        results = find_similar_projects(
            title=row.get("project_title", ""),
            abstract=row.get("abstract", ""),
            description=row.get("description", ""),
            features=row.get("features", []),
            top_k=1
        )

        if "project_id" in results.columns:

            pred = int(results.iloc[0]["project_id"])

            if pred == i:
                success += 1

    score = success / total

    print("\n==============================")
    print("SELF RETRIEVAL TEST")
    print("==============================")
    print(f"Projects Tested : {total}")
    print(f"Top1 Accuracy   : {score:.2%}")
    print("==============================")

    return score


# =====================================================
# REAL QUERY TEST
# =====================================================
def run_real_queries():

    queries = [

        {
            "title":
            "AI Clinic Management System",

            "description":
            """
            Smart clinic with booking,
            chatbot, patient records,
            doctor dashboard.
            """
        },

        {
            "title":
            "Smart Library Assistant",

            "description":
            """
            Library app with chatbot,
            recommendation system,
            qr code borrowing.
            """
        },

        {
            "title":
            "Attendance Face Recognition",

            "description":
            """
            Attendance system using
            face recognition and reports.
            """
        },

        {
            "title":
            "E-commerce Recommendation Platform",

            "description":
            """
            Online shopping website with
            recommendation engine,
            payments and dashboard.
            """
        }

    ]

    print("\n==============================")
    print("REAL QUERY TEST")
    print("==============================")

    total_score = 0
    count = 0

    for q in queries:

        results = find_similar_projects(
            title=q["title"],
            description=q["description"],
            top_k=1
        )

        if "hybrid_score" in results.columns:

            score = float(
                results.iloc[0]["hybrid_score"]
            )

            risk = str(
                results.iloc[0]["duplicate_risk"]
            )

            top_title = str(
                results.iloc[0]["project_title"]
            )

            total_score += score
            count += 1

            print()
            print("Query:", q["title"])
            print("Top Match:", top_title)
            print("Score:", round(score, 4))
            print("Risk:", risk)

    avg = total_score / count if count else 0

    print("\n==============================")
    print(f"Average Query Score: {avg:.4f}")
    print("==============================")

    return avg


# =====================================================
# FINAL JUDGMENT
# =====================================================
def final_status(
    self_score,
    query_score
):

    print("\n==============================")
    print("FINAL MODEL STATUS")
    print("==============================")

    final_score = (
        0.60 * self_score +
        0.40 * query_score
    )

    if final_score >= 0.90:
        print("EXCELLENT ✅")

    elif final_score >= 0.75:
        print("VERY GOOD ✅")

    elif final_score >= 0.60:
        print("GOOD ⚠️")

    else:
        print("NEEDS IMPROVEMENT ❌")

    print("Overall Score:", round(final_score, 4))
    print("==============================\n")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":

    self_score = run_self_test()

    query_score = run_real_queries()

    final_status(
        self_score,
        query_score
    )