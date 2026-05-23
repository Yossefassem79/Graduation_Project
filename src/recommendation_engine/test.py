
### idea generator test

# from src.recommendation_engine.idea_generator import generate_ideas

# result = generate_ideas(
#     domain="healthcare",
#     top_k=5
# )

# print("\nFINAL IDEAS:")
# for i, idea in enumerate(result["final_ideas"], 1):
#     print(f"{i}. {idea}")

###featuregeneratoretest

# from src.recommendation_engine.feature_generator import generate_features

# # =========================================
# # TEST 1: First generation
# # =========================================
# print("\n========== FIRST RUN ==========\n")

# first = generate_features(
#     title="Smart Hospital System",
#     description="""
#     AI-based hospital system with chatbot,
#     patient management, QR booking,
#     and smart recommendations.
#     """,
#     features=["chatbot", "search", "qr booking"],
#     top_k=2
# )

# for i, f in enumerate(first["recommended_features"], 1):
#     print(f"{i}. {f}")


# # =========================================
# # TEST 2: Second generation (NO duplicates)
# # =========================================
# print("\n========== SECOND RUN (NO DUPLICATES) ==========\n")

# second = generate_features(
#     title="Smart Hospital System",
#     description="""
#     AI-based hospital system with chatbot,
#     patient management, QR booking,
#     and smart recommendations.
#     """,
#     features=["chatbot", "search", "qr booking"],
#     previous_generated_features=first["recommended_features"],
#     top_k=2
# )

# for i, f in enumerate(second["recommended_features"], 1):
#     print(f"{i}. {f}")


# # =========================================
# # TEST 3: Check overlap manually
# # =========================================
# print("\n========== DUPLICATE CHECK ==========\n")

# duplicates = set(first["recommended_features"]) & set(second["recommended_features"])

# if duplicates:
#     print(" DUPLICATES FOUND:")
#     for d in duplicates:
#         print("-", d)
# else:
#     print(" NO DUPLICATES — SYSTEM WORKING PERFECT")


## chatbot generator test



# from src.recommendation_engine.chatbot_engine import chatbot

# user_id = "user_1"

# print("\n=== IDEA ===")
# print(chatbot(user_id, "I want healthcare project idea"))

# print("\n=== FEATURES ===")
# print(chatbot(user_id, "add smart features"))

# print("\n=== DESCRIPTION ===")
# print(chatbot(user_id, "describe it"))

# print("\n=== MEMORY TEST ===")
# print(chatbot(user_id, "improve it more"))




# run_test.py

from src.recommendation_engine.chatbot_engine import chatbot

USER_ID = "test_user"



def interactive():
    print("\n===== INTERACTIVE MODE =====\n")

    while True:
        user_input = input("YOU: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        response = chatbot(USER_ID, user_input)

        print("\nBOT:")
        print(response)
        print("\n----------------------\n")



interactive()