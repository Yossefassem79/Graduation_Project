from .preprocessing import (
    normalize_text,
    extract_features,
    preprocess_dataset
)

from .semantic_search import (
    load_model,
    load_faiss_index,
    load_metadata,
    search_by_text,
    search_by_project_id,
    compare_two_ideas
)

from .feature_similarity import (
    load_feature_model,
    safe_feature_list,
    compute_feature_similarity,
    compare_projects,
    compare_project_against_many
)

from .hybrid_ranker import (
    compute_hybrid_score,
    compute_originality,
    compute_confidence,
    risk_label,
    rank_candidates
)

from .embedding_engine import (
    train_embedding_engine,
    ProjectEmbedder
)

from .similarity_engine import (
    find_similar_projects
)