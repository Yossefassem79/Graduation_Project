# api/schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field


# =====================================================
# Request Schema
# =====================================================
class AnalyzeRequest(BaseModel):
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=5)

    abstract: Optional[str] = ""
    features: Optional[List[str]] = []
    top_k: Optional[int] = 5


# =====================================================
# Result Item
# =====================================================
class SimilarProject(BaseModel):
    project_id: int
    project_title: str

    semantic_score: float
    feature_score: float
    hybrid_score: float

    originality_score: float
    duplicate_risk: str


# =====================================================
# Response Schema
# =====================================================
class AnalyzeResponse(BaseModel):
    extracted_features: List[str]
    results: List[SimilarProject]