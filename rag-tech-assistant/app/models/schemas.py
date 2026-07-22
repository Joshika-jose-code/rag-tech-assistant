from pydantic import BaseModel, Field
from typing import Optional, List


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    source: str
    snippet: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    is_fallback: bool
    retries_used: int
    grounded: Optional[bool] = None
    hallucination_retries_used: int
    used_web_search: bool


class IngestURLRequest(BaseModel):
    urls: List[str]


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = None
