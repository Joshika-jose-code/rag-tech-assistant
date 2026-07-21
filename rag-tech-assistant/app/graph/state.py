# app/graph/state.py
from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    query: str
    query_type: Optional[str]
    documents: List[Document]
    graded_documents: List[Document]
    retry_count: int
    max_retries: int
    generation: Optional[str]
    sources: List[dict]
    is_fallback: bool
