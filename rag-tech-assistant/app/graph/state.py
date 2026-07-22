
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
    grounded: Optional[bool]        # set by hallucination_check_node; None until it has run
    hallucination_retry_count: int  # incremented only in generate_node when regenerating
    max_hallucination_retries: int  # regeneration budget, set by caller (main.py: DEFAULT_MAX_HALLUCINATION_RETRIES=2)
