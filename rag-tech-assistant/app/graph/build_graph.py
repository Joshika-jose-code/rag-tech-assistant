# app/graph/build_graph.py
from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.nodes import (
    query_analysis_node,
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    generate_node,
    hallucination_check_node,
    generate_fallback_node,
)


def decide_next_step(state: GraphState) -> str:
    """Conditional edge after grading: route based on relevance + retry budget."""
    if state["graded_documents"]:
        return "generate"
    if state["retry_count"] < state["max_retries"]:
        return "transform_query"
    return "generate_fallback"


def decide_after_hallucination_check(state: GraphState) -> str:
    """Conditional edge after the hallucination check: accept a grounded
    answer, regenerate if it's ungrounded and retry budget remains, or fall
    through to the canned fallback once that budget is exhausted."""
    if state["grounded"]:
        return "accept"
    if state["hallucination_retry_count"] < state["max_hallucination_retries"]:
        return "generate"
    return "generate_fallback"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("generate_fallback", generate_fallback_node)

    workflow.set_entry_point("query_analysis")

    workflow.add_edge("query_analysis", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "generate_fallback": "generate_fallback",
        },
    )

    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "hallucination_check")

    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination_check,
        {
            "accept": END,
            "generate": "generate",
            "generate_fallback": "generate_fallback",
        },
    )

    workflow.add_edge("generate_fallback", END)

    return workflow.compile()


compiled_graph = build_graph()


if __name__ == "__main__":
    initial_state = {
        "question": "How do I use retry logic in this library?",
        "query": "How do I use retry logic in this library?",
        "query_type": None,
        "documents": [],
        "graded_documents": [],
        "retry_count": 0,
        "max_retries": 2,
        "generation": None,
        "sources": [],
        "is_fallback": False,
        "grounded": None,
        "hallucination_retry_count": 0,
        "max_hallucination_retries": 2,
    }
    result = compiled_graph.invoke(initial_state)
    print("Answer:", result["generation"])
    print("Retries used:", result["retry_count"])
    print("Fallback triggered:", result["is_fallback"])
    print("Grounded:", result["grounded"])
    print("Hallucination retries used:", result["hallucination_retry_count"])
