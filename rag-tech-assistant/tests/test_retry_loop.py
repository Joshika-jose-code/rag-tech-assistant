"""
Tests for the retry-loop termination logic — the core self-corrective
mechanism of the graph.

These tests never call a real LLM or vector store. Node functions are
monkeypatched at the app.graph.build_graph module level (where they were
imported into) so build_graph() wires up fakes when we construct a fresh
graph inside each test.
"""
from langchain_core.documents import Document

from app.graph import build_graph as bg


IRRELEVANT_DOC = Document(page_content="unrelated content", metadata={"source": "x.md"})
RELEVANT_DOC = Document(page_content="the actual answer", metadata={"source": "quickstart.md"})


def _base_state(max_retries=2):
    return {
        "question": "How do I do the thing?",
        "query": "How do I do the thing?",
        "query_type": None,
        "documents": [],
        "graded_documents": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "generation": None,
        "sources": [],
        "is_fallback": False,
        "grounded": None,
        "hallucination_retry_count": 0,
        "max_hallucination_retries": 2,
    }


# ---------------------------------------------------------------------------
# Fake node implementations
# ---------------------------------------------------------------------------

def fake_query_analysis(state):
    return {"query": state["question"], "query_type": "how-to"}


def fake_retrieve(state):
    return {"documents": [IRRELEVANT_DOC]}


def fake_retrieve_relevant(state):
    return {"documents": [RELEVANT_DOC]}


def fake_grade_always_irrelevant(state):
    return {"graded_documents": []}


def fake_grade_always_relevant(state):
    return {"graded_documents": [RELEVANT_DOC]}


def fake_transform_query(state):
    return {
        "query": state["query"] + " (rewritten)",
        "retry_count": state["retry_count"] + 1,
    }


def fake_generate(state):
    return {
        "generation": "a grounded answer",
        "sources": [{"source": d.metadata["source"], "snippet": d.page_content[:20], "score": None}
                    for d in state["graded_documents"]],
        "is_fallback": False,
    }


def fake_generate_fallback(state):
    return {
        "generation": "I don't know",
        "sources": [],
        "is_fallback": True,
    }


def fake_hallucination_check_grounded(state):
    """Always accepts the answer on the first pass — keeps these tests
    focused on the retrieval retry loop, not the hallucination-check loop
    (see test_hallucination_loop.py for that)."""
    return {"grounded": True}


def _patch_nodes(monkeypatch, *, retrieve=fake_retrieve, grade=fake_grade_always_irrelevant,
                  transform=fake_transform_query, generate=fake_generate,
                  hallucination_check=fake_hallucination_check_grounded,
                  fallback=fake_generate_fallback):
    monkeypatch.setattr(bg, "query_analysis_node", fake_query_analysis)
    monkeypatch.setattr(bg, "retrieve_node", retrieve)
    monkeypatch.setattr(bg, "grade_documents_node", grade)
    monkeypatch.setattr(bg, "transform_query_node", transform)
    monkeypatch.setattr(bg, "generate_node", generate)
    monkeypatch.setattr(bg, "hallucination_check_node", hallucination_check)
    monkeypatch.setattr(bg, "generate_fallback_node", fallback)
    return bg.build_graph()


# ---------------------------------------------------------------------------
# decide_next_step: pure routing logic, no graph execution needed
# ---------------------------------------------------------------------------

class TestDecideNextStep:
    def test_routes_to_generate_when_relevant_docs_present(self):
        state = _base_state()
        state["graded_documents"] = [RELEVANT_DOC]
        assert bg.decide_next_step(state) == "generate"

    def test_routes_to_transform_query_when_no_docs_and_retries_remain(self):
        state = _base_state(max_retries=2)
        state["graded_documents"] = []
        state["retry_count"] = 0
        assert bg.decide_next_step(state) == "transform_query"

    def test_routes_to_fallback_when_no_docs_and_retries_exhausted(self):
        state = _base_state(max_retries=2)
        state["graded_documents"] = []
        state["retry_count"] = 2
        assert bg.decide_next_step(state) == "generate_fallback"

    def test_routes_to_fallback_when_retry_count_exceeds_max(self):
        # defensive: should never happen given the loop structure, but the
        # routing function itself should still terminate safely if it does
        state = _base_state(max_retries=2)
        state["graded_documents"] = []
        state["retry_count"] = 5
        assert bg.decide_next_step(state) == "generate_fallback"


# ---------------------------------------------------------------------------
# Full graph execution: retry loop actually terminates
# ---------------------------------------------------------------------------

class TestRetryLoopTermination:
    def test_terminates_via_fallback_when_never_relevant(self, monkeypatch):
        """If every retrieval attempt is graded irrelevant, the graph must
        stop after exactly max_retries loops and return the fallback."""
        graph = _patch_nodes(monkeypatch, grade=fake_grade_always_irrelevant)

        result = graph.invoke(_base_state(max_retries=2))

        assert result["is_fallback"] is True
        assert result["retry_count"] == 2
        assert result["generation"] == "I don't know"
        assert result["sources"] == []

    def test_terminates_via_generate_on_first_success(self, monkeypatch):
        """If the first retrieval already finds relevant docs, no retry
        should occur at all."""
        graph = _patch_nodes(
            monkeypatch,
            retrieve=fake_retrieve_relevant,
            grade=fake_grade_always_relevant,
        )

        result = graph.invoke(_base_state(max_retries=2))

        assert result["is_fallback"] is False
        assert result["retry_count"] == 0
        assert result["generation"] == "a grounded answer"
        assert len(result["sources"]) == 1

    def test_succeeds_after_one_retry(self, monkeypatch):
        """Grading fails on attempt 1, succeeds on attempt 2 — should use
        exactly one retry and then generate normally."""
        call_count = {"n": 0}

        def flaky_grade(state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"graded_documents": []}
            return {"graded_documents": [RELEVANT_DOC]}

        graph = _patch_nodes(monkeypatch, grade=flaky_grade)

        result = graph.invoke(_base_state(max_retries=2))

        assert result["is_fallback"] is False
        assert result["retry_count"] == 1
        assert result["generation"] == "a grounded answer"

    def test_respects_custom_max_retries(self, monkeypatch):
        """Retry ceiling is read from state, not hardcoded — confirm a
        lower max_retries stops the loop earlier."""
        graph = _patch_nodes(monkeypatch, grade=fake_grade_always_irrelevant)

        result = graph.invoke(_base_state(max_retries=0))

        # zero retries allowed: should go straight to fallback with no loop
        assert result["is_fallback"] is True
        assert result["retry_count"] == 0

    def test_retry_count_increments_exactly_once_per_loop(self, monkeypatch):
        """Guards against the off-by-one bug where retry_count gets bumped
        in more than one place — query is rewritten each loop and retry_count
        should track that 1:1."""
        queries_seen = []

        def tracking_retrieve(state):
            queries_seen.append(state["query"])
            return {"documents": [IRRELEVANT_DOC]}

        graph = _patch_nodes(
            monkeypatch,
            retrieve=tracking_retrieve,
            grade=fake_grade_always_irrelevant,
        )

        result = graph.invoke(_base_state(max_retries=3))

        assert result["retry_count"] == 3
        # original query + 3 rewrites = 4 distinct retrieve calls
        assert len(queries_seen) == 4
        assert len(set(queries_seen)) == 4  # each rewrite actually changed the query
