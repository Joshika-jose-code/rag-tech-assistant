"""
Tests for the hallucination-check loop — the Self-RAG style verification
step that runs after generation: accept a grounded answer, regenerate if
it's ungrounded and budget remains, or fall through to generate_fallback
once that budget is exhausted.

These tests never call a real LLM. Node functions are monkeypatched at the
app.graph.build_graph module level (where they were imported into) so
build_graph() wires up fakes when we construct a fresh graph inside each
test. Mirrors the structure of tests/test_retry_loop.py.
"""
from langchain_core.documents import Document

from app.graph import build_graph as bg


RELEVANT_DOC = Document(page_content="the actual answer", metadata={"source": "quickstart.md"})


def _base_state(max_hallucination_retries=2):
    return {
        "question": "How do I do the thing?",
        "query": "How do I do the thing?",
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
        "max_hallucination_retries": max_hallucination_retries,
        "used_web_search": False,
    }


# ---------------------------------------------------------------------------
# decide_after_hallucination_check: pure routing logic, no graph execution
# ---------------------------------------------------------------------------

class TestDecideAfterHallucinationCheck:
    def test_accepts_when_grounded(self):
        state = _base_state()
        state["grounded"] = True
        assert bg.decide_after_hallucination_check(state) == "accept"

    def test_regenerates_when_ungrounded_and_retries_remain(self):
        state = _base_state(max_hallucination_retries=2)
        state["grounded"] = False
        state["hallucination_retry_count"] = 0
        assert bg.decide_after_hallucination_check(state) == "generate"

    def test_falls_back_when_ungrounded_and_retries_exhausted(self):
        state = _base_state(max_hallucination_retries=2)
        state["grounded"] = False
        state["hallucination_retry_count"] = 2
        assert bg.decide_after_hallucination_check(state) == "generate_fallback"

    def test_falls_back_when_retry_count_exceeds_max(self):
        # defensive: should never happen given the loop structure, but the
        # routing function itself should still terminate safely if it does
        state = _base_state(max_hallucination_retries=2)
        state["grounded"] = False
        state["hallucination_retry_count"] = 5
        assert bg.decide_after_hallucination_check(state) == "generate_fallback"


# ---------------------------------------------------------------------------
# Fake node implementations
# ---------------------------------------------------------------------------

def fake_query_analysis(state):
    return {"query": state["question"], "query_type": "how-to"}


def fake_retrieve_relevant(state):
    return {"documents": [RELEVANT_DOC]}


def fake_grade_always_relevant(state):
    return {"graded_documents": [RELEVANT_DOC]}


def fake_generate(state):
    # Mirrors the real generate_node's is_retry gate (state["grounded"] is
    # False only after hallucination_check_node has rejected a prior
    # answer) so the loop-termination behavior under test matches reality,
    # without invoking an actual LLM.
    is_retry = state.get("grounded") is False
    result = {
        "generation": "an answer" + (" (revised)" if is_retry else ""),
        "sources": [{"source": d.metadata["source"], "snippet": d.page_content[:20], "score": None}
                    for d in state["graded_documents"]],
        "is_fallback": False,
    }
    if is_retry:
        result["hallucination_retry_count"] = state["hallucination_retry_count"] + 1
    return result


def fake_hallucination_check_always_grounded(state):
    return {"grounded": True}


def fake_hallucination_check_always_ungrounded(state):
    return {"grounded": False}


def fake_generate_fallback(state):
    return {"generation": "I don't know", "sources": [], "is_fallback": True}


def _patch_nodes(monkeypatch, *, generate=fake_generate,
                  hallucination_check=fake_hallucination_check_always_grounded):
    monkeypatch.setattr(bg, "query_analysis_node", fake_query_analysis)
    monkeypatch.setattr(bg, "retrieve_node", fake_retrieve_relevant)
    monkeypatch.setattr(bg, "grade_documents_node", fake_grade_always_relevant)
    monkeypatch.setattr(bg, "generate_node", generate)
    monkeypatch.setattr(bg, "hallucination_check_node", hallucination_check)
    monkeypatch.setattr(bg, "generate_fallback_node", fake_generate_fallback)
    return bg.build_graph()


# ---------------------------------------------------------------------------
# Full graph execution: hallucination loop actually terminates
# ---------------------------------------------------------------------------

class TestHallucinationLoopTermination:
    def test_accepts_on_first_grounded_check(self, monkeypatch):
        graph = _patch_nodes(monkeypatch, hallucination_check=fake_hallucination_check_always_grounded)

        result = graph.invoke(_base_state())

        assert result["is_fallback"] is False
        assert result["grounded"] is True
        assert result["hallucination_retry_count"] == 0
        assert result["generation"] == "an answer"

    def test_terminates_via_fallback_when_never_grounded(self, monkeypatch):
        """If the check always rejects the answer, the graph must stop
        after exactly max_hallucination_retries regenerations and return
        the fallback."""
        graph = _patch_nodes(monkeypatch, hallucination_check=fake_hallucination_check_always_ungrounded)

        result = graph.invoke(_base_state(max_hallucination_retries=2))

        assert result["is_fallback"] is True
        assert result["generation"] == "I don't know"
        assert result["hallucination_retry_count"] == 2

    def test_succeeds_after_one_regeneration(self, monkeypatch):
        """Check fails on attempt 1, passes on attempt 2 — should use
        exactly one regeneration and then accept the answer."""
        call_count = {"n": 0}

        def flaky_check(state):
            call_count["n"] += 1
            return {"grounded": call_count["n"] > 1}

        graph = _patch_nodes(monkeypatch, hallucination_check=flaky_check)

        result = graph.invoke(_base_state())

        assert result["is_fallback"] is False
        assert result["hallucination_retry_count"] == 1
        assert result["generation"] == "an answer (revised)"

    def test_respects_custom_max_hallucination_retries(self, monkeypatch):
        """Retry ceiling is read from state, not hardcoded — confirm a
        lower max_hallucination_retries stops the loop earlier."""
        graph = _patch_nodes(monkeypatch, hallucination_check=fake_hallucination_check_always_ungrounded)

        result = graph.invoke(_base_state(max_hallucination_retries=0))

        # zero regenerations allowed: should go straight to fallback with no loop
        assert result["is_fallback"] is True
        assert result["hallucination_retry_count"] == 0

    def test_hallucination_retry_count_increments_exactly_once_per_loop(self, monkeypatch):
        """Guards against an off-by-one bug where the counter gets bumped
        in more than one place — generate_node's is_retry gate should fire
        only on regeneration passes, 1:1 with loop iterations."""
        seen_is_retry = []

        def tracking_generate(state):
            is_retry = state.get("grounded") is False
            seen_is_retry.append(is_retry)
            result = {"generation": "an answer", "sources": [], "is_fallback": False}
            if is_retry:
                result["hallucination_retry_count"] = state["hallucination_retry_count"] + 1
            return result

        graph = _patch_nodes(
            monkeypatch,
            generate=tracking_generate,
            hallucination_check=fake_hallucination_check_always_ungrounded,
        )

        result = graph.invoke(_base_state(max_hallucination_retries=3))

        assert result["hallucination_retry_count"] == 3
        # first generation + 3 regenerations = 4 distinct generate calls
        assert seen_is_retry == [False, True, True, True]
