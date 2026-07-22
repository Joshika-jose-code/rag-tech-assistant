"""
Tests for the web-search last resort: when the local-corpus retry loop
(grade_documents/transform_query) is exhausted, web_search_node queries
Tavily before falling through to the canned "I don't know" response.

Covers three layers, mirroring the style of tests/test_retry_loop.py and
tests/test_hallucination_loop.py:
  - web_search_node itself, with a fake Tavily client (no real API calls)
  - decide_after_web_search: pure routing logic
  - the full graph, to confirm the retry loop actually reaches web_search
    and routes onward correctly depending on what it finds
"""
from langchain_core.documents import Document

from app.graph import build_graph as bg
from app.graph import nodes as gn


WEB_DOC = Document(page_content="web answer content", metadata={"source": "https://example.com/answer", "title": "Answer"})


# ---------------------------------------------------------------------------
# web_search_node: fake Tavily client, no real network/API calls
# ---------------------------------------------------------------------------

class FakeTavilyClient:
    def __init__(self, results=None, error=None):
        self._results = results if results is not None else []
        self._error = error
        self.queries_received = []

    def search(self, query, max_results=4):
        self.queries_received.append(query)
        if self._error:
            raise self._error
        return {"results": self._results}


def _node_state(question="How do I do the thing?", query=None):
    # query defaults to a deliberately different string from question, so
    # tests catch a regression back to searching state["query"].
    return {"question": question, "query": query if query is not None else "unrelated rewritten query"}


class TestWebSearchNode:
    def test_converts_results_to_documents(self, monkeypatch):
        monkeypatch.setattr(gn, "get_tavily_client", lambda: FakeTavilyClient(results=[
            {"title": "CORS Guide", "url": "https://example.com/cors", "content": "Some CORS content"},
        ]))

        result = gn.web_search_node(_node_state(question="how to fix cors errors"))

        assert result["used_web_search"] is True
        docs = result["graded_documents"]
        assert len(docs) == 1
        assert docs[0].page_content == "Some CORS content"
        assert docs[0].metadata["source"] == "https://example.com/cors"
        assert docs[0].metadata["title"] == "CORS Guide"

    def test_searches_original_question_not_rewritten_query(self, monkeypatch):
        """Regression test: web_search_node must query Tavily with
        state["question"] (the original, never-mutated question), not
        state["query"] — the latter has been through transform_query_node's
        LLM rewrites by the time this node runs and can drift far from, or
        degenerate below, the original question's meaning."""
        fake_client = FakeTavilyClient(results=[])
        monkeypatch.setattr(gn, "get_tavily_client", lambda: fake_client)

        gn.web_search_node(_node_state(
            question="How do I connect FastAPI to Postgres with async SQLAlchemy?",
            query="what",
        ))

        assert fake_client.queries_received == ["How do I connect FastAPI to Postgres with async SQLAlchemy?"]

    def test_skips_results_with_no_content(self, monkeypatch):
        monkeypatch.setattr(gn, "get_tavily_client", lambda: FakeTavilyClient(results=[
            {"title": "Empty", "url": "https://example.com/empty", "content": ""},
            {"title": "Has Content", "url": "https://example.com/ok", "content": "actual text"},
        ]))

        result = gn.web_search_node(_node_state())

        assert len(result["graded_documents"]) == 1
        assert result["graded_documents"][0].metadata["source"] == "https://example.com/ok"

    def test_handles_no_results(self, monkeypatch):
        monkeypatch.setattr(gn, "get_tavily_client", lambda: FakeTavilyClient(results=[]))

        result = gn.web_search_node(_node_state())

        assert result["graded_documents"] == []
        assert result["used_web_search"] is True

    def test_handles_search_errors_gracefully(self, monkeypatch):
        """A Tavily error (missing API key, network failure, rate limit)
        must not crash the node — it should fail through to an empty
        result so the graph can fall back to the canned response."""
        monkeypatch.setattr(gn, "get_tavily_client", lambda: FakeTavilyClient(error=RuntimeError("boom")))

        result = gn.web_search_node(_node_state())

        assert result["graded_documents"] == []
        assert result["used_web_search"] is True

    def test_handles_client_construction_errors_gracefully(self, monkeypatch):
        """Missing TAVILY_API_KEY raises at TavilyClient() construction
        time, not at .search() time — must be caught too."""
        def raise_missing_key():
            raise RuntimeError("MissingAPIKeyError")
        monkeypatch.setattr(gn, "get_tavily_client", raise_missing_key)

        result = gn.web_search_node(_node_state())

        assert result["graded_documents"] == []
        assert result["used_web_search"] is True


# ---------------------------------------------------------------------------
# decide_after_web_search: pure routing logic, no graph execution needed
# ---------------------------------------------------------------------------

def _base_state(max_retries=1):
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
        "used_web_search": False,
    }


class TestDecideAfterWebSearch:
    def test_generates_when_web_results_found(self):
        state = _base_state()
        state["graded_documents"] = [WEB_DOC]
        assert bg.decide_after_web_search(state) == "generate"

    def test_falls_back_when_no_web_results(self):
        state = _base_state()
        state["graded_documents"] = []
        assert bg.decide_after_web_search(state) == "generate_fallback"


# ---------------------------------------------------------------------------
# Full graph execution: local retry loop exhausts, then web search runs
# ---------------------------------------------------------------------------

def fake_query_analysis(state):
    return {"query": state["question"], "query_type": "how-to"}


def fake_retrieve_irrelevant(state):
    return {"documents": []}


def fake_grade_always_irrelevant(state):
    return {"graded_documents": []}


def fake_transform_query(state):
    return {"query": state["query"] + " (rewritten)", "retry_count": state["retry_count"] + 1}


def fake_web_search_with_results(state):
    return {"graded_documents": [WEB_DOC], "used_web_search": True}


def fake_web_search_no_results(state):
    return {"graded_documents": [], "used_web_search": True}


def fake_generate(state):
    return {
        "generation": "answer from the web",
        "sources": [{"source": d.metadata["source"], "snippet": d.page_content[:20], "score": None}
                    for d in state["graded_documents"]],
        "is_fallback": False,
    }


def fake_hallucination_check_grounded(state):
    return {"grounded": True}


def fake_generate_fallback(state):
    return {"generation": "I don't know", "sources": [], "is_fallback": True}


def _patch_nodes(monkeypatch, *, web_search=fake_web_search_with_results, generate=fake_generate):
    monkeypatch.setattr(bg, "query_analysis_node", fake_query_analysis)
    monkeypatch.setattr(bg, "retrieve_node", fake_retrieve_irrelevant)
    monkeypatch.setattr(bg, "grade_documents_node", fake_grade_always_irrelevant)
    monkeypatch.setattr(bg, "transform_query_node", fake_transform_query)
    monkeypatch.setattr(bg, "web_search_node", web_search)
    monkeypatch.setattr(bg, "generate_node", generate)
    monkeypatch.setattr(bg, "hallucination_check_node", fake_hallucination_check_grounded)
    monkeypatch.setattr(bg, "generate_fallback_node", fake_generate_fallback)
    return bg.build_graph()


class TestWebSearchLoopIntegration:
    def test_uses_web_results_when_local_retries_exhausted(self, monkeypatch):
        graph = _patch_nodes(monkeypatch, web_search=fake_web_search_with_results)

        result = graph.invoke(_base_state(max_retries=1))

        assert result["is_fallback"] is False
        assert result["used_web_search"] is True
        assert result["generation"] == "answer from the web"
        assert result["sources"][0]["source"] == "https://example.com/answer"

    def test_falls_back_when_web_search_also_finds_nothing(self, monkeypatch):
        graph = _patch_nodes(monkeypatch, web_search=fake_web_search_no_results)

        result = graph.invoke(_base_state(max_retries=1))

        assert result["is_fallback"] is True
        assert result["used_web_search"] is True
        assert result["generation"] == "I don't know"

    def test_web_search_not_invoked_when_local_docs_are_relevant(self, monkeypatch):
        """web_search must be a last resort, not a parallel path — if the
        local corpus loop already found relevant docs, web_search_node
        should never be called."""
        calls = {"n": 0}

        def counting_web_search(state):
            calls["n"] += 1
            return fake_web_search_with_results(state)

        monkeypatch.setattr(bg, "query_analysis_node", fake_query_analysis)
        monkeypatch.setattr(bg, "retrieve_node", lambda state: {"documents": [WEB_DOC]})
        monkeypatch.setattr(bg, "grade_documents_node", lambda state: {"graded_documents": [WEB_DOC]})
        monkeypatch.setattr(bg, "transform_query_node", fake_transform_query)
        monkeypatch.setattr(bg, "web_search_node", counting_web_search)
        monkeypatch.setattr(bg, "generate_node", fake_generate)
        monkeypatch.setattr(bg, "hallucination_check_node", fake_hallucination_check_grounded)
        monkeypatch.setattr(bg, "generate_fallback_node", fake_generate_fallback)
        graph = bg.build_graph()

        result = graph.invoke(_base_state(max_retries=1))

        assert calls["n"] == 0
        assert result["used_web_search"] is False
