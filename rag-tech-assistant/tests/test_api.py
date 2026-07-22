import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.main as main_module


client = TestClient(app)


def _fake_invoke_success(initial_state):
    return {
        "generation": "The answer is X.",
        "sources": [{"source": "quickstart.md", "snippet": "X is...", "score": None}],
        "is_fallback": False,
        "retry_count": 0,
        "grounded": True,
        "hallucination_retry_count": 0,
        "used_web_search": False,
    }


def _fake_invoke_fallback(initial_state):
    return {
        "generation": "I don't know.",
        "sources": [],
        "is_fallback": True,
        "retry_count": 2,
        "grounded": None,
        "hallucination_retry_count": 0,
        "used_web_search": True,
    }


class TestQueryEndpoint:
    def test_query_success(self, monkeypatch):
        monkeypatch.setattr(main_module.compiled_graph, "invoke", _fake_invoke_success)

        resp = client.post("/query", json={"question": "How do path params work?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "The answer is X."
        assert body["is_fallback"] is False
        assert body["retries_used"] == 0
        assert len(body["sources"]) == 1
        assert body["grounded"] is True
        assert body["hallucination_retries_used"] == 0
        assert body["used_web_search"] is False

    def test_query_fallback_response_shape(self, monkeypatch):
        monkeypatch.setattr(main_module.compiled_graph, "invoke", _fake_invoke_fallback)

        resp = client.post("/query", json={"question": "How do I do something unrelated?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["is_fallback"] is True
        assert body["retries_used"] == 2
        assert body["sources"] == []
        assert body["grounded"] is None
        assert body["hallucination_retries_used"] == 0
        assert body["used_web_search"] is True

    def test_query_rejects_empty_question(self):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_graph_exception_returns_500(self, monkeypatch):
        def raise_error(initial_state):
            raise RuntimeError("boom")

        monkeypatch.setattr(main_module.compiled_graph, "invoke", raise_error)

        resp = client.post("/query", json={"question": "anything"})
        assert resp.status_code == 500


class TestIngestEndpoints:
    def test_ingest_urls_rejects_empty_list(self):
        resp = client.post("/ingest/urls", json={"urls": []})
        assert resp.status_code == 400

    def test_ingest_urls_success(self, monkeypatch):
        monkeypatch.setattr(main_module, "load_and_index_urls", lambda urls: 5)

        resp = client.post("/ingest/urls", json={"urls": ["https://example.com/doc"]})

        assert resp.status_code == 200
        assert resp.json() == {"status": "success", "chunks_added": 5}

    def test_ingest_urls_bad_url_returns_422(self, monkeypatch):
        def raise_value_error(urls):
            raise ValueError("Failed to fetch https://bad.example.com")

        monkeypatch.setattr(main_module, "load_and_index_urls", raise_value_error)

        resp = client.post("/ingest/urls", json={"urls": ["https://bad.example.com"]})
        assert resp.status_code == 422


class TestFeedbackEndpoint:
    def test_feedback_accepts_valid_rating(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resp = client.post("/feedback", json={
            "question": "q", "answer": "a", "rating": "up"
        })
        assert resp.status_code == 200
        assert resp.json() == {"status": "recorded"}
        assert (tmp_path / "feedback_log.jsonl").exists()

    def test_feedback_rejects_invalid_rating(self):
        resp = client.post("/feedback", json={
            "question": "q", "answer": "a", "rating": "sideways"
        })
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
