"""
Tests for app/ingestion/embed_store.py — exercised directly against the real
Chroma + HuggingFace embedding stack, independent of the LangGraph pipeline
(no query_analysis/retrieve/generate nodes involved).
"""
import importlib

from langchain_core.documents import Document

import app.ingestion.embed_store as embed_store


def test_vectorstore_is_not_built_at_import_time():
    """Regression test: the embedding model + Chroma client used to be built
    eagerly at module import, which made importing this module trigger a
    network call (downloading the embedding model) as a side effect. That
    must stay lazy — only get_vectorstore() should build it, on first use."""
    importlib.reload(embed_store)
    assert embed_store._vectorstore is None


def test_get_vectorstore_builds_once_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_store, "PERSIST_DIR", str(tmp_path / "vectorstore"))
    monkeypatch.setattr(embed_store, "_vectorstore", None)

    first = embed_store.get_vectorstore()
    second = embed_store.get_vectorstore()

    assert first is second


def test_add_documents_and_list_indexed_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_store, "PERSIST_DIR", str(tmp_path / "vectorstore"))
    monkeypatch.setattr(embed_store, "_vectorstore", None)

    docs = [
        Document(
            page_content="FastAPI path parameters are declared with type hints.",
            metadata={"source": "quickstart.md", "chunk_id": "quickstart.md-0"},
        ),
        Document(
            page_content="A 422 error means validation failed on the request body.",
            metadata={"source": "troubleshooting.md", "chunk_id": "troubleshooting.md-0"},
        ),
        Document(
            page_content="Query() lets you declare validation and metadata for query params.",
            metadata={"source": "api_reference.md", "chunk_id": "api_reference.md-0"},
        ),
    ]

    embed_store.add_documents_to_store(docs)

    counts = {d["filename"]: d["chunk_count"] for d in embed_store.list_indexed_docs()}
    assert counts == {
        "quickstart.md": 1,
        "troubleshooting.md": 1,
        "api_reference.md": 1,
    }


def test_add_documents_to_store_noop_on_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_store, "PERSIST_DIR", str(tmp_path / "vectorstore"))
    monkeypatch.setattr(embed_store, "_vectorstore", None)

    embed_store.add_documents_to_store([])

    assert embed_store.list_indexed_docs() == []
