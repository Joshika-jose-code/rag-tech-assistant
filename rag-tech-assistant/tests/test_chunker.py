from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents


def test_chunk_documents_produces_at_least_one_chunk():
    doc = Document(page_content="A short piece of text.", metadata={"source": "a.md"})
    chunks = chunk_documents([doc])
    assert len(chunks) >= 1


def test_chunk_metadata_preserves_source_and_adds_chunk_id():
    doc = Document(page_content="Some content here about FastAPI.", metadata={"source": "a.md"})
    chunks = chunk_documents([doc])
    for chunk in chunks:
        assert chunk.metadata["source"] == "a.md"
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith("a.md-")


def test_long_document_splits_into_multiple_chunks():
    # ~2000 words, comfortably over the 300-token chunk size
    long_text = "This is a sentence about FastAPI path operations. " * 400
    doc = Document(page_content=long_text, metadata={"source": "long.md"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1


def test_chunk_ids_are_unique_within_a_document():
    long_text = "FastAPI dependency injection lets you share logic. " * 200
    doc = Document(page_content=long_text, metadata={"source": "b.md"})
    chunks = chunk_documents([doc])
    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_multiple_documents_get_independent_chunk_numbering():
    doc_a = Document(page_content="Content A. " * 50, metadata={"source": "a.md"})
    doc_b = Document(page_content="Content B. " * 50, metadata={"source": "b.md"})
    chunks = chunk_documents([doc_a, doc_b])

    sources = {c.metadata["source"] for c in chunks}
    assert sources == {"a.md", "b.md"}
