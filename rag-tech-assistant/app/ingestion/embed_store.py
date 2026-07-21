# app/ingestion/embed_store.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIR = "vectorstore"
COLLECTION_NAME = "tech_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_vectorstore = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore


def add_documents_to_store(chunks: list[Document]) -> None:
    if not chunks:
        return
    get_vectorstore().add_documents(chunks)


def list_indexed_docs() -> list[dict]:
    """Aggregate chunk counts per source document for GET /documents."""
    data = get_vectorstore().get()
    counts: dict[str, int] = {}
    for meta in data.get("metadatas", []):
        source = meta.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return [{"filename": src, "chunk_count": cnt} for src, cnt in counts.items()]
