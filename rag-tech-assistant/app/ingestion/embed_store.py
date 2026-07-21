# app/ingestion/embed_store.py
from chromadb.config import Settings
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
            # Disables Chroma's anonymous usage telemetry (paired with
            # pinning `posthog` in requirements.txt - see that comment for
            # why an unpinned posthog crashes chromadb's telemetry code).
            # is_persistent=True is required here: langchain_chroma only
            # defaults to persistent mode itself when client_settings is
            # NOT passed. Passing our own client_settings without it
            # silently downgrades to ephemeral/in-memory mode - data looks
            # fine within a single process but vanishes the moment it
            # exits, which a from-scratch process-boundary check caught.
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
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
