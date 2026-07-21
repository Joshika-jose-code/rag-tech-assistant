# app/ingestion/chunker.py
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4

MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    ". ",
    " ",
]

_splitter = None


def _get_splitter():
    global _splitter
    if _splitter is not None:
        return _splitter

    try:
        _splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=MARKDOWN_SEPARATORS,
        )
    except Exception as e:
        logger.warning(
            "Falling back to character-based splitting — could not load "
            "tiktoken encoder (%s). Chunk sizes will be approximate rather "
            "than exact token counts.", e
        )
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_CHARS,
            chunk_overlap=CHUNK_OVERLAP_CHARS,
            separators=MARKDOWN_SEPARATORS,
        )

    return _splitter


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = _get_splitter().split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}-{i}"
    return chunks
