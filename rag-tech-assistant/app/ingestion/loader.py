# app/ingestion/loader.py
import os
import tempfile
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from fastapi import UploadFile

from app.ingestion.chunker import chunk_documents
from app.ingestion.embed_store import add_documents_to_store

ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}


def _read_file_content(path: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    if ext in (".html", ".htm"):
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator="\n")
    return raw  # markdown/text: keep as-is, chunker handles structure


async def load_and_index_files(files: list[UploadFile]) -> int:
    documents = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}' for {file.filename}. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        contents = await file.read()
        if not contents:
            raise ValueError(f"File {file.filename} is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            text = _read_file_content(tmp_path, file.filename)
        finally:
            os.unlink(tmp_path)

        documents.append(
            Document(page_content=text, metadata={"source": file.filename})
        )

    chunks = chunk_documents(documents)
    add_documents_to_store(chunks)
    return len(chunks)


def load_and_index_urls(urls: list[str]) -> int:
    documents = []

    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch {url}: {str(e)}")

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n")
        else:
            text = resp.text

        if not text.strip():
            raise ValueError(f"No content extracted from {url}")

        documents.append(Document(page_content=text, metadata={"source": url}))

    chunks = chunk_documents(documents)
    add_documents_to_store(chunks)
    return len(chunks)
