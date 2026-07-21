"""
Standalone ingestion script.

Run this once before starting the API to build the vector store from the
local corpus in data/corpus/. Also supports ingesting from URLs directly,
without going through the FastAPI /ingest/urls endpoint.

Usage:
    python ingest.py                          # ingest everything in data/corpus/
    python ingest.py --urls url1 url2 url3    # ingest from URLs instead
    python ingest.py --clear                  # wipe the existing vector store first
"""
import argparse
import os
import shutil
import sys

from dotenv import load_dotenv

load_dotenv()

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents
from app.ingestion.embed_store import add_documents_to_store, PERSIST_DIR

CORPUS_DIR = "data/corpus"
ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}


def load_local_corpus(corpus_dir: str) -> list[Document]:
    if not os.path.isdir(corpus_dir):
        print(f"Error: corpus directory '{corpus_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    documents = []
    filenames = sorted(os.listdir(corpus_dir))

    if not filenames:
        print(f"Warning: '{corpus_dir}' is empty. Nothing to ingest.")
        return documents

    for filename in filenames:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            print(f"Skipping unsupported file: {filename}")
            continue

        path = os.path.join(corpus_dir, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        if not raw.strip():
            print(f"Skipping empty file: {filename}")
            continue

        if ext in (".html", ".htm"):
            text = BeautifulSoup(raw, "html.parser").get_text(separator="\n")
        else:
            text = raw

        documents.append(Document(page_content=text, metadata={"source": filename}))
        print(f"Loaded: {filename} ({len(text)} chars)")

    return documents


def load_from_urls(urls: list[str]) -> list[Document]:
    import requests

    documents = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}", file=sys.stderr)
            continue

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            text = BeautifulSoup(resp.text, "html.parser").get_text(separator="\n")
        else:
            text = resp.text

        if not text.strip():
            print(f"No content extracted from {url}", file=sys.stderr)
            continue

        documents.append(Document(page_content=text, metadata={"source": url}))
        print(f"Fetched: {url} ({len(text)} chars)")

    return documents


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--urls", nargs="+", default=None, help="Additional URLs to fetch and ingest alongside the local corpus")
    parser.add_argument("--skip-local", action="store_true", help="Skip data/corpus/ and ingest only the given --urls")
    parser.add_argument("--clear", action="store_true", help="Delete the existing vector store before ingesting")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
        sys.exit(1)

    if args.clear and os.path.isdir(PERSIST_DIR):
        print(f"Clearing existing vector store at '{PERSIST_DIR}'...")
        shutil.rmtree(PERSIST_DIR)

    documents = []

    if not args.skip_local:
        documents.extend(load_local_corpus(CORPUS_DIR))

    if args.urls:
        documents.extend(load_from_urls(args.urls))

    if not documents:
        print("No documents loaded — nothing to ingest.")
        sys.exit(0)

    print(f"\nChunking {len(documents)} document(s)...")
    chunks = chunk_documents(documents)
    print(f"Produced {len(chunks)} chunks.")

    print("Embedding and storing...")
    add_documents_to_store(chunks)

    print(f"\nDone. Indexed {len(chunks)} chunks from {len(documents)} document(s) into '{PERSIST_DIR}'.")


if __name__ == "__main__":
    main()
