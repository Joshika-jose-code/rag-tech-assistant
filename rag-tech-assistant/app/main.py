import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from dotenv import load_dotenv

load_dotenv()

from app.models.schemas import (
    QueryRequest, QueryResponse, IngestURLRequest,
    DocumentInfo, FeedbackRequest
)
from app.graph.build_graph import compiled_graph
from app.ingestion.embed_store import list_indexed_docs
from app.ingestion.loader import load_and_index_files, load_and_index_urls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Technical Documentation Assistant")

DEFAULT_MAX_RETRIES = 2


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        initial_state = {
            "question": request.question,
            "query": request.question,
            "query_type": None,
            "documents": [],
            "graded_documents": [],
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "generation": None,
            "sources": [],
            "is_fallback": False,
        }
        result = compiled_graph.invoke(initial_state)
        return QueryResponse(
            answer=result["generation"],
            sources=result["sources"],
            is_fallback=result["is_fallback"],
            retries_used=result["retry_count"],
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/ingest/files")
async def ingest_files(files: list[UploadFile] = File(...)):
    try:
        added = await load_and_index_files(files)
        return {"status": "success", "chunks_added": added}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("File ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/urls")
async def ingest_urls(request: IngestURLRequest):
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    try:
        added = load_and_index_urls(request.urls)
        return {"status": "success", "chunks_added": added}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("URL ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/documents", response_model=list[DocumentInfo])
async def documents():
    try:
        return list_indexed_docs()
    except Exception as e:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        with open("feedback_log.jsonl", "a") as f:
            f.write(request.model_dump_json() + "\n")
        return {"status": "recorded"}
    except Exception as e:
        logger.exception("Failed to record feedback")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
