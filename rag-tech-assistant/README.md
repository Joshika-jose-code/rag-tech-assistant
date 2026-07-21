# RAG-Based Technical Documentation Assistant

A Retrieval-Augmented Generation system that answers questions about a small
corpus of FastAPI documentation, built with a self-corrective LangGraph
workflow (query analysis → retrieval → document grading → generation, with a
conditional retry loop) and served via FastAPI.

Built for the Express Analytics AI/ML Engineer Intern take-home assignment.

---

## 1. Project Overview

The system ingests a mixed corpus of local Markdown files and fetched
official documentation pages, indexes them in a Chroma vector store, and
answers natural-language questions through a LangGraph pipeline that:

1. Rewrites/classifies the incoming question for better retrieval
2. Retrieves the top-k most similar chunks
3. **Grades each chunk for actual relevance with an LLM** (the self-corrective
   step) — irrelevant chunks are filtered out
4. If nothing relevant survives grading, rewrites the query and retries
   (bounded by a retry limit) before falling back to an honest "I don't know"
   response
5. Generates a final answer grounded only in the surviving relevant chunks,
   with inline citations

## 2. Architecture

```
                          ┌─────────────────┐
                          │  query_analysis  │   rewrite + classify query
                          └────────┬─────────┘
                                   │
                                   ▼
                    ┌────────────────────────┐
              ┌────► │        retrieve         │  vector similarity search
              │      └────────────┬────────────┘
              │                   │
              │                   ▼
              │      ┌────────────────────────┐
              │      │    grade_documents      │  LLM grades each chunk
              │      └────────────┬────────────┘
              │                   │
              │        (conditional edge: decide_next_step)
              │                   │
              │     ┌─────────────┼──────────────┐
              │     ▼             ▼              ▼
              │  relevant    no match,       no match,
              │  docs found  retries left    retries exhausted
              │     │             │              │
              │     ▼             ▼              ▼
              │  generate   transform_query  generate_fallback
              │     │             │              │
              └─────┴─────────────┘              │
                     │                            │
                     ▼                            ▼
                    END                          END
```

**State schema** (`app/graph/state.py`) tracks: the original `question`
(never mutated) separately from the current `query` (mutated on retry),
`retry_count` / `max_retries` for loop control, `documents` vs
`graded_documents` (raw retrieval vs. post-grading), and `is_fallback` so the
API layer knows whether the answer came from real context or the fallback
path.

Full reasoning behind the state design and node choices is in
[Section 7](#7-design-decisions--tradeoffs).

## 3. Project Structure

```
rag-tech-assistant/
├── app/
│   ├── main.py                 # FastAPI app + routes
│   ├── graph/
│   │   ├── state.py            # GraphState TypedDict
│   │   ├── nodes.py            # all 6 node functions + prompts
│   │   ├── build_graph.py      # StateGraph wiring + conditional edge
│   ├── ingestion/
│   │   ├── loader.py           # file/URL loading for API endpoints
│   │   ├── chunker.py          # token-based splitting strategy
│   │   ├── embed_store.py      # Chroma embedding + storage
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
├── data/
│   └── corpus/                 # local Markdown docs (quickstart, troubleshooting)
├── vectorstore/                # persisted Chroma index (created on ingest)
├── ingest.py                   # standalone CLI ingestion script
├── requirements.txt
├── .env.example
└── feedback_log.jsonl          # created at runtime by POST /feedback
```

## 4. Setup

### Prerequisites
- Python 3.11+
- An OpenAI API key (used for both chat completions and embeddings)

### Install

```bash
git clone <your-repo-url>
cd rag-tech-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env and add your OPENAI_API_KEY
```

### Ingest the corpus

```bash
python ingest.py --urls \
  https://fastapi.tiangolo.com/tutorial/path-params/ \
  https://fastapi.tiangolo.com/tutorial/dependencies/ \
  https://fastapi.tiangolo.com/tutorial/query-params-str-validations/
```

This indexes the 2 local files in `data/corpus/` plus the 3 fetched URLs —
5 documents total. Add `--clear` to wipe and rebuild the vector store, or
`--skip-local` to index only URLs.

### Run the API

```bash
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

## 5. Example Requests

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add a path parameter with type validation?"}'
```

```json
{
  "answer": "You can add a path parameter by declaring it in the route path using curly braces, e.g. `/items/{item_id}`, and adding a matching function parameter with a type hint such as `item_id: int` [1]. FastAPI will automatically validate and convert the value...",
  "sources": [
    {
      "source": "https://fastapi.tiangolo.com/tutorial/path-params/",
      "snippet": "Path parameters with types...",
      "score": null
    }
  ],
  "is_fallback": false,
  "retries_used": 0
}
```

### Query that should trigger the fallback path

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I configure WebSocket authentication in FastAPI?"}'
```

```json
{
  "answer": "I wasn't able to find information about WebSocket authentication in the currently indexed documentation. You could try rephrasing your question, or check the official FastAPI docs directly.",
  "sources": [],
  "is_fallback": true,
  "retries_used": 2
}
```

(The corpus deliberately doesn't cover WebSockets — this is a documented
negative-case test; see Section 8.)

### `POST /ingest/urls`

```bash
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://fastapi.tiangolo.com/tutorial/first-steps/"]}'
```

```json
{ "status": "success", "chunks_added": 6 }
```

### `POST /ingest/files`

```bash
curl -X POST http://localhost:8000/ingest/files \
  -F "files=@my_notes.md"
```

### `GET /documents`

```json
[
  { "filename": "quickstart.md", "chunk_count": 9 },
  { "filename": "troubleshooting.md", "chunk_count": 7 },
  { "filename": "https://fastapi.tiangolo.com/tutorial/path-params/", "chunk_count": 4 }
]
```

### `POST /feedback`

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add a path parameter?", "answer": "...", "rating": "up"}'
```

```json
{ "status": "recorded" }
```

## 6. Chunking & Embedding Strategy

- **Splitter**: `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, with a
  separator priority of markdown headers → paragraphs → lines → sentences →
  words. Token-based sizing (300 tokens, 50 overlap) rather than raw
  character count, since token count is what actually governs embedding
  input limits and LLM context budget — a character-based split can
  silently produce wildly different token counts depending on content
  density (code vs. prose).
- **Why prioritize header boundaries**: technical docs are structured around
  headers far more than narrative prose is; splitting on `##`/`###`
  boundaries first keeps a concept and its explanation together rather than
  slicing mid-thought.
- **Overlap (50 tokens, ~15%)**: preserves continuity across a chunk
  boundary — e.g. a sentence that references "the previous example" doesn't
  lose that antecedent entirely.
- **Embedding model**: `text-embedding-3-small` (OpenAI) — good
  quality-to-cost ratio for a corpus this size; no need for the larger model
  here.
- **Vector store**: ChromaDB, chosen over FAISS because it persists to disk
  with metadata natively and its `.get()` method makes listing indexed
  sources for `GET /documents` straightforward, without maintaining a
  separate metadata store as FAISS would require.

## 7. Design Decisions & Tradeoffs

**Explicit-node state machine over tool-calling agent.** LangGraph's official
"Agentic RAG" pattern lets the LLM itself decide whether to call a retriever
tool, using `MessagesState`. I didn't use that pattern here — the assignment
explicitly specifies four named nodes and calls out state-schema design
(especially retry tracking) as a core evaluation criterion, which maps
directly onto the explicit `TypedDict` + conditional-edge pattern used in
LangGraph's CRAG/Adaptive RAG references instead. The tool-calling pattern is
a reasonable alternative for a different kind of assignment, but it makes
retry counting and node-level responsibility much harder to point to
explicitly.

**`question` vs `query` are separate fields.** The original question is never
mutated; `query` is what gets rewritten on each retry. Generation is prompted
against the *original* question even though retrieval used the *rewritten*
query — otherwise a multi-hop rewrite could drift the final answer away from
what the user actually asked.

**Grading is per-chunk, not batched.** Each retrieved chunk gets its own LLM
grading call rather than grading all k chunks in a single call. This costs
more tokens/latency but avoids the failure mode where a single "grade these
4 chunks" call silently conflates or drops one. Batching is a reasonable
optimization if the LLM cost becomes a real constraint.

**`transform_query_node` is the single point where `retry_count` increments.**
Keeping the increment in exactly one place was a deliberate choice to avoid
an off-by-one bug that would burn the retry budget faster than intended.

**`generate_fallback` is its own LLM-driven node, not a hardcoded string.**
This lets the "I don't know" response still reference the original question
naturally, rather than returning a generic canned message.

**Prompt-injection guard on grading and generation prompts.** Both prompts
explicitly instruct the model to treat retrieved content as data, not
instructions, and wrap chunks in `<context>` tags. This is a mitigation, not
a hard guarantee — a sufficiently adversarial chunk could still partially
influence output. For a corpus of trusted official docs the risk is low, but
this matters more once `/ingest` accepts arbitrary user-submitted URLs/files.

**Split `/ingest/files` and `/ingest/urls` instead of one combined endpoint.**
FastAPI doesn't cleanly mix multipart file uploads with a JSON body in a
single endpoint; splitting also gives each path its own validation logic and
error semantics (400 for missing input, 422 for ingestion-specific failures
like unreachable URLs or unsupported file types, 500 for genuine server
errors).

**Feedback storage is a flat `.jsonl` file**, not a database. Sufficient for
a 2-day assignment; a SQLite table would be the natural upgrade if this went
further.

## 8. Assumptions

- The corpus is small enough (5 documents) that full-corpus re-ingestion on
  `--clear` is cheap; no incremental-update/dedup logic was built for
  re-ingesting an already-indexed file.
- Single-turn queries only — no conversation memory across requests (see
  below).
- OpenAI is the only LLM/embedding provider wired up; swapping providers
  would mean changing `ChatOpenAI`/`OpenAIEmbeddings` instantiations in
  `nodes.py` and `embed_store.py`.

## 9. What I'd Improve With More Time

- **Hallucination check** (Self-RAG style): a node after `generate` that
  verifies the answer is actually supported by the retrieved context before
  returning it, looping back to regenerate or falling through to
  `generate_fallback` if not.
- **Web search fallback**: if grading exhausts retries with nothing relevant,
  fall back to a live web search (Tavily/Serper) before generating, rather
  than going straight to "I don't know."
- **Conversation memory**: add `chat_history` to `GraphState` and feed it
  into `query_analysis` so follow-up questions ("what about the other one?")
  can resolve pronouns/context from prior turns.
- **Batched grading** to cut latency/cost once corpus size and query volume
  grow.
- **Score-aware retrieval**: switch `similarity_search` to
  `similarity_search_with_score` so `sources[].score` in the API response is
  a real number instead of always `null`.
- **Ingestion dedup**: hash-check before re-adding a previously-ingested
  source to avoid duplicate chunks on repeated `/ingest` calls.
