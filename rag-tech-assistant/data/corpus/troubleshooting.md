# FastAPI Troubleshooting Guide

This guide covers common errors developers encounter when building FastAPI
applications, and how to resolve them.

## 422 Unprocessable Entity on a request that looks correct

This almost always means the request body, query parameters, or path
parameters don't match what your path operation function expects. Check that:

- The `Content-Type` header is `application/json` when sending a JSON body.
- Field names in your request body match your Pydantic model exactly
  (case-sensitive).
- Required fields (fields without a default value) are actually present in
  the request.
- Numeric or boolean fields aren't being sent as quoted strings when the
  model expects a raw type, unless you've configured coercion.

FastAPI returns a `detail` array in the 422 response body describing exactly
which field failed and why — always read this before guessing.

## "RuntimeError: There is no current event loop in thread" in background tasks

This typically happens when a synchronous function inside a `BackgroundTasks`
callback tries to call async code, or when mixing `asyncio.run()` inside an
already-running event loop (which FastAPI provides via Uvicorn/Starlette).
Fix: keep background task functions either fully sync or fully async — don't
call `asyncio.run()` from within a request handler.

## CORS errors in the browser console despite adding CORSMiddleware

Common causes:

- `CORSMiddleware` must be added before other middleware that might return a
  response early (ordering matters).
- `allow_origins` must exactly match the scheme + host + port of the
  frontend origin; wildcards (`*`) cannot be combined with
  `allow_credentials=True`.
- Preflight `OPTIONS` requests failing silently — check the Network tab for a
  separate `OPTIONS` request and confirm it returns 200, not 404/405.

## Pydantic model changes not reflected in the interactive docs (/docs)

The OpenAPI schema is generated at app startup and cached. If you're running
with `--reload` this should regenerate automatically on file save; if not,
fully restart the server process rather than relying on hot-reload, and hard
refresh the browser tab (`/docs` caches the schema client-side too).

## "Field required" errors for fields that have defaults in the database model

This is a common confusion between your Pydantic **request** model and your
**database/ORM** model. A field can be optional in your database schema (e.g.
nullable column with a server-side default) while still being required in
the Pydantic model used to validate incoming requests, because Pydantic only
sees what you declared. Use `Optional[str] = None` (or the equivalent
`str | None = None`) explicitly in the request model if the client should be
allowed to omit the field.

## Slow response times on endpoints with `def` instead of `async def`

Synchronous (`def`) path operations run in a thread pool, not the main event
loop, so a single slow synchronous endpoint won't block other requests — but
if you have many concurrent slow sync endpoints, you can exhaust the thread
pool. If your endpoint does I/O (database calls, HTTP requests to other
services), prefer `async def` with an async-compatible client library so it
yields control back to the event loop instead of occupying a thread.

## ImportError or circular import errors when splitting routers into multiple files

Common in larger FastAPI projects using `APIRouter`. Usually caused by two
modules importing from each other directly. Fix by centralizing shared
dependencies (like a database session getter) in their own module that both
routers import from, rather than routers importing from each other.
