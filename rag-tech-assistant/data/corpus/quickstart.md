# FastAPI Quickstart & Core Concepts

This document gives a conceptual overview of FastAPI's core building blocks,
for developers who are new to the framework.

## What FastAPI is

FastAPI is a Python web framework for building APIs, built on top of
Starlette (for the web parts) and Pydantic (for the data validation parts).
Its main selling points are automatic request validation from Python type
hints, automatic interactive API documentation (Swagger UI at `/docs` and
ReDoc at `/redoc`), and native support for asynchronous request handling.

## Minimal application

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

Run it with `uvicorn main:app --reload`. The `--reload` flag restarts the
server automatically when source files change, which is useful during
development but should be disabled in production.

## Path operations

A "path operation" is FastAPI's term for a function decorated with an HTTP
method and a URL path, like `@app.get("/items/{item_id}")`. The decorator
tells FastAPI which HTTP verb and path this function handles; the function
itself contains the logic that runs when a matching request arrives.

## Type hints drive everything

FastAPI reads your function's Python type hints to determine:

- Where each parameter comes from (path, query string, request body, header)
- How to validate and convert incoming data
- What to show in the generated OpenAPI schema and interactive docs

For example, a parameter typed as `item_id: int` in a path like
`/items/{item_id}` tells FastAPI to extract the value from the URL and
convert it to an integer, returning a validation error automatically if the
value isn't a valid integer.

## Pydantic models for request bodies

For anything more complex than simple scalar parameters, you define a
Pydantic model:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

@app.post("/items/")
def create_item(item: Item):
    return item
```

FastAPI parses the incoming JSON body against this model, validates every
field, and gives you a fully-typed Python object to work with inside your
function. Invalid input results in an automatic 422 response with details
about what failed.

## Dependency injection

FastAPI has a built-in dependency injection system, invoked via `Depends()`.
A dependency is just a callable (usually a function) that FastAPI calls on
your behalf before your path operation runs, and whose return value gets
passed into your function as an argument. This is commonly used for shared
logic like database session management, authentication, or pagination
parameters that many endpoints need without duplicating code.

## Async vs sync path operations

You can declare a path operation with either `def` or `async def`. Use
`async def` when your function calls other async code (e.g. an async
database driver or HTTP client) with `await`. Use plain `def` for
CPU-bound or blocking synchronous work — FastAPI runs these in an external
thread pool automatically so they don't block the event loop.

## Automatic interactive docs

Every running FastAPI app exposes `/docs` (Swagger UI) and `/redoc` (ReDoc)
by default, generated from the OpenAPI schema FastAPI builds from your route
definitions, type hints, and Pydantic models. These are live — you can send
real requests to your API directly from the `/docs` page during development.
