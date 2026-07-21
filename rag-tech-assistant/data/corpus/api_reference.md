# FastAPI API Reference: Path Operations & Parameter Declarations

This is a reference listing of the parameters accepted by FastAPI's path
operation decorators and parameter-declaration functions. Unlike the
quickstart guide, this document doesn't explain concepts — it documents the
exact keyword arguments each function accepts.

## Path operation decorators

`@app.get(path, ...)`, `@app.post(path, ...)`, `@app.put(path, ...)`,
`@app.delete(path, ...)`, `@app.patch(path, ...)`, `@app.options(path, ...)`,
`@app.head(path, ...)`, and `@app.trace(path, ...)` all accept the same
keyword arguments:

- `response_model`: the Pydantic model (or type) used to filter and
  validate the response, independent of the function's return type hint.
- `status_code`: the default HTTP status code for a successful response
  (e.g. `201` for a creation endpoint). Can be overridden per-request via
  the `Response` object.
- `tags`: a list of strings used to group endpoints in the generated docs
  (`List[str]`).
- `dependencies`: a list of `Depends()` calls that run for this path
  operation but whose return values aren't injected as arguments — useful
  for auth checks that don't need the result.
- `summary` / `description`: short and long text shown in the OpenAPI docs.
  If `description` is omitted, FastAPI uses the function's docstring.
- `response_description`: docs text for the response section (defaults to
  `"Successful Response"`).
- `responses`: a `Dict[int, dict]` describing additional possible responses
  (e.g. `{404: {"description": "Item not found"}}`) for the OpenAPI schema.
- `deprecated`: marks the endpoint as deprecated in the docs UI.
- `operation_id`: overrides the auto-generated OpenAPI `operationId`.
- `response_model_exclude_unset`: when `True`, fields never explicitly set
  on the response model are omitted from the JSON output (useful for
  PATCH-style partial responses).
- `response_model_exclude_none`: when `True`, fields whose value is `None`
  are omitted from the JSON output.
- `include_in_schema`: when `False`, hides the endpoint from `/docs` and
  `/redoc` entirely while still leaving it reachable.

## `Path()`, `Query()`, `Header()`, `Cookie()` — parameter declaration

These all share the same validation/metadata keyword arguments, imported
from `fastapi`:

- `default`: the default value if the parameter is optional. `Path()`
  parameters are always required (they come from the URL), so `default` is
  not meaningful there — use `...` or omit it.
- `alias`: use a different name in the incoming request than the Python
  parameter name (e.g. `alias="item-id"` to accept a hyphenated query key).
- `title` / `description`: shown in the generated OpenAPI schema.
- `gt`, `ge`, `lt`, `le`: numeric constraints (greater-than, greater-or-equal,
  less-than, less-or-equal) enforced automatically, returning a 422 on
  violation.
- `min_length`, `max_length`: string length constraints.
- `pattern`: a regex the string value must match (named `regex` in FastAPI
  versions prior to 0.100 — both raise a 422 on mismatch).
- `deprecated`: marks the individual parameter as deprecated in the docs.
- `include_in_schema`: exclude this specific parameter from the OpenAPI
  schema while still parsing it.

## `Body()` — request body fields

Same metadata arguments as above, plus:

- `embed`: when `True`, expects the body to be nested under a key matching
  the parameter name (`{"item": {...}}`) instead of the model's fields being
  merged directly into the top-level JSON body. FastAPI embeds automatically
  when a path operation takes multiple `Body()` parameters, regardless of
  this flag.
- `media_type`: defaults to `"application/json"`; override for endpoints
  that accept a different content type.

## `Form()` and `File()`

Same base arguments as `Body()`. `File()` parameters are typed as `bytes` or
`UploadFile`; `UploadFile` is preferred for anything beyond small payloads
since it streams to a spooled temp file instead of loading the whole upload
into memory. Both require `python-multipart` to be installed — omitting it
raises a runtime error the first time such an endpoint is hit, not at
import time.

## `Depends()`

- `dependency`: the callable to invoke. If omitted, FastAPI uses the type
  annotation of the parameter itself as the callable.
- `use_cache`: defaults to `True`. When multiple parameters in the same
  request depend on the same callable, FastAPI calls it once per request
  and reuses the cached result unless `use_cache=False` is set.

## Response status code constants

FastAPI re-exports Starlette's `status` module (`from fastapi import
status`) with named constants such as `status.HTTP_201_CREATED`,
`status.HTTP_404_NOT_FOUND`, and `status.HTTP_422_UNPROCESSABLE_ENTITY`,
preferred over hardcoding integers for readability in `status_code=` and
`responses=` arguments.
