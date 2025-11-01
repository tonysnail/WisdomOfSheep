# `backend.schemas`

`backend.schemas` defines the Pydantic models exposed by the FastAPI layer.
Splitting them out keeps `app.py` lean and makes it easier to reuse the models in
unit tests or other tooling.  The module depends on `backend.interest` for the
`InterestRecord` type and groups related response payloads together:

- List and detail views for posts
- Calendar aggregations
- Batch run and council job request/response envelopes
- Research payload representations used by the researcher endpoints

With the models in a dedicated module we avoid circular imports and keep schema
definitions close to (but separate from) the business logic that populates them.
