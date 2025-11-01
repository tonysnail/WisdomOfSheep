# `backend.interest`

`backend.interest` encapsulates the logic for loading and constructing
`InterestRecord` instances from the `council_stage_interest` table.  It exposes
`build_interest_record`, which normalises raw database values, parses embedded
debug JSON, and produces the Pydantic object consumed by the API responses.

Normalisation responsibilities include:

- Coercing numeric scores and boolean flags from loosely typed database columns
- Preserving auxiliary metrics surfaced in the debug payload
- Handling absent or malformed data gracefully while signalling the record’s
  status

By treating interest-score handling as its own subsystem, the rest of the
backend can work with strongly typed objects instead of reimplementing parsing
logic whenever interest metadata is needed.
