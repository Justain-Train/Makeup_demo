# Security Delta Review: Face Segmentation Phase 1, Iteration 2

Agent: `security-reviewer`. This is a read-only delta review.

**Overall assessment: PASS.** The changes introduce no new vulnerabilities. F1 and F2 are unchanged, and both are now documented as pre-deployment blockers, as the Evaluator required.

## Scope

**Files read:**

- `main.py`
- `services/segmentation/face_parser.py`
- `services/segmentation/sides.py`
- `api/face_segmentation.py`, read to trace how errors map to HTTP responses
- `docs/face_segmentation.md`
- `README.md`

**Files grepped:** `regions.py`, `face_regions.py`, `derive.py`. `sides.py` has one call site, `face_regions.py:219`.

**Measurement:** one bounded run with a synthetic 24.5 MP JPEG (5531×4429, 1.6 MB), through the in-process TestClient. Scratch files were deleted afterwards and no server was left running.

## Findings

There are no new findings at MEDIUM or above.

### Lifespan (`main.py:36-52`)

**The generic `except Exception` does not mask a security failure.**

- A SHA-256 mismatch raises `FaceParserUnavailableError`, which is a `FaceParserError`.
- That error is caught by the first except clause and logged at ERROR, and the endpoint returns 503.

**The parser is never partially initialised.**

- `app.state.face_parser = None` is set before the `try`.
- The parser is assigned only after `OnnxFaceParser(...)` returns.
- Any failure inside it leaves the parser as `None`, and the route then returns 503.

**Cleanup.** `service.close()` is in a `finally` block.

**Logs.**

- The generic branch logs only `type(exc).__name__`.
- The `FaceParserError` branch logs the message, which includes the server's own configured weights path. It goes to server logs only and contains no file contents.

**Residual (LOW/INFO).** Any failure, including a tampered weights file, degrades to 503 plus an ERROR log line. That is fail-closed, which is correct. Operators should alert on that log line.

### OSError path (`face_parser.py:129-136`)

- **What is logged:** only the exception type.
- **Where the message goes:** the message contains the path, but it is raised only at startup, so it never reaches a client.
- **What clients see:** a fixed 503 string with no path. 500 responses use the generic message and are raised `from None`.
- **INFO:** the 503 text names `scripts/download_face_parser.py`. This is mild layout disclosure, it predates this change, and a deployment build could use a generic message instead.

### `sides.py` resource use

**Memory is bounded by chunking.**

- Each chunk is at most 20000×17×2 float64, about 5 MB.
- The per-call arrays scale with the number of eye, brow and ear pixels. At the 25 MP cap that is at most about 6.7M pixels, roughly 0.3–0.4 GB in the adversarial worst case.
- In practice that pixel count is a few hundred thousand.

**Measured at 24.5 MP:**

| Stage | Time |
|---|---|
| total | 932 ms |
| zones | 425 ms |
| encode | 125 ms |
| inference | 173 ms |
| postprocess | 64 ms |

Peak RSS was about **1.13 GB**, against about 1.22 GB in iteration 1, so side assignment does not worsen F1.

**Abuse resistance.** No index is user-controlled. The landmark indices are constants with a maximum of 473, so there is no out-of-bounds risk.

## Positive controls (retained)

- **Fail-closed parser:** the parser stays `None` on a hash mismatch, a read error, or any exception.
- **Weights integrity:** the bytes are hashed and loaded from the same buffer, so there is no TOCTOU gap.
- **Error hygiene:** error responses are generic, and logs record only exception types.
- **Busy handling:** 503 with `Retry-After`.
- **Cleanup:** the landmarker is closed in a `finally`.

## Blockers documented (confirmed)

`docs/face_segmentation.md` has a "Pre-deployment blockers" section. It lists:

- **F1:** unbounded in-flight memory.
- **F2:** no pre-parse body cap.
- **Rate limiting** and **authentication**.
- **Biometric legal review.**

The licence warning appears at the top of both the docs and the README, and the blockers section references it.

## Missing protections (unchanged)

- A request-level limiter covering decode through encode.
- A body-size cap at the server or proxy.
- Rate limiting and authentication.
- A retention policy. Nothing is persisted today.

## Deferred

- F1 and F2, both required before deployment.
- A cap on working resolution, which would change the API contract.
- Alerting on parser-unavailable log lines.
- Concurrent memory was not measured, and the docs say so.
