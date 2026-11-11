# Security Review: Face Segmentation Phase 1

Agent: `security-reviewer`. This was a read-only review.

**Overall assessment: PASS.** There are no critical or high findings for the local prototype. F1 and F2 must be fixed, and rate limiting and authentication added, before any deployment.

## Scope

- The requirements (section 13), the approved plan's Security section and the Coder report.
- Backend:
  - `api/{uploads,face_segmentation,facial_analysis}.py`
  - `services/image_loading.py`
  - `services/segmentation/face_parser.py`
  - `services/face_regions.py`
  - `main.py`, `config.py`
  - `scripts/download_face_parser.py`
  - `requirements.txt`, `.env.example`, `.gitignore`
- Frontend:
  - `lib/{api,segmentation}.ts`
  - `components/SegmentationControls.tsx`
  - `app/landmark-test/page.tsx`
- Docs, and a check of the repository tree for images.

Commands run:

- `pip-audit -r backend/requirements.txt` reported **no known vulnerabilities**.
- A bounded local load test on port 8002, using a synthetic 24.2 MP JPEG (4400×5500, 777 KB) upscaled from the fixture. The server was stopped afterwards and scratch files were deleted.

## Findings

### F1: Unbounded concurrent in-flight memory. The semaphore covers only inference. Severity: MEDIUM now, HIGH before deployment.

**Location:** `api/uploads.py`, `services/image_loading.py`, `services/face_regions.py`, and `services/segmentation/face_parser.py` lines 190-194.

`BoundedSemaphore(2)` is held only inside `OnnxFaceParser.parse`. These steps all run outside it, in the FastAPI threadpool of about 40 threads:

- the upload read;
- decode, EXIF transpose and RGB conversion (about 3 copies of up to 75 MB each);
- MediaPipe;
- zone derivation;
- PNG encoding.

Measured process RSS with concurrent 24.2 MP requests (idle baseline about 286 MB):

| Concurrent requests | Peak RSS | Result |
|---|---|---|
| 1 | 1.22 GB | all 200 |
| 3 | 1.74 GB | all 200 |
| 6 | 1.93 GB | all 200 |
| 12 | 3.07 GB | all 200, 4.3 s latency vs 0.9 s |

- The 5 s queue timeout never fired, because each request holds a slot for only about 100 ms.
- Memory is bounded by the thread count, not by the concurrency setting.
- RSS stayed at 2.45 GB after the runs. This is allocator retention.
- Extrapolating, about 40 parallel 25 MP uploads would reach roughly 8-10 GB. This was not run.

**Mitigation:**

- Put a request-level limiter around the whole heavy path, from decode to encode.
- Downscale after decoding when the longest side is over about 4096 px, or lower `MAX_IMAGE_PIXELS` to about 12 MP.
- Set an explicit threadpool or worker limit.
- Add rate limiting before deployment.

### F2: The upload byte limit is applied after the whole body has been spooled. Severity: LOW now, MEDIUM before deployment.

**Location:** `api/uploads.py` lines 28-33.

- Starlette's multipart parser stores the full body in a `SpooledTemporaryFile` (on disk above 1 MB) before the handler runs.
- A 150 MB upload was fully received, then rejected with 413 after 0.22 s.
- The declared type is also checked only after the full body is received.
- `Content-Length` is not checked.
- There was no RSS blow-up. The cost is temp-disk space and bandwidth.

**Mitigation:** a body cap in ASGI middleware or at a reverse proxy, plus `uvicorn --limit-concurrency`.

### F3: The declared Content-Type is echoed back in the 415 error. Severity: INFORMATIONAL.

**Location:** `api/uploads.py` line 24.

The response is `application/json`, and the frontend renders it as React-escaped text. Optionally, stop echoing it.

### F4: The weights hash is trust-on-first-use. Severity: LOW, already documented.

The upstream publishes no checksums. Cross-check the hash against an independent source when one becomes available.

### F5: The download script has no size cap or timeout. Severity: LOW.

**Location:** `scripts/download_face_parser.py` lines 58-71.

- `urlretrieve` has no size cap and no timeout.
- The release tag is mutable, but the hash check would detect tampering.
- The `.part` file is cleaned up correctly, and the final file is moved into place only after verification.

The same weakness exists in `download_model.py`, which was out of scope.

### F6: The frontend has no size sanity cap on decoded server PNGs. Severity: INFORMATIONAL.

The server is trusted at this stage.

## Positive controls (verified)

- **Format checking:** the decoder allow-list uses the real format. A GIF declared as `image/png` returns a generic 400.
- **Pixel cap:** checked from the header before decoding. Bomb, `OSError`, `SyntaxError`, `ValueError` and `EOFError` all map to 400.
- **Byte limit:** 413 fires correctly.
- **EXIF:** orientation is applied. Metadata is neither returned nor stored.
- **Model integrity:**
  - The weights are hashed in memory, and the session is built from the same bytes, so there is no TOCTOU window.
  - The path comes from settings only.
  - On a mismatch the parser is disabled: `/segmentation` returns 503 and `/landmarks` keeps working.
  - Warm-up validates the output layout.
- **Busy handling:** 503 with `Retry-After`.
- **Error and log hygiene:**
  - 500 responses are generic, and exceptions are raised `from None`.
  - Logs hold exception types and fixed strings only, with no image, filename, landmark or mask data.
  - The server log contains only startup noise.
- **No persistence:** the application writes nothing to disk. Tests use synthetic data or `tmp_path`. The only image in the tree is `tests/fixtures/portrait.jpg`. Weights and `.env` files are gitignored.
- **CORS:**
  - Allowed origin is `http://localhost:3000`, with GET and POST only.
  - A disallowed origin receives no `Access-Control-Allow-Origin` header.
  - Credentials are not enabled.
- **Frontend:**
  - There is no `dangerouslySetInnerHTML` or `innerHTML`.
  - Server strings are React-escaped.
  - PNGs are decoded with `createImageBitmap(Blob)`, and `bitmap.close()` runs in a `finally` block.
  - Object URLs are revoked.
  - Label values are validated.
- **Dependencies:** exact pins, and `pip-audit` is clean.
- **License warning:** shown in every `/segmentation` response, in the UI, in the docs and in `.env.example`.

## Missing protections

- A request-level limiter for the heavy path, and a stricter pixel policy (F1).
- A pre-parse body cap (F2).
- Rate limiting and authentication (deferred by design).
- Per-request timeouts.
- `pip-audit` in CI.
- A frontend `npm audit`. This was not run.

## Deferred risks (acceptable for a local prototype)

- Rate limiting, authentication and ownership.
- Biometric and privacy legal review.
- TLS and deployment settings. The server is not bound to `0.0.0.0`.
- Weights provenance (F4, F5).
- Licensing, which is not a security finding and is surfaced in the product.
