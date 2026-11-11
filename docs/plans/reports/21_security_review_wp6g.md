# Security review: WP6-G glasses signal (2026-11-11)

Assessment: PASS, no findings on the diff.
- No new logging in `face_regions.py` or the schema; the only new service line derives the hint from the existing warnings list.
- Field is additive (`default_factory=list`), frontend type optional; nothing rejects a photo on the hint.
- Hint is set only after the `status != "ok"` early return, so non-ok responses default to [].
- Frontend renders a literal string gated on `.includes("glasses")`; no `any`, no `dangerouslySetInnerHTML`.
- Docs carry aggregate numbers only; no home-directory paths or per-image data.
Pre-existing, already documented, unchanged: no auth or rate limiting, no pre-parse body cap (deployment work, D6).
