# Evaluation: Phase 4 Batch 3a (2026-11-11)

Verdict: **PASS** after fix loop 1. Criteria 1-10 are met.
- 415 passed, with nothing loosened. The ac5 explicit set is equivalent to the old prefix.
- The goldens were captured before the edit.
- `face_regions.py` changed only the iris predicate.

Non-blocking notes:
- The 20% glasses threshold lets a lash line stay `ok` below that share.
- The name-based model-only check will not catch future `eye_`/`brow_` polygon zones.
