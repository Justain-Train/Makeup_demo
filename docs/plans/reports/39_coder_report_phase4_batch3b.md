# Coder report: Phase 4 Batch 3b (2026-11-11)

Modified files:
- `catalog.py`: 10 specs and the stage lists.
- `derive.py`: constants, an optional `parent` on `_eye_area_guard`, and `_brow_eyelid_zones`. A brow parent gives a
  reason that says "brow".
- Tests:
  - new `test_phase4_batch3b_verification.py`;
  - `test_phase4_batch3a_iris.py`: added an eyelid iris case;
  - `test_phase1_independent.py`, `test_application_zones.py` and `test_phase4_batch3a_verification.py`: id sets, an
    explicit `eyelid_crease` check (`eyelid_lower` stays banned through `PHASE2_ZONES`), exact highlighter lists, and
    a brow-part branch in the background-only test.
- Docs.

Fix loop 1:
- The 3a iris-predicate expected set now includes `eyelid_upper`.
- `_window` bbox helper: the brow-part and under-eye splits now use it instead of full-grid `np.indices`, which was
  security's memory note.
- Stale highlighter docs and docstrings fixed.
