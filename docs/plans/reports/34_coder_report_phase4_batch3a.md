# Coder report: Phase 4 Batch 3a (2026-11-11)

Modified files:
- `catalog.py`: 12 `_lr` specs; stage wiring; a highlighter TODO comment.
- `derive.py`:
  - untuned constants: `EYE_CORNER_RADIUS` 0.10, `EYE_CORNER_TOLERANCE` 0.05, `MIN_EYE_CORNER_COVERAGE` 0.6,
    `LASH_STROKE` 0.03 and `GLASSES_ZONE_FRACTION` 0.20;
  - `_under_eye_geometric`, extracted from `_under_eye`;
  - new `_disc_polygon`, `_eye_area_guard` (demote-only) and `_eye_zones`.
- `face_regions.py`: only the iris-demotion predicate changed, to `z.spec.group == "eye" and z.spec.id.endswith(f"_{side}")`.
- Tests:
  - new `test_phase4_batch3a_verification.py`;
  - `test_phase1_independent.py`: ids moved to `PHASE1_ZONES` and out of `PHASE2_ZONES` (`lash_line_*` too); explicit
    `eyebrow_eye` list;
  - `test_application_zones.py`: `CLIPPED_TO_SKIN` and `GEOMETRIC_ONLY` updated; the background-only test gained a
    branch for the new zones.
- Docs: `face_segmentation.md` and `ROADMAP.md`.

Fix loop 1 (test only): `test_ac5_..._model_zones_have_none`.
- The prefix rule was replaced with an explicit `MODEL_ONLY_ZONES = {eye_r, eye_l, brow_r, brow_l}`.
- Added `LANDMARK_EYE_ZONES`, whose 8 entries must carry polygons.
