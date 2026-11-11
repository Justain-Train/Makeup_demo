# Coder report: Phase 4 Batch 2 (2026-11-11)

Modified files:
- `catalog.py`: added the 6 specs and `_FOREHEAD_TEMPLE_JAW`, which goes in the primer, foundation and setting_spray
  stages.
- `derive.py`:
  - named untuned constants;
  - `_forehead_parts` and `_forehead_polygon`, with `_forehead` refactored onto them;
  - `_forehead_center`: a quad at landmarks 10 and 9, intersected with the forehead polygon;
  - `_forehead_perimeter`: the forehead polygon minus the arc shifted down by 0.12 ipd;
  - `_temple`: a quad through landmarks 70, 162, 127 and 33 with offsets;
  - `_jawline_outer`: a 0.07 ipd strip inward from the jaw path. Fix loop 1 added a zero-norm guard.
- `test_phase1_independent.py`: the ids moved to `PHASE1_ZONES`, and "temple" was dropped from the banned tokens.
- `test_application_zones.py`: added `BATCH2_SKIN_ZONES`.
- New file `test_phase4_batch2_verification.py`.
- `face_segmentation.md` and `ROADMAP.md`.

`frame.py` is unchanged, because all the indices used are already mirrored or on the midline.

Fix loop 1:
- The vacuous 0.3 ipd shift test was replaced by `test_zones_with_low_skin_coverage_are_never_ok`, which carries a
  `checked >= 2` guard.
- Added a jaw-band test.

Fix loop 2:
- `test_forehead_zone_matches_pre_refactor_golden`: 6 cases, exact equality.
- The jaw-band test is now direct: distance to the jaw path ≤ 0.07·ipd + 1.5 px, and at most 1% outside `jawline`.
