# Planner report: Phase 4 Batch 2, forehead/temples/outer jaw (2026-11-11)

Zones: `forehead_center`, `forehead_perimeter`, `temple_l/r`, `jawline_outer_l/r`. All six are clipped to parser skin
and derived from landmarks. The API change is additive.

- These zones have no parser positive mask, so each is validated by the coverage rule: `MIN_COVERAGE` 0.5, and under
  16 px it is `unavailable`. Tolerance-coverage against a parser mask does not apply here.
- A fringe or hair lowers coverage, so the zone becomes `low_confidence` or `unavailable`. Nothing is guessed.
  `hair_fraction` is set on both forehead zones.
- `_forehead` is refactored onto a `_forehead_polygon()` helper. Its behaviour must stay unchanged.
- Stages: primer, foundation and setting_spray only. Contour stays `["jawline"]` until Batch 4.

Target files: `application_zones/catalog.py` and `application_zones/derive.py`; tests in
`test_phase1_independent.py`, `test_application_zones.py` and the new `test_phase4_batch2_verification.py`; docs in
`face_segmentation.md` and `ROADMAP.md`.

Acceptance criteria:
1. The suite passes with no loosening. The baseline is 372.
2. All 6 ids are present, the 24 existing zones are unchanged, and `_forehead` is behaviour-identical.
3. Each new zone is a subset of the clip region. It is disjoint from eye, lip, nose, hair, background and ear.
   `jawline_outer` lies in the outer part of the jaw band, and the temple is disjoint from the central forehead and the eye.
4. Sides are correct at ±20° roll.
5. Real negative tests exist for: fringe, hair on one temple, no jaw skin, and low skin coverage.
6. The zones appear only in their stages.
7. No model class ids appear, and the protected files are untouched.
8. Fixture results, payload and timings are reported.
