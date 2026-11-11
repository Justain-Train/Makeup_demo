# Tester report: Phase 4 Batch 3a (2026-11-11)

Command: `backend/.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short`.

| Run | Result |
|---|---|
| First run | 414 passed, 1 failed |
| After fix loop 1 (test only) | **415 passed, 0 failed** |

First-run failure: `test_ac5_landmark_zone_polygons_are_in_image_and_model_zones_have_none`.
- The test's `eye_`/`brow_` prefix check caught the new landmark-built `eye_corner_*` zones, which correctly carry polygons.
- The prefix had matched exactly {eye_r, eye_l, brow_r, brow_l} before this batch, and the explicit set is
  equivalent, so nothing was loosened.
- Caveat: future `eye_*`/`brow_*` polygon zones are not caught by name.

Results per criterion:
- Golden: `golden_batch3.json` was captured by the orchestrator at 01:14:14 from the pre-edit code. The first edit
  was at 01:15. It holds hashes, counts and ratios only. The test loads the file and does not recompute. 5 scenarios pass.
- Partition: inner ∪ outer = `under_eye_x`, and the two are disjoint, at 0° and ±20°.
- Containment and sides: pass. Inner is closer to the midline than outer.
- Negative tests (each guarded by a clean-map precondition):
  - glasses over both eyes: all 12 zones not `ok`;
  - glasses over one eye: only that side changes;
  - 0.3 ipd eye shift: the corners are not `ok`;
  - background only: the halves are `unavailable`.
- Iris demotion, behavioural: new `test_phase4_batch3a_iris.py`, 3 tests. They run the real API with `iris_checks`
  monkeypatched. The failing side's eye-group zones are all not `ok`, and the other side is unchanged.
- `face_regions.py`: the orchestrator diffed it against three identical Nov 10 copies. Only 3 lines differ: the iris
  predicate, plus the two WP6-G `retake_suggested` lines (report 20).

Fixture (`portrait.jpg`): all 12 new zones are `ok`.

| Zones | Area | Validation value |
|---|---|---|
| Under-eye halves | 319-324 px | coverage 0.93-0.95 |
| Corners | 50-61 px | tolerance coverage 0.84-1.0 |
| Lash lines | 190-207 px | — |

Payload is 94.3 kB (was 86 kB). Zones take 33.7 ms (was 23.8) and encode takes 5.3 ms.
