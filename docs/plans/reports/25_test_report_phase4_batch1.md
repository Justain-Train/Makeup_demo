# Tester report: Phase 4 Batch 1 (2026-11-11)

Command: `backend/.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short`.

First run: 360 passed, 4 failed. All four failures had one cause. On the fixture, `lip_corner_l` came out
`low_confidence`, with an IoU of 0.17 against the 0.40 lip gate.
- Orchestrator diagnosis: inside the left corner disc the landmark lip edge sits 2-4 px outside the parser lip edge, and
  the parser has only 27 lip px in that disc.
- The table gives the share of the corner covered by parser lip at each tolerance:

  | Corner | 0 px | 2 px | 4 px (about 0.05 ipd) |
  |---|---|---|---|
  | Right | 0.50 | 0.87 | 1.00 |
  | Left | 0.18 | 0.49 | 0.79 |

- The coder had loosened the `lip_liner` assert to check only `[0]`. This was flagged and later restored.

After fix loop 1: 371 passed. The new `tests/test_phase4_batch1_verification.py` adds 6 tests, all passing:
- the skin zones lie inside SKIN and do not touch eye, lip or nose;
- the lip zones lie inside `lip_full` dilated by 0.15 ipd;
- `mouth_surround` does not overlap `lip_full`;
- corner sides are correct at ±20° roll;
- with a background-only map, the zones are `unavailable` with a reason and no mask.

After fix loop 2: **372 passed, 0 failed**.
- With the parser lip shifted by 0.3 ipd, both corners come out `low_confidence` with confidence 0.0. `z.confidence`
  equals the tolerance coverage (`derive.py:284`).
- The setting_spray `_MOUTH_CHIN` addition was made later, and the suite was re-run: 372 passed.

Fixture results (`portrait.jpg`). All six new zones are `ok`:

| Zone | Area (px) | Notes |
|---|---|---|
| `chin_center` | 719 | |
| `under_lower_lip` | 421 | |
| `cupid_bow` | 395 | |
| `lip_corner_l` | 98 | coverage 0.786, strict IoU 0.168 |
| `lip_corner_r` | 105 | coverage 1.0, strict IoU 0.46 |
| `mouth_surround` | 4594 | |

The user checked the left corner visually and judged its placement correct.

Payload: 79.6 kB, up from 74 kB for 18 zones. Timings: zones 19.95 ms, encode 4.85 ms, total 152 ms.

Criterion 7 was confirmed by the orchestrator:
- The `.pyc` caches for `face_regions.py` (11,153 B) and `schemas/face_segmentation.py` (4,830 B) were written at
  00:05:09 and match the current source mtime and size.
- Batch 1 began at 00:42.
- The 00:04 edits are WP6-G work (report 20).
