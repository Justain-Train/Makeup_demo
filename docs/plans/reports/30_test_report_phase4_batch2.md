# Tester report: Phase 4 Batch 2 (2026-11-11)

Command: `backend/.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short`.

| Run | Result |
|---|---|
| First run | 384 passed. The 0.3 ipd shift test was vacuous: no zone dropped under 0.5, so its assertion never ran. |
| After fix loop 1 | 386 passed. The low-coverage test checked 7 zones, all `low_confidence`. |
| After fix loop 2 | **392 passed, 0 failed** |

Checks:
- Disjointness: `test_batch2_zones_are_skin_subsets_disjoint_from_non_skin_regions` covers all 6 zones against eye, lip,
  nose, hair, background and ear. Temples are disjoint from `forehead_center`. The nearest eye pixel to a temple is
  13 px away, against a 4.3 px tolerance.
- Outer jaw band, bound 0.07·ipd + 1.5 = 7.56 px:

  | Side | Max distance to jaw path | Outside `jawline` |
  |---|---|---|
  | Left | 6.61 px | 0.1% |
  | Right | 6.86 px | 0.0% |

- Sides were tested at 0° and ±20° roll.

`_forehead` refactor equivalence (orchestrator):
- Three pre-batch copies of `derive.py` were found in an earlier session's scratch directories. They date from
  Nov 10 17:58:08, are 15,008 B, and have identical `_forehead` bodies.
- The old and new code gave identical results in 7 scenarios, comparing status, reason, mask hash, polygon hash,
  confidence, coverage and `hair_fraction`:

  | Scenario | Result |
  |---|---|
  | Synthetic | 8548 px, `ok` |
  | Roll −20° | 8539 px |
  | Roll +20° | 8578 px |
  | Fringe to brows | `unavailable` |
  | Partial hairline | 2692 px, `low_confidence` |
  | Background only | `unavailable` |
  | Real fixture via the API | identical |

- These values are pinned as exact goldens.

Fixture (`portrait.jpg`), all `ok`:

| Zone | Area | Coverage |
|---|---|---|
| `forehead_center` | 2347 px | 1.0 |
| `forehead_perimeter` | 1881 px | 0.96 |
| `temple_l` | 406 px | 0.99 |
| `temple_r` | 545 px | 0.99 |
| `jawline_outer_l` | 865 px | 0.90 |
| `jawline_outer_r` | 935 px | 0.94 |

Payload is 86.1 kB, up from 79.6 kB. Zones take 23.8 ms and encode 5.1 ms. Fix loop 2 changed tests only, so these
results are unchanged.
