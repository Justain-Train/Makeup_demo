# Tester report: Phase 4 Batch 3b (2026-11-11)

First run: 438 passed, 1 failed. The 3a iris-predicate set test lacked `eyelid_upper`. That zone is group "eye", so
the demotion is correct, and the expected set was extended.

After fix loop 1: **439 passed, 0 failed**.

Changed assertions. All are extensions; none is loosened:
- the "eyelid" token check is now an explicit `eyelid_crease` check;
- the highlighter checks use exact equality;
- the background-only test has a branch for the new brow parts.

Checks:
- Goldens: `golden_batch3b.json` was captured by the orchestrator from the pre-edit code at 01:43:52. It covers 20
  ids across 5 synthetic scenarios plus the fixture API, compared by exact equality.
- Partition: exact at 0° and ±20°, ordered head→arch→tail.
- Disjointness: the eyelid, brow bone, eye, brow and brow polygon are pairwise disjoint.
- Sides: correct.
- Guarded negatives:
  - glasses over the eyes;
  - glasses over the brows, with the eyelid as control;
  - a 0.3 ipd brow shift;
  - background only;
  - eyelid iris demotion.
- The default-parent guard is unchanged.
- Stages are exact.

Memory fix: the 10 new zones were snapshotted before the fix (`derive.py` sha 419b6bfc) and after it (c594b2f1)
across 6 scenarios. The output is byte-identical. The estimated peak temporaries fell from about 3.7 MB per call on
the fixture (100-250 MB at 25 MP) to under 100 KB.

Fixture results (52 zones):

| Zone | Status | Area | Coverage |
|---|---|---|---|
| Brow parts | `ok` | 127-237 px | tolerance coverage 1.0 |
| eyelid_upper l / r | `ok` | | 0.70 / 0.74 |
| brow_bone_l | `ok` | | **0.536**, near the 0.5 threshold |
| brow_bone_r | `ok` | | 0.70 |

Payload is 101 kB, up from 94 kB. Zone derivation takes 43.3 ms, up from 33.7 ms.
