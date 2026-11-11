# Security review: Phase 4 Batch 3b (2026-11-11)

Initial review: the reviewer read all files. The code has no findings:
- no logging;
- reason strings hold only ratios and constants;
- sizes scale with ipd;
- the guards are demote-only;
- only `Region` names are used;
- no global state;
- the golden JSON holds only hashes and counts.

MEDIUM: image-derived bbox literals at `test_phase1_independent.py:861`. The orchestrator found the identical line in
the Nov 10 copy of that file, so it predates this batch. It belongs to the user-deferred `portrait.jpg` provenance item.

Notes:
- A landmark zone can stay `ok` while glasses cover up to 20% of it.
- The full-grid `np.indices` used about 100-250 MB at 25 MP. This was fixed in loop 1.

Delta review: PASS, no findings. Memory is bounded by the bounding box, and empty rasters are handled.
