# Security review: Phase 4 Batch 3a (2026-11-11)

Result: no findings.
- Files reviewed: `derive.py`, `face_regions.py`, `catalog.py` and `golden_batch3.json`.
- No logging.
- Reason strings hold percentages and constants only.
- Radii and strokes scale with ipd.
- `_eye_area_guard` only demotes and never adds pixels.
- The warnings and `retake_suggested` do not read zone status, so they are unaffected.
- The golden JSON holds hashes, counts and ratios only, with no coordinates.

The reviewer did not open the tests or the docs. The orchestrator scanned them and found no paths, emails, names,
coordinates or numeric class ids. The only `print` is a pre-existing "DONE" in a subprocess script.

Notes:
- At or below a 20% glasses share, a landmark-only lash line can stay `ok`.
- Stroke width has a floor but no cap. This is bounded by the 25 MP image cap.

Fix loop 1 changed tests only.
