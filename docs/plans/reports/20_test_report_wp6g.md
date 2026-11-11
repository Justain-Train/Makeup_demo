# Tester report: WP6-G glasses signal (2026-11-11)

Commands: `backend/.venv/bin/python -m pytest -q --tb=short` and `frontend: npx tsc --noEmit`.

First run: 356 passed, 1 failed. tsc exit 0.
- Criteria 1-5, 7-10 PASS (glasses >= 1 % gives warning + `retake_suggested == ["glasses"]`; < 1 % gives neither;
  no_face / multiple_faces give []; `GLASSES_MIN_AREA_FRACTION` still 0.01; only `face_regions.py` and the schema changed
  under backend/app; prompt text exact and gated on the hint; no new logging; docs carry all measurement numbers).
- Criterion 6 FAIL: `test_phase3_independent.py::test_report_regeneration_needs_no_network_and_matches_committed_json`.
  The regenerated Markdown differs from the committed report in exactly one line: the "current tree at report time"
  source fingerprint, which changes with any edit under backend/. The JSON (pilot-run fingerprint only) matched.
- Coder's uncertainty confirmed: `assign_sides` rewrites only the unsided EYE/BROW/EAR ids, so the GLASSES stub holds.

Fix loop 1 (orchestrator): the test now compares the Markdown with that one report-time line removed. No production
code changed. Re-run: **357 passed, 1 warning (StarletteDeprecationWarning, pre-existing)**.

Not covered: rendering of the note in a browser (static inspection only).
