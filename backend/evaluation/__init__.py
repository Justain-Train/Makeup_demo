"""Ground-truth evaluation harness for the face-parsing pipeline (roadmap Phase 3).

Nothing here is imported by `app/`. Run from backend/:  python -m evaluation.cli --help

Privacy: image-derived data (pixels, annotations, per-image rows) never goes into the repo or into
logs. Only opaque dataset ids and aggregates are written under docs/evaluation/.
"""
