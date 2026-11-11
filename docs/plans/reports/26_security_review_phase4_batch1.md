# Security review: Phase 4 Batch 1 (2026-11-11)

Initial review: PASS, no findings. Files reviewed: `catalog.py` and `derive.py`.
- No logging, print or I/O. Masks, polygons and landmarks stay in memory.
- Only the canonical `Region` enum is used. There are no model class ids.
- All new masks use the existing label-map grid. Radii scale with ipd, and ipd is derived from landmarks on an image
  that has already passed the Phase 1 caps.
- No mutable global state.
- The reason strings contain only percentages, IoU values and constants.

The orchestrator checked the docs (`face_segmentation.md`, `ROADMAP.md`): they contain no paths, emails, coordinates
or image-derived data.

Fix-loop delta (`_lip_corner` tolerance coverage): PASS, no findings. The new reason string contains only the percentage
and the constant 0.05. The dilation radius is `round(0.05·ipd)`, which is bounded.

Optional hardening, not blocking: clamp dilation radii to `max(shape)`.
