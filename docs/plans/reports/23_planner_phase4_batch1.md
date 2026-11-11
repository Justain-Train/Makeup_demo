# Planner report: Phase 4 Batch 1, lip/mouth/chin zones (2026-11-11)

Phase 4 was unfrozen by the user on 2026-11-11. The Phase 3 deferred items are listed in `CLAUDE.md`.

No schema or route change is needed. `Zone.group` is a free `str`, and new `ZONE_SPECS` entries flow through
`derive_zones` -> `_encode_zone` -> `segment_face` unchanged. The API stays additive.

Batches (the eyelid crease is excluded throughout, because it cannot be derived from the landmarks or the current classes):
1. Lips/mouth/chin: `chin_center`, `under_lower_lip`, `mouth_surround`, `cupid_bow`, `lip_corner_l/r`.
2. Forehead center/perimeter, temples, outer jawline.
3. Eyes/brows: under-eye inner/outer, eye corners, eyelids, lash lines, brow head/arch/tail, brow bone.
4. Cheeks: apples, upper/outer cheeks, cheekbones, contour hollows. Also wire in the `blush`, `bronzer`, `highlighter`
   and `contour` stages.
5. Nose: bridge, tip, sides, nostrils.

Batch 1 design:
- `chin_center`, `under_lower_lip` and `mouth_surround` are landmark bands or dilations clipped to parser skin.
- `cupid_bow` is a geometric stroke through the upper-lip peaks.
- `lip_corner_l/r` are small regions around landmarks 61 and 291, inside `lip_full`, validated against the parser lip classes.

Target files: `application_zones/catalog.py`, `application_zones/derive.py`, `application_zones/frame.py`
(MIRROR_PAIRS, if needed), `tests/test_application_zones.py`, `tests/test_phase1_independent.py`,
`docs/face_segmentation.md`, `docs/ROADMAP.md`.

Acceptance criteria:
1. The suite passes. The baseline was 357, measured before editing.
2. The new ids are in the response. The 18 old zones are unchanged.
3. Skin zones lie inside the clip region and do not touch eye, lip or nose. The lip zones are confined to the lip area.
   `mouth_surround` excludes `lip_full`.
4. Sides are correct under roll.
5. With a background-only map, the zones are `unavailable` with a reason.
6. The stage and catalog tests pass, and no model class ids appear outside `services/segmentation/`.
7. Phase 1/2, `services/segmentation/`, `face_regions.py` and the schemas are unmodified.
8. No eyelid-crease zone exists.
9. Payload size and timings are reported.
