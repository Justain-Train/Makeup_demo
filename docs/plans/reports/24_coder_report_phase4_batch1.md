# Coder report: Phase 4 Batch 1 (2026-11-11)

Modified files:
- `backend/app/services/application_zones/catalog.py`:
  - 6 `ZoneSpec`s were added, using `_lr` for the lip corners;
  - `_MOUTH_CHIN` was added to primer, foundation, setting_powder and (after the evaluation) setting_spray;
  - `cupid_bow` and `lip_corner_r/l` were added to the lipstick and lip_liner layers;
  - `bronzer`, `blush` and `highlighter` are still empty;
  - a stale docstring was fixed.
- `backend/app/services/application_zones/derive.py`:
  - new named constants in ipd units;
  - new `_chin_center`, `_under_lower_lip`, `_mouth_surround` and `_lip_corner` methods;
  - `cupid_bow` is a stroke through landmarks [40, 39, 37, 0, 267, 269, 270] with `geometric_only=True`.
- `backend/tests/test_application_zones.py`: tests were updated and new ones added, including the lip-corner tolerance
  test and the 0.3 ipd negative test.
- `backend/tests/test_phase1_independent.py`: the zone sets were extended, and the lip_liner check is a full-list equality.
- `docs/face_segmentation.md` and `docs/ROADMAP.md` were updated.

`frame.py` was not changed, because every index used is already mirrored or lies on the midline.

Fix loop 1: lip-corner agreement was changed from IoU to tolerance coverage.
- The coverage is the share of the corner mask that lies within `LIP_CORNER_TOLERANCE` = 0.05·ipd of the parser lip.
  The minimum is `MIN_LIP_CORNER_COVERAGE` = 0.6.
- The tolerance matches the Phase 3 evaluation's τ = 0.05·IOD.
- The `lip_upper`, `lip_lower` and `lip_full` zones are unchanged.

Fix loop 2: added `test_lip_corner_is_not_ok_when_parser_lip_is_translated_by_0_3_ipd`.
