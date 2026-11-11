# Coder Report: Face Segmentation Phase 1, Iteration 2

Agent: `coder`. This iteration fixes the Evaluator's iteration-1 findings. See `04_evaluation_iter1.md`.

**Final counts:**

- Backend: `pytest` **174 passed, 0 xfail**. That is 164 + 4 flipped + 6 new.
- Frontend: typecheck clean, build OK.

The orchestrator independently confirmed the 174 passing tests and the mirrored-fixture behaviour.

## Fixes

### BUG-1: mirrored selfies collapsed left and right

- **Side-agnostic classes.** The model's paired classes (brow 2/3, eye 4/5, ear 7/8) now map to side-agnostic internal ids `Unsided.BROW`, `Unsided.EYE` and `Unsided.EAR` (values 200–202). These sit outside the `Region` enum and never appear in a response.
- **Side assignment.** New module `services/segmentation/sides.py`.
  - Each pixel gets the side of its **nearest landmark of that feature**, using the MediaPipe convention: subject-right is the side containing landmarks 33, 133 and 468.
  - It does not depend on head roll, and it splits touching brows.
  - It is processed in chunks, so memory stays bounded.
- **Hard iris rule in `alignment_check`.** Both irises must fall in their eye region. If either fails, the check gives `passed=False` and a `parser_landmark_mismatch` warning, and that side's `eye_*` and `under_eye_*` zones become `low_confidence`.
- **Backstop on eye and brow zones.** A zone is never `ok` if more than 20% of its pixels lie inside the other side's landmark polygon, or if its bbox is wider than 1.5× its own landmark polygon.
- **Convention documented** in the README, `docs/face_segmentation.md`, the schema field descriptions and the TypeScript type comment. On a mirrored photo, sides follow the image geometry. Mirroring is not auto-detected.

### BUG-2: unreadable weights crashed startup

`read_bytes()` is now wrapped. On `OSError` it raises `FaceParserUnavailableError` with an actionable message and logs only the exception type.

### BUG-3: a failed startup leaked the landmarker

`lifespan` closes the landmark service in a single `finally` on any exit. A generic `Exception` during parser creation is also caught, so the app still starts: landmarks keep working and `/segmentation` returns 503.

### Docs

- **Pre-deployment blockers** section added: F1, F2 and the licence.
- **Lips:** a note that the `lip_full` polygon is the outer contour while its mask excludes the mouth opening, plus a description of the lip layers and `chin` in setting powder.
- **Known limitations** added:
  - half-face recall;
  - the roughly 5 px inner-canthus gap;
  - a truncated landmark result returns a generic 500;
  - Pillow's own header guard at about 179 MP.

## Mirrored fixture, before and after

| | Before | After |
|---|---|---|
| Iris 468 lands in | `eye_l` | `eye_r` |
| Iris 473 lands in | `eye_l` | `eye_l` |
| `eye_l` | ok, 115 px wide | ok, 29 px |
| `eye_r` | low_confidence, 16 px | ok, 31 px |
| `brow_l` | ok, 151 px | ok, 66 px |
| `brow_r` | ok, 59 px | ok, 59 px |
| `alignment_check` | passed, 0.833 | passed, 1.0 |

On the original fixture, nothing changed: every zone status, bbox width and centre, and the alignment result are identical to before.

## Test changes

- **Flipped:** the four xfail tests are now normal tests. The two startup tests skip when run as root.
- **New in `test_phase1_independent.py`:**
  - mirrored eyes and brows each on their own side of the midline, with each eye covering one iris;
  - original fixture unchanged against the iteration-1 baseline;
  - exactly one wrong iris gives `passed=False`.
- **New in `test_application_zones.py`:**
  - `assign_sides` at 0°, +30° and −40° roll;
  - a mask covering both eyes is never `ok`.
- **Changed:**
  - `test_face_parser.py` now checks the side-agnostic grouping.
  - The Tester's invariant `passed == (fraction >= 0.6)` is now `passed == (both irises in eyes and fraction >= 0.6)`, computed independently from the label map.

## Files

- **Created:** `backend/app/services/segmentation/sides.py`.
- **Backend source modified:**
  - `services/segmentation/regions.py`
  - `services/segmentation/face_parser.py`
  - `services/application_zones/derive.py`
  - `services/face_regions.py`
  - `main.py`
  - `schemas/face_segmentation.py`
- **Backend tests modified:**
  - `tests/test_phase1_independent.py`
  - `tests/test_face_parser.py`
  - `tests/test_application_zones.py`
- **Frontend:** `types/faceSegmentation.ts` (comment only).
- **Docs:** `README.md`, `docs/face_segmentation.md`.

## Browser verification

Headless Chrome, real backend, production frontend build. Screenshots are in the scratchpad `coder_iter2/` only.

The Coder reported 9/9 PASS, judged by canvas hash changes. The Coder did not look at the screenshots.

The orchestrator reviewed them and found two issues for the Tester to verify:

1. **Zone toggle not visible.** `03_zone_cheek_l.png` shows the `eye_l` *region* checked and the stage tree collapsed. The left-cheek zone overlay can't be visually confirmed at screenshot scale.
2. **Stale divider line.** `07_two.png`, the `multiple_faces_detected` state, correctly shows "Please upload a photo with only one face.", but a white wipe-compare divider line is still drawn over the image. This is stale compare state from an earlier step.

The only console error was a favicon 404.

## Known limitations

- On mirrored photos, sides follow the image geometry. This is documented, and there is no capture flag yet.
- Ear side assignment is a coarse heuristic. Ears are optional.
- F1 and F2 remain blockers before deployment.
