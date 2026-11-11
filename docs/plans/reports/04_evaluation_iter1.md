# Evaluation: Face Segmentation Phase 1, Iteration 1

Agent: `evaluator`.

**Status: FAIL** (iteration 1 of 3). There are 3 correctness defects: 1 high and 2 medium.

Evidence the evaluator re-ran itself:

- `pytest`: 164 passed, 4 xfailed. These are exactly the Tester's documented bugs.
- `npm run typecheck`: clean.
- Code read: `alignment_check` and `lifespan`.

## Acceptance criteria

| # | Criterion | Verdict |
|---|---|---|
| 1 | Model, classes, mapping and license warning documented | Met |
| 2 | Weights load once; missing or corrupt weights → 503 with landmarks still working; SHA-256 pinned | **Partly met.** Unreadable weights crash startup (BUG-2) |
| 3 | Mapping verified; alignment passes on the fixture | Met on the fixture. The check is too lax (BUG-1) |
| 4 | 1 px alignment | Met |
| 5 | Masks valid; no misleading `ok` | **Not met** (BUG-1): a mask covering both eyes is `ok` |
| 6 | Left and right distinguishable | **Not met for mirrored input** (BUG-1) |
| 7 | Honest zone statuses | **Violated** by BUG-1. Otherwise met |
| 8 | Unsupported items documented and absent | Met |
| 9 | Invalid input handled | Met, except the BUG-2 startup path |
| 10 | Frontend | Probably met. The evidence is thin, so a fresh browser check is required |
| 11 | Tests, typecheck and build | Met |
| 12 | Measurements recorded | Met |
| 13 | Security | Met for the local prototype |

## Required changes

### BUG-1: mirrored selfies collapse left and right

The candidate fix is sound and is adopted with refinements.

**Definition.** Left/right follows the MediaPipe landmark convention: subject-right is the side containing landmarks 33, 133 and 468. A mirrored photo cannot be detected from pixels. On a mirrored photo, the side is therefore the image-geometry side, which is the reverse of the person's real side. This must be documented. Do not try to auto-detect mirroring.

**Changes:**

1. Treat the model's paired classes (eye, brow and optionally ear) as side-agnostic. Merge the left and right classes. Model class ids must stay inside `services/segmentation/`.
2. Assign the side from landmark geometry, using the roll-aware midline through landmarks 10 and 152 and the `FaceFrame` axis `u`. Prefer connected-component or nearest-landmark assignment over a raw per-pixel split. Do this in post-processing, not in the model mapping.
3. In `alignment_check`, iris placement becomes a **hard** requirement: both irises must be in an eye class, otherwise `passed=False` regardless of the fraction. The other checks stay in the 0.6 fraction. On failure, emit `parser_landmark_mismatch` and degrade the affected zones.
4. As a backstop, an eye or brow mask may not be `ok` if more than about 20% of its pixels fall inside the other side's landmark polygon, or if it is wider than about 1.5× the landmark polygon width.
5. Document the convention in the README (around line 149), `docs/face_segmentation.md` and the schema field descriptions.
6. Tests:
   - Flip the two mirrored xfail tests into normal passing tests, and drop `_xfail` from their names.
   - Add mirrored-fixture tests for eyes and for brows.
   - Add a test that the original fixture result is unchanged.
   - Add a unit test where exactly one iris is wrong, which must give `passed=False`.

**Trade-off.** The model's own left/right signal is discarded. That is acceptable, because it was shown to be unreliable.

### BUG-2: unreadable weights crash startup

- Wrap `read_bytes()` and the session creation in `try/except OSError` (and onnxruntime errors), and raise `FaceParserError`.
- Log only the exception type.
- Flip the test to a normal test, and guard it for root, where chmod 000 has no effect.

### BUG-3: a failed startup leaks the landmarker

- Restructure `lifespan` so `service.close()` runs on any exception after the service is created: one `try/finally`, or an `ExitStack`.
- Flip the test to a normal test.

### Frontend

- Run one fresh headless-browser check after the fixes, covering:
  - upload and segment;
  - a region toggle and a zone toggle;
  - an opacity change and the compare slider;
  - landmarks shown alongside the overlay;
  - a non-ok message (`no_face_detected` or `multiple_faces_detected`).
- Write screenshots to the scratchpad only.
- Run `npm run build`.

## Answers to the orchestrator's questions

**1. BUG-1 fix.** Sound, with the refinements above. Ear handling is optional.

**2. Security F1.** Defer it, but document it as a pre-deployment blocker together with F2 and the license. A request-level limiter is acceptable as an optional cheap improvement.

**3. Tester observations.** None is required. The README wording on mirrored photos is required as part of BUG-1. Document these:

- The lipstick layer membership, or align it to the plan.
- `lip_full`: the polygon is the outer contour, while the mask excludes the mouth opening.
- Known limitations: half-face recall, the inner-canthus gap, a truncated landmark result producing a 500, and the Pillow header cap.

**4. Scope.**

- Compliant. All 18 Phase 1 zones are present, Phase 2 is absent, and nothing from the exclusions list is implemented.
- Frontend items 1–10 appear to be implemented. The runtime evidence is thin, so the browser check above is required.

## Next steps

1. Coder fixes BUG-1, BUG-2 and BUG-3, and does the browser check.
2. Tester re-runs the flipped tests, the full suite and the mirrored regression.
3. Security Reviewer does a delta review of startup and error handling.
4. Evaluator re-evaluates.

An iteration-2 PASS is expected if the fixes land.
