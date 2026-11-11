# Evaluation: Face Segmentation, Iteration 2

Agent: `evaluator`.

**Status: FAIL** (iteration 2 of 3). One small frontend defect blocks PASS, and the fix is about 2 lines.

Evidence re-run by the evaluator:

- `pytest`: 191 passed, 0 xfail, 0 skipped.
- `npm run typecheck`: clean.

## Iteration-1 required changes

| Change | Verdict | Evidence |
|---|---|---|
| BUG-1: mirrored selfie | **Satisfied** | Classes are side-agnostic, and side is assigned from the nearest landmark. Iris placement is a hard check, with a backstop of 20% overlap and 1.5× width. On the mirrored fixture: `eye_l` 29 px, `eye_r` 31 px, alignment 1.0. The original fixture is unchanged. The tests pass, the 14 mirrored robustness variants pass, and the convention is documented. |
| BUG-2: unreadable weights | **Satisfied** | 503 and 200 returned as required. The test ran rather than skipping. |
| BUG-3: landmarker leak | **Satisfied** | `close()` is in a `finally`. A spy test checks it is called exactly once. |
| Docs | **Satisfied** | Mirroring convention, blockers (F1, F2, licence), `lip_full` note, known limitations. |
| Fresh browser verification | **Satisfied**, and it found a defect | Screenshots were inspected visually. All checks pass except the stale state. |

Acceptance criteria 1–13 are met. Criteria 5–7 are now met. Criterion 10 is met except for the divider.

## Blocking defect: stale compare divider

The Tester's diagnosis is confirmed:

- `page.tsx:172` passes `compare={compareEnabled ? compare : null}` without checking whether an overlay exists.
- `FaceLandmarkViewer.tsx:106-109` draws the divider whenever `compare !== null`.
- The compare checkbox only appears when `seg?.status === "ok"` (`page.tsx:177`).
- `onFileChange` (`page.tsx:64-72`) never resets the compare state.

**Why it blocks:**

- The divider can be read as a comparison result over an error message or an unsegmented photo.
- The user cannot clear it once that happens.
- The requirements stress that inaccuracies must be easy to spot and that results must not mislead.

## Required changes (Coder)

1. `page.tsx:172`: `compare={overlay && compareEnabled ? compare : null}`.
2. `onFileChange`: add `setCompareEnabled(false)` and `setCompare(0.5)`.
3. Optional defensive guard in `FaceLandmarkViewer.tsx:106`: `if (overlay && compare !== null)`.
4. Re-check in the browser:
   1. Segment a photo, then enable compare.
   2. Upload a blank image, then a two-face image.
   3. Confirm that no divider appears and that the message is shown.
   4. Then run `npm run typecheck`, `npm run build` and a `pytest` sanity run.

## Tester observations

All of these are deferred:

- **Zone outline:** not distinct at 55% opacity. Minor.
- **Angular zone polygons:** visual tuning, deferred.
- **±35° roll:** alignment fails, which is a documented and honest degradation.

## Outstanding (not failures)

- **Ground truth:** the roadmap requires a ground-truth quality evaluation before Phase 3 is complete. This is a separate follow-up feature.
- **Known limitations:**
  - mirrored photos follow image geometry;
  - alignment fails at ±35° roll;
  - half-face recall;
  - the gap of about 5 px at the inner canthus.
- **Before deployment:** F1, F2 and the licence.
- **Docs:** rename the phases to the roadmap numbering.

## Recommendation

PASS is expected at iteration 3 once the divider fix and a short browser re-check are done. No further backend review is needed beyond a `pytest` sanity run. The backend is a sound Phase 3 deliverable, apart from the ground-truth evaluation.
