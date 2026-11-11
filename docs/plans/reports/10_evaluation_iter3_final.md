# Final Evaluation: Face Segmentation, Iteration 3

Agent: `evaluator`.

**Status: PASS.** The divider fix was not verified at runtime, because the user waived re-testing. That is a stated residual risk.

## Code trace

Every path that sets `seg`, `overlay` or `compare` was traced. This was code inspection only.

### When `overlay` is null

`overlay` is a `useMemo` (`page.tsx:52-55`) that is non-null only when both `seg` and `decoded` exist. `decoded` is set only when `seg.status === "ok"` (`page.tsx:34-48`).

So `overlay` is null in all of these states:

- no face detected;
- multiple faces detected;
- `onAnalyze`, which calls `setSeg(null)`;
- the error path in `onSegment`, which calls `setSeg(null)`;
- right after a new file is chosen;
- while decoding is still pending.

### The guards

- **Page:** `compare={overlay && compareEnabled ? compare : null}` (`page.tsx:174`). In every null-overlay state above, the viewer receives `null`.
- **Viewer:** the divider is drawn only when `overlay && compare !== null` (`FaceLandmarkViewer.tsx:106`). This is a second, independent guard. The clip sits inside `if (overlay)`.

### State reset and the checkbox

- **New file:** `onFileChange` calls `setCompareEnabled(false)` and `setCompare(0.5)` (`page.tsx:70-71`).
- **No stuck state:** the checkbox renders only when `seg?.status === "ok"`, so compare can no longer be left on with no way to turn it off.
- **Re-segmenting the same file:** `compareEnabled` stays true. That is acceptable: the divider appears only once the new overlay exists, and the checkbox is visible.

### Regression check

With an overlay present, the compare wipe behaves exactly as before. No regression path was found.

## Evidence

- `npm run typecheck`: clean, re-run by the evaluator as a compile check.
- `npm run build`: OK, per the Coder.
- Backend: 191 tests passed at iteration 2. No backend changes since.
- Browser checks a–e passed at iteration 2, except the divider.
- The fix itself was not run (user waiver). It is a two-line conditional change.

## Acceptance criteria

All criteria met at iteration 2 remain met. Criterion 10 (frontend) is now satisfied, based on code inspection.

## Required changes

None.

## Outstanding issues (none blocking)

- **Runtime verification:** skipped by user choice. See the manual check below.
- **Ground-truth quality evaluation:** required by the roadmap before Phase 3 is complete. This is a separate follow-up feature.
- **Pre-deployment blockers:** F1 (concurrent memory), F2 (pre-parse body cap), the CelebAMask-HQ licence, rate limiting and auth, and a biometric legal review.
- **Known limitations:**
  - mirrored photos follow image geometry;
  - alignment fails at ±35° roll;
  - half-face recall;
  - a gap of about 5 px at the inner canthus.
- **Deferred:**
  - the zone outline is not distinct at 55% opacity;
  - angular zone polygons need visual tuning.
- **Docs:** rename the phases to the roadmap numbering.

## Recommendation

The feature is ready for human approval as the **Phase 3 segmentation deliverable**. Phase 3 is not complete until the ground-truth evaluation is done.

## Manual check (about 1 minute)

1. Segment a face photo and tick Compare. The divider and the wipe should appear.
2. Choose a blank image and segment it. The no-face message should show, with **no** divider.
3. Choose a two-face image and segment it. The multiple-faces message should show, with **no** divider and no compare checkbox.
4. Segment a good photo again. Compare should start unticked, at 0.5.

If step 2 or 3 shows a divider, reopen the issue at `frontend/app/landmark-test/page.tsx:174`.
