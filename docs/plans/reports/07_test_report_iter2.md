# Test Report: Face Segmentation, Iteration 2

Agent: `tester`.

**Overall result: FAIL.** One defect remains: a low-severity frontend bug. All backend fixes pass.

## Summary

- **Backend suite:** **191 passed, 0 xfail, 0 skipped**. The Tester appended 17 tests to `test_phase1_independent.py`.
- **Frontend:** typecheck and build pass.
- **UI:** checked in headless Chrome against a real backend. Screenshots were **visually inspected**. They are stored in the scratchpad `tester_iter2/` only.

## Verified passing

### BUG-1: mirrored selfie

- **Mirrored fixture:**
  - Iris 468 lands in `eye_r` and iris 473 in `eye_l`, and alignment passes.
  - Each eye and brow is on its own side of the roll-aware midline.
  - Neither eye covers the other iris.
- **Mirrored robustness:** 14 variants were run (±15°, ±35°, EXIF, scales, crops, grayscale, dark). Statuses are honest and the invariants hold.
- **Original fixture:** unchanged (statuses, bbox centres and widths within 2 px, alignment 1.0, no warnings).
- **One wrong iris:**
  - Unit level: `passed=False`.
  - End to end (both sides, via dependency override):
    - `passed=False` and `parser_landmark_mismatch`.
    - That side's `eye_*` and `under_eye_*` zones are degraded, and the other side stays `ok`.

### BUG-2: unreadable weights

- Unreadable weights now give startup OK, `/segmentation` 503 and `/landmarks` 200. The test ran rather than skipping (uid 501).
- New spy test: on a failed startup, the lifespan itself closes the landmark service exactly once.

### BUG-3: GC hang

- The subprocess test passes in seconds.
- The iteration-1 repro script now completes.
- Caveat: the hang could not be re-created in isolation. The spy test, which checks that `close()` runs, is the stronger evidence.

### Test-file review

- The four former xfail tests are now normal tests. Their assertions are at least as strong as before.
- The changed invariant `passed == (irises_ok and fraction >= 0.6)` is a **tightening**, not a weakening.
- No weakened assertion was found.

### UI (visual)

| Check | Result |
|---|---|
| Left cheek zone draws on the subject's left (image right); unchecking removes it | PASS |
| `eye_l` and `eye_r` regions draw on the correct eyes | PASS |
| Left and right have distinct colours (brows, cheeks, eyes) | PASS |
| Opacity visibly changes; compare wipe shows the original on one side | PASS |
| Landmarks shown alongside segmentation | PASS |
| Non-ok states show the message and no segmentation panel | PASS |
| No stale overlay or divider in non-ok states | **FAIL** |

## Failure: stale compare divider (frontend, low severity, cosmetic but with a stuck state)

**Mechanism:**

- `app/landmark-test/page.tsx` passes `compare={compareEnabled ? compare : null}` whether or not an overlay exists.
- `FaceLandmarkViewer` draws the divider whenever `compare !== null`.
- The compare checkbox lives in `SegmentationControls`, which renders only when `status === "ok"`.

As a result, once compare is enabled:

- the divider is drawn over later no-face and two-face results, and over a newly uploaded, unsegmented photo;
- the user cannot turn it off without reloading the page.

**Reproduce:**

1. Segment the fixture.
2. Enable compare.
3. Then upload a blank image and segment it, upload a two-face composite and segment it, or upload any new photo without segmenting.

**Control:** a fresh page with compare never enabled shows no divider.

**Suggested fix:**

- Pass `compare` only when an overlay exists.
- Reset `compareEnabled` when a new file is chosen.

## Observations (not failures)

- **Rolled heads:** at ±35° roll, alignment fails and the eyes degrade, with a `head_tilt` warning. This existed before this iteration and is the documented limitation.
- **Zone outline:** the 1 px outline is not visually distinct from the fill at 55% opacity. The plan asks for "fill plus a 1–2 px outline", so this is minor.
- **Zone polygons:** coarse and angular. The left-cheek wedge runs over the nasolabial fold. Visual tuning is deferred.
- **Frontend tests:** there is no automated frontend test harness.
- **Data:** only one real face, and no ground truth.

## Environment note

Port 3000 was held by a `next-server` (PID 28452, started 18:07 from `frontend/`) that was not started by the Tester. It was left untouched.
