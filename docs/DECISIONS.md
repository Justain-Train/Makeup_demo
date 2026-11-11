# Decision log

This log records decisions, approvals, revised criteria and deferred items that cannot be recovered from the code,
tests or git history. Write one short entry when a phase, batch or work package closes. Day-to-day agent hand-off
reports are scratch files and are not kept in the repo (see `docs/ROADMAP.md`, "Important Rules").

Entries up to 2026-11-11 were condensed from the agent reports that used to live in `docs/plans/reports/` (01–43).

---

## Phase 3 — Face segmentation

### Segmentation deliverable (iterations 1–3, ending 2026-11-10): PASS

The first face-parsing pipeline, together with the initial 18 zones. Final suite: 191 tests.

**Iteration 1 bugs, all fixed:**
- **Mirrored selfies collapsed left and right.**
  - The model's paired classes (eye, brow) are now treated as side-agnostic.
  - The side is assigned from the landmark geometry, using the roll-aware midline (10–152) and `FaceFrame.u`.
  - Iris placement is a **hard** requirement of `alignment_check`.
  - Backstop: an eye or brow mask may not be `ok` if more than 20% of it overlaps the other side, or if it is wider than 1.5× the landmark polygon.
- **Unreadable weights crashed startup.** They now return 503 while `/landmarks` keeps working.
- **A failed startup leaked the landmarker.** `close()` now runs in a `finally`.
- **The compare divider stayed on screen** over error and no-face states. It is now guarded on both the page and the viewer.

**Convention decided.** Left and right follow MediaPipe: subject-right is the side holding landmarks 33/133/468. A mirrored photo cannot be detected, so it follows the image geometry. The README documents this. Do not try to auto-detect mirroring.

**Waived by the user.** The divider fix was not re-run in the browser; it was verified by code trace only. To check it by hand:
1. Segment a photo and tick Compare.
2. Load a blank image. No divider should appear.
3. Load a two-face image. No divider should appear.
4. Segment again. Compare should start unticked, at 0.5.

**Known limitations, accepted:**
- Mirrored photos follow the image geometry.
- Alignment fails at ±35° roll.
- Half-face recall is weak.
- There is a gap of about 5 px at the inner canthus.

### Phase 3 completion: EasyPortrait evaluation (WP1–WP5, 2026-11-10): PASS after 3 iterations

**WP4a.** Parser registration fixed (worst residual 0.002 px). Truncated landmark results now get a status instead of an error. `tests/test_phase3_independent.py` must stay in the suite: it catches registration mutants that the coder's own test missed.

**Pilot.**
- **Data:** EasyPortrait v2 test annotations, N=600 stratified, seed 20261110, 598 faces detected.
- **Agreement (IoU):** skin 0.866, lips 0.787, eye 0.717, brow 0.710.
- **Side swaps:** 0.
- **Detection failures:** 0.33%.
- **Files:** report and sample ids in `docs/evaluation/`.
- **Weighting:** the aggregates are over the stratified sample and are not re-weighted to the 6000-image split.

**Provisional targets (D2), against the pilot:**
- Skin IoU 0.90: **missed**. The cause is precision, not neck pixels.
- Lips 0.75, eye 0.65, brow 0.60: met.
- Boundary F ≥ 0.80 at 0.02·IOD: **missed** for lips (0.683), eye (0.751) and brow (0.653).
- Side swap ≤ 1%: met.
- 1–2 MP p95 latency ≤ 300 ms: **provisional**. It was measured on battery. The benchmark p95 is 176 ms at 2 MP.

**Decision rules:**
- **Rule 1 (acceptable):** not met.
- **Rule 2 (post-processing fix):** triggered. Small regions show a consistent best shift of +1 to +2 px downward. The cause is unexplained; it could be the annotation convention or the pipeline. Landmark polygons sit at (−1, −1), so the two signs disagree. Roll ±30° costs 0.12–0.16 IoU and ±45° costs 0.35–0.54.
- **Rule 3 (replace or fine-tune):** not triggered on the headline numbers. Caveat: eye IoU is 0.00 on the 48 glasses images, and 0.168 on the 93 alignment failures.
- **Rule 4 (detector):** not triggered. Detection loss is 4.5% of skin loss.
- **Rule 5:** the model must be replaced before any commercial release.

**Data handling decided:**
- The dataset is used for evaluation only. Only aggregates and the UUID list are committed. No training, no redistribution.
- The cache lives outside the repo, at `~/.cache/makeup_demo/`.
- EasyPortrait licence: a reworked CC BY-SA 4.0 variant. Attribution is in the report.

**To reproduce,** from `backend/` with the venv:
```
python -m evaluation.cli sample      # deterministic
python -m evaluation.cli fetch --threads 4 --budget-gb 3
python -m evaluation.cli run
python -m evaluation.cli report --run-dir ~/.cache/makeup_demo/easyportrait_runs/<date>
python -m evaluation.cli clean --yes # remove the cache
```

**Deferred findings:**
- **Security F1:** map zlib and struct errors to `FetchError`, and cancel pending futures.
- **Security F3:** add a wall-clock cap per request.
- **Tester F3:** decide whether 479 landmarks should be accepted.
- **Tester F7:** decide whether unknown prediction ids should raise.

### WP6-G glasses signal (2026-11-11): PASS

**Measurement:**
- All 48 images flagged with glasses have eye IoU 0.00. The parser labels the whole eye area as GLASSES.
- 0 false alarms and 0 missed glasses on that proxy.
- The glasses area is 3.1–8.9% on flagged images and at most 0.59% on the rest, so the 1% threshold needs no tuning.

**Built:** the existing warning, plus the hint `retake_suggested: ["glasses"]` and a generic UI prompt. Nothing is guessed behind the lenses. A prompt that depends on the user's makeup choice waits for Phases 6, 10 and 11.

**Revised criterion:** the committed-report byte comparison ignores the volatile "current tree" fingerprint line. This was a test-only change.

**Residual:** glasses the parser does not see at all cannot be counted, because EasyPortrait has no glasses label.

### Phase 3 closed for the prototype (user, 2026-11-11)

Skin-tone testing is done.

**Deferred by the user:**
- WP6 fixes: roll-aligned crop, y-offset, and the `present` floor.
- A provenance and consent record for `backend/tests/fixtures/portrait.jpg`. This is **required before any launch**.
- The fine-tuning decision. No custom model is trained without approval.
- A latency re-measure on AC power.

---

## Phase 4 — Makeup application zones

Every batch kept the earlier zones bit-identical to a `tests/golden_batch*.json` snapshot taken **before** that batch's edits.

### Batch 1 (2026-11-11): PASS after 2 fix loops, 372 tests

**Zones:** `chin_center`, `under_lower_lip`, `cupid_bow`, `lip_corner_l/r`, `mouth_surround`.

**Rejected once** on three points:
- Lip-corner confinement was not demonstrated after the metric change. A 0.3 ipd negative test was added.
- The corner coverage values were not reported.
- The "Phase 1/2 unchanged" claim rested on mtimes only. Bytecode-cache evidence was accepted.

**Note:** the left corner has a strict IoU of 0.168, because the parser lip is thin there. The user confirmed the placement visually.

### Batch 2 (2026-11-11): PASS after 2 fix loops, 392 tests

**Zones:** `forehead_center/perimeter`, `temple_l/r`, `jawline_outer_l/r`.

**Rejected once** on three points:
- `_forehead` equivalence was not proven. It is now proven against the pre-batch code, with goldens pinned.
- The jaw-band proxy was weak. The band is now checked directly.
- Disjointness evidence was missing from the summary. It is now tested for all 6 zones.

### Batch 3a (2026-11-11): PASS after 1 fix loop, 415 tests

**Zones:** under-eye inner and outer halves, inner and outer eye corners, upper and lower lash lines (12 zones).

**Code change:** in `face_regions.py`, only the iris-demotion predicate changed. It now selects eye-group zones by side suffix.

**Notes:**
- The 20% glasses threshold lets a lash line stay `ok` while glasses cover less than that share of it.
- A name-based check for model-only zones will not catch future `eye_`/`brow_` polygon zones.

### Batch 3b (2026-11-11): PASS after 1 fix loop, 439 tests

**Zones:** `eyelid_upper`, brow head, arch and tail, and `brow_bone` (10 zones).

**Approved by the user:** `highlighter = [brow_bone_r, brow_bone_l]`, the first highlighter entries.

**Notes:**
- `brow_bone_l` coverage on the fixture is near its threshold.
- The MEDIUM finding at `test_phase1_independent.py:861` is deferred. That test holds bbox literals measured on the real portrait.

### Batch 4 (2026-11-11): PASS, 463 tests

**Zones:** cheek apple, upper cheek, outer cheek, cheekbone and contour hollow, left and right (10 zones).

**Staging, approved by the user:**
- `blush` = apples, upper cheeks and outer cheeks.
- `highlighter` = brow bone + cheekbone.
- `contour` = jawline + hollows.
- `bronzer` = [], intentionally empty.

**Third fix loop, test-only, approved by the user:**
- The L/R mirror test runs on a yaw-symmetrized fixture, with shared bounds of 0.75–1.33.
- The real fixture is bounded at 0.65–1.5, because it is slightly yawed.

### Batch 5 (2026-11-11): PASS after 1 fix loop, 489 tests

**Zones:** `nose_bridge`, `nose_tip`, `nose_side_l/r`, `nostril_l/r`.

**Staging:** the sides join `contour`, and the bridge and tip join `highlighter`. The nostrils are unstaged exclusion areas.

**Bug fix, approved by the user.** `NOSE_HULL` was ordered so that the polygon crossed itself. It is now `[168,358,327,2,98,129]`. This changed `nose`, `cheek_l` and `mouth_surround` in synthetic scenarios only. Those 15 `golden_batch4/5` entries were re-captured with a dated note.

### Batch 6 (2026-11-11): ACCEPTED by the user, 512 tests

**Zones:** `lower_eyelid_r/l`.

**Geometry.** The zone is a band 0.07 ipd deep below the lower lid line. It is intersected with the geometric `under_eye_{k}` band, excludes the 1 px-dilated eye polygon, and is clipped to skin.

**Staging, approved by the user:** the zones join the `eye` layer after `eyelid_upper`.

**Fix loop 1:**
- At roll ±20°, 1 px fell outside `under_eye`. Fixed by intersecting with the under-eye band.
- The Batch 3a eye-group expected set was extended to include the new zones.
- **Criterion revised:** under glasses, the lower eyelid's reason must equal `eyelid_upper_{k}`'s, instead of containing the word "Glasses".

**Evaluator REJECT ×2, overridden by the user.** The only issue was that "Phase 1/2 unmodified" could not be proven from file contents, because there was no git repo and no pre-batch hashes. The evidence offered:
- No Phase 1/2 file is newer than the pre-edit golden snapshot.
- The edits to `test_phase1_independent.py` only add the two new ids.

**Phase 4 zone list complete: 70 zones.** The eyelid crease cannot be derived and is never added.

**Optional follow-ups:**
- A glasses-specific reason for both eyelids.
- A ratio test on a real photo. The current one uses synthetic labels.

---

## Standing pre-deployment blockers (open)

- **Licence:** the CelebAMask-HQ-derived parser weights are non-commercial only. Replace the model, and never fine-tune from it or pseudo-label with it.
- **Test fixture:** no provenance or consent record exists yet for the test portrait.
- **Hardening:**
  - Bound concurrent inference memory (F1).
  - Cap the request body before parsing (F2).
  - Add rate limiting and authentication.
- **Legal:** a review of biometric data handling.
