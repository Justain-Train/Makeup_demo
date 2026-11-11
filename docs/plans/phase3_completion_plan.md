# Phase 3 completion plan: ground-truth evaluation and two bug fixes (APPROVED 2026-11-10)

Sources:
- the Planner reports, summarised in `reports/11_planner_phase3_completion.md`;
- the orchestrator's verification;
- the user's decisions, listed below.

Phase numbering follows the roadmap in `CLAUDE.md`.

## User decisions (2026-11-10)

| # | Decision |
|---|---|
| D1 | **Data:** the EasyPortrait v2 **test** split, sampled for diversity (stratified), **N = 600**. Own photos and CelebAMask-HQ are **not** part of this iteration. |
| D2 | **Decision rules are fixed now** (below). The numeric targets are **provisional** and will be confirmed by the user after this first ("pilot") run. |
| D3 | **Report both scorings:** full image (pixels outside the parse window count as "not predicted"), and inside the parse window only. |
| D4 | **WP4a approved:** fix the half-pixel registration shift and the truncated-landmark 500. |
| D5 | **Commit only the sample UUID list** under `docs/evaluation/`. Commit the aggregate report there too. Never commit images, annotations or per-image rows. |
| D6 | **F1, F2 and rate limiting stay deployment work.** They are not part of Phase 3. |
| D7 | **Pilot results accepted "for now" for the prototype** (user, 2026-11-10, after the iteration-3 PASS). The provisional targets are not formally revised. The decisions on WP6, WP6-G and fine-tuning are still open. |

Defaults the user did not change:
- the cache is `~/.cache/makeup_demo/easyportrait`, overridable with `EVAL_CACHE_DIR` or `--cache-dir`;
- a `--crop-scale` flag exists, but only 2.0 is run;
- teeth and mouth metrics are secondary;
- the landmark-polygon comparison is included.

## Licence (read by the orchestrator from `license/en_us.pdf` in hukenovs/easyportrait, 2026-11-10)

- **What it is:** "Public license with attribution and conditions reserved". The footnote says it is *not* a Creative Commons licence. It is a reworking of CC BY-SA 4.0.
- **Grants:** a worldwide, royalty-free, irrevocable licence to reproduce the material and create adapted material. There is **no non-commercial clause**.
- **Attribution and share-alike (§3):** these apply when the material, or adapted material, is *submitted* (distributed).
- **Not granted (§2(b)(1)):** the privacy and portrait rights of the people depicted.
- **Database rights:** governed by the Civil Code of the Russian Federation.

Consequences:
- Evaluation-only use with aggregate results is fine.
- Never redistribute images or annotations.
- Never use the data for training without the separate legal review that is already listed as a pre-deployment blocker.
- Attribute in the report and docs: cite the BibTeX of arXiv:2304.13509 (Kapitanov, Kvanchiani, Kirillova, 2023), give the repository URL, and name the licence with a link.

## Verified findings that drive the fixes (orchestrator, 2026-11-10)

1. **Half-pixel registration shift (confirmed by measurement).**
   - Cause: the forward warp in `face_parser.py:229-235` is corner-aligned (`dst = k*src`), while the inverse map at `:267-273` and `cv2.resize` are centre-aligned. The `INTER_AREA` pre-shrink at `:226` also uses the nominal factor, not the size actually produced.
   - Measurement: a synthetic smooth vertical edge, window equal to the image, a stub `_infer` that returns the input's channel as the skin logit.
   - Results:

     | Window | Shift in the network input | Label-map edge |
     |---|---|---|
     | 600 | +0.11 px | |
     | 1000 | +0.53 px | +1 px |
     | 1500 | +0.80 px | +1 px |
     | 2000 | +0.98 px | +1 px |
     | 2650 | +1.08 px | +2 px |

   - The shift is to the right and down. The existing 1 px tolerance test does not catch it.
2. **Truncated landmarks (confirmed in the code).** `landmarks.py:140-143` has no length check, and `face_regions.py` and `frame.py` index 468 and 473. Fewer than 478 landmarks therefore raises an `IndexError`, which becomes a generic 500.

## Work packages for this iteration

### WP4a: bug fixes (production code; the only `backend/app/` changes allowed)

**1. Registration.** Make the forward warp centre-consistent:
- dst = a·(src + 0.5) − 0.5, where a = the actual scale;
- for the pre-shrink, use the real produced size (`sub.shape`), not the nominal `shrink`, to derive the scale and offset.

The result must keep the existing border, padding and `MAX_WORK_SIZE` behaviour.

Add a regression test in `tests/test_face_parser.py`:
- a stub `_infer` that echoes the input;
- smooth synthetic edges, both vertical and horizontal;
- windows of 600, 1000, 2000 and 2650 px, including one window that extends past the image and one with the pre-shrink path;
- assert the sub-pixel edge shift in the *network input* is ≤ 0.1 original px, and that the label-map edge is within ±1 px of the correct pixel with no systematic sign.

Re-run the whole suite. If the zone tests' landmark/parser IoU expectations move, report the before and after values. Do not loosen thresholds silently.

**2. Truncated landmarks.**
- In `FaceLandmarkService` (`landmarks.py`), if the selected face has fewer than `EXPECTED_LANDMARKS` (478) points, return `status="no_face_detected"` with the message "Face landmarks were incomplete; please retake the photo with your whole face visible." Set `face_detected=False` and `landmark_count=0`.
- Do not change the status Literal or the API contract.
- Log the event as a count only, with no coordinates.
- Add unit tests with a stubbed landmarker: 477 and 0 points both produce this status on `/landmarks` and `/segmentation`, and 478 is unchanged.

**3.** Update `docs/face_segmentation.md`:
- record the registration fix;
- change "Truncated landmark result" in Limitations to the new behaviour;
- add the **±35° roll alignment failure** observed in report 07, which is missing from the doc.

### WP1 + WP5: EasyPortrait evaluation harness (new package `backend/evaluation/`; must not modify `backend/app/`)

The design follows the earlier Planner design (preserved in `reports/11_planner_phase3_completion.md`, "Plan A"). In summary:

- **Modules:**
  - `labels.py`: the only place that knows EasyPortrait ids. It is written independently of `MODEL_ID_TO_CANONICAL`.
  - `metrics.py`: pure numpy functions.
  - `dataset.py`
  - `runner.py`
  - `report.py`
  - `cli.py`, run as `python -m evaluation.cli {sample,fetch,run,report,clean}`.
- **EasyPortrait classes:** 0 BACKGROUND, 1 PERSON, 2 SKIN, 3 LEFT_BROW, 4 RIGHT_BROW, 5 LEFT_EYE, 6 RIGHT_EYE, 7 LIPS, 8 TEETH. These are verbatim from the README.
- **Comparison classes:**
  - skin: EP 2 vs our SKIN ∪ NOSE;
  - lips: EP 7 vs our LIP_UPPER ∪ LIP_LOWER;
  - eye_R, eye_L, brow_R and brow_L, with the EP LEFT_* classes mapped to our `_R`. **Verify this per image** with centroid ordering. If more than 2 % of images are exceptions, flag the sided rows and rely on the merged ones;
  - merged eye and merged brow;
  - secondary: recall of EP teeth by MOUTH_INTERIOR, and the whole mouth (LIPS ∪ TEETH vs lips ∪ mouth interior), both labelled "not comparable";
  - not scored: PERSON and BACKGROUND.
- **Per image:**
  - a 9×20 confusion matrix (EP 0-8 × canonical 0-18 + NOT_PARSED), from which all class metrics are derived;
  - boundary F-score for eyes, brows and lips at τ = 0.02 and 0.05 × IOD, computed on a bounding-box crop;
  - the best-IoU (dx, dy) shift per small region, within ±3 px;
  - the area ratio;
  - the GT inter-ocular distance (IOD);
  - the mean luma of GT skin pixels;
  - side-swap overlaps;
  - the landmark-polygon versus GT comparison for eyes and the lip outline;
  - per-stage latency from `timings_ms`, plus wall time and decode time.
- **D3, two scorings:**
  1. **Full image:** NOT_PARSED counts as "not predicted".
  2. **Inside the parse window:** pixels outside window ∩ image are excluded from both the GT and the prediction.

  Also report, per class, the share of GT pixels outside the window.
- **Denominators:**
  - A: detected images (`ok`) only;
  - B: all valid images, with failures counted as empty predictions.

  Report outcome counts: ok, no_face, multiple_faces, pipeline_error, invalid_sample.
- **Aggregates:**
  - dataset-level (pixel-weighted) IoU, Dice, precision and recall, with 95 % bootstrap confidence intervals (1000 resamples over images, seeded);
  - per-image mean, median, p10 and p90 of IoU, using images whose GT region has ≥ 25 px;
  - the false-alarm rate.
- **Strata ("diverse" sampling):**
  - the sample is drawn by brightness quartile (meta `brightness`, quantile edges over all test rows) × resolution group, allocated proportionally with a floor of min(cell, 5), using `numpy.random.default_rng(20261110)` over sorted ids, and trimmed or topped up to exactly 600;
  - the report also stratifies by GT skin-luma tercile, GT-IOD face-size tercile, glasses-detected and alignment-passed;
  - cells with n < 30 are flagged as low-sample;
  - luma is **not** a proxy for skin tone; skin-tone bias remains unquantified and must be stated.
  - **Status (2026-11-11):** skin-tone stratification done by ITA band on the pilot rows plus 35 public-domain
    portraits, see `docs/evaluation/skin_tone_2026-11-11.md`; gap targets met except eyes in the very_light band when
    glasses images are included (0.102 lower-CI gap, a glasses artefact).
- **Dev/test split for later tuning:**
  - assign each sampled id deterministically: dev if `sha256(id)` mod 10 < 4, otherwise test. Store this in the sample file;
  - the pilot report gives the numbers for the full sample, plus dev and test;
  - future fixes (WP6) are tuned on dev only.
- **Fetch:**
  - standard library only, using HTTP Range reads of the members from `test.zip` and `annotations.zip`. Use `meta.zip` for the sample; resolve the README's inconsistency about its location with a HEAD request;
  - https only, with no redirects across scheme or host;
  - require 206 and a matching `Content-Range`, and abort on 200;
  - pin Content-Length, verified by HEAD at implementation time;
  - a byte budget, default 3 GB;
  - member size caps;
  - CRC verified on read;
  - never `extract`; member names are built only from ids that match a strict UUID regex;
  - 4 threads; resumable;
  - throughput is printed.
- **Cache:**
  - must be outside the repo; refuse a path that resolves inside it, with a test;
  - directory mode 0700, files 0600;
  - a marker file, and a `clean` command that requires the marker;
  - `manifest.json` with SHA-256s.
- **Run:**
  - offline and sequential, with 3 warm-ups;
  - build the services the way `app/main.py` lifespan does, without editing it;
  - decode with `load_rgb_image`;
  - decode annotations as PNG only, mode L or P, values ≤ 8, with the shape equal to the decoded image. Any mismatch is an `invalid_sample`. Never rotate labels to fit. If invalid samples exceed 1 %, stop and report;
  - call the real `segment_face` and decode the label map from the response;
  - `per_image.jsonl` goes outside the repo;
  - logs contain ids, outcome categories, timings and exception type names only. Meta `user_id` is never read into outputs.
- **Report:** `docs/evaluation/easyportrait_test_<date>.md` and `.json`, with aggregates and opaque ids only. Contents:
  - the run header, with model and landmarker SHA-256s, package versions, CPU and OS (no hostname, username or home paths), crop scale and threads, a source-tree fingerprint, and a note that no git hash exists;
  - outcomes;
  - headline tables (A/B × full/window);
  - side handling;
  - the confusion view;
  - stratified tables;
  - latency by megapixel bucket;
  - landmark polygon versus parser;
  - the worst 10 ids per class;
  - a weak-region ranking with the dominant error mode;
  - the **provisional targets table and the decision rules** below, applied to the numbers and labelled provisional;
  - limitations;
  - licence and attribution.

  The word "accuracy" never appears unqualified. The wording is "agreement with EasyPortrait annotations".
- **Sample ids:** `docs/evaluation/easyportrait_test_sample_seed20261110_n600.txt`, one UUID per line, with a split column.
- **`.gitignore`:** add defensive entries for evaluation caches and outputs.
- **Execute** `sample` → `fetch` → `run` → `report` at N = 600. Record the download size, the elapsed times and the peak RSS.

### WP2: reproducible benchmark (`backend/evaluation/bench.py`; no `app/` changes)

- The fixture upscaled to 0.8, 2, 10 and 25 MP, labelled as synthetic scale.
- 30 warm runs after 1 discarded, giving p50, p95 and p99 per stage.
- Concurrency 1, 2 and 4, with peak RSS per configuration measured in a fresh subprocess. Note that `ru_maxrss` is in bytes on macOS and in KB on Linux.
- Environment capture.
- Offline experiment, measurement only: on the EasyPortrait dev split, compare agreement when the working label-map resolution is capped at 1024 and 2048. Do this inside the evaluation package by downscaling the decoded label map output, not by changing production code.
- Output: `docs/evaluation/benchmark_<date>.md`.

### WP3: robustness suite (`backend/evaluation/perturb.py`)

- Perturbations, each applied to the image and the GT together: horizontal mirror (swap GT sides), roll ±15/30/45°, scale 0.5×, JPEG q30, Gaussian blur, gamma 0.5/2.0, grayscale.
- Run on a seeded 100-image subset of the **dev** split.
- Report the change in agreement and the failure-rate change per perturbation, plus a GT-free consistency score: flip equivariance and rotation equivariance.
- Output: a section of the main report, or `docs/evaluation/robustness_<date>.md`.

### WP6-G: glasses handling (added by the user 2026-11-10; part of WP6, gated)

**Status (2026-11-11):** measurement done (`reports/18_wp6g_glasses_measurement.md`): 0 false alarms, 0 missed
glasses, 1 % threshold left unchanged. Phase 3 signal implemented and passed the workflow (`reports/19`-`22`):
additive `retake_suggested: ["glasses"]`, generic UI prompt, tests, docs. The makeup-choice-dependent prompt (item 4)
and the Phase 9 preview (item 5) remain future work.

**Evidence:**
- Eye IoU is 0.00 on all 48 pilot images where `glasses_detected` fired.
- The alignment check failed on 93 of 598 images, with eye IoU 0.17. How many of those images have glasses is not yet known.

**Behaviour.** This is a soft prompt. The photo is never rejected.

1. **Glasses detected and eye-area makeup wanted.** If `glasses_detected` fires and the user wants eye-area makeup, prompt them: "To get eye-makeup results, retake the photo without your glasses." Eye-area makeup covers eyeshadow, liner, lashes, brows, under-eye concealer and anything else whose zones touch the eyes.
2. **Everything else is still returned.** Skin, lips, cheeks and the other zones are unaffected by the prompt.
3. **The user may choose to keep their glasses on.** For example, they may want to see the generated preview while wearing them. In that case:
   - the eye and under-eye zones stay `unavailable` or `low_confidence`, as they do today;
   - the response and UI say clearly that eye-area results are limited while glasses are on;
   - nothing is guessed behind the lenses.
4. **The prompt depends on the user's makeup choice**, which arrives in later phases (6, 10 and 11). Phase 3 provides only the signal. The signal is the existing warning plus, optionally, an additive response hint, `retake_suggested: ["glasses"]`. The UI shows a generic version of the prompt until a makeup choice exists.
5. **Generative preview with glasses on** belongs to Phase 9. Two requirements carry forward:
   - preserve the glasses unchanged;
   - only apply makeup to eye-area pixels that are visible.

   How well this works is **unknown**. It must be measured in Phase 9 on photos with glasses before it is offered as reliable. Until then, the UI labels it "experimental with glasses".
6. **Reference images (Phase 7)** cannot be asked to remove glasses. That case is handled separately in Phase 7.

**Measurement before building.** This is Evaluator or WP6 work. It uses only the existing `per_image.jsonl` and needs no re-run.

- **Missed glasses:** count the images with no glasses warning where a noticeable share of EP eye pixels was predicted as GLASSES.
- **False alarms:** count the images with the warning where eye IoU is still good.
- **Overlap with alignment failures:** check how many of the 93 alignment-failed images also have the glasses warning.
- **Threshold:** tune the 1 % glasses-area threshold on the dev split only, if the results justify it.

**Implementation.** WP6 after user approval:

- an additive response field, if adopted;
- frontend messaging;
- tests.

The API contract keeps working for existing clients.

### Not in this iteration

- WP6 fixes: the `present` floor, speck cleanup, a crop-scale change, the roll-aligned crop, a resolution cap. Each needs the pilot results and user approval.
- WP7: the final test-split decision.
- Own photos, CelebAMask-HQ, any training, any Phase 4 zones, frontend changes, F1/F2.

## Decision rules (FIXED NOW, D2)

Evaluated on the dev split when tuning and on the test split once for the final decision. "Agreement" means dataset-level IoU against EasyPortrait annotations, full-image scoring, denominator A, unless stated.

1. **Acceptable for the prototype:** every provisional target is met, or any miss is ≤ 0.03 and its 95 % confidence interval overlaps the target. Detection failure rate ≤ 2 % on images where the face is fully visible.
2. **Post-processing or geometry fix is enough:** misses are concentrated in a fixable error mode shown on dev and confirmed once on test. That is any of:
   - boundary F at 0.05·IOD ≥ 0.85 while IoU is low;
   - area ratio outside 0.85-1.15;
   - a consistent best-shift ≠ 0;
   - stray false positives;
   - failures concentrated at |roll| > 15° (perturbation suite);
   - losses mainly outside the parse window.
3. **Model replacement or fine-tuning is needed** (commercial path only; never from the CelebAMask-HQ weights): any of
   - a small region (eye, brow or lips) still below IoU 0.50 after the dev-tuned fixes;
   - a brightness or luma-stratum gap > 0.10 that survives the fixes;
   - boundary F at 0.05·IOD < 0.70 for a small region (wrong shape or place, not a convention).
4. **Detector lever:** if most of the denominator B loss is detection failure, the next work is the landmarker or detector, not the parser.
5. **Licence caveat:** regardless of the outcome, the CelebAMask-HQ model is replaced before any commercial release. The harness becomes the comparison tool for that replacement.

## Provisional numeric targets (to be confirmed by the user after the pilot, D2)

| Comparison class | IoU target | Boundary F @ 0.02·IOD |
|---|---|---|
| skin (EP skin vs our skin ∪ nose) | ≥ 0.90 | — |
| lips | ≥ 0.75 | ≥ 0.80 |
| eye (merged) | ≥ 0.65 | ≥ 0.80 |
| brow (merged) | ≥ 0.60 | ≥ 0.80 |
| Side-swap rate (eyes, brows) | ≤ 1 % | — |
| Worst-stratum gap (brightness and luma) | ≤ 0.05 (skin) / 0.08 (small regions), using the lower CI bound | — |
| Latency, 1-2 MP, reference machine | p95 ≤ 300 ms | — |

EasyPortrait boundary conventions differ from CelebAMask-HQ's, so the IoU targets may be revised after the pilot. That is why boundary F and the area ratio are reported alongside.

## Acceptance criteria for this iteration

1. WP4a: the registration test passes (input shift ≤ 0.1 px). The truncated-landmark tests pass. The full suite passes (previously 191), with any changed expectations reported.
2. `sample` is deterministic and gives exactly 600 ids covering every non-empty stratum. `fetch` downloads only the sampled members, with every guard covered by an offline unit test that uses a fake range fetcher and an in-memory zip. `run` is offline.
3. Metric unit tests pass with no network, dataset or weights:
   - perfect, disjoint, partial, empty GT and empty prediction cases;
   - confusion-derived results equal direct mask computation;
   - NOT_PARSED handling;
   - both scorings;
   - boundary F behaviour as the shift crosses τ;
   - bootstrap reproducibility;
   - the EP side mapping;
   - UUID and path guards.
4. A weights-gated plumbing test: GT synthesised from the fixture's own prediction gives agreement ≈ 1. Swapping EP left and right is detected.
5. The report exists with every section, both scorings, both denominators, confidence intervals, low-sample flags, the side-semantics exception count, the invalid-sample count, provisional targets and decision rules applied, attribution, and no unqualified "accuracy". The JSON and Markdown agree.
6. The benchmark and robustness outputs exist.
7. No image, annotation, per-image row or pixel data is in the repo. No log contains image-derived data. `backend/app/` is changed only for WP4a.
