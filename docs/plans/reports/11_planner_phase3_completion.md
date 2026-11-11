# Planner Report: Phase 3 Completion Assessment (2026-11-10)

Agent: `planner`, two runs.
- **Run B** was commissioned in this session: a full Phase 3 assessment.
- **Run A** came from an earlier session: an EasyPortrait evaluation design.

The orchestrator verified their claims. The approved plan, with the user's decisions, is in `../phase3_completion_plan.md`.

## Findings (Run B, verified by the orchestrator where marked)

### 1. Implemented and working

- **Class mapping:** complete, and model ids are confined to `segmentation/` (`regions.py:87-111`).
- **Weights:** hashed, and the session is built from the same bytes. Input and output shapes are checked (`face_parser.py:127-183`).
- **Crop and inverse map:** within 1 px in the stub tests.
- **EXIF:** applied before both landmarks and parsing (`image_loading.py:40`).
- **Sides:** assigned by nearest landmark and roll-independent (`sides.py`).
- **Iris gate:** both irises must fall in an eye region (`face_regions.py:90`).
- **Concurrency:** bounded, with a 503 on timeout.

### 2. Implemented but needs improvement

- **Registration offset.** The forward warp is corner-aligned while the inverse map is centre-aligned (`face_parser.py:229-235` vs `:267-273`), and the pre-shrink uses the nominal factor.
  **Verified by measurement:**

  | Window | Shift in the network input | Label edge |
  |---|---|---|
  | 1000 | +0.53 px | +1 px |
  | 2000 | +0.98 px | +1 px |
  | 2650 | +1.08 px | +2 px |

- **No roll compensation.** The crop is axis-aligned, and alignment fails at ±35° (reports 07:86, 08:56, 10:62). This failure is missing from `docs/face_segmentation.md`.
- **`present = n > 0`** (`face_regions.py:110`). There is no noise floor and no cleanup of stray components.
- **Confidence and thresholds.** `mean_probability` and the zone confidence are uncalibrated. The zone thresholds and `crop_scale = 2.0` were tuned on one photo.
- **Window coverage.** The label map covers window ∩ image only, so hair, neck and cloth beyond the window are never labelled.
- **Large photos.** The full-resolution label map makes memory and latency grow, and concurrent memory has not been measured.

### 3. Missing

- Ground truth.
- A benchmark script in the repo.
- A robustness suite.
- A reproducible check of class 9 (earring) and class 15 (necklace).
- Subgroup analysis.
- Calibration.
- A golden regression test against the real model.
- A test for truncated landmarks.
- A record of where the test fixture came from and its licence.

### 4. Bugs and architectural concerns

- **Truncated landmarks give a generic 500. Verified in the code:** there is no length check at `landmarks.py:140-143`, and later code indexes 468 and 473.
- **Rolled crops.** A roll-aligned crop would need an optional argument on the `FaceParser.parse` Protocol and an additive `parse_window.angle` field.
- **`area_fraction`** uses the window area as its denominator, so it is not comparable across photos.
- **No git repo.** There are no evaluation-data entries in `.gitignore`, and the project is not a git repository.

## Ground-truth options considered

| Source | Assessment |
|---|---|
| Own consented, hand-annotated photos | Cleanest. Costs roughly 20-40 minutes per image. Deferred by the user. |
| EasyPortrait v2 test split | Out of distribution for the model, and close to the product's selfie and webcam domain. 9 classes. **Chosen.** |
| CelebAMask-HQ test split | In distribution, so optimistic, and non-commercial. Useful only for mapping verification. Not chosen. |
| LaPa, FFHQ, Fitzpatrick17k, Casual Conversations | Non-commercial, have no parsing masks, or are not applicable. |

**Licence.** The orchestrator read `license/en_us.pdf` in full. It is a custom licence reworked from CC BY-SA 4.0 and is not a CC licence. It has no non-commercial clause. Attribution and share-alike apply when the material is distributed. Privacy and portrait rights are not granted. Database rights fall under Russian law.

## Plan A (earlier-session Planner): key design points adopted

- **Sampling:** stratified by brightness quartile × resolution group, proportional allocation with a floor, seeded.
- **Fetching:** HTTP Range fetches of individual zip members, with guards: https only, 206 required, pinned sizes, byte budget, size caps, CRC check, UUID-only member names, no extraction.
- **Two phases:** fetch, then an offline run.
- **Cache:** outside the repo, with permissions 0700/0600 and a marker-guarded clean command.
- **Metrics:** a per-image 9×20 confusion matrix; boundary F at τ = 0.02 and 0.05 × IOD; area ratio; side-semantics and side-swap checks; landmark-polygon vs GT comparison; teeth treated as a secondary metric.
- **Denominators:** A (detected images only) and B (all valid images).
- **Statistics and reporting:** bootstrap confidence intervals; strata tables; worst-k ids; weak-region diagnosis; an if-then guide for choosing between boundary fixes and fine-tuning; a report with aggregates only.

## Plan B additions adopted

- A dev/test split, so that later fixes are tuned on dev only.
- The best-shift diagnostic.
- Both scorings: full image and inside the window.
- Pre-registered decision rules with provisional numeric targets.
- The benchmark (WP2) and the robustness suite (WP3).
- The WP4a bug fixes.
- An evaluation adapter written independently of `MODEL_ID_TO_CANONICAL`.
