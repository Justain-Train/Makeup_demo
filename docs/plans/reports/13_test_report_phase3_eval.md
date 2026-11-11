# Test report 13: Phase 3 completion (WP4a + WP1/WP5), independent verification (2026-11-10)

Tester: independent testing agent. Scope: `docs/plans/phase3_completion_plan.md` acceptance criteria 1-7, with WP6 and WP6-G out of scope. No production code under `backend/app/` or `backend/evaluation/` was modified. The only file added is `backend/tests/test_phase3_independent.py`. Scratch probes, mutant copies and run outputs are under `/private/tmp/claude-501/-Users-test-Makeup-demo/be1f5eaa-15ab-4e74-9511-6c4706153e4d/scratchpad/` and are not in the repo. No network was used.

## 1. Test summary

| Area | What was done | Result |
|---|---|---|
| WP4a registration | Own float32 edge probe (no quantisation), least-squares shift estimator with ideal centre-aligned sampling, 9 geometries x 2 axes x 4 sub-pixel phases, injected-shift sensitivity check. Two scratch mutants tested for detection. | Fixed code: worst input shift 0.002 px. Both mutants detected (M1 corner-aligned: 56 cases over 0.1 px, worst 1.29 px; M3 nominal pre-shrink scale: 16 cases, worst 0.90 px; 8-geometry scratch probe). The coder's own registration test misses M3 (finding F1). |
| WP4a truncated landmarks | 0/1/477/478/479 points through `FaceLandmarkService`, `/landmarks` and `/segmentation` via TestClient, parser not invoked, log content checked. | PASS. Mutant with the length check removed gives failures for 0/1/477. |
| Harness: mapping, metrics, scorings, denominators, bootstrap, boundary F, best shift, side swap | Unit tests written from the spec, random confusion matrices vs direct masks, NOT_PARSED/Unsided handling, both scorings, both denominators, seeded bootstrap, boundary F across tau, best-shift sign, side swap. | PASS. |
| Real-data spot check | 4 cached samples (glasses, smallest, largest, no-face) recomputed by hand with PIL/numpy and compared with `per_image.jsonl`. | PASS, every field equal. |
| Determinism | Sampler re-run twice and diffed against the committed sample file; 3 fresh pipeline runs compared with pilot rows. | PASS, byte-identical / identical on all deterministic fields. |
| Report regeneration | `evaluation.cli report` rebuilt from `per_image.jsonl` into scratch, and also from the test. | PASS, JSON and Markdown byte-identical to the committed files. |
| Guards (fetch/cache) | Fake range fetchers and fake HTTP openers, in-memory zips: UUID hostile list, cache location (repo, repo parent, relative path that resolves inside the repo, sibling prefix), marker refusal, 0700/0600 modes, https/redirect refusal, 206/Content-Range/200/short/long body, error messages without URLs, budget under 40 threads, pins, member size cap before download and during decompression, CRC, name mismatch, unsupported method. | PASS. |
| Privacy | Greps over `docs/`, `backend/`; cache modes; log output of a real run; committed artefacts. | PASS, with an observation on the test fixture (F9). |
| Full suite | `pytest -q` on backend. | 350 passed (273 baseline + 77 new), 0 failed, 0 skipped. |

## 2. Test cases

Results are for `backend/tests/test_phase3_independent.py` (40 functions, 77 cases, all executed, all PASS on the current code).

### 2.1 Registration (WP4a)

| Test | Expected | Actual | Result |
|---|---|---|---|
| `test_network_input_shift_is_within_0p1_px[9 geometries]`: square 600; non-square 700x500 past right/bottom; 900x700 past left/top; odd 1001x777 past left/top; pre-shrink 1300x1200 past right/bottom; pre-shrink 1777x1500 odd past left/top; pre-shrink 2650 == image; pre-shrink 3001x2400 past top-left; window 1026 (> MAX_WORK_SIZE, pre-shrink) | Estimated input shift <= 0.1 px for every axis and phase (0.1, 0.3, 0.6, 0.9); worst asserted < 0.02 | Worst 0.002 px (2650 case) | PASS |
| `test_label_map_edge_is_within_1px_with_no_systematic_sign` | Label-map edge error within +/-1 px; exactly 0 for windows <= 1024 (work resolution == window); |mean| < 0.25 | Passes | PASS |
| `test_estimator_sees_an_injected_0p2_px_shift[x, y]` | Injected +0.2 px measured as +0.2 +/- 0.03 px (and -0.2 for -0.2) | Passes | PASS |

Sensitivity note: the estimator is only useful if it can see an error above the 0.1 px limit. The injected-shift test shows it can.

### 2.2 Truncated landmarks (WP4a)

| Test | Expected | Actual | Result |
|---|---|---|---|
| `test_truncated_landmarks_status_on_both_endpoints[0, 1, 477]` | 200 on `/landmarks` and `/segmentation`; `status=no_face_detected`, `face_detected=false`, `face_count=0`, `landmark_count=0`, `landmarks=[]`, exact message; segmentation `anatomical=null`, `zones=[]`; parser never called; log args are ints/strings only | Passes | PASS |
| `test_478_landmarks_are_unchanged_on_both_endpoints` | `ok`, 478 landmarks, message null, segmentation 200 | Passes | PASS |
| `test_479_points_do_not_crash_and_are_documented_as_ok` | No crash; accepted as ok with count 479 (see F3) | `ok`, 479 | PASS (documents F3) |
| `test_parser_unavailable_is_503_even_when_landmarks_are_truncated` | 503 (parser checked first) | 503 | PASS |

### 2.3 Evaluation harness

| Test | Expected | Actual | Result |
|---|---|---|---|
| `test_ep_side_mapping_is_image_left_equals_our_right` | EP LEFT_EYE(5)/LEFT_BROW(3) -> our `_R`; RIGHT -> `_L`; merged sets | Passes | PASS |
| `test_labels_module_does_not_import_the_model_mapping` | `labels.py` imports no `MODEL_ID_TO_CANONICAL` / `build_label_lut` (AST check) | Passes | PASS |
| `test_sided_eval_detects_a_true_swap_and_not_a_match` | Aligned: no swap, tp=100; swapped: swap detected | Passes | PASS |
| `test_confusion_derived_counts_equal_direct_mask_counts[seed 0-5]` | For every comparison class, in full, window and valid scopes, tp/fp/fn from the confusion matrix equal direct mask counts (Unsided 200-202 and 255 are "not predicted") | Passes | PASS |
| `test_not_parsed_and_unsided_predictions_count_as_not_predicted_only` | 255 and 200 counted only in the NOT_PARSED column (50 TP, 0 FP, 50 FN) | Passes | PASS |
| `test_full_and_window_scoring_differ_only_by_pixels_outside_the_window` | Full FN = 200, window FN = 0 | Passes | PASS |
| `test_bootstrap_is_seeded_and_brackets_the_estimate` | Same seed identical; other seed differs; CI brackets point estimate; empty -> None | Passes | PASS |
| `test_scores_edge_cases` | Empty -> None; perfect -> 1; disjoint -> 0 | Passes | PASS |
| `test_per_image_iou_uses_only_images_with_enough_gt_and_false_alarms_use_absent_gt` | GT < 25 px excluded from IoU stats; absent = GT < 25 px; alarm = pred >= 25 px | Passes | PASS |
| `test_boundary_f_is_one_within_tau_and_drops_beyond_it` | F ~1 for shifts <= tau = 2 px; F drops for shifts > tau; monotone | Passes | PASS |
| `test_boundary_f_edge_cases` | Both empty -> None; one empty -> 0; identical -> 1 | Passes | PASS |
| `test_best_shift_returns_the_move_that_aligns_pred_to_gt[6 cases]` | If pred = gt moved by (dx, dy) (+ = right/down), best shift is (-dx, -dy), IoU 1 | Passes. Sign convention confirmed: positive dy means the prediction sits above the GT and must move down. | PASS |
| `test_best_shift_none_for_empty_and_tiny_masks` | None when a mask is empty | Passes | PASS |
| `test_side_swap_is_undefined_without_overlap` | None with no overlap | Passes | PASS |
| `test_denominators_and_failure_rows_count_as_empty_predictions` | A = ok only (1 of 3 rows); B = all but invalid (2 of 3) | Passes | PASS |
| `test_report_regeneration_needs_no_network_and_matches_committed_json` | Rebuilt JSON and Markdown identical to committed | Passes (not skipped; the pilot outputs are present) | PASS |
| `test_sample_is_exact_stratified_and_reproducible_from_the_meta_table` | Sample identical on re-draw and to the committed file; every non-empty (brightness quartile x resolution) cell covered with at least min(cell, 5); 600 rows | Passes | PASS |
| `test_committed_sample_split_follows_the_sha256_rule_independently` | Every split matches sha256(id) mod 10 < 4; 600 lines; 239 dev | Passes | PASS |

### 2.4 Guards (fake fetchers, in memory, no network)

| Test | Expected | Actual | Result |
|---|---|---|---|
| `test_uuid_guard_rejects_hostile_ids_before_any_path_is_built[12 cases]` | Upper case, whitespace, NUL, newline, `../`, `..\`, percent-encoding, no hyphens, short/long, Arabic-Indic digits, embedded path: all rejected in `require_id`, `image_path`, `annotation_path` | Passes | PASS |
| `test_cache_location_guard_refuses_repo_parents_and_relative_paths` | Refused: repo, a path inside docs, repo parent, relative `backend/../docs`; allowed: sibling `<repo>_sibling_check` and tmp dir | Passes | PASS |
| `test_cache_marker_guard_on_clean_and_on_foreign_directories` | Non-empty foreign directory refused by `ensure_cache` and `clean_cache`; file kept | Passes | PASS |
| `test_cache_directories_and_files_are_private` | 0700 directories, 0600 files and marker | Passes | PASS |
| `test_range_fetch_accepts_only_a_matching_206_with_exact_length` | Accepted | Passes | PASS |
| `test_range_fetch_aborts_on_any_mismatch[6 cases]` | 200, wrong start, missing or malformed Content-Range, short body, long body: all FetchError | Passes | PASS |
| `test_range_fetch_http_errors_do_not_leak_the_url` | FetchError; message has no URL, includes the HTTP code | Passes | PASS |
| `test_plain_http_and_redirects_are_refused_without_contacting_the_server` | http:// refused for get and head with zero opener calls; redirect handler raises | Passes | PASS |
| `test_head_requires_200_and_a_length` | Missing length or non-200 refused | Passes | PASS |
| `test_budget_is_reserved_before_the_request_and_never_overshoots_under_threads` | Budget 1500 bytes: exactly 15 of 40 threads succeed; refused requests never reach the inner fetcher | Passes | PASS |
| `test_pins_refuse_unpinned_urls_and_ranges_past_the_archive_end` | Unpinned URL and range past end refused, no request | Passes | PASS |
| `test_central_directory_and_member_roundtrip_with_fetch_accounting` | Member bytes equal the original | Passes | PASS |
| `test_member_checks_run_before_any_download_and_inside_decompression` | 5 MB zeros member: refused before any request when declared size is over the cap; a lying directory is refused during bounded decompression | Passes | PASS |
| `test_member_crc_and_name_and_method_checks` | Wrong CRC, wrong local name, bzip2 method refused | Passes | PASS |
| `test_committed_evaluation_artefacts_hold_ids_and_aggregates_only` | No `user_id`, no `/Users/`, no `Makeup_demo`, no data URI or PNG base64, no per-image list (longest list 20) | Passes | PASS |

## 3. Test execution

Working directory `/Users/test/Makeup_demo/backend`, venv `.venv`. `-p no:cacheprovider` used so no cache files are written.

```
.venv/bin/python -m pytest -q -p no:cacheprovider                       # baseline before my file: 273 passed
.venv/bin/python -m pytest -q -p no:cacheprovider                       # after my file: 350 passed (41 s)
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_phase3_independent.py   # 77 passed (11 s)
.venv/bin/python -m pytest -q tests/test_face_parser.py                 # on scratch mutant copies (see 5)
```

Probe and mutation runs, with scratch scripts in `.../scratchpad/scripts/` (`reg_probe.py`, `mut_probe.py`, `spot_check.py`) and mutant copies in `.../scratchpad/mut_m1`, `.../scratchpad/mut_m3`, `.../scratchpad/mut_m4`. Each mutant is a full copy of `backend/` without `.venv`; the repo was not touched.

```
PYTHONPATH=<repo>/backend .venv/bin/python mut_probe.py <face_parser variant>   # fixed: 0 cases > 0.1 px; M1: 56; M3: 16
python -m pytest tests/test_face_parser.py                                      # M1: 1 failed (registration), M3: 15 passed
python -m pytest tests/test_phase3_independent.py                               # M1: registration fails; M3: pre-shrink fails; M4: truncation 0/1/477 fail
.venv/bin/python -m evaluation.cli report --run-dir ~/.cache/makeup_demo/easyportrait_runs/2026-11-10 --date 2026-11-10 --out-prefix <scratch>/regen/easyportrait_test_2026-11-10   # cmp: JSON and MD identical
.venv/bin/python -m evaluation.cli sample --out <scratch>/regen/sample_a.txt   # twice; cmp identical and identical to committed sample
.venv/bin/python -m evaluation.cli fetch          # offline: "fetched 0 (skipped 600 already cached, failed 0)", 0 requests
.venv/bin/python -m evaluation.cli run --sample <scratch>/three.txt --out-dir <scratch>/run3 --warmups 1   # 3 ok, exit 0
.venv/bin/python <scratch>/scripts/spot_check.py   # ALL_CHECKS_PASS on 4 samples
caffeinate -dims ...   # used for every run
```

Not run, on purpose: the 600-image pilot, the benchmark and the robustness suite (the brief said they were slow). Their committed outputs were checked for existence and privacy only.

## 4. Failures and findings

No test failed on the current code. The findings below are ordered by severity. None is a defect in the WP4a fix.

**F1. Medium (test strength): the coder's registration regression test misses the pre-shrink scale regression.**
- Test: `backend/tests/test_face_parser.py::test_registration_is_centre_consistent_in_network_input_and_label_map`.
- Mutant M3: in `face_parser.py` replace `sx = sub.shape[1] / (rx1 - rx0)` and `sy = sub.shape[0] / (ry1 - ry0)` with `sx = shrink` and `sy = shrink`. This keeps the explicit produced size, so the nominal scale is wrong by the rounding of the output size.
- Reproduction: copy `backend/` to a scratch directory, apply M3, run `pytest tests/test_face_parser.py`. Result: 15 passed.
- Effect: for the 2650 pre-shrink case the true input shift is -0.294 px (my probe; the model is `error = k*(p - rx0 + 0.5)*(sx_true/sx_nom - 1)`, which matches). The coder's measurement gives +0.072, +0.033 and -0.075 px on M3. This is under its 0.1 px limit because the measurement is uint8-quantised, uses an edge sigma of about 44 px, and fits a line through the crossing, which gives roughly 0.07 px of noise.
- Fixed code: this is the bug the coder found and fixed (report 12, "a second cause"), so the implementation is correct. The regression test would not stop it coming back.
- Mitigation in place: `test_phase3_independent.py` uses float32 input, a 10 px edge, least-squares fitting and an injected-shift sensitivity check. It fails on M3 (2650 and 1777 pre-shrink cases) and on M1. Recommend the orchestrator either replace the coder's test with this one or keep both.
- Severity: Medium because the regression guard is weaker than the report implies, not because of a current defect.

**F2. Low: the existing 1 px alignment test does not detect the half-pixel bug.**
- `test_crop_and_inverse_map_align_with_the_image_within_1px` passes on the corner-aligned mutant M1 (only the registration test fails). This matches the plan's note. The 1 px test is still valid for gross misalignment and should be kept.

**F3. Low (contract and documentation): 479 landmarks are accepted.**
- `landmarks.py` checks only `len(face) < 478`. A 479-point result returns `status=ok` with `landmark_count=479`, while the schema says 478. MediaPipe does not produce more than 478, and nothing crashes. Suggest `!= EXPECTED_LANDMARKS` if the contract is meant to be exact, or a documented note.

**F4. Low (reproducibility narrative): the source fingerprint is stale, and the coder's account conflicts with Run 2.**
- `source_fingerprint_sha256` in the committed JSON is `4718831d...`. The current tree gives `d829a240...`. This is expected, because `evaluation/` gained `bench.py`, `perturb.py` and `predict.py`, and `report.py` gained a view after the run.
- Report 12, section 8, says "report.py was not edited after the run". Run 2 in the same report says it was. The committed JSON and Markdown regenerate byte-identically from the current code, so the artefacts are internally consistent. Only the narrative needs correcting.

**F5. Info (WP6-G evidence not in report 12): the glasses and alignment problems overlap completely.**
- Recomputed from `per_image.jsonl`: all 48 images with `glasses_detected` are among the 93 `alignment_passed=false` images. Eye IoU is 0.000 for glasses images (n=48) against 0.765 without (n=550). Eye IoU is 0.168 for alignment failures (n=93) against 0.778 for passes (n=505).
- The plan's open question, "how many of the 93 alignment failures have glasses", is answered: 48.
- Missed glasses at the 1 % threshold: 0 images with no warning and at least 1 % of EP eye pixels predicted GLASSES. That is a count of candidates only, and tuning is for WP6-G on dev.

**F6. Info (semantics): in-window scoring for failed images.** The denominator-B window case counts the whole GT of failed images as missed, because they have no window. The report documents this. It has no numeric effect in the pilot: 0.00 % of EP skin GT pixels fall outside the window (recomputed; the report states the same for every class, which I did not recompute for the other classes).

**F7. Low (robustness of the metric code): unknown prediction ids are silently scored as "not predicted".** `labels.pred_to_columns` maps any value that is not a canonical id, including the internal `Unsided` ids 200 to 202, to the NOT_PARSED column. It cannot happen today because `assign_sides` runs first. If a future change leaked an `Unsided` id, recall would drop without any error. Suggest raising on unexpected ids other than 255.

**F8. Info (provenance): no git history.** The "only WP4a in `backend/app/`" claim can only be checked by mtimes. Modified in the WP4a window: `face_parser.py` (18:55) and `landmarks.py` (18:56) only. Other `app/` files are dated 17:05 to 18:01, before the Planner report (18:53), so they predate this iteration. The Phase 3 files cannot be diffed.

**F9. Info (privacy observation).** `backend/tests/fixtures/portrait.jpg` is a photo of a person committed to the repo. It predates this iteration and is used by the existing tests. Its provenance and consent should be confirmed under the biometric-data blocker. Not a finding against this iteration.

## 5. Mutation evidence

| Mutant | What it changes | Fixed code | Existing `test_face_parser.py` | My `test_phase3_independent.py` |
|---|---|---|---|---|
| M1 | Forward warp is corner-aligned (drops the +0.5 terms) | Passes | 1 failed (registration), 14 passed | Registration fails on all 9 geometries; label test fails |
| M3 | Pre-shrink uses the nominal scale with the explicit produced size | Passes | 15 passed (missed) | Pre-shrink geometries fail; sensitivity test fails |
| M4 | `landmarks.py` length check removed | Passes | n/a | Truncation tests fail for 0, 1, 477 |
| M2 (not a bug) | `cv2.resize(fx=...)` with nominal scale throughout | Passes | n/a | Passes. OpenCV samples on the nominal grid, so this is a consistent alternative |

Failures in the mutant copies of the path-dependent tests (labels, sample, artefacts) are expected: those copies have a different directory layout. They are not detections.

## 6. Spot checks (criterion 5 and item 5)

Four cached samples recomputed with PIL and numpy only (no `evaluation.metrics`), then compared with their `per_image.jsonl` rows:

| Sample (id prefix) | Case | Checks |
|---|---|---|
| 0c8728ab | glasses warning; 1440x1920 | GT counts, IOD (286.08 px), skin luma (140.96), eye side order, window (130,-61,1213), warnings, skin/lips/eye/brow TP/FP/FN: all equal. Eye TP 0 (the 0.00 glasses effect is visible) |
| 2b0f39bd | smallest image, 761x761 | All equal. Eye_R TP 1175 and eye_L TP 1176, so the EP-to-canonical mapping is right on real data |
| 01b7d081 | largest image, 3840x2880 | All equal |
| c77b7416 | no face | GT fields equal; outcome no_face |

The pipeline is deterministic. A fresh run of 3 samples gave rows identical to the pilot rows on every deterministic field.

## 7. Coverage gaps

1. Live HTTPS behaviour was not verified (no network). The guards were tested with fake fetchers only, and the 206/200 behaviour of the real server is only inferred from report 12.
2. The benchmark (`docs/evaluation/benchmark_2026-11-10.md`) and robustness (`robustness_2026-11-10.md`) outputs were checked for existence and privacy only. Their numbers were not re-derived.
3. Latency targets were not independently timed. The 1-2 MP bucket has n=59 on one machine, and the pilot latency is flagged as possibly distorted by sleep. The benchmark ran on battery.
4. Skin-tone bias is not quantified. EasyPortrait has no tone labels, and luma is not a proxy. The report says so.
5. The frontend (glasses messaging, `retake_suggested`) is out of scope and was not tested.
6. Eye and brow sided rows have no independent per-image side-swap verification beyond the 4 spot checks and the 0 % swap result in the report.
7. The registration test covers 9 geometries and 4 phases. The mutant-detection evidence is for two mutants; other regressions were not enumerated.

## 8. Regression results

- Full backend suite: 350 passed, 0 failed, 0 skipped (273 before my file, which still pass, plus my 77). The coder's registration and truncation tests pass. I did not diff the existing test files, so "no expectation moved" is the coder's claim, checked only by the suite passing.
- The pilot numbers reproduce exactly from `per_image.jsonl`: skin IoU 0.866, lips 0.787, eye 0.717, brow 0.710 (recomputed by hand), side swap 0 % (reported; not recomputed).
- The WP4a registration fix is verified in the network input (worst 0.002 px) and in the label map (exact for windows up to 1024, within 1 px above that).
- No regression found by the suite in the upload, landmark or segmentation contracts. I did not separately test the status values beyond the truncation and no-face cases.

## 9. Privacy results

- No `user_id` string in the repo outside code that excludes it and docs that describe the rule. None in `docs/evaluation/` or the committed JSON.
- No image-derived strings in `docs/` or `backend/`. No data URIs, no base64 blobs over 200 characters.
- No image, annotation, per-image row or `.zip` in the repo, except the pre-existing fixture (F9).
- `.gitignore` has `easyportrait*/`, `.makeup_demo_easyportrait_cache`, and `docs/evaluation/**/{jpg,jpeg,png,zip}` entries.
- Cache: the directory `easyportrait` is 0700, with 1,200 image and annotation files at 0600, plus manifest, meta, marker and run files at 0600. The run directory `easyportrait_runs/2026-11-10` is 0700. The parent `easyportrait_runs` is 0755 and holds no data.
- `meta_test.csv` has the columns `image_name, height, width, brightness` only.
- Logs: the run CLI prints `run 3/3 {outcome counts}` and a JSON summary with timings and RSS. The native TensorFlow/MediaPipe lines on stderr carry no image data. Warnings from the landmark path contain a count only.
- The committed report contains no hostname, no home path and no username. The environment block has the OS string `macOS-15.6-arm64...`, the CPU name and package versions only.
- "accuracy" appears once, in "They are not an accuracy ...". Every other use is qualified as agreement with EasyPortrait annotations.

## 10. Overall result

**PASS** for the plan's acceptance criteria 1 to 7, with the conditions below.

Evidence by criterion:
1. WP4a: registration verified independently (worst 0.002 px, both mutants caught by my tests). Truncated landmarks verified on both endpoints. Suite 350 passed. The coder's registration test is weak (F1). Recommended action before the next iteration: replace it with, or keep it alongside, the independent test.
2. `sample` is deterministic, gives exactly 600 ids and covers every non-empty stratum (checked against the meta table). `fetch` is resumable and offline-verified, and the guards have offline tests (fake fetchers). `run` is offline.
3. Metric tests: perfect, disjoint, partial, empty GT and empty prediction; confusion equals direct masks; NOT_PARSED; both scorings; boundary F across tau; bootstrap reproducibility; EP side mapping; UUID and path guards. All PASS.
4. Weights-gated plumbing test: exists in `test_eval_plumbing.py` and passes (the suite ran with weights present, none skipped).
5. The report has every section. JSON and Markdown agree and regenerate byte-identically. Both scorings and both denominators, CIs, low-sample flags, side exceptions (0), invalid samples (0), provisional targets and decision rules are present. "accuracy" is never unqualified.
6. Benchmark and robustness outputs exist (not re-run).
7. No image, annotation or per-image data in the repo. No image-derived log content. `backend/app/` changed only in WP4a, as far as mtimes show (F8).

Conditions and open items:
- F1 (Medium): fix or replace the coder's registration regression test. My test file already covers this.
- F3 and F7 (Low): decide the 479-point contract and whether unknown prediction ids should raise.
- F4 (Low): correct the report 12 section 8 claim, and note that the source fingerprint is from run time.
- Coverage gaps 1 to 3 need live or extended runs that were out of scope or not possible offline.

Files:
- New tests: `/Users/test/Makeup_demo/backend/tests/test_phase3_independent.py`
- This report: `/Users/test/Makeup_demo/docs/plans/reports/13_test_report_phase3_eval.md`
- Production files examined: `/Users/test/Makeup_demo/backend/app/services/segmentation/face_parser.py`, `/Users/test/Makeup_demo/backend/app/services/facial_analysis/landmarks.py`
- Harness examined: `/Users/test/Makeup_demo/backend/evaluation/` (`labels.py`, `metrics.py`, `remote_zip.py`, `dataset.py`, `runner.py`, `report.py`, `cli.py`)
- Committed outputs checked: `/Users/test/Makeup_demo/docs/evaluation/easyportrait_test_2026-11-10.{md,json}`, `/Users/test/Makeup_demo/docs/evaluation/easyportrait_test_sample_seed20261110_n600.txt`
