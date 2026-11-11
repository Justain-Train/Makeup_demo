# Coder report 12: Phase 3 completion, WP4a + WP1/WP5 (2026-11-10)

Scope: WP4a (two bug fixes) and WP1 + WP5 (EasyPortrait evaluation harness, executed at N = 600). Not WP2, WP3, WP6, WP7. Not committed.

## 1. Files

Created:
- `backend/evaluation/` : `__init__.py`, `labels.py` (only place with EasyPortrait ids), `metrics.py` (pure numpy), `remote_zip.py` (HTTP Range zip reader and guards), `dataset.py` (cache, meta, sampling, fetch, annotation decoding), `runner.py`, `report.py`, `cli.py`. WP2/WP3 can plug in as new modules reusing `dataset`, `runner.evaluate_image`, `report`.
- `backend/tests/test_eval_metrics.py` (15 tests), `test_eval_dataset.py` (37), `test_eval_plumbing.py` (9).
- `docs/evaluation/easyportrait_test_sample_seed20261110_n600.txt` (600 UUIDs, split, strata), `easyportrait_test_2026-11-10.md`, `easyportrait_test_2026-11-10.json`.
- this report.

Modified:
- `backend/app/services/segmentation/face_parser.py` (registration), `backend/app/services/facial_analysis/landmarks.py` (truncated landmarks). Nothing else in `backend/app/`.
- `backend/tests/test_face_parser.py` (+1 test), `backend/tests/test_face_segmentation_api.py` (+4 tests).
- `docs/face_segmentation.md` (registration fix, truncated-landmark limitation, roll failure at about +/-35 deg, "Evaluation results" pointer). `CLAUDE.md` not edited.
- `.gitignore` (defensive evaluation entries).

## 2. WP4a

Registration fix: forward warp is now centre-aligned (`dst = k*(src+0.5)-0.5`) using the scale actually produced by the pre-shrink (`sub.shape`). A second cause found while testing: `cv2.resize(fx=, fy=)` with INTER_AREA averages with the nominal factor even though the output size is rounded; the pre-shrink now passes an explicit output size. (Fixing only the matrix left +0.32 px at window 2650.)

Measurement (stub `_infer` echoing input channel 0; smooth logistic edge; window == image; mean edge shift in the network input in original px, + = right/down; averaged over x and y edges, 3 sub-pixel phases):

| Window | Before | After | Label-map edge error before / after |
|---|---|---|---|
| 600 | +0.09 | -0.01 | 0..1 / 0 |
| 1000 | +0.47 | -0.00 | 0..1 / 0 |
| 1000, extends past image | +0.47 | -0.00 | 0..1 / 0 |
| 2000 (pre-shrink) | +0.99 | +0.01 | 1..2 / 0 |
| 2650 (pre-shrink) | +1.43 | +0.01 | 0..3 / 0 |

(Before values used edge phases 0.0/0.3/0.6, after values 0.2/0.5/0.8; largest single-case |shift| after the fix was 0.075 px.) Regression test: `test_registration_is_centre_consistent_in_network_input_and_label_map` (asserts input shift <= 0.1 px, label edge within 1 px, mean signed error < 0.5).

Truncated landmarks: fewer than 478 points (including 0) gives HTTP 200, `status="no_face_detected"`, `face_detected=false`, `landmark_count=0`, the exact message from the plan; log is a count only. Empty-face area guard added. Tests: 477, 1 and 0 points on both endpoints, plus 478 unchanged.

Suite: 191 before; 257 now, all pass (191 + 5 WP4a + 61 evaluation). No pre-existing expectation moved; no thresholds changed.

## 3. Commands run

```
cd backend
.venv/bin/python -m evaluation.cli sample          # twice: output byte-identical (deterministic)
.venv/bin/python -m evaluation.cli fetch --threads 4 --budget-gb 3
.venv/bin/python -m evaluation.cli run --limit 12 --out-dir <scratch>/smoke   # smoke test, deleted
.venv/bin/python -m evaluation.cli run
.venv/bin/python -m evaluation.cli report --run-dir ~/.cache/makeup_demo/easyportrait_runs/2026-11-10
.venv/bin/pytest -q
```
Cache: `~/.cache/makeup_demo/easyportrait` (0700/0600, marker present, 1.2 GB; remove with `python -m evaluation.cli clean --yes`). Run output (outside repo): `~/.cache/makeup_demo/easyportrait_runs/2026-11-10/{per_image.jsonl,run_meta.json}`.

HEAD-resolved pins: `v2/test.zip` 12,794,142,394 B; `v2/annotations.zip` 657,142,345 B; `v2/meta.zip` 1,941,735 B (ETags in `dataset.py`). `annotations/meta.zip` (README alternative) returns 403; `v2/meta.zip` used. Members: `images/test/<uuid>.jpg` (6000), `annotations/test/<uuid>.png` (40,008 entries across splits). meta.tsv is comma-separated despite its name; height/width are floats ("1920.0"). Test split in meta: 6000 rows, 0 malformed.

## 4. Download

- Fetch: 600/600 samples, 1,233,132,390 bytes (1233.1 MB; includes 6.5 MB central directories) in 1204 requests, 457.0 s, 2.70 MB/s. Budget 3 GB. 0 failed.
- **Incident (my error):** while inspecting zip layout I ran two `curl -r -4000000-` commands with a malformed Range header (trailing dash). The server ignored it and streamed full files: annotations.zip was downloaded completely (657 MB) and test.zip partially (about 722 MB) before I killed it. This was outside the harness and outside its byte budget; total network use was therefore about 1.38 GB more than the harness figure. Both files were in the scratchpad (outside the repo), were not used for anything, and were deleted. The harness itself requires 206 and aborts on 200 (unit tested).

## 5. Run

- Elapsed 486.8 s for 600 images (about 0.81 s/image, sequential, 3 warm-ups; services built like the app lifespan). Peak RSS 1470 MB (process, includes MediaPipe landmarker; macOS `ru_maxrss` bytes converted; `/usr/bin/time -l` agrees: 1,470,382,080 B). The `time -l` "real" figure (5064 s) is larger than the run loop because the machine was suspended/the command was waiting while I was cut off; the loop timer is the reliable one. Latency outliers (max total 2584 ms) may include such effects.
- Outcomes: ok 598, no_face 2, multiple_faces 0, pipeline_error 0, invalid_sample 0 (limit was 1%). Split: dev 239, test 361. 16 strata, all populated.

## 6. Headline (dataset-level IoU, 95% bootstrap CI; full report has Dice/precision/recall/per-image stats)

| class | A full | A window | B full | B window |
|---|---|---|---|---|
| skin | 0.866 [0.855, 0.876] | 0.866 | 0.865 [0.854, 0.876] | 0.865 |
| lips | 0.787 [0.767, 0.805] | 0.787 | 0.786 [0.768, 0.803] | 0.786 |
| eye (merged) | 0.717 [0.695, 0.738] | 0.717 | 0.716 [0.695, 0.737] | 0.716 |
| brow (merged) | 0.710 [0.695, 0.723] | 0.710 | 0.709 [0.695, 0.722] | 0.709 |

Window = full image because 0.00% of GT pixels for every class fall outside the parse window (EasyPortrait faces fill the frame, window = 2x face box). Sided rows (eye_R/L, brow_R/L) are 0.70-0.72. Side semantics: 0 GT order exceptions, 0 swaps (545 eye, 593 brow images with defined overlap). Dev vs test (A full): skin 0.862/0.868, lips 0.791/0.785, eye 0.709/0.723, brow 0.700/0.715.

Other findings in the report: boundary F at 0.02 IOD (mean per image) lips 0.683, eye 0.751, brow 0.653; at 0.05 IOD 0.942/0.883/0.923. Best-shift: median (0, +2) px lips, (0, +1) eye, (0, +2) brow (prediction needs to move down), non-zero in 97% of images. Area ratios 0.99-1.10. Missed EP SKIN pixels go mostly to GLASSES (1.1% of GT); 10.1% of our skin+nose pixels lie in EP PERSON (see Run 2). Strata: eye IoU is 0.00 in the 48 glasses-detected images (0.76 without) and 0.17 in the 93 images where `alignment_check` failed (0.78 when passed); face-size low tercile skin 0.82 vs high 0.88. Landmark polygons vs GT: eye 0.730 vs parser 0.717; whole mouth 0.726 vs 0.695. Latency (total, p50/p95 ms): 1-2 MP (n=59) 215/291; 2-4 MP (n=366) 270/409; >=4 MP (n=163) 532/942.

## 7. Provisional targets and decision rules (labelled provisional in the report)

- Targets: skin IoU 0.90 **missed** (0.866, CI upper 0.876; miss 0.034 > 0.03 tolerance). Lips IoU 0.75, eye 0.65, brow 0.60 **met**. Boundary F @0.02 IOD >= 0.80 **missed** for lips (0.683), eye (0.751), brow (0.653). Side swap <= 1% met (0%). Worst-stratum gaps (brightness, luma) met on the lower-bound criterion (largest point gap 0.075 lips Q4 vs Q2, lower bound 0.021). Latency 1-2 MP p95 291 ms <= 300 met (n=59, this machine).
- Rule 1 (acceptable): not met (skin IoU and the three boundary-F targets). Detection failure rate 0.33% (<= 2%).
- Rule 2 (post-processing/geometry fix): triggered by a consistent best-shift of 1-2 px downward for lips, eye, brow. Not triggered: boundary F @0.05 is >= 0.85 but IoU is above target; area ratios are in 0.85-1.15; no stray false positives. Roll concentration not assessable (needs WP3).
- Rule 3 (replacement/fine-tuning): no trigger on the pilot numbers (no small region below 0.50; gaps small; boundary F @0.05 >= 0.70). Caveat: glasses and alignment-failure subsets are very poor (see above) and are not captured by the headline.
- Rule 4 (detector lever): detection failures are 4.5% of B skin loss, so no.
- Rule 5: model still replaced before commercial release.
These are pilot numbers on the full sample; per the protocol, tune on dev only and decide once on test.

## 8. Technical decisions and deviations

- Sample file has two extra columns (brightness quartile, resolution group) so the report can stratify without the meta table; resolution groups <1.5, 1.5-2.5, 2.5-3.0, >=3.0 MP (chosen from the observed distribution, 2.76 MP dominates).
- Allocation: floors first (min(cell,5)), remainder by largest remainder on spare capacity; gives exactly 600 without trimming.
- Fetch reads each member with one ranged GET (local header + data), my own CRC/size/zip-bomb checks via `zlib.decompressobj`; central directory parsed with `zipfile` over a tail-only file object.
- In-window scoring for failed images (denominator B) counts the whole GT as missed because no window exists; documented in the report. Boundary F and best-shift use full-image masks.
- Side-order exception = GT centroid x ordering; swap = overlap-based.
- Report text nit: the "A, window" heading repeats the sentence about failed images in B. The source-tree fingerprint in the report is that recorded at run time (`4718831d...`); (Corrected in Iteration 2: this was wrong. report.py was edited after the run in Run 2 (prediction view) and Iteration 2 (text); the recorded fingerprint is the pilot-run tree only.)
- Plan deviation: none intended besides the above. WP2 and WP3 not done.

## 9. Security and privacy

https only, no redirects, 206 + Content-Range required (200 aborts), pinned Content-Length via HEAD, byte budget, member caps (images 25 MB, annotations 2 MB), CRC verified, no extraction, UUID-regex names, cache outside repo (in-repo and symlink-into-repo paths refused, tested), 0700/0600, marker-guarded `clean`. `user_id` never read into outputs (parsed by header name, discarded; tested). Logs carry ids, counts, timings, exception type names. Only the UUID list and aggregates are under `docs/evaluation/` (verified: no `user_id` string in the report; every UUID in the report belongs to the sample). `.gitignore` has defensive entries.

## 10. Licence and attribution

EasyPortrait (Kapitanov, Kvanchiani, Kirillova, arXiv:2304.13509), https://github.com/hukenovs/easyportrait. Licence: "a variant of Creative Commons Attribution-ShareAlike 4.0" (reworked, not a CC licence; no non-commercial clause; attribution and share-alike apply on distribution; portrait/privacy rights not granted; database rights under Russian law): https://github.com/hukenovs/easyportrait/blob/master/license/en_us.pdf . Used for evaluation only, aggregate results only, no redistribution, no training. README BibTeX is reproduced verbatim in the report.

## 11. Open issues

1. Consistent 1-2 px downward best-shift for small regions: could be annotation convention or a residual vertical offset in the pipeline (the registration fix was in place during the run). Not investigated; needs a decision (WP6 candidate, tune on dev).
2. Eyes under glasses (IoU 0.0 in 48 images) and alignment-failed images (93, eye IoU 0.17): the dominant weakness hidden by the headline; partly a convention difference (EP labels eyes behind glasses, our model labels glasses).
3. Skin IoU 0.866 vs target 0.90: precision issue (corrected in Run 2): precision 0.888, recall 0.973, area ratio 1.10.
4. Boundary-F targets (0.80 @0.02 IOD) not met; may reflect EP vs CelebAMask conventions.
5. Skin-tone bias unquantified; EP has no tone labels. 1-2 MP latency bucket is small (n=59), single machine.
6. Latency outliers possibly inflated by machine suspend during the run.
7. WP2 (benchmark), WP3 (robustness) pending; `evaluate_image`, `dataset`, `report` are reusable.

6. (Run 2 correction) The pilot's per-stage latency numbers (run 19:18-20:43) were taken on battery with sleep periods in the window; treat them as unreliable. See Run 2.

## Run 2: WP2 benchmark and WP3 robustness (2026-11-10)

Offline, existing cache, no network; `backend/app/` untouched. New: `evaluation/bench.py`, `evaluation/perturb.py`, `evaluation/predict.py`; `metrics.confusion_matrix` gained an optional `valid` mask (rotation fill is not scored); `report.py` gained a "where our predictions fall" view (pilot report regenerated from the existing `per_image.jsonl`; all other content byte-identical in JSON). Tests: `tests/test_eval_perturb_bench.py` (16: image+GT transformed together, side swap on mirror and none on vflip, rotation fill excluded from scoring, rotation direction, other perturbations, equivariance, seeded dev subset, aggregation, `ru_maxrss` units bytes vs KB, worker-output parsing, label-map cap). Full suite: **273 passed** (257 + 16).

Commands: `caffeinate -dims .venv/bin/python -m evaluation.bench` (re-run), `.venv/bin/python -m evaluation.perturb` (one run, 100 dev images, seed 20261110, about 7 minutes of pipeline time), `.venv/bin/python -m evaluation.cli report --run-dir ~/.cache/makeup_demo/easyportrait_runs/2026-11-10`.

### Benchmark (docs/evaluation/benchmark_2026-11-10.md)
The first benchmark run overlapped a system sleep (20:53-21:09) on battery and was discarded. The re-run was under `caffeinate -dims`, still **on battery power** (76-77%, discharging), and `pmset -g log` shows **0 sleep events** in its window (23:16-23:19). Versus the first run the numbers differ by at most 6% (25 MP total p50 728 vs 769 ms; 2 MP 173 vs 179 ms; peak RSS within 1.5%), so the first run's latency and memory were not materially distorted; its sleep gap fell in the cap experiment.
- Total latency p50/p95 (ms, synthetic upscales of one face): 0.8 MP 141/147; 2 MP 173/176; 10 MP 374/380; 25 MP 728/746. Inference is flat at about 82 ms; zones and encode grow with pixels.
- Concurrency (parser slots 2, fresh process each): 2 MP peak RSS 680/713/736 MB for 1/2/4 threads, throughput 5.7/7.8/8.2 req/s; 25 MP peak RSS 1521/1880/2214 MB, throughput 1.31/2.50/3.88 req/s. The F1 memory concern is confirmed: a 4-way burst of 25 MP images peaks at about 2.2 GB (landmarker and decode/label-map memory are not covered by the parser semaphore).
- Label-map cap experiment (dev, 239 detected): IoU skin/lips/eye/brow with no cap 0.8624/0.7912/0.7087/0.7003; cap 2048 (29 images changed) 0.8627/0.7933/0.7109/0.7027; cap 1024 (180 images changed) 0.8624/0.7968/0.7150/0.7099. Capping the output resolution did not lower agreement here (it rose by up to 0.01, likely smoothing; no CI computed). This supports a resolution cap as a memory fix without an agreement cost on this data; it does not test large phone photos, and it changes the API contract (not done).

### Robustness (docs/evaluation/robustness_2026-11-10.md; 100 dev images)
- Baseline B IoU: skin 0.866, lips 0.780, eye 0.716, brow 0.712. Mirror, scale 0.5, JPEG q30, blur, gamma 0.5, gamma 2.0, grayscale: eye/brow/lips deltas between +0.003 and -0.040; failure rate 0-2%.
- Roll: +/-15 costs 0.02-0.04; +/-30 costs 0.12-0.16 (eye +30: -0.125 [-0.161, -0.088]); +/-45 costs 0.35-0.54 (brow +45: 0.173 vs 0.712; lips 0.277). Detection stays 97-99%, so the loss is parsing/alignment, not the detector: this supports a roll-aligned crop (decision rule 2 trigger, for WP6 approval).
- GT-free: mirror equivariance IoU eye 0.855, brow 0.854, lips 0.890, skin 0.970 (sides consistent: eye_R 0.852, eye_L 0.858); roll 30 degrees 0.60-0.73 for small regions, roll 45 degrees 0.18-0.52.
- Vertical flip: only 36 of 100 upside-down faces were detected (64% failure), so the comparison rests on small, selected subsets and the parser itself is poor on upside-down faces (eye B IoU 0.18). Median best shift (dx, dy), upright vs flipped: eye (0, +1) n=93 vs (0, -1) n=32; lips (+1, +1) n=100 vs (0, +1) n=11; brow (0, +2) n=100 vs (0, 0) n=26. Eye flips sign (convention-like); lips do not (but n=11); brow shrinks to 0. **Inconclusive**, leaning towards a convention effect for eyes; not evidence of a pipeline coordinate bug, but the check is weak. Mirror shifts are similar to upright (dy +1..+2 persists).

### Skin precision (correction of the pilot write-up)
My earlier explanation (EP SKIN includes neck/body skin) was wrong. The confusion view shows EP PERSON, not SKIN, contains our NECK (8.1% of PERSON), and EP SKIN has 0.00% of its pixels outside the parse window. The skin shortfall is precision: precision 0.888, recall 0.973, area ratio 1.10. The new view of where our predictions fall: of our skin + nose pixels (252.3 M), 88.8% lie in EP SKIN, **10.1% in EP PERSON**, 0.22% in BACKGROUND, about 0.9% in brows/eyes/lips. EP PERSON is the catch-all for hair, neck, clothing, so the extra skin is skin-labelled pixels in regions EP calls person-but-not-skin (the data cannot say which sub-part: possibly beard/hairline/neck boundary; per-class breakdown inside PERSON is not available from EP). Lips: 84.2% in EP LIPS, 10.1% PERSON, 5.5% SKIN; eyes: 84% eye, 8.2% SKIN, 7.8% PERSON.

### Open findings (not fixed)
- `app/services/facial_analysis/geometry.py:33` converts landmarks as `x*w` without the -0.5 pixel-centre correction before `cv2.fillPoly` (orchestrator finding; consistent with the landmark-polygon best shift of about (-1, -1) px for eye and mouth). Affects zone polygons (Phase 4). Not fixed per instructions.
- Parser residual y offset of about +1..+2 px: orchestrator diagnostic on 40 images was eye (0, +2), lips (+0.5, +2); y-only, so unlikely to be a coordinate bug like the one fixed; the flip test is inconclusive.
- Pilot timing caveat recorded in `docs/face_segmentation.md`.
- Benchmark ran on battery power; the machine's performance state may differ from AC.

## Iteration 2 (documentation fixes after Evaluator FAIL; text and labelling only)

Text edits were made with the editor tools (no shell was available in this run). Regeneration, the JSON numeric diff and pytest were NOT run by me: results filled in by the orchestrator, 2026-11-10:

- **Regeneration:** OK, using `cli report` from the existing `per_image.jsonl`.
- **Numeric diff:** none. The JSON before and after was compared field by field; the only change is the text field `/targets/items[17]/note` (the latency caveat). The diff script was self-checked on a known pair.
- **pytest:** 350 passed.
- **Current-tree fingerprint** (printed in the report header): `c143cb9b0f41780f73ee48c863c3f18ab8ed5879064f4ebe3e27d0c83732d238`.
- **Pilot-run fingerprint:** `4718831d…`, unchanged.

All report.py locations are in `backend/evaluation/report.py`; no numeric logic and nothing in `backend/app/` changed.

| # | Item | Where fixed |
|---|---|---|
| 1 | Wrong "EP SKIN may include neck" limitation replaced by the precision finding (0.888 / 0.973 / 1.10; 10.1% of our skin+nose in EP PERSON) | `render_markdown`, first bullet of "## Limitations"; face_segmentation.md already had it |
| 2 | Latency "met" line: provisional, not measured on AC, pilot on battery with sleep events, cites re-run benchmark (2 MP p95 176 ms, 0 sleep events, battery) | `evaluate_targets`, `note` of the latency item (shown in the targets table); face_segmentation.md timing caveat |
| 3 | Rule-2 roll item updated with WP3 (+/-30: 0.12-0.16; +/-45: 0.35-0.54), met, needs WP6 approval | `render_markdown`, decision-rule line 2 (text appended after "Not assessed in this pilot run") |
| 4 | Rule 3: glasses (eye 0.00, n=48) and alignment-failure (0.168, n=93) strata, all 48 within the 93, decision not made, owed to user | `render_markdown`, decision-rule line 3 |
| 5 | Header prints both fingerprints: pilot-run tree (from run_meta.json) and current tree computed at report time; report 12 section 8 corrected | `render_markdown` run header (two bullets) + new helper `_current_fingerprint()` just above `render_markdown`; this report, section 8 |
| 6 | y-offset cause unexplained; landmark polygons (-1,-1) vs parser (0,+2), opposite signs; not fixed | `render_markdown`, third added bullet in "## Limitations"; face_segmentation.md open finding |
| 7 | Aggregates are over the stratified 600-image sample, not re-weighted to the 6000 test split | `render_markdown`, second added bullet in "## Limitations"; face_segmentation.md |

Note for the diff: the only JSON change expected is the `note` text of the latency item in `targets.items`; the current-tree fingerprint appears in the Markdown only. Also, `tests/test_eval_plumbing.py` checks for the "accuracy" wording: the new text contains none.
EOF


## Iteration 3 (2026-11-10)

This iteration fixes the Evaluator's FAIL in report 16, §2.

**Coder changes (`backend/evaluation/report.py`):**
- In `apply_decision_rules`, `rule2.not_assessed` is now `[]`.
- New text field `rule2.roll_from_wp3`.
- New text field `rule3.caveat`.
- `render_markdown` now renders both from the JSON rather than from hard-coded text, and prints the "Not assessed" clause only when that list is non-empty.

**Orchestrator verification (the Coder had no shell):**
- **Regeneration:** regenerated from the existing `per_image.jsonl`.
- **JSON diff against the iteration-2 copy:** the numeric diff is **empty**. The only structural or text changes are the three expected ones under `decision_rules`.
- **Markdown diff:** only the current-tree fingerprint (now `1c022e37e7099cb797bc3f5112a47459f682f80eeb51479dbfeab4bdb7c34f48`) and the removal of the stale "Not assessed" clause in rule 2.
- **pytest:** 350 passed.
