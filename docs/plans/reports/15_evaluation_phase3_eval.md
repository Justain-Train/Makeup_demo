# Evaluation: Phase 3 Completion, Iteration 1 (2026-11-10)

Agent: `evaluator`. The agent has no write tool, so the orchestrator saved this report from its returned text. It did not run pytest itself and relied on the Tester's result: 350 passed, 0 failed.

## Verdict

**FAIL.** All required fixes are documentation only. The pilot does not need to be re-run.

## 1. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | WP4a | **Met.** See note below. |
| 2 | Sample, fetch, run | **Met.** Deterministic; guards tested offline. |
| 3 | Metric unit tests | **Met.** |
| 4 | Weights-gated plumbing test | **Met.** |
| 5 | Report | **Partial.** Defects are listed in section 3. |
| 6 | Benchmark and robustness outputs | **Met** (existence checked). |
| 7 | No data in repo; `backend/app` changed only for WP4a | **Partial.** See note below. |

**Criterion 1 (WP4a).**
- The registration fix (`face_parser.py:226-244`) is centre-consistent. The worst residual shift is 0.002 px.
- The truncated-landmark status is in `landmarks.py:153-164`.
- The Coder's registration test does not catch mutant M3. The independent `test_phase3_independent.py` catches M1 and M3.

**Criterion 7 (repository data).**
- The claim that `backend/app` changed only for WP4a rests on file modification times, because there is no version control.
- `tests/fixtures/portrait.jpg` has no provenance record.

## 2. Decision rules, checked against the JSON

- **Rule 1: not acceptable. Correct.**
  - Skin IoU is 0.866 against a target of 0.90. It misses by 0.034, which exceeds the 0.03 tolerance, and the CI is [0.855, 0.876].
  - Boundary F at 0.02·IOD is 0.683 for lips, 0.751 for eyes and 0.653 for brows. All are below 0.80.
  - Detection failure is 0.33%.
- **Rule 2: triggered. Correct.**
  - The median best shift is (0, +2) for lips and brows and (0, +1) for eyes. About 97% of images have a non-zero shift.
  - The report's roll item is stale (see section 3).
- **Rule 3: "no trigger" is true for the headline numbers only.** The rule's wording does not cover strata, so these must be stated as caveats:
  - eye IoU is 0.00 on the 48 glasses images;
  - eye IoU is 0.168 on the 93 alignment failures.
- **Rule 4: not triggered. Correct.** Detection loss is 4.5% of the denominator-B skin loss.

## 3. Required changes before PASS (documentation only)

1. **Wrong skin explanation.** The report says EP SKIN "may include neck". The confusion view shows NECK falls under PERSON. Replace this with the precision finding from Run 2.
2. **Latency marked "met" without caveats.** The 1–2 MP p95 of 291 ms has no timing caveat, although the run was on battery and the machine slept during it. Add the caveat and cite the benchmark: 2 MP p95 176 ms, 0 sleep events, still on battery. Mark the latency target provisional and not yet measured on AC power.
3. **Stale rule-2 roll item.** It still says "not assessed (needs WP3)". WP3 is now done: roll ±30° costs 0.12–0.16 IoU and roll ±45° costs 0.35–0.54. That meets the rule-2 roll item, and acting on it needs WP6 approval.
4. **Rule-3 caveat.** Add the glasses and alignment-failure strata, and state that the fine-tuning decision has not been made.
5. **Fingerprint narrative.** The recorded fingerprint `4718831d…` is the pilot-run tree, not the current one (`d829a240…`). Say so in the report header. Also correct report 12 §8, which says "report.py was not edited after the run".
6. **Y-offset.** State that its cause is unexplained (labelling convention or pipeline). Also note that the landmark polygons are offset by (−1, −1) while the parser is offset by (0, +2), so the two disagree in sign.
7. **Weighting.** State that the aggregates are over the stratified 600-image sample and are not re-weighted to the 6000-image test split.

## 4. Findings triage

| Finding | Decision |
|---|---|
| Security F1 (LOW) | Defer |
| Tester F1 (Medium) | Defer, provided `test_phase3_independent.py` stays in the suite |
| Tester F3 | Defer |
| Tester F4 | Fix now (item 5 above) |
| Tester F7 | Defer |
| Tester F9 | Defer, but record provenance before Phase 3 is declared complete |
| Security F2 | Fixed in report 12. Reports 13 and 14 still contain absolute paths (cosmetic). |

## 5. Remaining before Phase 3 is complete

1. **Ground truth.** Evaluated on EasyPortrait only.
2. **Diverse faces.** Not met. Skin-tone bias is unquantified, and there are no tone labels yet. This needs a tone-labelled dataset or the user's own photos.
3. **Boundary improvements.** Measured but not fixed. They need dev-tuned WP6 fixes, including the y-offset, followed by one confirmation run on the test split (WP7).
4. **Fine-tuning decision.** The evidence is in hand. The decision belongs to the user, after they confirm the D2 targets and after the WP6 fixes.
5. **User actions.** The user needs to:
   - confirm the targets;
   - approve or reject WP6 and WP6-G;
   - record the fixture's provenance.
6. **Latency.** Re-measure on AC power.

## 6. Scope

No WP6, WP6-G or Phase 4 work was done. The pre-deployment blockers (F1/F2 memory and body cap, rate limiting, authentication, legal review) remain open.
