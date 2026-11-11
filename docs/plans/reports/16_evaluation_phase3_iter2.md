# Evaluation: Phase 3 Completion, Iteration 2 (2026-11-10)

Agent: `evaluator`. The orchestrator saved this report from the returned text, because the agent has no write tool. The agent did not re-run pytest; the orchestrator reports 350 passed.

**Verdict: FAIL.** One narrow fix to the JSON and Markdown is needed. The pilot does not need to be re-run.

## 1. The seven required items from report 15, section 3

Line numbers refer to `docs/evaluation/easyportrait_test_2026-11-10.md` ("md") and `docs/face_segmentation.md` ("fs").

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Skin explanation | Fixed | md 324; fs 152–154 |
| 2 | Timing caveat | Fixed | md 308; fs 158–160 |
| 3 | Rule-2 roll item | Fixed in the Markdown only | md 317; fs 161–163 |
| 4 | Rule-3 caveat | Fixed in the Markdown only | md 318; fs 164–165 |
| 5 | Fingerprints | Fixed | md 13–14 (pilot-run `4718831d…`, current `c143cb9b…`) |
| 6 | Y-offset cause | Fixed | md 326; fs 166–170 |
| 7 | Not re-weighted | Fixed | md 325; fs 161–162 |

## 2. Required change before PASS

The JSON and the Markdown disagree on rule 2.

- `backend/evaluation/report.py:432` still writes the roll item into `decision_rules.rule2.not_assessed` as needing WP3, while the Markdown says it is met.
- The rule-2 and rule-3 caveats are hard-coded text in `render_markdown` (lines 703–707), so they are missing from the JSON.
- Acceptance criterion 5 requires the JSON and the Markdown to agree.

**Fix:**
1. Put the roll-met conclusion and the rule-3 caveat into the JSON `decision_rules` as text fields.
2. Render the Markdown from those fields.
3. Regenerate both files.
4. Confirm with a field-level diff that no number changed.

## 3. Changes in `report.py` are text only

The changes are:
- the `_current_fingerprint()` helper;
- the latency note (lines 411–412);
- Markdown text (lines 603–604, 703–707 and 711–714).

`aggregate_class`, `headline_tables`, `worst_stratum_gaps`, `apply_decision_rules` and the thresholds are unchanged.

## 4. Not checked in this pass

- The pytest run (taken from the orchestrator).
- The wording of report 12, §8.

## 5. Phase 3 is still not complete

The remaining work is unchanged from report 15:

1. **Skin tone:** bias across skin tones is not quantified.
2. **Boundaries and y-offset:** investigate them, apply the WP6 fixes after user approval, then confirm on the test split (WP7).
3. **Fine-tuning decision:** the user decides after confirming the targets.
4. **User actions:**
   - confirm the targets;
   - approve or reject WP6 and WP6-G;
   - record where the test fixture came from.
5. **Latency:** measure on AC power.

The deferred findings are unchanged: security F1, and tester F1, F3 and F7.
