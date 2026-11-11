# Evaluation: Phase 3 Completion, Iteration 3, Final (2026-11-10)

Agent: `evaluator`. The orchestrator saved this report from the agent's returned text, because the agent has no write tool.

**Verdict: PASS** for this iteration's deliverables. The iteration-2 defect, where the JSON and the Markdown disagreed (report 16, §2), is fixed. **Phase 3 itself is not complete** (see "What remains" below).

## Checks

1. **The JSON carries the decision-rule conclusions.** In `docs/evaluation/easyportrait_test_2026-11-10.json`, lines 341–353:
   - `rule2.not_assessed` is now an empty list, `[]`.
   - `rule2.roll_from_wp3` states that the roll item is met: ±30° costs 0.12–0.16 and ±45° costs 0.35–0.54. A roll-aligned crop is a WP6 candidate and needs user approval.
   - `rule3.caveat` covers the glasses stratum (eye 0.00, n=48) and the alignment-failure stratum (eye 0.168, n=93). It states that the fine-tuning decision belongs to the user.
   - Rules 1 and 4 are unchanged.
2. **The Markdown agrees with the JSON.** In the `.md`, lines 317–318 contain the same text. `report.py` lines 714–718 render both from the JSON fields. The fingerprints are on lines 13–14:
   - pilot-run fingerprint: `4718831d…`;
   - current fingerprint: `1c022e37…`.
3. **No regressions found.** The orchestrator found no numeric change against iteration 2, and all 350 tests passed. The Evaluator did not re-run either check.

## Minor, non-blocking

- `rule2.triggers` has no structured roll entry. The roll result is recorded only as text in `roll_from_wp3`.

## What remains before Phase 3 is complete

1. **Diverse faces (CLAUDE.md item 2).** Skin-tone bias is not quantified. This needs a dataset labelled for skin tone, or the user's own consented photos.
2. **Boundary improvements (item 3).** The y-offset and the landmark-polygon sign mismatch are unexplained. They need dev-tuned WP6 work, which needs user approval, followed by one confirmation on the test split (WP7).
3. **Fine-tuning decision (item 4).** This belongs to the user. It comes after the targets are confirmed and after the WP6 fixes, and it must take the glasses and alignment-failure strata into account.
4. **User actions.**
   - Confirm the provisional D2 targets.
   - Approve or reject WP6 and WP6-G.
   - Record the provenance and licence of `backend/tests/fixtures/portrait.jpg`.
5. **Latency.** Re-measure with the machine on AC power.
6. **Deferred findings.**
   - Security F1: map zlib and struct errors to `FetchError`, and cancel pending futures.
   - Tester findings F1, F3 and F7.
   - Keep `tests/test_phase3_independent.py` in the suite.

This was iteration 3, the last one allowed by the loop limit. Any further changes need direction from the user.
