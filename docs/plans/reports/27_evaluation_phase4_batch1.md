# Evaluation: Phase 4 Batch 1 (2026-11-11)

First evaluation: REJECTED. The evaluator raised three issues:
- lip-corner confinement was not demonstrated after the metric change, and there was no negative test;
- the corner coverage values were not reported;
- criterion 7 rested on mtimes only.

Final verdict, after fix loop 2: **PASS**. Criteria 1-9 are met, based on the Tester (`25`) and Security (`26`) reports.
- 372 passed, 0 failed. No assertions were loosened.
- The 0.3 ipd negative test makes both corners `low_confidence`. Coverage is left 0.786 and right 1.0.
- Criterion 7 is supported by the bytecode-cache evidence.

Non-blocking notes:
- The left corner's strict IoU is 0.168 because the parser lip is thin there. The user confirmed the placement visually.
- The dilation clamp is optional.
