# WP6-G glasses handling: measurement before building (2026-11-11)

Source: the pilot run's per-image results (outside the repo, `easyportrait_runs/2026-11-10/per_image.jsonl`),
598 detected images of the 600-image EasyPortrait test sample. No re-run was needed. No image, annotation
or per-image row is reproduced here; only aggregates.

Definitions: "eye IoU" is merged-eye agreement with EasyPortrait annotations (EP LEFT_EYE + RIGHT_EYE vs our
EYE_L + EYE_R), full-image scoring. "Glasses share" is the fraction of EP eye pixels our parser labelled GLASSES.
"Glasses area" is the fraction of the label map labelled GLASSES (the quantity the 1 % warning threshold uses).

| Question | Result |
|---|---|
| Images with the `glasses_detected` warning | 48 of 598 |
| Eye IoU on warned images | 0.00 on all 48 (max 0.00) |
| Glasses share of EP eye pixels on warned images | median 1.00, minimum 0.89 |
| Glasses area on warned images | 3.1 % to 8.9 % (median 4.5 %) |
| Glasses area on non-warned images | at most 0.59 %; 541 of 550 have 0 % (9 have a trace, 0.01-0.59 %) |
| False alarms (warned and eye IoU >= 0.3) | 0 |
| Missed glasses (no warning and >= 5 % of EP eye pixels labelled GLASSES) | 0 |
| Alignment-check failures | 93; all 48 warned images are among them; 0 warned images passed alignment |
| Alignment failures without glasses | 45, mean eye IoU 0.41 |

Threshold sweep on the dev split (239 detected images; positive = eye IoU < 0.30): every threshold from 0.05 %
to 2 % fires on the same 19 images, all of which are collapsed, with 0 good images among them, and leaves the same
6 collapsed images unfired (those are not glasses images: their glasses area is 0). The gap between 0.59 % and
3.1 % means the 1 % threshold needs no tuning.

Interpretation:
- Behind glasses the parser labels the whole eye area GLASSES and both eye classes are empty. That is why the
  iris checks fail (hence every glasses image is an alignment failure) and why eye and under-eye zones are already
  `unavailable`. Nothing is guessed behind the lenses today, which is the behaviour WP6-G requires.
- The alignment-failure stratum (eye IoU 0.168, n=93) is therefore two populations: glasses (48, IoU 0.00) and
  other mismatches (45, IoU 0.41). The glasses population is handled by the WP6-G prompt, not by a parser fix.
- Phase 3 work is the signal only: keep the warning, add the additive `retake_suggested: ["glasses"]` hint and a
  generic UI prompt. The makeup-choice-dependent prompt waits for Phases 6, 10 and 11.

Caveat: EasyPortrait has no glasses annotation, so "missed glasses" is measured through our own GLASSES label on EP
eye pixels; glasses the parser does not see at all cannot be counted. Visual inspection of the no-warning images
is not possible without redistributing the data, so this remains an unquantified residual.
