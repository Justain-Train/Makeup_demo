# Robustness suite on EasyPortrait dev images (pilot)

Date: 2026-11-10. 100 images, seeded subset of the **dev** split (seed 20261110); perturbations are applied to image and annotation together. Numbers are *agreement with EasyPortrait annotations* (dataset-level IoU, full image; rotation fill is not scored), not an accuracy figure. Positive roll = counter-clockwise.

Source fingerprint `91d43308e2b690809f23d1f9410099efbb8c369b7e7135a74e42578af0849c4a`; crop scale 2.0; one machine (Apple M2 Pro).

## Detection and agreement per perturbation

B = all images (failures are empty predictions); delta = paired difference to the unperturbed baseline with a paired 95% bootstrap CI.

| perturbation | detected | failure rate | skin B | lips B | eye B | brow B | eye B delta [CI] | brow B delta [CI] | lips B delta [CI] |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 100/100 | 0.0% | 0.866 | 0.780 | 0.716 | 0.712 | - | - | - |
| mirror | 99/100 | 1.0% | 0.866 | 0.768 | 0.708 | 0.708 | -0.007 [-0.017, 0.002] | -0.004 [-0.014, 0.007] | -0.012 [-0.024, 0.000] |
| roll+15 | 99/100 | 1.0% | 0.858 | 0.752 | 0.681 | 0.680 | -0.035 [-0.048, -0.023] | -0.031 [-0.051, -0.009] | -0.029 [-0.043, -0.016] |
| roll-15 | 99/100 | 1.0% | 0.858 | 0.748 | 0.694 | 0.670 | -0.022 [-0.039, 0.003] | -0.042 [-0.057, -0.028] | -0.032 [-0.050, -0.016] |
| roll+30 | 98/100 | 2.0% | 0.825 | 0.658 | 0.591 | 0.551 | -0.125 [-0.161, -0.088] | -0.160 [-0.199, -0.124] | -0.122 [-0.158, -0.089] |
| roll-30 | 99/100 | 1.0% | 0.817 | 0.665 | 0.585 | 0.597 | -0.131 [-0.186, -0.058] | -0.115 [-0.141, -0.091] | -0.116 [-0.155, -0.081] |
| roll+45 | 97/100 | 3.0% | 0.706 | 0.277 | 0.361 | 0.173 | -0.355 [-0.427, -0.276] | -0.539 [-0.592, -0.482] | -0.504 [-0.591, -0.422] |
| roll-45 | 98/100 | 2.0% | 0.710 | 0.429 | 0.361 | 0.286 | -0.355 [-0.415, -0.303] | -0.426 [-0.486, -0.374] | -0.351 [-0.426, -0.280] |
| scale0.5 | 99/100 | 1.0% | 0.865 | 0.775 | 0.707 | 0.702 | -0.009 [-0.013, -0.005] | -0.010 [-0.017, -0.004] | -0.005 [-0.008, -0.002] |
| jpeg30 | 100/100 | 0.0% | 0.865 | 0.778 | 0.714 | 0.708 | -0.002 [-0.006, 0.001] | -0.004 [-0.008, 0.001] | -0.003 [-0.007, 0.001] |
| blur | 99/100 | 1.0% | 0.861 | 0.772 | 0.711 | 0.703 | -0.005 [-0.012, 0.002] | -0.009 [-0.016, -0.002] | -0.008 [-0.014, -0.004] |
| gamma0.5 | 99/100 | 1.0% | 0.860 | 0.785 | 0.719 | 0.695 | +0.003 [-0.003, 0.010] | -0.017 [-0.031, -0.006] | +0.005 [-0.003, 0.013] |
| gamma2.0 | 98/100 | 2.0% | 0.866 | 0.751 | 0.689 | 0.697 | -0.027 [-0.046, -0.014] | -0.014 [-0.024, -0.005] | -0.029 [-0.045, -0.016] |
| gray | 98/100 | 2.0% | 0.860 | 0.740 | 0.696 | 0.700 | -0.020 [-0.030, -0.009] | -0.012 [-0.022, -0.001] | -0.040 [-0.062, -0.019] |
| vflip | 36/100 | 64.0% | 0.305 | 0.035 | 0.184 | 0.000 | -0.531 [-0.618, -0.438] | -0.712 [-0.752, -0.656] | -0.745 [-0.798, -0.681] |

Denominator A (detected images only) is in the JSON. Failures are `no_face` / `multiple_faces` outcomes: mirror: {'ok': 99, 'no_face': 1, 'multiple_faces': 0, 'pipeline_error': 0}; roll+15: {'ok': 99, 'no_face': 1, 'multiple_faces': 0, 'pipeline_error': 0}; roll-15: {'ok': 99, 'no_face': 1, 'multiple_faces': 0, 'pipeline_error': 0}; roll+30: {'ok': 98, 'no_face': 2, 'multiple_faces': 0, 'pipeline_error': 0}; roll-30: {'ok': 99, 'no_face': 1, 'multiple_faces': 0, 'pipeline_error': 0}; roll+45: {'ok': 97, 'no_face': 3, 'multiple_faces': 0, 'pipeline_error': 0}; roll-45: {'ok': 98, 'no_face': 2, 'multiple_faces': 0, 'pipeline_error': 0}; scale0.5: {'ok': 99, 'no_face': 1, 'multiple_faces': 0, 'pipeline_error': 0}; blur: {'ok': 99, 'no_face': 1, 'multiple_faces': 0, 'pipeline_error': 0}; gamma0.5: {'ok': 99, 'no_face': 1, 'multiple_faces': 0, 'pipeline_error': 0}; gamma2.0: {'ok': 98, 'no_face': 2, 'multiple_faces': 0, 'pipeline_error': 0}; gray: {'ok': 98, 'no_face': 2, 'multiple_faces': 0, 'pipeline_error': 0}; vflip: {'ok': 36, 'no_face': 64, 'multiple_faces': 0, 'pipeline_error': 0}

## GT-free consistency (equivariance)

Mirror: IoU between the prediction on the mirrored image and the mirrored prediction on the original (sides swapped back). Roll: between the prediction on the rotated image and the rotated original prediction. Pixels parsed in both and not rotation fill only; images detected in both.

| perturbation | images | skin | lips | eye | brow | eye_R | eye_L | brow_R | brow_L |
|---|---|---|---|---|---|---|---|---|---|
| mirror | 99 | 0.970 | 0.890 | 0.855 | 0.854 | 0.852 | 0.858 | 0.854 | 0.855 |
| roll+15 | 99 | 0.957 | 0.848 | 0.800 | 0.795 | 0.787 | 0.812 | 0.782 | 0.807 |
| roll-15 | 99 | 0.955 | 0.851 | 0.806 | 0.803 | 0.824 | 0.788 | 0.806 | 0.801 |
| roll+30 | 98 | 0.911 | 0.715 | 0.671 | 0.612 | 0.664 | 0.677 | 0.571 | 0.648 |
| roll-30 | 99 | 0.889 | 0.732 | 0.605 | 0.660 | 0.676 | 0.536 | 0.662 | 0.659 |
| roll+45 | 97 | 0.753 | 0.273 | 0.375 | 0.180 | 0.254 | 0.481 | 0.126 | 0.224 |
| roll-45 | 98 | 0.769 | 0.461 | 0.389 | 0.296 | 0.523 | 0.247 | 0.356 | 0.238 |
| vflip | 36 | 0.753 | 0.103 | 0.505 | 0.000 | 0.482 | 0.523 | 0.000 | 0.000 |

## Roll: where the failures start

| roll | detected | eye B | brow B | lips B |
|---|---|---|---|---|
| baseline | 100/100 | 0.716 | 0.712 | 0.780 |
| roll+15 | 99/100 | 0.681 | 0.680 | 0.752 |
| roll-15 | 99/100 | 0.694 | 0.670 | 0.748 |
| roll+30 | 98/100 | 0.591 | 0.551 | 0.658 |
| roll-30 | 99/100 | 0.585 | 0.597 | 0.665 |
| roll+45 | 97/100 | 0.361 | 0.173 | 0.277 |
| roll-45 | 98/100 | 0.361 | 0.286 | 0.429 |

## Vertical-flip sanity check for the residual y offset

Median best integer shift of the parser's prediction to the GT within +-3 px (+dy = prediction must move down in that image). A convention bias of the model (labels sit lower on the face than the annotators draw them) moves with the face, so its dy flips sign when the face is upside down; a pipeline coordinate bug is tied to the image axes and keeps its sign.

| region | upright dx, dy (n) | vertically flipped dx, dy (n) |
|---|---|---|
| eye | 0.000, 1.000 (93) | 0.000, -1.000 (32) |
| lips | 1.000, 1.000 (100) | 0.000, 1.000 (11) |
| brow | 0.000, 2.000 (100) | 0.000, 0.000 (26) |

Detection rate on upside-down faces: 36.0% (36 of 100). Enough detect for a comparison, but only the detected (right-way-up-compatible) subset enters it.

## Mirror: median best shifts

| region | upright | mirrored |
|---|---|---|
| eye | 0.000, 1.000 | 1.000, 1.000 |
| lips | 1.000, 1.000 | 0.000, 2.000 |
| brow | 0.000, 2.000 | 0.000, 2.000 |

## Limitations

- 100 images from the dev split only; per-perturbation intervals are wide. Not a test-split decision.
- Blur sigma 3 px and rotation fill value 0 are single settings. Gamma and grayscale change the image only.
- Rotation canvases are expanded, so large rolls enlarge the image; fill is excluded from scoring and from the equivariance counts.
- GT is rotated with nearest-neighbour interpolation, which adds a boundary error of up to about 1 px.

Data: EasyPortrait (Kapitanov, Kvanchiani, Kirillova, arXiv:2304.13509), https://github.com/hukenovs/easyportrait ; licence variant of CC BY-SA 4.0 (see the main evaluation report). Evaluation only, aggregates only.
