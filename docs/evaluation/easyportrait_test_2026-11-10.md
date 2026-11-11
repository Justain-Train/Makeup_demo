# Agreement of the face-parsing pipeline with EasyPortrait v2 test annotations (pilot)

Date: 2026-11-10. Status: **pilot, provisional targets**. All numbers are *agreement with EasyPortrait annotations* on a stratified sample of the EasyPortrait v2 test split. They are not an accuracy figure for the product: the annotation conventions differ from the CelebAMask-HQ conventions the model was trained on, and skin-tone bias is not quantified (see Limitations).

## Run header

- Sample: `docs/evaluation/easyportrait_test_sample_seed20261110_n600.txt`: 600 ids (target 600, seed 20261110); dev 239, test 361.
- Parser: `bisenet-resnet18-celebamask-hq`, weights SHA-256 `0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f`.
- Landmarker SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- Crop scale 2.0, parser threads setting 0 (0 = onnxruntime default), OpenCV threads 10; sequential run, 3 warm-ups.
- Python 3.13.7; packages Pillow 13.0.0, mediapipe 1.2.2, numpy 2.5.4, onnxruntime 1.31.1.
- OS macOS-15.6-arm64-arm-64bit-Mach-O; CPU Apple M2 Pro (10 logical cores).
- Source-tree fingerprint of the **pilot-run tree** (recorded in run_meta.json when the run was made; SHA-256 over backend/app and backend/evaluation `*.py`): `4718831d1215e9cabaebba36e4bfd67023b1dccfede1682011c1e4e3b8c3a381`. not a git repository: no commit hash available.
- Source-tree fingerprint of the **current tree at report time** (the evaluation code has changed since the run, e.g. this report generator): `1c022e37e7099cb797bc3f5112a47459f682f80eeb51479dbfeab4bdb7c34f48`.
- Elapsed 486.8 s for 600 images; peak RSS 1470 MB (process, includes the MediaPipe landmarker).

## Outcomes

| outcome | images |
|---|---|
| ok | 598 |
| no_face | 2 |
| multiple_faces | 0 |
| pipeline_error | 0 |
| invalid_sample | 0 |

Invalid-sample reasons: none. Pipeline-error types: none.
Denominator A = images with outcome `ok` only. Denominator B = all valid images (everything except `invalid_sample`); a failed image counts as an empty prediction. `multiple_faces` is a deliberate rejection of the pipeline; it is still a miss in B.

## Headline tables

Dataset-level (pixel-weighted) scores with 95% bootstrap CIs (1000 resamples over images, seeded). Per-image statistics use images whose GT region has at least 25 px. The false-alarm column is the share of images with no GT region (< 25 px) where the prediction has at least 25 px. Comparison classes: skin = EP SKIN vs our skin + nose; lips = EP LIPS vs our upper + lower lip; eye / brow merged; sided rows map EP LEFT_* to our `_R` (see Side handling). PERSON and BACKGROUND are not scored.

### Denominator A, scoring: full image (pixels outside the parse window count as 'not predicted')

| class | images | IoU | IoU 95% CI | Dice | precision | recall | per-image IoU mean / median / p10 / p90 | area ratio | false alarms |
|---|---|---|---|---|---|---|---|---|---|
| skin | 598 | 0.866 | [0.855, 0.876] | 0.928 | 0.888 | 0.973 | 0.849 / 0.893 / 0.679 / 0.949 (n=598) | 1.10 | n/a |
| lips | 598 | 0.787 | [0.767, 0.805] | 0.881 | 0.842 | 0.923 | 0.800 / 0.843 / 0.645 / 0.901 (n=598) | 1.10 | n/a |
| eye | 598 | 0.717 | [0.695, 0.738] | 0.835 | 0.839 | 0.832 | 0.678 / 0.768 / 0.363 / 0.848 (n=598) | 0.99 | n/a |
| brow | 598 | 0.710 | [0.695, 0.723] | 0.830 | 0.821 | 0.839 | 0.682 / 0.715 / 0.515 / 0.817 (n=598) | 1.02 | n/a |
| eye_R | 598 | 0.710 | [0.686, 0.732] | 0.831 | 0.850 | 0.813 | 0.661 / 0.766 / 0.000 / 0.859 (n=598) | 0.96 | n/a |
| eye_L | 598 | 0.724 | [0.703, 0.744] | 0.840 | 0.830 | 0.851 | 0.679 / 0.769 / 0.017 / 0.861 (n=596) | 1.03 | 0.0% of 2 |
| brow_R | 598 | 0.703 | [0.686, 0.719] | 0.826 | 0.827 | 0.825 | 0.677 / 0.714 / 0.514 / 0.816 (n=597) | 1.00 | 0.0% of 1 |
| brow_L | 598 | 0.716 | [0.701, 0.730] | 0.834 | 0.815 | 0.854 | 0.680 / 0.718 / 0.510 / 0.832 (n=598) | 1.05 | n/a |

### Denominator A, scoring: inside the parse window only (outside pixels excluded from GT and prediction; for failed images in B the whole GT counts as missed)

| class | images | IoU | IoU 95% CI | Dice | precision | recall | per-image IoU mean / median / p10 / p90 | area ratio | false alarms |
|---|---|---|---|---|---|---|---|---|---|
| skin | 598 | 0.866 | [0.855, 0.876] | 0.928 | 0.888 | 0.973 | 0.849 / 0.893 / 0.679 / 0.949 (n=598) | 1.10 | n/a |
| lips | 598 | 0.787 | [0.767, 0.805] | 0.881 | 0.842 | 0.923 | 0.800 / 0.843 / 0.645 / 0.901 (n=598) | 1.10 | n/a |
| eye | 598 | 0.717 | [0.695, 0.738] | 0.835 | 0.839 | 0.832 | 0.678 / 0.768 / 0.363 / 0.848 (n=598) | 0.99 | n/a |
| brow | 598 | 0.710 | [0.695, 0.723] | 0.830 | 0.821 | 0.839 | 0.682 / 0.715 / 0.515 / 0.817 (n=598) | 1.02 | n/a |
| eye_R | 598 | 0.710 | [0.686, 0.732] | 0.831 | 0.850 | 0.813 | 0.661 / 0.766 / 0.000 / 0.859 (n=598) | 0.96 | n/a |
| eye_L | 598 | 0.724 | [0.703, 0.744] | 0.840 | 0.830 | 0.851 | 0.679 / 0.769 / 0.017 / 0.861 (n=596) | 1.03 | 0.0% of 2 |
| brow_R | 598 | 0.703 | [0.686, 0.719] | 0.826 | 0.827 | 0.825 | 0.677 / 0.714 / 0.514 / 0.816 (n=597) | 1.00 | 0.0% of 1 |
| brow_L | 598 | 0.716 | [0.701, 0.730] | 0.834 | 0.815 | 0.854 | 0.680 / 0.718 / 0.510 / 0.832 (n=598) | 1.05 | n/a |

### Denominator B, scoring: full image (pixels outside the parse window count as 'not predicted')

| class | images | IoU | IoU 95% CI | Dice | precision | recall | per-image IoU mean / median / p10 / p90 | area ratio | false alarms |
|---|---|---|---|---|---|---|---|---|---|
| skin | 600 | 0.865 | [0.854, 0.876] | 0.928 | 0.888 | 0.971 | 0.846 / 0.893 / 0.676 / 0.949 (n=600) | 1.09 | n/a |
| lips | 600 | 0.786 | [0.768, 0.803] | 0.880 | 0.842 | 0.922 | 0.797 / 0.843 / 0.644 / 0.901 (n=600) | 1.09 | n/a |
| eye | 600 | 0.716 | [0.695, 0.737] | 0.835 | 0.839 | 0.830 | 0.675 / 0.767 / 0.332 / 0.848 (n=600) | 0.99 | n/a |
| brow | 600 | 0.709 | [0.695, 0.722] | 0.830 | 0.821 | 0.838 | 0.680 / 0.715 / 0.512 / 0.817 (n=600) | 1.02 | n/a |
| eye_R | 600 | 0.710 | [0.686, 0.732] | 0.830 | 0.850 | 0.812 | 0.659 / 0.766 / 0.000 / 0.859 (n=600) | 0.96 | n/a |
| eye_L | 600 | 0.723 | [0.702, 0.743] | 0.839 | 0.830 | 0.849 | 0.677 / 0.768 / 0.000 / 0.861 (n=598) | 1.02 | 0.0% of 2 |
| brow_R | 600 | 0.703 | [0.686, 0.719] | 0.825 | 0.827 | 0.824 | 0.675 / 0.713 / 0.508 / 0.816 (n=599) | 1.00 | 0.0% of 1 |
| brow_L | 600 | 0.715 | [0.700, 0.728] | 0.834 | 0.815 | 0.853 | 0.678 / 0.718 / 0.505 / 0.832 (n=600) | 1.05 | n/a |

### Denominator B, scoring: inside the parse window only (outside pixels excluded from GT and prediction; for failed images in B the whole GT counts as missed)

| class | images | IoU | IoU 95% CI | Dice | precision | recall | per-image IoU mean / median / p10 / p90 | area ratio | false alarms |
|---|---|---|---|---|---|---|---|---|---|
| skin | 600 | 0.865 | [0.854, 0.876] | 0.928 | 0.888 | 0.971 | 0.846 / 0.893 / 0.676 / 0.949 (n=600) | 1.09 | n/a |
| lips | 600 | 0.786 | [0.768, 0.803] | 0.880 | 0.842 | 0.922 | 0.797 / 0.843 / 0.644 / 0.901 (n=600) | 1.09 | n/a |
| eye | 600 | 0.716 | [0.695, 0.737] | 0.835 | 0.839 | 0.830 | 0.675 / 0.767 / 0.332 / 0.848 (n=600) | 0.99 | n/a |
| brow | 600 | 0.709 | [0.695, 0.722] | 0.830 | 0.821 | 0.838 | 0.680 / 0.715 / 0.512 / 0.817 (n=600) | 1.02 | n/a |
| eye_R | 600 | 0.710 | [0.686, 0.732] | 0.830 | 0.850 | 0.812 | 0.659 / 0.766 / 0.000 / 0.859 (n=600) | 0.96 | n/a |
| eye_L | 600 | 0.723 | [0.702, 0.743] | 0.839 | 0.830 | 0.849 | 0.677 / 0.768 / 0.000 / 0.861 (n=598) | 1.02 | 0.0% of 2 |
| brow_R | 600 | 0.703 | [0.686, 0.719] | 0.825 | 0.827 | 0.824 | 0.675 / 0.713 / 0.508 / 0.816 (n=599) | 1.00 | 0.0% of 1 |
| brow_L | 600 | 0.715 | [0.700, 0.728] | 0.834 | 0.815 | 0.853 | 0.678 / 0.718 / 0.505 / 0.832 (n=600) | 1.05 | n/a |

### Share of GT pixels outside the parse window (denominator A)

| class | GT pixels outside window |
|---|---|
| skin | 0.00% |
| lips | 0.00% |
| eye | 0.00% |
| brow | 0.00% |
| eye_R | 0.00% |
| eye_L | 0.00% |
| brow_R | 0.00% |
| brow_L | 0.00% |

### Secondary classes (not comparable; denominator A, full image)

| class | images | IoU | IoU 95% CI | Dice | precision | recall | per-image IoU mean / median / p10 / p90 | area ratio | false alarms |
|---|---|---|---|---|---|---|---|---|---|
| teeth_vs_mouth_interior | 598 | 0.110 | [0.087, 0.141] | 0.198 | 0.111 | 0.908 | 0.311 / 0.120 / 0.014 / 0.817 (n=61) | 8.19 | 2.2% of 537 |
| whole_mouth | 598 | 0.695 | [0.648, 0.741] | 0.820 | 0.734 | 0.930 | 0.793 / 0.846 / 0.644 / 0.905 (n=598) | 1.27 | n/a |

EP TEETH vs our `mouth_interior` and EP lips + teeth vs our lips + mouth interior are not the same definitions; read recall of teeth only.

## Boundary agreement and registration diagnostics (denominator A, per-image means)

| class | F @0.02 IOD mean | median | F @0.05 IOD mean | median | images | best (dx, dy) within +-3 px (+ = prediction moves right/down) |
|---|---|---|---|---|---|---|
| lips | 0.683 | 0.709 | 0.942 | 0.997 | 596 | median (+0, +2); mean (+0.20, +1.28); non-zero 97% |
| eye | 0.751 | 0.865 | 0.883 | 1.000 | 596 | median (+0, +1); mean (+0.16, +1.31); non-zero 97% |
| brow | 0.653 | 0.677 | 0.923 | 0.969 | 596 | median (+0, +2); mean (+0.31, +1.73); non-zero 98% |
| whole_mouth | 0.665 | 0.702 | 0.926 | 0.998 | 596 | median (+0, +2); mean (+0.19, +1.26); non-zero 97% |

## Side handling

EP LEFT_* is assumed to be the image-left structure (our `_R`, MediaPipe convention). Checked per image by centroid x ordering of the GT; the swap rate is judged by overlap.

| pair | GT order defined | GT order exceptions | exception rate | swap defined | swapped | swap rate |
|---|---|---|---|---|---|---|
| eye | 598 | 0 | 0.000 | 545 | 0 | 0.000 |
| brow | 599 | 0 | 0.000 | 593 | 0 | 0.000 |

Sided rows flagged (exceptions > 2%): **False**. Sided rows are usable.

## Confusion view (denominator A, full image): where each EP class's pixels go

Row = EasyPortrait class; entries are the share of that class's pixels assigned to each canonical class (only shares >= 0.5% shown). `NOT_PARSED` = outside the parse window.

- **BACKGROUND** (1,302,024,275 px): NOT_PARSED 53.8%, BACKGROUND 44.0%, CLOTH 0.8%, HAIR 0.8%, HAT 0.6%
- **PERSON** (883,694,441 px): NOT_PARSED 47.3%, CLOTH 23.0%, HAIR 13.7%, NECK 8.1%, SKIN 2.8%, BACKGROUND 2.5%, HAT 1.1%
- **SKIN** (230,253,923 px): SKIN 88.8%, NOSE 8.5%, GLASSES 1.1%, HAIR 0.6%
- **LEFT_BROW** (4,371,281 px): BROW_R 82.5%, SKIN 14.9%, HAIR 1.3%, GLASSES 1.3%
- **RIGHT_BROW** (4,369,005 px): BROW_L 85.4%, SKIN 13.3%, GLASSES 0.8%
- **LEFT_EYE** (1,891,192 px): EYE_R 81.3%, SKIN 11.2%, GLASSES 7.5%
- **RIGHT_EYE** (1,886,432 px): EYE_L 85.1%, SKIN 8.0%, GLASSES 6.8%
- **LIPS** (9,199,357 px): LIP_LOWER 53.6%, LIP_UPPER 38.7%, SKIN 7.2%
- **TEETH** (227,423 px): MOUTH_INTERIOR 90.8%, LIP_UPPER 5.4%, LIP_LOWER 3.7%

### Where our predictions fall (denominator A, full image): share of each predicted group lying in each EasyPortrait class

Complement of the view above (precision side). EP PERSON is the catch-all for the person that is not skin, brows, eyes, lips or teeth (hair, neck, clothing and the like).

- **our skin + nose** (252,306,698 px): SKIN 88.8%, PERSON 10.1%
- **our lips** (10,087,150 px): LIPS 84.2%, PERSON 10.1%, SKIN 5.5%
- **our eye** (3,743,049 px): RIGHT_EYE 42.9%, LEFT_EYE 41.1%, SKIN 8.2%, PERSON 7.8%
- **our brow** (8,934,730 px): RIGHT_BROW 41.8%, LEFT_BROW 40.3%, PERSON 10.8%, SKIN 7.1%

## Stratified agreement (denominator A, full image; IoU with 95% CI)

Cells with n < 30 are marked `low-sample` and excluded from the gap computation. Luma is the mean Rec.601 luma of GT skin pixels, a **lighting** indicator and **not a skin-tone measure**; skin-tone bias remains unquantified.

### brightness_quartile

| cell | images | skin | lips | eye | brow |
|---|---|---|---|---|---|
| Q1 | 149 | 0.873 [0.853, 0.890] | 0.779 [0.734, 0.817] | 0.713 [0.658, 0.756] | 0.712 [0.683, 0.741] |
| Q2 | 149 | 0.870 [0.849, 0.889] | 0.763 [0.720, 0.803] | 0.721 [0.676, 0.759] | 0.699 [0.663, 0.730] |
| Q3 | 150 | 0.879 [0.863, 0.895] | 0.777 [0.735, 0.814] | 0.738 [0.706, 0.764] | 0.711 [0.690, 0.732] |
| Q4 | 150 | 0.835 [0.802, 0.867] | 0.837 [0.823, 0.849] | 0.692 [0.628, 0.742] | 0.717 [0.693, 0.737] |

### resolution

| cell | images | skin | lips | eye | brow |
|---|---|---|---|---|---|
| 1.5-2.5MP | 85 | 0.852 [0.825, 0.875] | 0.794 [0.729, 0.839] | 0.677 [0.567, 0.749] | 0.708 [0.680, 0.730] |
| 2.5-3.0MP | 452 | 0.868 [0.855, 0.880] | 0.784 [0.763, 0.806] | 0.724 [0.698, 0.746] | 0.715 [0.699, 0.729] |
| ge3.0MP | 31 | 0.853 [0.812, 0.889] | 0.835 [0.804, 0.862] | 0.697 [0.632, 0.754] | 0.646 [0.551, 0.713] |
| lt1.5MP | 30 | 0.864 [0.813, 0.908] | 0.741 [0.665, 0.803] | 0.688 [0.604, 0.759] | 0.662 [0.577, 0.750] |

### luma

| cell | images | skin | lips | eye | brow |
|---|---|---|---|---|---|
| high | 200 | 0.872 [0.856, 0.889] | 0.781 [0.745, 0.813] | 0.717 [0.669, 0.760] | 0.721 [0.697, 0.742] |
| low | 198 | 0.864 [0.849, 0.879] | 0.789 [0.756, 0.818] | 0.724 [0.693, 0.752] | 0.684 [0.656, 0.707] |
| mid | 200 | 0.860 [0.835, 0.881] | 0.793 [0.757, 0.825] | 0.713 [0.681, 0.739] | 0.717 [0.691, 0.738] |

### face_size

| cell | images | skin | lips | eye | brow |
|---|---|---|---|---|---|
| high | 199 | 0.885 [0.870, 0.899] | 0.777 [0.745, 0.804] | 0.725 [0.692, 0.757] | 0.720 [0.701, 0.740] |
| low | 198 | 0.821 [0.803, 0.839] | 0.810 [0.787, 0.828] | 0.699 [0.672, 0.727] | 0.678 [0.661, 0.694] |
| mid | 199 | 0.840 [0.824, 0.855] | 0.797 [0.770, 0.821] | 0.705 [0.677, 0.730] | 0.698 [0.680, 0.715] |

### glasses_detected

| cell | images | skin | lips | eye | brow |
|---|---|---|---|---|---|
| no | 550 | 0.878 [0.866, 0.888] | 0.788 [0.769, 0.807] | 0.765 [0.754, 0.775] | 0.717 [0.705, 0.730] |
| yes | 48 | 0.704 [0.676, 0.728] | 0.775 [0.698, 0.840] | 0.000 [0.000, 0.000] | 0.606 [0.509, 0.692] |

### alignment_passed

| cell | images | skin | lips | eye | brow |
|---|---|---|---|---|---|
| no | 93 | 0.757 [0.728, 0.788] | 0.752 [0.692, 0.804] | 0.168 [0.104, 0.244] | 0.628 [0.559, 0.683] |
| yes | 505 | 0.882 [0.871, 0.892] | 0.793 [0.772, 0.812] | 0.778 [0.768, 0.787] | 0.722 [0.710, 0.734] |

### Worst-stratum gaps (brightness quartile and luma tercile)

| stratifier / class | best cell | worst cell | gap | lower bound |
|---|---|---|---|---|
| brightness_quartile/skin | Q3 | Q4 | 0.044 | 0.000 |
| brightness_quartile/lips | Q4 | Q2 | 0.075 | 0.021 |
| brightness_quartile/eye | Q3 | Q4 | 0.046 | 0.000 |
| brightness_quartile/brow | Q4 | Q2 | 0.018 | 0.000 |
| luma/skin | high | mid | 0.012 | 0.000 |
| luma/lips | mid | high | 0.012 | 0.000 |
| luma/eye | low | mid | 0.011 | 0.000 |
| luma/brow | high | low | 0.037 | 0.000 |

## Latency (denominator A; ms; this machine only; `total` = `segment_face` end to end excluding HTTP and decode)

| megapixels | n | landmarks | preprocess | inference | postprocess | zones | encode | total | decode | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 598 | 10.2 / 15.4 | 7.7 / 13.5 | 84.6 / 132.2 | 40.8 / 58.3 | 89.3 / 370.0 | 29.3 / 115.5 | 292.3 / 765.2 | 21.6 / 79.7 | 293.7 / 770.2 |
| <1 MP | 10 | 9.2 / 14.6 | 4.8 / 7.5 | 85.1 / 132.5 | 32.8 / 46.4 | 29.5 / 46.8 | 10.1 / 16.4 | 181.9 / 267.7 | 4.7 / 7.6 | 183.3 / 268.5 |
| 1-2 MP | 59 | 9.4 / 11.3 | 5.1 / 7.8 | 84.2 / 131.3 | 38.8 / 43.4 | 43.7 / 82.3 | 14.7 / 31.2 | 215.3 / 291.4 | 11.2 / 14.0 | 216.4 / 293.9 |
| 2-4 MP | 366 | 9.9 / 13.4 | 7.4 / 9.5 | 84.4 / 119.3 | 40.5 / 55.9 | 76.9 / 150.4 | 25.0 / 50.8 | 270.4 / 408.9 | 20.5 / 28.0 | 272.2 / 411.3 |
| >=4 MP | 163 | 11.9 / 18.0 | 11.1 / 16.2 | 85.0 / 157.3 | 43.2 / 63.3 | 238.5 / 455.9 | 77.5 / 140.2 | 532.0 / 942.3 | 55.6 / 95.0 | 536.1 / 949.7 |

Cells are p50 / p95.

## Landmark polygons vs parser (same images, same GT, full image)

Images: 598.

| region | GT | landmark polygon IoU [CI] | parser IoU [CI] |
|---|---|---|---|
| eye | EP eyes (LEFT+RIGHT) | 0.730 [0.719, 0.740] | 0.717 [0.695, 0.738] |
| mouth | EP lips + teeth (whole mouth) | 0.726 [0.687, 0.761] | 0.695 [0.648, 0.741] |

## Worst 10 images per class (opaque dataset ids; per-image IoU, GT >= 25 px)

- **skin**: `ae1e71d8-3e09-4e3b-b626-c8e95878cceb` (0.434), `439f0136-cada-4c0b-af96-3deb1a216d85` (0.443), `b3dc4d73-8ab8-46e4-be81-50e220c4c653` (0.466), `17c071c3-36ea-496e-aad2-acc80d16f5a6` (0.478), `668399b5-bbb4-42db-9311-61f351f8df10` (0.489), `369334d2-bb13-4e13-9155-469a40a36521` (0.498), `f05263ae-3624-4b00-929e-e9229db37172` (0.506), `173925cd-9aa4-43ea-b6d1-6fc9ed5e7999` (0.514), `8a0fa024-4c0a-4c05-b0eb-457aec9bb05a` (0.514), `33696bfb-c275-4e1b-8475-70b86920102a` (0.539)
- **lips**: `eb1afb42-af19-4eaf-825e-ea2795c93bc4` (0.116), `933d3490-6359-4503-b349-1e5310e3027a` (0.181), `f91f54cf-3dc8-46a0-837e-60325f608e28` (0.185), `02ec7863-650e-4047-aad6-c621a1aed928` (0.189), `5f0e80f3-a36e-460c-b51c-7f5159d30ddd` (0.222), `0c104242-1221-42e3-95a0-fa6994b2f61d` (0.233), `147e9fd4-c6ce-4d60-afa5-f363411bae56` (0.241), `8a95efe3-bfcb-4bac-9665-d5e56ee4890c` (0.245), `1ace5c08-2f7e-4e90-87ef-1eb9a7910f87` (0.255), `d5d1184d-887a-4c5a-84b2-eeef9d46d263` (0.297)
- **eye**: `0c8728ab-96f9-4937-8ba9-75c7deec5d00` (0.000), `0f536fe6-86dc-4fe1-bb07-055d94b8067e` (0.000), `1282134c-ffd6-49a4-a595-f25948a535b3` (0.000), `1341f980-7306-4fb9-ac35-8f2b664eab2c` (0.000), `13831237-60f0-4858-8e68-738cf5b1d35e` (0.000), `14e4e5c7-16fe-48cc-9725-e6b428554c9b` (0.000), `188e80b4-5fea-40c4-ac97-0498519804c8` (0.000), `18edf397-83c4-4d28-8d47-6c31d47312e9` (0.000), `1c31d0b8-e45c-4145-bebe-2d40509e379a` (0.000), `1cd76bc3-c040-4487-8324-6130ae306447` (0.000)
- **brow**: `227949f2-81ac-414f-9371-ad2a48da0e53` (0.000), `30ee9c3b-5c7b-47f8-9f1b-68fa7897fe9b` (0.000), `6b539bb2-1fb0-4bbe-a914-ada0a457f357` (0.000), `8261f91d-6068-4fdc-9021-c8b0412d1a24` (0.000), `9eed18eb-17d2-482f-ad29-64801bc50776` (0.000), `d5d1184d-887a-4c5a-84b2-eeef9d46d263` (0.001), `ea703959-5daa-474f-a349-e6e63ad8d9e4` (0.039), `a194102b-7640-4841-ae5e-afb399add5ef` (0.040), `17e58d41-21a1-42ad-a051-ecf06340ec30` (0.069), `1fda97a4-2369-452e-8c83-cb17c3c9825c` (0.083)
- **eye_R**: `0c8728ab-96f9-4937-8ba9-75c7deec5d00` (0.000), `0ea7b5d0-f309-4143-9744-3cdbd5d69685` (0.000), `0f536fe6-86dc-4fe1-bb07-055d94b8067e` (0.000), `0fadadd9-d4e6-4d6b-b766-482e2d431d12` (0.000), `1282134c-ffd6-49a4-a595-f25948a535b3` (0.000), `1341f980-7306-4fb9-ac35-8f2b664eab2c` (0.000), `13831237-60f0-4858-8e68-738cf5b1d35e` (0.000), `14e4e5c7-16fe-48cc-9725-e6b428554c9b` (0.000), `17e58d41-21a1-42ad-a051-ecf06340ec30` (0.000), `188e80b4-5fea-40c4-ac97-0498519804c8` (0.000)
- **eye_L**: `0c8728ab-96f9-4937-8ba9-75c7deec5d00` (0.000), `0f536fe6-86dc-4fe1-bb07-055d94b8067e` (0.000), `1282134c-ffd6-49a4-a595-f25948a535b3` (0.000), `1341f980-7306-4fb9-ac35-8f2b664eab2c` (0.000), `13831237-60f0-4858-8e68-738cf5b1d35e` (0.000), `14e4e5c7-16fe-48cc-9725-e6b428554c9b` (0.000), `173925cd-9aa4-43ea-b6d1-6fc9ed5e7999` (0.000), `188e80b4-5fea-40c4-ac97-0498519804c8` (0.000), `18edf397-83c4-4d28-8d47-6c31d47312e9` (0.000), `1c31d0b8-e45c-4145-bebe-2d40509e379a` (0.000)
- **brow_R**: `0374a515-3499-4287-ab22-f52c6a7b999f` (0.000), `225eb720-7888-4edd-aa82-20af17f12d8b` (0.000), `227949f2-81ac-414f-9371-ad2a48da0e53` (0.000), `2a7cc351-36c6-4c5c-8d87-75b70624672c` (0.000), `2cfbb212-bb47-4b12-99a3-fa341c32f9bf` (0.000), `30ee9c3b-5c7b-47f8-9f1b-68fa7897fe9b` (0.000), `439f0136-cada-4c0b-af96-3deb1a216d85` (0.000), `6b539bb2-1fb0-4bbe-a914-ada0a457f357` (0.000), `9eed18eb-17d2-482f-ad29-64801bc50776` (0.000), `b2ad4622-5d40-4308-b859-5660ffc8f8eb` (0.000)
- **brow_L**: `17e58d41-21a1-42ad-a051-ecf06340ec30` (0.000), `1da1f112-413d-4444-be41-38d52e81d891` (0.000), `227949f2-81ac-414f-9371-ad2a48da0e53` (0.000), `2c773853-316a-45c6-b31b-5844bed45dbf` (0.000), `30ee9c3b-5c7b-47f8-9f1b-68fa7897fe9b` (0.000), `6b539bb2-1fb0-4bbe-a914-ada0a457f357` (0.000), `8261f91d-6068-4fdc-9021-c8b0412d1a24` (0.000), `9eed18eb-17d2-482f-ad29-64801bc50776` (0.000), `a81f43a0-2f06-4b2d-b7ab-127801b9a030` (0.000), `b6aad415-f838-49fa-9889-93408e3a6bc9` (0.000)

## Weak regions and dominant error mode (denominator A, full image)

| class | IoU | target | shortfall | missed GT pixels go to | false pixels come from |
|---|---|---|---|---|---|
| skin | 0.866 | 0.90 | 0.034 | GLASSES (1.1% of GT; total missed 2.7%) | PERSON (10.1% of prediction; total false 11.2%) |
| lips | 0.787 | 0.75 | -0.037 | SKIN (7.2% of GT; total missed 7.7%) | PERSON (10.1% of prediction; total false 15.8%) |
| eye | 0.717 | 0.65 | -0.067 | SKIN (9.6% of GT; total missed 16.8%) | SKIN (8.2% of prediction; total false 16.1%) |
| brow | 0.710 | 0.60 | -0.110 | SKIN (14.1% of GT; total missed 16.1%) | PERSON (10.8% of prediction; total false 17.9%) |

## Dev and test splits (IoU, denominator A and B, full image)

Future fixes are tuned on dev only; the test split is for the single final decision.

### dev (239 images; outcomes {'ok': 239})

| class | A full | B full | A window | B window |
|---|---|---|---|---|
| skin | 0.862 [0.847, 0.877] | 0.862 [0.847, 0.877] | 0.862 | 0.862 |
| lips | 0.791 [0.766, 0.816] | 0.791 [0.766, 0.816] | 0.791 | 0.791 |
| eye | 0.709 [0.666, 0.742] | 0.709 [0.666, 0.742] | 0.709 | 0.709 |
| brow | 0.700 [0.673, 0.724] | 0.700 [0.673, 0.724] | 0.700 | 0.700 |
| eye_R | 0.702 [0.653, 0.741] | 0.702 [0.653, 0.741] | 0.702 | 0.702 |
| eye_L | 0.715 [0.676, 0.746] | 0.715 [0.676, 0.746] | 0.715 | 0.715 |
| brow_R | 0.696 [0.662, 0.724] | 0.696 [0.662, 0.724] | 0.696 | 0.696 |
| brow_L | 0.705 [0.677, 0.732] | 0.705 [0.677, 0.732] | 0.705 | 0.705 |

### test (361 images; outcomes {'ok': 359, 'no_face': 2})

| class | A full | B full | A window | B window |
|---|---|---|---|---|
| skin | 0.868 [0.852, 0.881] | 0.867 [0.851, 0.881] | 0.868 | 0.867 |
| lips | 0.785 [0.759, 0.810] | 0.782 [0.753, 0.808] | 0.785 | 0.782 |
| eye | 0.723 [0.694, 0.747] | 0.721 [0.693, 0.745] | 0.723 | 0.721 |
| brow | 0.715 [0.701, 0.730] | 0.714 [0.698, 0.729] | 0.715 | 0.714 |
| eye_R | 0.716 [0.685, 0.742] | 0.715 [0.685, 0.740] | 0.716 | 0.715 |
| eye_L | 0.730 [0.701, 0.756] | 0.728 [0.702, 0.752] | 0.730 | 0.728 |
| brow_R | 0.708 [0.689, 0.725] | 0.707 [0.688, 0.724] | 0.708 | 0.707 |
| brow_L | 0.723 [0.705, 0.739] | 0.721 [0.703, 0.737] | 0.723 | 0.721 |

## Provisional targets (decision D2: to be confirmed by the user after this pilot)

| item | target | value | 95% CI / lower bound | result |
|---|---|---|---|---|
| skin IoU | 0.90 | 0.866 | [0.855, 0.876] | missed |
| lips IoU | 0.75 | 0.787 | [0.767, 0.805] | met |
| lips boundary F @ 0.02*IOD (mean per image) | 0.80 | 0.683 | n/a | missed |
| eye IoU | 0.65 | 0.717 | [0.695, 0.738] | met |
| eye boundary F @ 0.02*IOD (mean per image) | 0.80 | 0.751 | n/a | missed |
| brow IoU | 0.60 | 0.710 | [0.695, 0.723] | met |
| brow boundary F @ 0.02*IOD (mean per image) | 0.80 | 0.653 | n/a | missed |
| side-swap rate (eye) | 0.01 | 0.000 | n/a | met |
| side-swap rate (brow) | 0.01 | 0.000 | n/a | met |
| worst-stratum gap brightness_quartile/skin (Q3 vs Q4) | 0.05 | 0.044 | 0.000 | met |
| worst-stratum gap brightness_quartile/lips (Q4 vs Q2) | 0.08 | 0.075 | 0.021 | met |
| worst-stratum gap brightness_quartile/eye (Q3 vs Q4) | 0.08 | 0.046 | 0.000 | met |
| worst-stratum gap brightness_quartile/brow (Q4 vs Q2) | 0.08 | 0.018 | 0.000 | met |
| worst-stratum gap luma/skin (high vs mid) | 0.05 | 0.012 | 0.000 | met |
| worst-stratum gap luma/lips (mid vs high) | 0.08 | 0.012 | 0.000 | met |
| worst-stratum gap luma/eye (low vs mid) | 0.08 | 0.011 | 0.000 | met |
| worst-stratum gap luma/brow (high vs low) | 0.08 | 0.037 | 0.000 | met |
| latency p95, 1-2 MP images, this machine (segment_face total) | 300.00 | 291.408 | n/a | met (n = 59; PROVISIONAL, not measured on AC power: the pilot ran on battery with system sleep events in its window, so these timings are unreliable. The re-run benchmark (docs/evaluation/benchmark_2026-11-10.md; 2 MP p95 176 ms, 0 sleep events, still on battery) is the latency reference; the 2-4 MP bucket is the nearest large one) |

EasyPortrait boundary conventions differ from CelebAMask-HQ's, so the IoU targets may be revised after the pilot; boundary F and area ratio are reported alongside for that reason.

## Decision rules applied (provisional)

Evaluated on: pilot: full sample (this is not the final test-split decision, which follows the protocol).

1. **Acceptable for the prototype:** all targets met or within tolerance: **False**. Unmet: ['skin IoU', 'lips boundary F @ 0.02*IOD (mean per image)', 'eye boundary F @ 0.02*IOD (mean per image)', 'brow boundary F @ 0.02*IOD (mean per image)']. Not assessable: none. Detection failure rate 0.003 (limit 0.02; ok: True). failure rate is over ALL valid images; 'fully visible face' is not annotated, so it cannot be restricted.
2. **Post-processing or geometry fix is enough:** triggers: ['lips: consistent best shift (median dx +0, dy +2 px)', 'eye: consistent best shift (median dx +0, dy +1 px)', 'brow: consistent best shift (median dx +0, dy +2 px)']. Update from the robustness suite (WP3, 100 dev images): roll of +/-30 deg costs 0.12-0.16 IoU and +/-45 deg costs 0.35-0.54 while detection stays 97-99%, so the roll item of rule 2 IS met; a roll-aligned crop is a WP6 candidate and needs user approval.
3. **Model replacement or fine-tuning is needed (commercial path only):** triggers: none. Caveat: no trigger fires on the headline numbers, but the glasses stratum (eye IoU 0.00, n=48) and the alignment-failure stratum (eye IoU 0.168, n=93) are very poor; all 48 glasses images are among the 93 alignment failures. The fine-tuning / replacement decision is NOT made here; it is owed to the user.
4. **Detector lever:** share of the B loss (skin GT pixels) from detection failures 0.045; detector lever: **False**.
5. **Licence caveat:** Regardless of the outcome, the CelebAMask-HQ model is replaced before any commercial release; this harness is the comparison tool for that replacement.

## Limitations

- Agreement with another dataset's annotation conventions, not hand-checked ground truth for this product. The skin shortfall is a precision issue (precision 0.888, recall 0.973, area ratio 1.10): about 10.1% of our skin + nose pixels fall in EP PERSON (see 'Where our predictions fall'), not in EP SKIN.
- All aggregates are over the stratified 600-image sample (brightness quartile x resolution group, floor-based allocation); they are not re-weighted to the 6000-image test split.
- The y-offset cause is unexplained (label convention vs pipeline). Landmark polygons show a best shift of about (-1, -1) px while the parser shows about (0, +2) px: opposite signs. Not fixed. A vertical-flip check was inconclusive (see the robustness report).
- Sample of 600 of 6000 test images, stratified by brightness and resolution only. EasyPortrait is a webcam/portrait set with many near-frontal faces: large pose, heavy occlusion and extreme lighting are under-represented.
- **Skin-tone bias is not quantified.** The dataset has no skin-tone labels; luma strata measure lighting, not tone.
- The perturbation (robustness) suite and the benchmark are not part of this report. Roll failures (about +/-35 deg) are known from earlier tests and not measured here.
- One machine; latency applies to this CPU only. EasyPortrait images are mostly 2.8 MP, so the 1-2 MP latency target is barely sampled.
- Boundary F and best-shift are computed on full-image masks; best shift searches integer shifts within +-3 px.
- The weights remain CelebAMask-HQ (non-commercial); this evaluation does not change that restriction.

## Licence and attribution

Ground truth: EasyPortrait (Kapitanov, Kvanchiani, Kirillova, 2023), https://github.com/hukenovs/easyportrait . Licence: "a variant of Creative Commons Attribution-ShareAlike 4.0 International" (a reworked, non-Creative-Commons public licence with attribution and conditions reserved; no non-commercial clause; the privacy and portrait rights of the people depicted are not granted): https://github.com/hukenovs/easyportrait/blob/master/license/en_us.pdf . Use here: evaluation only, aggregate results only. No images or annotations are redistributed, and none are used for training (that needs the separate legal review listed as a pre-deployment blocker).

```
@article{EasyPortrait,
    title={EasyPortrait - Face Parsing and Portrait Segmentation Dataset},
    author={Kapitanov, Alexander and Kvanchiani, Karina and Kirillova Sofia},
    journal={arXiv preprint arXiv:2304.13509},
    year={2023}
}
```
