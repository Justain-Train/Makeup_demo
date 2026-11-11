# Face segmentation (roadmap Phase 3) and initial application zones (early Phase 4)

Phase numbering follows the roadmap in `CLAUDE.md`. Earlier plans and agent reports in `docs/plans/` say
"Phase 1" for this work and "Phase 2" for the remaining zones. In roadmap terms, those are Phase 3 (segmentation),
the initial Phase 4 zones, and the remaining Phase 4 zones.

Status: segmentation and the initial zones are implemented. They passed the agent workflow, with the final
evaluation in `docs/plans/reports/10_evaluation_iter3_final.md`. Phase 3 is **not complete** until a ground-truth
quality evaluation has been done. Phase 4 Batch 1 (`chin_center`, `under_lower_lip`, `mouth_surround`, `cupid_bow`, `lip_corner_l/r`) , Batch 2 (`forehead_center`, `forehead_perimeter`, `temple_l/r`, `jawline_outer_l/r`) and Batch 3a (`under_eye_inner/outer_l/r`, `eye_corner_inner/outer_l/r`, `lash_line_upper/lower_l/r`) and Batch 3b (`eyelid_upper_l/r`, `brow_head/arch/tail_l/r`, `brow_bone_l/r`) are implemented; the other Phase 4 zones are NOT implemented. All numbers below are **measurements on one
machine (Apple M2 Pro, 10 cores, macOS, CPU only, onnxruntime 1.31.1, Python 3.13)** unless marked otherwise. No
accuracy claim is made anywhere in this document: the only ground-truth comparison so far is a pilot against another
dataset's annotations (see [Evaluation results](#evaluation-results)), and there is no hand-annotated ground truth of
our own yet (see [Evaluation protocol](#evaluation-protocol)).

> **LICENSE WARNING.** The BiSeNet weights were trained on **CelebAMask-HQ, non-commercial research data. Not
> cleared for commercial use.** Accepted for the prototype only (approved decision 1). The warning is shown in the
> README, in this document and in `model.license_warning` of every `/segmentation` response. Code licence (MIT) does
> not change this. A commercial release needs a different model (see [Replacing the model](#replacing-the-model)).

## Architecture

```
Selfie -> shared upload validation -> MediaPipe landmarks (largest face) -> FaceParser (Layer A)
                                                                  |             |
                                                                  +--> FaceFrame + zone derivation (Layer B)
                                                                                |
                                                                       FaceSegmentationResponse
```

- **Layer A, anatomical segmentation** (`app/services/segmentation/`): square *parse window* around the landmark bbox
  (side = `FACE_PARSER_CROP_SCALE` x bbox side, default 2.0, may extend past the image and is padded with the
  ImageNet mean), 512x512 network input, logits upsampled bilinearly to the window (long side capped at 1024),
  argmax, mapped to canonical region ids, nearest-mapped to the original pixels of window ∩ image. The label map
  covers exactly window ∩ image (`anatomical.origin/width/height`); `parse_window` may stick out of the image.
- **Layer B, application zones** (`app/services/application_zones/`): landmark polygons built in a roll-aware,
  interpupillary-distance-relative `FaceFrame`, clipped to the anatomical masks, validated (coverage, minimum size,
  parser/landmark IoU). Zones are defined once for the subject's right side and mirrored with `MIRROR`.
- Everything outside `segmentation/` uses canonical region names only.

### Replacing the model

No model-specific class id exists outside `app/services/segmentation/`. To swap in another face-parsing model
(e.g. one the project trains on commercially usable data), change **only**:

1. `regions.py`: `MODEL_ID_TO_CANONICAL` (and `VERIFIED_MODEL_IDS`, `NUM_MODEL_CLASSES`), adding canonical regions if the
   new model has classes this one lacks;
2. `face_parser.py`: the pre/post-processing if the model's I/O differs (input size, normalisation, output layout).

The zone engine, API, schemas and frontend do not change. If the new model lacks a class (e.g. no separate brows), the
corresponding zone simply reports `unavailable`. Re-run the verification below (class order, left/right) on the new
model before trusting any mapping; `tests/test_face_parser.py` has the regression tests.

## Model selection

| Candidate | Classes | Licence | Decision |
|---|---|---|---|
| **yakhyo BiSeNet ResNet18 ONNX** | 19 (CelebAMask-HQ) | code MIT; training data non-commercial | **Selected** |
| yakhyo BiSeNet ResNet34 ONNX | same 19 | same | Benchmarked only (below) |
| SegFormer-b5, FaRL/facer | 19 / 11 | non-commercial / unclear | Not used (size, PyTorch) |
| MediaPipe selfie_multiclass_256 | 6 (skin and hair, no features) | Apache 2.0 | **Documented only**, not implemented. Commercially clean but only a partial fallback (face-skin, hair) |
| EasyPortrait SegFormer-B0 | 9 (skin, brows, eyes, lips, teeth...) | CC BY-SA 4.0 variant, no non-commercial clause; needs legal review | Candidate for the own-model path; needs a one-time export to ONNX |

Weights: `https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx` (53,205,364 bytes, SHA-256
`0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f`). ResNet34: 93,632,554 bytes, SHA-256
`5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e`. **The upstream project publishes no
checksums: these hashes were computed from files downloaded on 2026-11-10 (trust on first use).** They pin the file so
any later change is detected; they do not prove the upstream file is benign. Weights are gitignored
(`backend/models/*.onnx`) and are loaded from bytes that were hashed in memory (no verify/load race).

## Empirical verification of the model (done on the real weights)

Scratch scripts only; nothing from memory was trusted.

**ONNX metadata (ResNet18; ResNet34 identical apart from output names):**

| | name | shape | dtype |
|---|---|---|---|
| input | `input` | `[batch_size, 3, 512, 512]` (H/W fixed, batch dynamic) | float32 |
| output 0 (used) | `output` | `[batch, 19, 512, 512]` at runtime (declared dims are symbolic) | float32 |
| outputs 1, 2 | `414`, `424` (ResNet34: `566`, `576`) | also 19 x 512 x 512 | float32 |

Output 0 holds **logits** (values -4.4 .. 9.0 on the fixture, not probabilities, not argmaxed). Outputs 1 and 2 are
auxiliary training heads; they are ignored. The parser takes the input/output names from the session at load time and
rejects a model whose input is not `[*, 3, 512, 512]` or whose warm-up output is not `(19, 512, 512)`.
Preprocessing: RGB, 512x512, ImageNet mean/std (x255), NCHW float32. Using this preprocessing the maps are plausible
on the fixture and on a second photo (glasses, hat, beard).

**Class order** (rendered each class over `tests/fixtures/portrait.jpg` and over a second photo with glasses and a hat):
0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g (glasses), 7 l_ear, 8 r_ear, 9 ear_r, 10 nose,
11 mouth (teeth / open mouth interior), 12 u_lip, 13 l_lip, 14 neck, 15 neck_l, 16 cloth, 17 hair, 18 hat.
Confirmed visually: 0-8, 10-14, 16, 17 on the fixture; 6 (glasses) and 18 (hat) on the second photo.
**Not confirmed on any photo: 9 (`ear_r`, believed to be earrings) and 15 (`neck_l`, believed to be necklace).** They
are mapped to `earring` / `necklace` from the dataset's documented names and listed outside `VERIFIED_MODEL_IDS`.

**Left/right (convention, changed in iteration 2).** `left`/`right` and the `_l`/`_r` suffixes follow the **MediaPipe
landmark convention**: subject-right is the side containing landmarks 33, 133 and 468 (the image's left in an
unmirrored photo). The model's own `l_*`/`r_*` classes are **not used**: on the upright fixture they agree with this
convention (iris 468 falls in class 5 `r_eye`, iris 473 in class 4 `l_eye`), but on a horizontally mirrored copy of the
fixture the model labelled **both** irises `l_eye` (one eye mask covered both eyes). The paired classes (brow 2/3,
eye 4/5, ear 7/8) are therefore merged into side-agnostic internal ids in `MODEL_ID_TO_CANONICAL`, and
`segmentation/sides.py` assigns each pixel the side of the nearest landmark of that feature (roll-independent;
touching brows are still split). **Mirrored photos:** mirroring cannot be detected from pixels, so on a mirrored
selfie the sides follow the image geometry, i.e. they are swapped relative to the real person. A future capture flag
(front-camera mirroring) could correct this; it is not implemented. Guards: both irises inside their eye region is a
**hard** requirement of `alignment_check` (either failing gives `passed=false`, a `parser_landmark_mismatch`
warning and `low_confidence` for that side's eye and under-eye zones); an eye/brow zone is never `ok` if more than 20 %
of its pixels lie inside the other side's landmark polygon or it is wider than 1.5x its landmark polygon.

**Crop scale** (square window = scale x longest landmark-bbox side). IoU between the parser's class(es) and the
*landmark* polygon on the single fixture. This measures agreement with landmarks, **not accuracy**:

| window | right eye | left eye | lips (u+l+mouth) |
|---|---|---|---|
| 1.4x | 0.00 (class missed) | 0.35 | 0.88 |
| 1.7x | 0.63 | 0.76 | 0.87 |
| **2.0x (chosen)** | 0.73 | 0.76 | 0.84 |
| full frame, stretched to 512x512 | 0.00 | 0.00 | 0.77 |
| full frame, letterboxed | 0.00 | 0.00 | 0.19 |

2.0x was chosen because it was best for both eyes and also covers more hair/ears. Caveat: one image only, so the
choice is weakly supported. Tight crops (1.4x) lose an eye; the full-frame path misses both eyes on this 820x1024
portrait, i.e. the crop is required.

**What `skin` and `nose` cover** (observed on the two photos): `nose` is the whole nose including the nostril area (no
nostril openings are resolved). `neck`, `l_ear`/`r_ear`, `hair`, `cloth` are separate classes, so skin stops at the
jaw/neck boundary and at the ears. Beard hair is parsed as `skin` (second photo), and on that bearded photo the
lip polygon/parser agreement was poor (IoU 0.03 upper, 0.22 lower), so lips are reported `low_confidence`. Glasses
are parsed as `eye_g`; behind thick frames both eye classes were empty, so eye and under-eye zones are
`unavailable`.  A brimmed hat was parsed as `hat`.

**Registration fix (2026-11-10).** Measured with a stub network that echoes its input and a smooth synthetic edge
(window equal to the image), the network input was shifted to the right and down by 0.09 px (window 600), 0.47 px
(1000), 0.99 px (2000) and 1.43 px (2650) of the original image, and the label-map edge by up to 3 px at 2650. Cause:
the forward warp into the 512x512 input was corner-aligned while `cv2.resize` and the inverse map back to the image
are pixel-centre-aligned, and the `INTER_AREA` pre-shrink used the nominal factor instead of the size actually
produced (OpenCV rounds the output size, and with `fx`/`fy` it also averages with the nominal factor). Now the forward
warp is centre-aligned (`dst = k*(src + 0.5) - 0.5`), the pre-shrink passes an explicit output size, and the warp uses the
scale actually produced. After the fix the measured input shift is within 0.11 px at every tested window (600 to 2650, including a window
extending past the image and the pre-shrink path) and the label-map edge is on the correct pixel. Regression test:
`test_registration_is_centre_consistent_in_network_input_and_label_map`. The earlier landmark-polygon IoU
measurements in this document were taken before the fix and were not re-measured (all 191 pre-existing tests pass unchanged).

## Evaluation results

Pilot against EasyPortrait v2 test annotations (N = 600, stratified, seed 20261110, 598 detected): report
`docs/evaluation/easyportrait_test_2026-11-10.md` (and `.json`), sample ids in
`docs/evaluation/easyportrait_test_sample_seed20261110_n600.txt`, how it was produced in
`docs/plans/reports/12_coder_report_phase3_eval.md`. Headline *agreement with EasyPortrait annotations*
(dataset-level IoU, full-image scoring, detected images): skin 0.866, lips 0.787, eye 0.717, brow 0.710; left/right
swaps 0 of 545 (eyes) and 0 of 593 (brows). Weak spots: eyes in the 48 glasses images (IoU 0.0) and in the 93 images
where the landmark/parser alignment check failed (0.17); a consistent best-shift of about 1-2 px downward.
The skin miss against the provisional 0.90 target is a precision issue (precision 0.888, recall 0.973, area ratio
1.10): about 10% of our skin + nose pixels lie in EasyPortrait's PERSON class (the catch-all for hair, neck, clothing
and the like) and under 1% in BACKGROUND. Robustness (100 dev images, `docs/evaluation/robustness_2026-11-10.md`):
mirror, scale 0.5, JPEG q30, blur, gamma and grayscale change eye/brow/lips agreement by at most 0.04; roll of
+/-30 degrees costs about 0.12-0.16 and +/-45 degrees 0.35-0.54 (detection stays at 97-98%); upside-down faces are
detected in only 36% of images. Benchmark: `docs/evaluation/benchmark_2026-11-10.md`.
**Timing caveat:** the per-stage timings in the pilot report (19:18-20:43) were taken on battery power and the machine
slept at times during that window, so its latency columns (especially p95 and maximum) are unreliable; the re-run
benchmark (2 MP p95 176 ms; on battery, `caffeinate -dims`, 0 sleep events in its window) is the reference for latency.
The latency target is provisional and was not measured on AC power. All aggregates are over the stratified 600-image
sample, not re-weighted to the 6000-image test split. Roll costs of 0.12-0.16 (+/-30) and 0.35-0.54 (+/-45) meet the
roll item of decision rule 2 (a roll-aligned crop is a WP6 candidate needing user approval). Rule 3 shows no trigger on
the headline numbers, but the glasses stratum (eye IoU 0.00, n=48; all within the 93 alignment failures, eye IoU 0.168)
is very poor; the fine-tuning decision is not made and is owed to the user.
**Open finding (Phase 4, not fixed):** `app/services/facial_analysis/geometry.py:33` converts normalised landmarks to
pixels as `x*w` without the pixel-centre (-0.5) correction before `cv2.fillPoly`, which is consistent with the
symmetric (-1, -1) px offset of landmark polygons against EasyPortrait annotations (parser offset is y-only, about
+2 px, opposite in sign to the landmark polygons; its cause (label convention vs pipeline) is unexplained, and a
vertical-flip check was inconclusive, see the robustness report).
This is agreement with another dataset's annotation conventions on a stratified sample, not an accuracy figure for the
product. Skin-tone stratification of the same run (ITA bands of the GT skin) plus a 35-portrait qualitative check is in
`docs/evaluation/skin_tone_2026-11-11.md`: no measurable disadvantage for darker skin in any class; all worst-band gaps
meet the plan's targets once the glasses stratum is excluded; two watch items (2 dark-band `no_face` failures on
underexposed images, and the nose-zone threshold at 0.30 sitting at the typical confidence). The numeric targets in
the plan are provisional until confirmed.
Data licence: EasyPortrait (Kapitanov, Kvanchiani, Kirillova, arXiv:2304.13509),
https://github.com/hukenovs/easyportrait, a variant of CC BY-SA 4.0; evaluation only, nothing redistributed.

## Region support matrix (Phase 3 regions + initial Phase 4 zones)

| Region | Support in this implementation |
|---|---|
| Facial skin, forehead, left/right cheek, nose, chin, jawline | Parser `skin`/`nose` + landmark polygons (zones `face_skin`, `forehead`, `cheek_l/r`, `nose`, `chin`, `jawline`) |
| Under-eye (left/right) | Landmark band below the lower lid, clipped to skin (`under_eye_l/r`) |
| Brow head/arch/tail, upper eyelid, brow bone (left/right) | `brow_head/arch/tail_l/r`: the landmark brow polygon split at 35% and 70% of its head-to-tail axis, checked by coverage within 0.05 ipd of the parser brow mask (coverage within a tolerance, not IoU), not intersected with it. `eyelid_upper_l/r`: strip above the upper eye contour (0.18 ipd), minus eye and brow, clipped to skin. `brow_bone_l/r`: strip below the lower brow contour (0.12 ipd), minus brow, eye and eyelid, clipped to skin; staged in `highlighter`. Eyelid and brow bone have no parser positive mask (coverage rule only). Demoted when glasses cover 20% or more or the parent eye/brow is not ok; the eyelid is also iris-demoted via group `eye`. Offsets are untuned. No eyelid crease, no lower eyelid |
| Under-eye inner/outer halves, inner/outer eye corners, upper/lower lash lines (left/right) | `under_eye_inner/outer_l/r`: the under-eye band split by a line through the eye centre, clipped to skin (coverage rule only). `eye_corner_inner/outer_l/r`: small disc at landmark 133/33 inside the landmark eye polygon, checked by coverage within 0.05 ipd of the parser eye mask (coverage within a tolerance, not IoU). `lash_line_upper/lower_l/r`: geometric strokes along the eye contour. All 12 are demoted to low_confidence when glasses cover 20% or more of the zone, or when the parent eye is not ok; nothing is inferred behind lenses. Offsets are untuned approximations |
| Left/right eye, left/right eyebrow | Parser mask (`eye_l/r`, `brow_l/r`), validated by landmark IoU |
| Upper/lower/full lips, outer/inner lip boundary | Landmark polygons (`lip_upper/lower/full`), validated against parser lip classes; boundaries are geometric strokes (`lip_outer_boundary`, `lip_inner_boundary`) |
| Chin center, under lower lip, mouth surround | Landmark bands/dilation clipped to parser `skin` (`chin_center`, `under_lower_lip`, `mouth_surround`; the last excludes `lip_full`) |
| Forehead center/perimeter, left/right temple, left/right outer jawline | Landmark geometry clipped to parser `skin` (`forehead_center`, `forehead_perimeter`, `temple_l/r`, `jawline_outer_l/r`). No parser positive mask exists for these, so they are validated only by the coverage rule (at least 50% of the expected area is skin); offsets are untuned approximations. Stages: primer, foundation, setting spray only |
| Cupid's bow, left/right lip corners | Landmarks only: `cupid_bow` is a geometric stroke through the upper-lip peaks; `lip_corner_l/r` are small regions around 61/291 inside `lip_full`, validated against the parser lip classes |
| Hair, background, ears, neck, cloth, hat, glasses | Anatomical regions only (used for exclusion and warnings) |
| Earring, necklace | Mapped but unverified (see above) |
| **Not supported, not returned** | Eyelid crease, eyelid anatomy, nostril openings, hairline under bangs, blemishes/discoloration |
| **Remaining Phase 4 zones, not implemented** | cheek apple/hollow/bone, nose bridge/tip/sides, lower eyelid |

Zone geometry offsets (ipd multiples) are the plan's approximations, tuned only by eye on one photo.

## Stages

`stages` is an ordered list of `{number, key, name, layers}` where `layers` maps a product layer key to zone ids
(declared once in `application_zones/catalog.py`, `STAGES`). The 11 stages: 1 Primer, 2 Color Corrector,
3 Foundation / Skin Tint, 4 Concealer, 5 Setting Powder, 6 Bronzer & Contour (separate `contour` and `bronzer`
layers), 7 Blush, 8 Highlighter, 9 Eyebrow & Eye Makeup (`eyebrow`, `eye`), 10 Lips (separate `lipstick`, `lip_liner`,
`gloss` layers), 11 Setting Spray. A layer whose zones are remaining Phase 4 work (`bronzer`, `blush`) is present
with an empty list so the UI can show "no zones yet"; no stub zones exist. Currently `contour` holds `jawline`, and
`highlighter` holds `brow_bone_r/l` (Batch 4 extends it).
A test checks that every referenced zone exists and every implemented zone is used by at least one stage.

## Validation rules

- `coverage` = clipped area / in-image geometric area. < 0.5 gives `low_confidence` with a reason.
- Fewer than 16 pixels gives `unavailable`: no mask and a `reason` (the reason says when glasses/hair covered the area).
- Eyes, brows, nose, lips: confidence = IoU between the landmark polygon and the parser mask. Below the threshold
  (eye 0.30, brow 0.20, lip 0.40, nose 0.30, heuristics, not tuned) the zone is `low_confidence`.
- Other zones: confidence = mean parser probability inside the zone. **Uncalibrated.**
- `alignment_check`: iris 468 in `eye_r`, 473 in `eye_l`, nose tip in `nose`/`skin`, mouth centre (13) in
  mouth/lip classes, both brow midpoints in the brow classes; fraction < 0.6 gives `parser_landmark_mismatch`.
- Warnings: `face_partially_outside_image`, `low_resolution_face` (ipd < 40 px), `head_tilt` (|roll| > 25 deg, not
  corrected), `glasses_detected` (>= 1 % of the parsed area), `parser_landmark_mismatch`, `low_overlap_zone`,
  `hair_occludes_forehead` (>= 15 % of the expected forehead), `background_face_ignored`.
- `retake_suggested`: `["glasses"]` exactly when `glasses_detected` is raised, otherwise `[]` (always `[]` unless
  status is `ok`). It is a soft hint: the photo is never rejected, the UI shows a generic retake prompt, eye and
  under-eye zones stay `unavailable` or `low_confidence`, and nothing is guessed behind the lenses. A generative
  preview with glasses on is Phase 9 "experimental".
- WP6-G measurement (pilot `per_image.jsonl`, 598 detected images, 2026-11-11): the warning fires on 48 images; on
  every one 89-100 % of the ground-truth eye pixels are labelled GLASSES and eye IoU is 0.00. False alarms 0, missed
  glasses 0. Glasses area is >= 3.1 % on all warned images and <= 0.59 % on all others, so the 1 % threshold needs no
  tuning. All 48 are among the 93 alignment failures (eye classes empty behind the lenses, iris checks fail); the
  other 45 alignment failures have mean eye IoU 0.41.

## Multiple faces

MediaPipe runs with `num_faces=2`. The primary face is the one with the largest landmark bbox area (index order is not
assumed; observed to vary). A second face with area >= `SECONDARY_FACE_AREA_RATIO` (0.20) x primary rejects the photo
(`multiple_faces_detected`, HTTP 200, empty landmarks) on both `/landmarks` and `/segmentation`; a smaller one is
ignored with `background_face_ignored`. Observed limitation: the detector only finds a second face that is not tiny
relative to the frame (in a synthetic side-by-side test using the full portrait, a second face at 70 % scale was not
detected at all; with head crops it was), so very small background faces are neither rejected nor warned about. A
very small face is simply not seen.

## Measurements

Image: `tests/fixtures/portrait.jpg` (820x1024). Windowed parse = 492 px square, label map 492x471. Median of
**30** warm requests per stage after 1 discarded request, `segment_face` end to end excluding HTTP. Hardware as above.
Peak RSS is the process maximum resident set (`ru_maxrss`) of one process that also holds the MediaPipe landmarker
(~200 MB before the parser is loaded).

| | ResNet18 (selected) | ResNet34 |
|---|---|---|
| Weights file | 53.2 MB | 93.6 MB |
| Read + SHA-256 | 39 ms | 65 ms |
| `InferenceSession` creation | 35 ms | 34 ms |
| First inference (warm-up) | 89 ms | 140 ms |
| Parser constructor total (read, hash, session, warm-up) | 142 ms | 244 ms |
| landmarks (median / p95) | 9.0 / 9.1 ms | 9.0 / 9.5 ms |
| preprocess | 4.6 / 4.8 ms | 4.5 / 4.9 ms |
| inference | **83.1 / 83.6 ms** | **136.0 / 138.9 ms** |
| postprocess | 24.1 / 24.7 ms | 24.2 / 25.0 ms |
| zones | 12.8 / 13.2 ms | 13.1 / 14.0 ms |
| encode | 4.5 / 4.6 ms | 4.6 / 4.8 ms |
| **total** | **139 / 141 ms** | **193 / 199 ms** |
| First request after startup | 156 ms | 225 ms |
| RSS after landmarker only | 206 MB | 192 MB |
| RSS after parser loaded | 573 MB | 689 MB |
| Peak RSS after 30 requests | 652 MB | 770 MB |
| JSON payload (`ok`, all zones) | 74.0 kB | 73.9 kB |

The payload is 74 kB for this image: one label-map PNG plus 18 zone PNGs (base64) plus polygons.
(An early planning estimate was 50-200 ms inference; measured 83 ms for ResNet18.)

Larger photos (ResNet18, median of 7; same face upscaled, JPEG q90; window = 2x the face):

| Image | decode | window | label map | zones | encode | total | payload | peak RSS (process) |
|---|---|---|---|---|---|---|---|---|
| 0.8 MP | 4 ms | 492 | 492x472 | 13 ms | 5 ms | 144 ms | 74 kB | 606 MB |
| 10.3 MP (2870x3584) | 55 ms | 1716 | 1716x1647 | 159 ms | 49 ms | 361 ms | 141 kB | 964 MB |
| 24.5 MP (4428x5529) | 118 ms | 2653 | 2653x2545 | 392 ms | 115 ms | 676 ms | 208 kB | 1350 MB |

Inference is constant (~86 ms) because the network always sees 512x512. For large photos the **label map is produced
at original resolution**, which makes zone derivation, PNG encoding and memory grow with the face's pixel size. At the
25 MP cap one request peaks near 1.35 GB (process), and `FACE_PARSER_MAX_CONCURRENCY=2` allows two such requests at
once. The measured numbers above are for sequential requests; concurrent memory was not measured. A cap on the working
resolution (e.g. 2048 on the window's long side with a scale factor in the response) would remove this, but it changes
the API contract (masks no longer 1:1 with image pixels), so it was not done.

Not measured: concurrent request latency, other CPUs, GPU/CoreML, quantisation (deferred).

## Pre-deployment blockers

Besides the licence (above), these must be resolved before any deployment (Security review iteration 1):

- **F1, unbounded in-flight memory:** the semaphore covers inference only; decoded images and label maps of concurrent
  requests are not bounded (about 1.35 GB peak for one 25 MP request, measured sequentially). Needs a request-level
  limiter and a stricter pixel policy.
- **F2, no pre-parse body cap:** the byte limit is applied after the multipart body has been spooled. Needs a cap at the
  server/proxy or before parsing.
- Rate limiting, authentication, legal review of biometric data.

## Security notes

- Uploads: declared type allow-list, byte limit, **decoder allow-list (JPEG/PNG/WebP only, by real content)**, pixel
  cap checked from the header before decoding (25 MP), Pillow decompression-bomb errors mapped to 400.
- Weights are loaded only from the configured path, SHA-256 verified, built from the verified bytes.
- Inference is bounded by a semaphore; waiting requests time out with 503 + `Retry-After`.
- Logs contain only exception types and timings; 500 responses are generic. Nothing is persisted.
- `pip-audit` (environment and `requirements.txt`, 2026-11-10): no known vulnerabilities.
- Deferred until before any deployment: rate limiting, authentication, and legal review of biometric data.

## Limitations

- Licensing (above) blocks commercial use.
- Occlusions degrade results: glasses (eyes can vanish), bangs/hats (forehead), beards (parsed as skin, lips poorly
  separated), strong shadows. Zones then report `low_confidence`/`unavailable` rather than guessing.
- Lighting, age and pose bias are **unquantified**. Skin-tone agreement is stratified in
  `docs/evaluation/skin_tone_2026-11-11.md` (apparent tone under the photo's lighting, EasyPortrait + 35 studio
  portraits); low-light selfies of dark skin remain untested.
- Head roll above 25 degrees only warns. Large yaw/pitch is not handled. **At about +/-35 degrees of roll the
  parser/landmark alignment check fails and the eye masks degrade** (observed in test reports 07 and 08, synthetic
  rotations of the fixture; the response carries `head_tilt` and `parser_landmark_mismatch`). The parse window is
  axis-aligned and there is no roll compensation. Not yet quantified on ground truth (planned robustness suite).
- Parser masks are not resolution-independent: small regions (eyes, lips) are coarse at low resolution.
- Landmark-derived zones are approximations; ipd offsets were tuned visually on one photo.
- Large photos: memory/latency growth described above.
- Classes 9 and 15 unverified. The compare/overlay UI was exercised in headless Chrome on the fixture only.
- **Mirrored photos** get geometry-based sides (swapped vs. the real person); see the Left/right convention.
- **Half-face recall:** when only part of the face is in the frame, parser regions and landmark-derived zones can be
  partly missing; zones then report `low_confidence`/`unavailable` and the face-outside warning is raised.
- **Inner-canthus gap:** the parsed eye mask stops about 5 px short of the landmark inner eye corner on the fixture.
- **Truncated landmark result (fixed 2026-11-10):** if MediaPipe returns fewer than 478 landmarks for the selected
  face (including zero), `/landmarks` and `/segmentation` answer HTTP 200 with `status="no_face_detected"`,
  `face_detected=false`, `landmark_count=0` and the message "Face landmarks were incomplete; please retake the photo
  with your whole face visible." Only the point count is logged. The response contract is unchanged. (Earlier
  behaviour: an `IndexError` and a generic 500.)
- **Pillow's own header guard** (about 179 MP) also applies; the 25 MP cap is stricter and checked first.
- `lip_full`: its `polygon` is the outer lip contour, while its `mask` is the union of the upper and lower lip rings and
  therefore **excludes the mouth opening** (teeth).
- Layer membership: `lipstick` = `lip_full`, `lip_upper`, `lip_lower`, `lip_inner_boundary`; `gloss` = `lip_full`;
  `lip_liner` = `lip_outer_boundary`. `chin` is in `setting_powder` (requirement text) although the plan's zone table
  does not list stage 5 for it.

## Evaluation protocol

No accuracy can be claimed without ground truth. Proposed protocol:

1. Collect >= 200 photos covering skin tones (e.g. Fitzpatrick I-VI), ages, lighting, pose, glasses, beards,
   bangs/hats, makeup, and phone vs. studio images. Record consent and licence.
2. Hand-annotate per-region masks (skin, forehead, cheeks, nose, chin, jaw, eyes, brows, lips, hair, glasses...) with at
   least double annotation on a subset to estimate inter-annotator agreement.
3. Per region and per subgroup report IoU, Dice, precision and recall, with confidence intervals. Put extra weight on
   **small regions** (eyes, brows, lips), where a few pixels dominate IoU, and use boundary-tolerant metrics as well.
4. Evaluate the landmark-derived zones against annotated zone masks, and check whether the IoU-based confidence and
   status thresholds actually separate good from bad cases (calibration).
5. Keep a held-out set that is never used for tuning offsets or thresholds.

## Fine-tuning recommendations (nothing was trained)

- **Never fine-tune from, or pseudo-label with, the CelebAMask-HQ-trained weights**: that carries the non-commercial
  restriction into the new model.
- Commercial path: train/fine-tune from a permissively licensed backbone on data you own or are licensed to use
  (own annotated photos, and datasets whose licence permits commercial use after legal review, e.g. EasyPortrait: share-alike
  CC BY-SA variant, needs review). Export to ONNX with a 512x512 NCHW float input and per-class logits, then replace
  the model via the parser + mapping module only.
- Add classes the zones need that this model lacks (eyelid, nostril, hairline) rather than approximating them.
- Re-run the class-order and left/right verification and the evaluation protocol before switching.
