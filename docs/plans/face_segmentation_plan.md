# Approved Plan: Makeup-Aware Facial Segmentation, Phase 1

Status: **APPROVED by the user on 2026-11-10.** Author: Planner agent. Orchestrator amendments are marked **[AMENDED]**.

Approved decisions:

| # | Decision | Approved choice |
|---|---|---|
| 1 | Licensing | Accept the CelebAMask-HQ **non-commercial** license risk for the prototype. Show the warning in the README, docs and every API response |
| 2 | Model | BiSeNet **ResNet18** ONNX. Benchmark ResNet34 for the docs only |
| 3 | Weights | Download via script with SHA-256 verification. Gitignore them; never commit |
| 4 | Pixel cap | 25 MP |
| 5 | Multiple faces | **[AMENDED] Implement now and reject** (see "Multiple faces" below) |
| 6 | MediaPipe multiclass | Document only as the commercially clean partial fallback. Do not implement |
| 7 | Phasing | **Phase 1 now.** Phase 2 zones after Phase 1 passes the full workflow |
| 8 | Zone geometry | ipd-relative approximations, tuned visually later |
| 9 | UI location | Extend the existing `/landmark-test` page |

**Future-proofing rule** (the user plans to replace CelebA with their own commercially trained model):

- No model-specific class ids may leak outside `services/segmentation/` (the parser plus `MODEL_ID_TO_CANONICAL`).
- The zone engine, API and frontend use only canonical region names.
- Swapping the model must be a change to the parser and mapping module only.

---

## Goal

Add one new endpoint. Given a selfie, it returns two layers, kept architecturally separate:

- **Layer A, anatomical segmentation.** A pretrained face-parsing model (local ONNX inference, no training) produces pixel-level region masks.
- **Layer B, application-zone derivation.** MediaPipe landmarks, face geometry and the Layer A masks derive makeup-zone masks. Every zone is clipped and validated against the anatomical masks.

A zone catalog plus a stage-to-zone map covers the 11 makeup stages. Nothing is rendered. The existing landmark test page is extended to show the overlays, with per-region toggles, an opacity slider and a compare slider.

Findings from the code (orchestrator-verified where marked ✔):

- ✔ **No pixel guard.** `image_loading.load_rgb_image` calls `img.load()` with no size check. Pillow's `DecompressionBombError` is **not** an `OSError` (verified), so it escapes the `except` and becomes a 500.
- **Declared content type only.** The route checks the declared content type. Pillow then decodes with every plugin it has (TIFF, BMP, GIF and others).
- **Single face only.** `FaceLandmarkService` uses `num_faces=1`, so multiple faces cannot currently be detected.
- **Landmark startup is strict.** A missing landmark model aborts startup.
- ✔ **OpenCV is already installed.** `opencv-contrib-python==5.0.0.93` comes in as a mediapipe dependency. Do **not** add `opencv-python-headless`: two cv2 distributions overwrite each other's `cv2/` directory. Pin `opencv-contrib-python==5.0.0.93` explicitly in requirements.txt.
- **`.gitignore` covers only `backend/models/*.task`.** `*.onnx` must be added.

## Current Architecture

- `backend/app/main.py`
  - `create_app(use_lifespan)` loads `FaceLandmarkService` once in `lifespan` and stores it in `app.state.landmark_service`.
  - Tests use `use_lifespan=False` plus `dependency_overrides`.
- `backend/app/api/facial_analysis.py`
  - Sync routes, which FastAPI runs in a threadpool.
  - The upload checks (content type, size read, decode) are inline in the route. They need extracting into a helper shared by both routes, without behaviour change.
  - `get_landmark_service` is the DI hook.
- `backend/app/services/image_loading.py` decodes to an RGB uint8 array and applies EXIF transpose. Mask coordinates must be in this post-EXIF space.
- `backend/app/services/facial_analysis/landmarks.py` holds the only MediaPipe code and returns the `FacialLandmarkResult` Pydantic model. Landmarks are normalized: 478 points, irises at 468 (subject right) and 473 (subject left).
- `backend/app/services/facial_analysis/geometry.py` has `_to_pixels` (private) and documented indices. Add a public `landmark_pixels()` wrapper.
- `backend/app/config.py` uses pydantic-settings with `resolved_*_path` helpers.
- Frontend:
  - `FaceLandmarkViewer.tsx` draws one canvas at natural image resolution.
  - `page.tsx` owns state, layer checkboxes and the Detect button.
  - `lib/api.ts` has `request<T>`.
  - `types/facialLandmarks.ts` manually mirrors the backend schemas.
- Tests: 9 pass. They use `FakeService` plus one real MediaPipe test.

## Multiple faces [AMENDED, approved by the user]

- `FaceLandmarkService` uses `num_faces=2`.
- The **primary face** is the face with the largest landmark bounding-box area. Do not assume index 0.
- A second face is **significant** if its bbox area is ≥ `secondary_face_area_ratio` (config, default **0.20**) × the primary face's area.
- **Significant second face:**
  - `status = "multiple_faces_detected"`
  - `face_detected = true`
  - `landmarks = []`, `measurements = null`
  - `message = "Please upload a photo with only one face."`
  - HTTP 200, the same pattern as `no_face_detected`.
- **Insignificant second face:** analyze the primary face normally. The segmentation response adds a warning `background_face_ignored`.
- **Additive field** on `FacialLandmarkResult`: `face_count: int` (faces found by MediaPipe, 0-2).
- Applies to **both** `/landmarks` and `/segmentation`. For `/segmentation`, a multiple-face rejection returns that status with empty anatomical data and zones.
- **Contract impact on `/landmarks`:**
  - Additive only: a new `face_count` field and a new `status` literal value.
  - Existing fields are unchanged.
  - A two-face photo that previously returned the first face's landmarks now returns the rejection. This is an intentional behaviour change.
- **Frontend:** the test page shows `message` for any non-`ok` status. It must no longer assume that every non-ok result means "no face".
- **Tests:**
  - Fake-service tests for both endpoints.
  - A unit test for the primary-face selection and ratio rule.
  - If feasible, a synthetic two-face image built by placing the portrait fixture twice side by side.

## Model Recommendation

All latencies below are estimates, not measurements. The Coder must measure real numbers.

| Candidate | Classes | License | Runtime dep | Size | Est. CPU latency (UNMEASURED) |
|---|---|---|---|---|---|
| **yakhyo BiSeNet ResNet18 ONNX** | 19 (CelebAMask-HQ) | Code MIT; training data non-commercial | onnxruntime (+cv2, present) | 53.2 MB | roughly 50-200 ms at 512² |
| yakhyo BiSeNet ResNet34 ONNX | same 19 | same | onnxruntime | 93.6 MB | roughly 100-400 ms |
| jonathandinu SegFormer-b5 | same 19 | non-commercial | onnxruntime | about 330 MB | roughly 1-4 s |
| FaRL via facer (LaPa 11 / CelebM) | 11 or 19 | code MIT; weights unclear; LaPa non-commercial | PyTorch plus facer | hundreds of MB | roughly 1-4 s |
| MediaPipe selfie_multiclass_256 | 6 (background, hair, body-skin, face-skin, clothes, others) | Apache 2.0 | installed | small | roughly 10-40 ms |
| EasyPortrait SegFormer-B0 (researched by the orchestrator) | 9 (background, person, skin, L/R brow, L/R eye, lips, teeth) | custom CC BY-SA 4.0 variant, **no non-commercial clause**; needs legal review (share-alike, Russian law, sanctions) | needs a one-time mmseg→ONNX export | 45 MB | unmeasured |

**Selected: BiSeNet ResNet18 ONNX via onnxruntime.**

- Weights: `https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx` (53,205,364 bytes). ResNet34 is the same with `resnet34.onnx` (93,632,554 bytes).
- Make the weights path and expected SHA-256 configurable.

**The Coder must verify these on the real model and record the results in `docs/face_segmentation.md`:**

1. Read the ONNX session metadata for input and output names, shapes, dtype and dynamic or fixed H/W. Do not hardcode names from memory.
2. Confirm 19 output channels and the preprocessing (RGB, 512×512, ImageNet mean/std, NCHW float32). Check that output is logits, not argmaxed.
3. Confirm the class order by rendering each class over the portrait fixture and looking at it. The expected order is 0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g, 7 l_ear, 8 r_ear, 9 ear_r, 10 nose, 11 mouth, 12 u_lip, 13 l_lip, 14 neck, 15 neck_l, 16 cloth, 17 hair, 18 hat.
4. Confirm whether `l_*` means subject-left or image-left, using irises 468 (subject right, image left) and 473 (subject left). Encode the result in `MODEL_ID_TO_CANONICAL`, with a regression test. The runtime `alignment_check` also guards this.
5. Test crop scale at 1.4, 1.7 and 2.0 times the landmark-bbox side (square) on the fixture. Choose one and record it.
6. Test the full-frame, no-crop path and record how much worse it is.
7. Measure model load time, warm-up, steady-state latency (median of at least 20 runs, per stage), peak RSS, and ResNet34 on the same image. Report them as measured.
8. Check what `skin` and `nose` cover: nostrils, neck boundary, ears, beards, glasses.

Discovery and debug scripts that render images must write **only** to the session scratchpad: `/private/tmp/claude-501/-Users-test-Makeup-demo/61b8fe69-19b2-4eda-8907-06c60cd1f728/scratchpad/`. Never write them to the repo.

## Region Support Matrix

| Requested region | Support | Notes |
|---|---|---|
| Facial skin | Model (`skin` ∪ `nose`) | minus eyes, brows and lips |
| Forehead | Model + landmarks | Landmark polygon (extended above 10) ∩ skin; the hairline comes from the model |
| Left/right cheek | Model + landmarks | Landmark polygon ∩ skin |
| Nose | Model direct, refined by landmarks | Nostril openings not resolved |
| Chin | Model + landmarks | Polygon ∩ skin |
| Jawline | Model + landmarks | Band along the face oval ∩ skin; unreliable with beards or neck shadow |
| Hair | Model direct | Inside the parse window only, otherwise `not_parsed` |
| Background | Model direct | Same limitation |
| Left/right eye | Model direct + landmark eye polygon | Visible eye opening only |
| Upper/lower eyelid | Landmark-derived band only | No model class. Low confidence, not claimed as anatomy |
| Inner/outer eye corners | Landmarks | 133/362 inner, 33/263 outer |
| Left/right eyebrow | Model direct + landmark polygon | |
| Upper/lower lip | Model direct + landmark polygons | The landmark polygon is the zone; the model is validation |
| Cupid's bow | Landmarks | Around 37, 0, 267; verify visually |
| Inner/outer lip boundary | Landmarks | Mesh lip contours plus the model's `mouth` class |
| Glasses, ears, earrings, neck, necklace, cloth, hat | Model direct | Used for exclusion and warnings |
| **Not supported** | Eyelid crease, eyelid anatomy, nostril openings, hairline under bangs | No class or landmark. Not fabricated |

## Zone construction

### Face frame

All geometry is built in a `FaceFrame`:

- `ipd` is the interpupillary distance in pixels (468↔473).
- `u` is the unit vector from the subject's right eye to the left eye. It is roll-aware.
- `v` is the perpendicular, pointing down the face.
- Offsets are multiples of `ipd`, so zones scale with face size and rotate with head roll.
- Zones are defined once for the subject's right side. The left side is generated through `MIRROR`.

### MIRROR pairs (subject right ↔ subject left)

| Right | Left |
|---|---|
| 33 | 263 |
| 133 | 362 |
| 70 | 300 |
| 107 | 336 |
| 46 | 276 |
| 55 | 285 |
| 61 | 291 |
| 129 | 358 |
| 234 | 454 |
| 93 | 323 |
| 132 | 361 |
| 58 | 288 |
| 172 | 397 |
| 136 | 365 |
| 150 | 379 |
| 149 | 378 |
| 176 | 400 |
| 148 | 377 |
| 127 | 356 |
| 162 | 389 |
| 21 | 251 |
| 54 | 284 |
| 103 | 332 |
| 67 | 297 |
| 109 | 338 |
| 71 | 301 |
| 34 | 264 |
| 156 | 383 |
| 50 | 280 |
| 205 | 425 |
| 123 | 352 |
| 145 | 374 |
| 153 | 380 |
| 144 | 373 |
| 159 | 386 |
| 37 | 267 |
| 39 | 269 |
| 40 | 270 |
| 146 | 375 |
| 91 | 321 |
| 181 | 405 |
| 84 | 314 |
| 78 | 308 |
| 80 | 310 |
| 81 | 311 |
| 82 | 312 |
| 95 | 324 |
| 88 | 318 |
| 178 | 402 |
| 87 | 317 |

- **Midline:** 10, 152, 0, 13, 14, 17, 1, 4, 5, 6, 8, 9, 168, 197, 195.
- **Required test:** the pairs must be symmetric about the midline on the real fixture landmarks.
- Also cross-check the eye, brow and lip pairs against `vision.FaceLandmarksConnections` (LEFT_EYE/RIGHT_EYE, etc.), which is the authoritative source in mediapipe.

### Contour index sets

Verify these visually. Prefer deriving them from `FaceLandmarksConnections` where possible.

- **Eye, subject right.**
  - Upper lid: 33, 246, 161, 160, 159, 158, 157, 173, 133.
  - Lower lid: 33, 7, 163, 144, 145, 153, 154, 155, 133.
- **Brow, subject right.**
  - Upper: 70, 63, 105, 66, 107.
  - Lower: 46, 53, 52, 65, 55.
- **Lips.**
  - Outer upper: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291.
  - Outer lower: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291.
  - Inner upper: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308.
  - Inner lower: 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308.
- **Face oval:** 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109.

### Phase 1 zones (implement now)

All masks are clipped to `face_skin` unless stated. L/R suffixes are `_l`/`_r`, meaning the subject's side.

| id | Source | Construction | Clipped / validated against | Stages |
|---|---|---|---|---|
| `face_skin` | segmentation + landmarks | skin ∪ nose, minus model eye/brow/lip/glasses and minus landmark eye/lip polygons dilated about 1 px | itself; reports mean parser probability | 1, 3, 5, 11 |
| `forehead` | seg + landmarks | Polygon: oval arc [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389] moved up along −v by about 0.35 × (brow-to-10 distance), closed along the brow upper contours shifted up 0.15 ipd | skin (the hairline comes from the model) | 1, 2, 3, 11 |
| `cheek_l/r` | seg + landmarks | Polygon [33 shifted down 0.3 ipd, 133 shifted down 0.3 ipd, ala 129, mouth corner 61, 58, 93, 234] | skin | 1, 2, 3, 11 |
| `nose` | segmentation | `nose` class, validated against the landmark nose region | itself | 1, 2, 3, 11 |
| `chin` | seg + landmarks | Polygon: oval [136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365] closed by the mouth corners and 17 shifted down 0.12 ipd | skin | 1, 2, 3, 11 |
| `jawline` | seg + landmarks | Band along the oval [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361], 0.18 ipd thick, inward | skin | 1, 3, 6, 11 |
| `under_eye_l/r` | seg + landmarks | Band below the lower lid contour (shifted down 0.22 ipd), minus the eye polygon | skin | 1, 2, 4, 5 |
| `eye_l/r` | segmentation | `l_eye`/`r_eye` (after the verified mapping), validated against the landmark eye polygon (IoU) | itself | 9 |
| `brow_l/r` | segmentation | brow classes, validated against the landmark brow polygon | itself | 9 |
| `lip_upper`, `lip_lower` | landmarks | Outer plus inner contour polygons | validated against `u_lip`/`l_lip` (IoU becomes confidence) | 10 |
| `lip_full` | landmarks | union of upper and lower (lipstick and gloss) | as above | 10 |
| `lip_outer_boundary`, `lip_inner_boundary` | landmarks | Polylines stored as polygon vertices plus a thin stroke mask (lip liner) | none (geometric) | 10 |

### Phase 2 zones (NOT now; listed so the catalog structure accommodates them)

- **Under-eye and eye corners:**
  - `under_eye_inner/outer_l/r`
  - `eye_corner_inner/outer_l/r`
- **Forehead:**
  - `forehead_center`
  - `forehead_perimeter`
- **Cheeks and sides of the face:**
  - `temple_l/r`
  - `cheek_apple_l/r` (ellipse at 50/280)
  - `cheek_upper_l/r`, `cheek_outer_l/r`
  - `cheekbone_l/r`
  - `cheek_hollow_l/r`
- **Nose:**
  - `nose_bridge` (168, 6, 197, 195)
  - `nose_tip` (1)
  - `nose_side_l/r`
  - `nose_crease_l/r`
- **Chin and mouth:**
  - `chin_center`
  - `under_lower_lip`
  - `mouth_surround`
  - `jawline_outer_l/r`
- **Lips:**
  - `cupids_bow`
  - `lip_corner_l/r`
- **Brows:**
  - `brow_head/arch/tail_l/r`
  - `brow_bone_l/r`
- **Eyelids and lashes:**
  - `eyelid_upper/lower_l/r`
  - `lash_line_upper/lower_l/r`

### Stage map

`STAGES` maps each stage to zone ids. Zones are defined once.

- Phase 1 populates the stages with the Phase 1 zones that exist.
- Stage entries for Phase 2 zones are omitted, not stubbed.
- `bronzer_contour` holds `{contour: [...], bronzer: [...]}`.
- Lipstick and gloss reference `lip_full`; lip liner references `lip_outer_boundary`.

### Deferred or unsupported (document; do not return)

- **Eyelid crease:** no landmark or class marks it.
- **Nostril exclusion:** no nostril points in the mesh.
- **Discoloration and blemishes:** future capabilities.
- **Placement styles:** future.

### Validation

- **Coverage** = clipped area ÷ in-image geometric area. Below 0.5 gives `low_confidence` with a reason.
- **Minimum size:** under about 16 px gives `unavailable`, with no mask and a `reason`.
- **Lips, eyes, brows:** confidence is the IoU between the landmark polygon and the model mask.
- **Model regions:** report `mean_probability`, labelled uncalibrated.
- **`alignment_check`:**
  - 468 must fall in `r_eye` and 473 in `l_eye`.
  - The nose tip (1) must be in nose or skin.
  - The mouth centre (13) must be in mouth or lip.
  - Brow midpoints must be in the brow classes.
  - A pass fraction under 0.6 gives a `parser_landmark_mismatch` warning.

## API Contract

`POST /api/facial-analysis/segmentation` runs the landmark service internally: one upload, one decode, one coordinate space. `/landmarks` and `/topology` are unchanged except for the approved multiple-faces amendment.

**Response (`FaceSegmentationResponse`).** All coordinates are original-image pixels (post-EXIF).

```
status: "ok" | "no_face_detected" | "multiple_faces_detected"
message: str | null
image: {width, height}
model: {id, architecture, weights_sha256, input_size,
        license_warning: "Trained on CelebAMask-HQ (non-commercial research data). Not cleared for commercial use."}
landmarks: FacialLandmarkResponse
parse_window: {x, y, size}
anatomical: {
  encoding: "png_label_map_v1", origin: {x, y}, width, height, not_parsed_value: 255,
  label_map_png_base64: str,       # 8-bit grayscale PNG, pixel value = canonical region id
  regions: [{id, name, side, source: "segmentation", pixel_count, area_fraction, mean_probability, present}]
}
zones: [{id, name, group, side, sources: [...], status: "ok"|"low_confidence"|"unavailable", reason,
         confidence, clipped_to: [...], bbox: {x, y, width, height}, pixel_count,
         mask: {encoding: "png_bbox_v1", png_base64} | null, polygon: [[x, y], ...] | null}]
stages: {stage_key: [zone ids]}
alignment_check: {passed, fraction}
warnings: [{code, message}]
timings_ms: {preprocess, inference, postprocess, zones, encode, total}
```

**Failure semantics:**

| Case | Result |
|---|---|
| Wrong type | 415 |
| Wrong format | 400 |
| Too many bytes | 413 |
| Corrupt image, or too many pixels | 400 |
| No face, or multiple faces | 200 with status |
| Parser unavailable | 503 with an actionable message (landmarks keep working) |
| Busy | 503 with `Retry-After` |
| Inference error | 500 with a generic message; log only the exception type |

**Warning codes:**

- `face_partially_outside_image`
- `low_resolution_face`
- `head_tilt`
- `glasses_detected`
- `parser_landmark_mismatch`
- `low_overlap_zone`
- `hair_occludes_forehead`
- `background_face_ignored`

**Mask encoding:**

- **Anatomical masks:** one label-map PNG, because the classes are mutually exclusive.
- **Zones:** bbox-cropped PNGs, because zones overlap.
- **Polygons:** supplementary, for landmark zones only.
- **Rejected:** raw JSON arrays, full-size per-region PNGs, custom RLE, polygons for model masks.
- Measure the real payload size.

## Files

**Backend, new** (under `backend/`):

- `app/schemas/face_segmentation.py`
- `app/services/segmentation/__init__.py`
- `app/services/segmentation/regions.py` (canonical vocabulary and `MODEL_ID_TO_CANONICAL`)
- `app/services/segmentation/face_parser.py` (`FaceParser` Protocol and `OnnxFaceParser`)
- `app/services/segmentation/masks.py`
- `app/services/application_zones/__init__.py`
- `app/services/application_zones/frame.py`
- `app/services/application_zones/catalog.py`
- `app/services/application_zones/derive.py`
- `app/services/face_regions.py` (orchestrator)
- `app/api/face_segmentation.py`
- `app/api/uploads.py`
- `scripts/download_face_parser.py`
- Tests:
  - `tests/test_face_segmentation_api.py`
  - `tests/test_face_parser.py`
  - `tests/test_application_zones.py`
  - `tests/fixtures/portrait_landmarks.json`

**Backend, modified:**

- `app/config.py`
  - `face_parser_model_path`, `face_parser_sha256`, `face_parser_crop_scale`
  - `max_image_pixels`
  - `face_parser_max_concurrency`, `face_parser_queue_timeout_s`
  - `secondary_face_area_ratio`
- `app/main.py`
- `app/services/image_loading.py`
- `app/api/facial_analysis.py`
- `app/services/facial_analysis/landmarks.py` (multiple faces)
- `app/schemas/facial_landmarks.py` (`face_count`, status literal)
- `app/services/facial_analysis/geometry.py`
- `requirements.txt` (`onnxruntime==1.31.1`, `opencv-contrib-python==5.0.0.93`, `pip-audit` under dev)
- `.env.example`
- `.gitignore`
- `README.md`

**Docs, new:** `docs/face_segmentation.md`.

**Frontend, new:**

- `types/faceSegmentation.ts`
- `lib/segmentation.ts`
- `components/SegmentationControls.tsx`

**Frontend, modified:**

- `lib/api.ts`
- `components/FaceLandmarkViewer.tsx` (optional props; behaviour unchanged when absent)
- `app/landmark-test/page.tsx`
- `types/facialLandmarks.ts` (`face_count`, status)

## Implementation Steps (Phase 1)

1. **Hardening and the shared upload helper.**
   - `Image.open(..., formats=["JPEG","PNG","WEBP"])`.
   - Check `width*height > max_image_pixels` before `load()`.
   - Map `DecompressionBombError`/`Warning`/`OSError` to `ImageValidationError`.
   - Extract `api/uploads.py`.
   - Rerun the existing 9 tests.
   - Add tests for the pixel cap, an unsupported real format (GIF/BMP bytes labelled image/png) and the bomb path.
2. **Multiple faces [AMENDED].** `num_faces=2`, primary-face selection, the ratio rule, `face_count`, the new status, and tests.
3. **Model acquisition and discovery.**
   - Download script with SHA-256; run it; pin the hash; add `*.onnx` to `.gitignore`.
   - Throwaway discovery in the scratchpad: metadata, class order and left/right.
   - Fill `MODEL_ID_TO_CANONICAL` from the verified results.
4. **Schemas and canonical regions.**
5. **Parser service.**
   - Square crop around the landmark bbox × `crop_scale`, padded with the ImageNet mean.
   - `warpAffine` to 512, with an exact inverse for the output.
   - Upsample logits bilinearly to the window, capping the long side at 1024; nearest-map to the original.
   - Load once in `lifespan` with warm-up; semaphore with timeout; typed `FaceParserError`.
   - Pixels outside the window are `not_parsed` (255).
6. **Mask utilities.** Rasterize, stroke, ellipse, morphology, largest component, bbox, PNG encode at a low compression level.
7. **`FaceFrame` and `MIRROR`,** with a symmetry test on the fixture.
8. **Zone engine.**
   - `ZoneSpec`, `derive_zones`, clipping, coverage and IoU validation, statuses, and `STAGES`.
   - Rasterize within the parse window only.
   - The Phase 1 zones; check each visually on the fixture (scratchpad renders).
9. **Orchestrator and router.** `alignment_check`, warnings, timings, the limiter, and 503 when the parser is `None`.
10. **Wiring:** main, config, requirements, `.env.example`.
11. **Backend tests.**
    - Stub-model geometry test: a synthetic non-square image with an off-centre rectangle and a brightness-classifier stub must align within 1 px.
    - API tests with fakes.
    - Zone property tests: subset of clip region, low-overlap flagging, unavailable status.
    - PNG round trip.
    - Real-model integration test that skips without weights: 19 channels, `alignment_check.passed`, dimensions, zones ⊆ clip regions.
12. **Frontend.**
    - Types, API call, decode and compose, controls, viewer extension.
    - Compose the overlay only on toggle change.
    - Left and right in different hues; zones drawn as fill plus a 1-2 px outline.
    - Wipe compare slider; stage select-all; hover shows region and zone.
    - Show `message` for every non-ok status.
    - Run `npm run typecheck` and `npm run build`.
13. **Measure and document** in `docs/face_segmentation.md`:
    - model selection and the licensing blocker, plus the future-proofing rule;
    - the verified class mapping and the support matrix;
    - measured latency, load time, RSS and payload, for ResNet18 and ResNet34;
    - limitations;
    - an evaluation protocol (hand-annotated ground truth; IoU, Dice, precision, recall per region; small-region focus);
    - fine-tuning recommendations, including the EasyPortrait and own-data path and the rule **never fine-tune from, or pseudo-label with, the CelebA weights**.
    - Update the README.

## Acceptance Criteria

1. The model is documented with its actual 19 classes and verified mapping. The licensing warning appears in the README, the docs and the API response.
2. Weights load once with warm-up. Missing or corrupt weights leave landmarks working and segmentation returns 503 with an actionable message. SHA-256 is verified, and the weights are not committed.
3. The class mapping, including left/right, is verified empirically with a regression test, and `alignment_check.passed` is true on the portrait fixture.
4. Label map dimensions equal the covered region. A reconstructed full-size map aligns with the landmarks within 1 px, proven by the stub-model test.
5. All masks are valid: binary or label values only, no out-of-bounds bbox, and no empty mask reported as `ok`.
6. Left and right are distinguishable for eyes, brows, cheeks and under-eyes, in colour and in naming.
7. Every Phase 1 zone has id, name, sources, status, bbox, mask and confidence where meaningful. Each is a subset of its clip region. Statuses are honest, and nothing is fabricated.
8. Unsupported items are documented and absent from the response.
9. Invalid input is handled:
   - wrong type, wrong format, corrupt, too many bytes, too many pixels;
   - no face, or multiple significant faces (200 with status);
   - parser unavailable or busy (503);
   - inference error (500, generic).
10. The frontend supports upload, segment, per-region and per-zone toggles, opacity, a wipe compare and landmarks alongside. Non-ok messages are shown.
11. The existing 9 tests pass. Any change to an existing test is limited to the approved additive contract change and is explained. New tests pass, and typecheck and build pass.
12. Latency, load time, memory and payload are recorded as measured values. No accuracy claim is made without ground truth.
13. The Security Reviewer reports no unresolved critical or high findings.

## Security (address now)

- Pixel cap and decompression-bomb handling.
- Decoder allow-list (JPEG, PNG, WebP).
- SHA-256-pinned weights loaded from settings only.
- Concurrency semaphore that returns 503.
- Log only exception types and timings; never images, filenames, landmarks or masks. Generic 500 messages.
- No image persistence. Debug renders go to the scratchpad only.
- Frontend: React-escaped text only; PNGs via `createImageBitmap(Blob)`; no `dangerouslySetInnerHTML`.
- Run `pip-audit` and record the result.
- **Deferred until before any deployment:** rate limiting, auth, and legal review of biometric data.

## Performance (address now)

- Inference on the face crop only, with warm-up at startup and configured thread count.
- Logit upsample capped at 1024.
- Zones rasterized in the parse window.
- Low-compression PNG; skip encoding `unavailable` zones.
- Measure the payload.
- Landmarks embedded in the response, so MediaPipe does not run twice.
- Frontend composes on toggle change only.
- **Deferred:** CoreML/GPU, quantization, caching.

## Risks

- **Licensing:** a commercial blocker.
- **Class order and left/right swap:** guarded by verification, a test and the runtime check.
- **Domain gap:** mitigated by cropping. Head roll over about 25° gives a `head_tilt` warning and is not corrected.
- **Occlusions:** glasses, bangs and beards degrade results. Skin-tone and lighting bias is documented as unquantified.
- **Mesh indices:** need visual verification.
- **Partial faces:** produce a warning and lowered status.
- **Small regions:** eyelids and lip edges get coarse masks, so they are landmark-based.
- **Memory:** about 80 MB transient per request, bounded by the semaphore.
- **Frontend compose cost:** bounded to the parse region.
