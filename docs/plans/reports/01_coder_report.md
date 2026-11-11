# Coder Report: Face Segmentation Phase 1

Agent: `coder`. Two passes were made: the initial implementation, then a stage-naming fix requested by the orchestrator. The orchestrator checked: 64 tests pass, typecheck is clean, no face images or weights outside gitignored or fixture paths, and on the fixture `alignment_check` is 1.0 with 18 zones and 11 named stages.

## 1. Implementation Summary

- **New endpoint:** `POST /api/facial-analysis/segmentation`. It returns:
  - a label-map PNG;
  - 18 zone masks, each a bbox-cropped PNG;
  - landmark polygons;
  - the stage list;
  - `alignment_check`, warnings and timings;
  - the license warning, in every response.
- **Parser:** BiSeNet ResNet18 ONNX.
  - Square parse window around the face.
  - Weights verified by SHA-256, with a warm-up at startup and a concurrency semaphore.
  - If the weights are missing or corrupt, `/landmarks` keeps working and `/segmentation` returns 503.
- **Zone engine:** Phase 1 zones only, built in a frame that follows head roll and scales with the interpupillary distance. Statuses and confidence are honest; nothing is faked.
- **Multiple faces (amendment):**
  - The landmarker looks for up to 2 faces (`num_faces=2`), and the largest face is the primary.
  - A second face of at least 0.20 of the primary's area returns `multiple_faces_detected`.
  - A smaller second face is ignored, and the response carries `background_face_ignored`.
  - `face_count` is added. All of this applies to both endpoints.
- **Upload hardening:**
  - Shared helper `api/uploads.py`.
  - Decoder allow-list of JPEG, PNG and WebP, checked against the real content.
  - 25 MP cap, checked before decoding.
  - Decompression bombs return 400.
- **Frontend:**
  - "Segment face" button on `/landmark-test`.
  - Toggles per region and per zone, grouped stage, then layer, then zone, with select-all at stage and layer level.
  - Opacity slider, wipe compare and a hover tooltip.
  - A `message` is shown for every non-ok status.
- **Docs:** `docs/face_segmentation.md` and the README.

## 2. Files Created

Backend:

- `app/schemas/face_segmentation.py`
- `app/services/segmentation/__init__.py`, `regions.py`, `face_parser.py`, `masks.py`
- `app/services/application_zones/__init__.py`, `frame.py`, `catalog.py`, `derive.py`
- `app/services/face_regions.py`
- `app/api/face_segmentation.py`
- `app/api/uploads.py`
- `scripts/download_face_parser.py`
- `tests/test_face_segmentation_api.py`, `tests/test_face_parser.py`, `tests/test_application_zones.py`, `tests/synthetic.py`
- `tests/fixtures/portrait_landmarks.json`

Frontend:

- `types/faceSegmentation.ts`
- `lib/segmentation.ts`
- `components/SegmentationControls.tsx`

Docs:

- `docs/face_segmentation.md`

Gitignored download: `backend/models/face_parser_resnet18.onnx`.

## 3. Files Modified

- `backend/app/config.py`
- `backend/app/main.py`
- `backend/app/services/image_loading.py`
- `backend/app/api/facial_analysis.py`
- `backend/app/services/facial_analysis/landmarks.py`
- `backend/app/services/facial_analysis/geometry.py`
- `backend/app/schemas/facial_landmarks.py`
- `backend/requirements.txt`
- `backend/.env.example`
- `.gitignore`
- `README.md`
- `frontend/lib/api.ts`
- `frontend/components/FaceLandmarkViewer.tsx` (optional props only)
- `frontend/app/landmark-test/page.tsx`
- `frontend/types/facialLandmarks.ts`

## 4. Technical Decisions

- **Mapping back to the image:**
  - The parse window has integer size, so mapping back is an exact shift when the window is at most 1024 px. Larger windows are mapped back with nearest-neighbour.
  - The label map covers exactly window ∩ image.
  - Large windows are shrunk with `INTER_AREA` before inference.
  - Softmax probabilities are computed at 512 px and upsampled bilinearly.
- **Weights integrity:** the file is hashed in memory, and the ONNX session is built from those same bytes, so the file can't change between the check and the load.
- **Region ids:** canonical ids are deliberately different from the model's class ids (for example `EYE_R=6` while the model's `r_eye` is 5), so a skipped mapping fails in tests.
- **Clipping:** face zones are clipped to the skin class (nose excluded) inside `face_skin`.
- **Lip boundaries:** `lip_outer_boundary` and `lip_inner_boundary` are geometric strokes with no confidence value.
- **Error order:** upload validation runs before the parser-availability check, so bad input still gets a 4xx even when the parser is down (503).
- **Stages:** an ordered list of `Stage{number, key, name, layers}`, defined declaratively in `catalog.py`. Bronzer and contour are separate layers, and lipstick, lip liner and gloss are separate layers. Empty layers stay in the list for Phase 2.

## 5. Security

**Mitigated:**

- The declared type is checked against an allow-list, then the real content against the decoder allow-list.
- Byte cap, plus a 25 MP header check before decoding.
- Decompression-bomb, `SyntaxError` and `ValueError` paths return 400.
- Weights are SHA-256-pinned and loaded only from settings.
- The concurrency semaphore returns 503 with `Retry-After`.
- Logs contain only exception types, and 500 responses are generic.
- Nothing is persisted.
- The frontend decodes PNGs with `createImageBitmap(Blob)` and renders only React-escaped text.
- `pip-audit` reports no known vulnerabilities.

**Remaining:**

- The pinned weight hashes are trust-on-first-use, because the upstream publishes no checksums.
- Rate limiting, authentication and the biometric-data legal review are deferred until before deployment.
- The CelebAMask-HQ license blocks commercial use.

## 6. Performance

Measured on an M2 Pro (CPU) with the 820×1024 fixture; values are the median of 30 runs.

| | ResNet18 (selected) | ResNet34 |
|---|---|---|
| Weights | 53.2 MB | 93.6 MB |
| Parser constructor total | 142 ms | 244 ms |
| Inference | 83.1 ms | 136.0 ms |
| Total request | 139 ms | 193 ms |
| Peak RSS after 30 requests | 652 MB | 770 MB |
| Payload | 74.0 kB | 73.9 kB |

**Large images:** a 24.5 MP photo took 676 ms in total. Its label map was 2653×2545, the payload was 208 kB, and peak process RSS reached 1.35 GB. Two concurrent 25 MP requests were not measured.

## 7. Tests

- **Backend:** 64 passed. The original 9 tests are unchanged.
- **New tests cover:**
  - the class mapping and the left/right regression, including a real-model check that iris 468 falls in the subject's right eye and 473 in the subject's left;
  - stub-model alignment within 1 px;
  - the busy (concurrency) path;
  - weights verification;
  - `MIRROR` symmetry, mutation-checked;
  - zone properties;
  - the stage catalog;
  - PNG round trip;
  - upload hardening;
  - multiple faces with real MediaPipe;
  - startup with missing or corrupt weights.
- **Frontend:** typecheck and build pass. The UI was checked once in headless Chrome against the real backend, but not re-checked after the stage UI change.

## 8. Known Limitations

- **Occlusion:** on a second photo with thick glasses and a beard, both eye classes were empty, so the eye zones report `unavailable`. Beard hair is parsed as skin, and lip IoU was low (0.03 / 0.22).
- **Unverified classes:** 9 (earring) and 15 (necklace) were not seen, so they are not marked verified.
- **Tuning:** the 2.0 crop scale and the zone offsets were tuned on one fixture.
- **Not measured:** bias and accuracy.
- **Small faces:** a tiny second face is not detected by MediaPipe.
- **Phase 2:** not implemented.

## Verified model facts

- **Input:** `input` `[batch, 3, 512, 512]`, float32.
- **Outputs:** `output`, `414` and `424`, each 19×512×512. `output` holds logits (not argmaxed), and the other two are auxiliary heads.
- **Class order:** 0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g, 7 l_ear, 8 r_ear, 9 ear_r, 10 nose, 11 mouth, 12 u_lip, 13 l_lip, 14 neck, 15 neck_l, 16 cloth, 17 hair, 18 hat.
- **Left/right:** `l_*` means the subject's left (image right).
- **Crop scale** (IoU of parser classes against landmark polygons, on one photo):

  | Window | Right eye | Left eye | Lips |
  |---|---|---|---|
  | 1.4× | 0.00 | 0.35 | 0.88 |
  | 1.7× | 0.63 | 0.76 | 0.87 |
  | **2.0× (chosen)** | 0.73 | 0.76 | 0.84 |
  | Full frame | 0.00 | 0.00 | 0.77 / 0.19 |

## Deviations from the plan

1. The stage map is named and layered per the user's requirements, after an orchestrator correction.
2. The crop scale is 2.0.
3. For non-ok responses, `anatomical`, `parse_window` and `alignment_check` are null, and `timings_ms` gains a `landmarks` key.
4. Face zones are clipped to the skin class, not `face_skin`.
5. Two extra settings: `FACE_PARSER_THREADS`, plus bounds on `FACE_PARSER_MAX_CONCURRENCY`.
6. `opencv-contrib-python` and `pip-audit` are pinned in `requirements.txt`.
7. Pillow's global `MAX_IMAGE_PIXELS` is left untouched.
