# AI Makeup Assistant — Facial Landmarking

Current scope: **selfie → MediaPipe Face Landmarker → structured JSON → canvas visualization**, plus
**face segmentation (roadmap Phase 3)**: pixel-level anatomical regions from a face-parsing model, plus
the initial makeup-application zones derived from them (early start of roadmap Phase 4). See `CLAUDE.md` for the roadmap. No makeup rendering and no feature classification yet.

> **LICENSE WARNING.** The face-parsing weights (BiSeNet, ONNX) were trained on **CelebAMask-HQ,
> which is non-commercial research data. They are not cleared for commercial use.** This is accepted
> for the prototype only. The warning is repeated in every `/segmentation` response. See
> [docs/face_segmentation.md](docs/face_segmentation.md).

```
backend/                          FastAPI (Python 3.11+; tested on 3.13)
  app/
    main.py                       app factory, CORS, model loaded once at startup
    config.py                     env-var settings (backend/.env)
    api/facial_analysis.py        HTTP routes only (/landmarks, /topology)
    api/face_segmentation.py      HTTP route for /segmentation
    api/uploads.py                shared upload validation (type, bytes, real format, pixels)
    schemas/
      facial_landmarks.py         Landmark, FacialLandmarkResult, FaceMeshTopology (raw data)
      face_measurements.py        BoundingBox, FaceMeasurements, FacialLandmarkResponse (derived)
    services/
      image_loading.py            bytes → RGB numpy (+ EXIF orientation fix)
      facial_analysis/
        landmarks.py              FaceLandmarkService — the only MediaPipe code
        geometry.py               pixel measurements from landmarks (no classification)
      face_regions.py             orchestrator: landmarks -> parser -> zones -> response
      segmentation/               Layer A: face parser (ONNX); the ONLY place model class ids exist
      application_zones/          Layer B: FaceFrame, zone catalog/stages, zone derivation
  scripts/download_model.py
  scripts/download_face_parser.py  face-parser weights + SHA-256 check
  tests/                          test_facial_analysis, test_face_segmentation_api,
                                  test_face_parser, test_application_zones
frontend/                         Next.js 16 + React 19 + TypeScript
  app/landmark-test/page.tsx      debug page
  components/FaceLandmarkViewer.tsx  canvas overlay
  lib/api.ts                      fetch helpers
  types/facialLandmarks.ts        TS mirror of the Pydantic schemas
```

## Setup

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py          # → backend/models/face_landmarker.task (~3.8 MB)
python scripts/download_face_parser.py    # → backend/models/face_parser_resnet18.onnx (~53 MB, SHA-256 verified; NON-COMMERCIAL weights)
cp .env.example .env                      # optional; defaults work for local dev

# Frontend
cd ../frontend
npm install
cp .env.local.example .env.local          # optional; defaults to http://localhost:8000
```

To download the model by hand instead:
`curl -L -o backend/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

## Run

```bash
# Terminal 1
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000     # API docs: http://localhost:8000/docs

# Terminal 2
cd frontend && npm run dev                    # http://localhost:3000/landmark-test
```

## API

`POST /api/facial-analysis/landmarks` — multipart field `file` (JPEG/PNG/WebP, ≤ `MAX_UPLOAD_MB`)

| Case | HTTP | Body |
|---|---|---|
| Face found | 200 | `face_detected: true`, 478 landmarks, measurements |
| No face | 200 | `face_detected: false`, `status: "no_face_detected"`, `message`, empty landmarks |
| Wrong content type | 415 | `{"detail": "..."}` |
| Too large | 413 | `{"detail": "..."}` |
| Corrupt image | 400 | `{"detail": "..."}` |

A photo with more than one significant face (second face bbox area ≥ `SECONDARY_FACE_AREA_RATIO`, default 0.20,
of the largest) returns 200 with `status: "multiple_faces_detected"`, `face_detected: true`, empty landmarks and
`message: "Please upload a photo with only one face."`. A smaller second face is ignored and the largest face is
analyzed. `face_count` (0-2) reports how many faces MediaPipe found. Uploads are limited to JPEG/PNG/WebP (checked on
the real file content, not just the declared type) and to `MAX_IMAGE_PIXELS` (25 MP); violations return 400.

`POST /api/facial-analysis/segmentation` — same upload rules. Runs landmarks internally and returns the anatomical
label map (one PNG), the application zones (bbox-cropped PNG masks), the stage map, an `alignment_check`, warnings
and timings. All coordinates are original-image pixels. Region/zone names are canonical (`eye_l` = the side containing landmarks 263/362/473; see the left/right convention below).

| Case | HTTP | Body |
|---|---|---|
| Face found | 200 | `status: "ok"`, `anatomical`, `zones`, `stages`, `alignment_check`, `warnings`, `timings_ms` |
| No face / multiple faces | 200 | that `status` + `message`, empty `anatomical`/`zones` |
| Wrong type / too large / corrupt / wrong real format / too many pixels | 415 / 413 / 400 | `{"detail": "..."}` |
| Face parser weights missing or failing verification | 503 | actionable message (landmarks keep working) |
| All inference slots busy | 503 + `Retry-After` | |
| Inference failure | 500 | generic message |

```bash
curl -X POST http://localhost:8000/api/facial-analysis/segmentation \
  -F "file=@backend/tests/fixtures/portrait.jpg" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d['alignment_check'], [(z['id'], z['status']) for z in d['zones']])"
```

`GET /api/facial-analysis/topology` — MediaPipe's mesh connections (`tesselation` + named `contours`) used for drawing.

```bash
curl -X POST http://localhost:8000/api/facial-analysis/landmarks \
  -F "file=@backend/tests/fixtures/portrait.jpg" | python3 -m json.tool | head -40
```

Example response (landmarks truncated; real output from the sample portrait):

```json
{
  "face_detected": true,
  "status": "ok",
  "message": null,
  "face_count": 1,
  "image": { "width": 820, "height": 1024 },
  "thresholds": { "min_face_detection_confidence": 0.5, "min_face_presence_confidence": 0.5 },
  "landmark_count": 478,
  "landmarks": [
    { "index": 0, "x": 0.4963, "y": 0.2503, "z": -0.0329 },
    { "index": 1, "x": 0.4966, "y": 0.2260, "z": -0.0639 }
  ],
  "measurements": {
    "face_bounding_box": { "x_min": 307.0, "y_min": 102.2, "x_max": 498.7, "y_max": 347.8, "width": 191.7, "height": 245.5 },
    "interpupillary_distance_px": 86.6,
    "inner_eye_corner_distance_px": 46.8,
    "left_eye_width_px": 35.0,
    "right_eye_width_px": 34.8,
    "mouth_width_px": 100.3,
    "face_width_px": 189.3,
    "face_height_px": 245.6
  }
}
```

### Coordinate conventions
- `x`, `y` are normalized by image width/height, with the origin at the top left. Pixel = `x * width`, `y * height`.
- `z` is relative depth (smaller = closer to the camera), on roughly the same scale as `x`.
- Indices 0–467 are the face mesh and 468–477 are the irises (468 = right iris center, 473 = left).
- **left/right follows the MediaPipe landmark convention:** subject-right is the side containing landmarks 33/133/468. In an unmirrored selfie the subject's left eye (263/362) is on the *right* of the image. **A horizontally mirrored photo cannot be detected, so its sides follow the image geometry (swapped vs. the real person)**; a future capture flag could correct this. Segmentation `_l`/`_r` regions and zones use the same convention (assigned from landmarks, not from the model's own left/right classes).
- Face Landmarker gives **no per-face confidence score**. A face is returned only if it passes both thresholds, so `thresholds` tells you what "detected" meant.

## Verifying the landmarks are correct

On `/landmark-test`, upload a photo and click **Detect landmarks**, then:
1. **Points** should cover the face only, hug the jaw, and not spill into the background.
2. **Contours**: the blue eye outlines sit on the eyelid margins, the pink lips trace the lip line, the yellow brows follow the brows, and the purple iris circles sit on the irises.
3. **Hover** a point to see its index. Spot checks: `1` = nose tip, `152` = chin, `10` = top of forehead, `33` = outer corner of the eye on the image's left, `263` = outer corner of the eye on the image's right.
4. **Measurements** layer: the orange box should enclose the mesh, with red dots on the eye corners, iris centers, and mouth corners.
5. Try a phone photo taken in portrait orientation. If you see a red "aspect ratio" warning or the mesh is rotated, EXIF handling is broken.
6. Try a photo with no face. You should see the "No face detected" message, not an error.

## Tests

```bash
cd backend && pytest
```
Unit tests use fake services and a stub parser. Integration tests run real MediaPipe and the real face parser on
`tests/fixtures/portrait.jpg` and are skipped if the respective weights are missing.

## Face segmentation (roadmap Phase 3) and initial zones (early Phase 4)

On `/landmark-test`, click **Segment face**. The panel lets you toggle anatomical regions and application zones
(left and right use different hues), select all zones of a makeup stage, change overlay opacity, and compare with a
wipe slider. Hover shows the region and zones under the cursor. Details, the verified class mapping, measured
performance, limitations and the evaluation protocol: [docs/face_segmentation.md](docs/face_segmentation.md).
