# AI Makeup Assistant

An anatomy-aware makeup assistant. It adapts reference makeup techniques to the user's own facial geometry
instead of transferring pixels. The current pipeline is **selfie → MediaPipe Face Landmarker (478 landmarks) →
face parsing (anatomical regions) → 70 makeup-application zones → structured JSON → canvas visualization**.
A reference makeup photo goes through the same pipeline and becomes a versioned **MakeupBlueprint** (Phase 7).
There is no makeup rendering and no feature classification yet. The full plan is in
[docs/ROADMAP.md](docs/ROADMAP.md).

> **LICENSE WARNING.** The face-parsing weights (BiSeNet, ONNX) were trained on **CelebAMask-HQ,
> which is non-commercial research data. They are not cleared for commercial use.** This is accepted
> for the prototype only. The warning is repeated in every `/segmentation` response. See
> [docs/face_segmentation.md](docs/face_segmentation.md).

## Milestones

This README is updated each time a roadmap milestone closes.

| Phase | Milestone | Status |
|---|---|---|
| 1 | Secure image upload and loading | **Done.** Validates magic bytes, byte limit and 25 MP pixel cap before decoding; fixes EXIF orientation. |
| 2 | Facial landmarks and measurements | **Done.** MediaPipe 478 landmarks, pixel measurements, and rejection when a second face is ≥ 20% of the largest. |
| 3 | Face segmentation | **Closed for the prototype (2026-11-11).** Pilot results on EasyPortrait (N=600), as IoU: skin 0.866, lips 0.787, eyes 0.717, brows 0.710. Glasses signal and skin-tone testing are done. Some fixes are deferred (see below). |
| 4 | Facial geometry and makeup application zones | **Done (2026-11-11).** 70 zones in batches 1–6, staged across 11 makeup steps; 512 backend tests. |
| 5 | FacialProfile construction | **Done (2026-11-12).** Versioned profile (`schema_version` 1.0) from `POST /api/facial-analysis/profile`. Each component (landmarks, frame, parsing, zones) has its own provider and status, so one failure leaves the rest valid. Nothing is persisted. Includes frontend types and an API client (no UI). 607 backend tests. |
| 6 | Makeup taxonomy and application mapping | **Done (2026-11-12).** Versioned vocabulary (1.1) of 12 categories with product types, finishes, techniques, placements and directions; a versioned `MakeupComponent` schema with sRGB hex and derived CIELAB; and `map_zones`, which resolves a product and placement to zone ids without leaving the product's `STAGES` layer. The highlighter and bronzer layers were filled. 930 backend tests. |
| 7 | Makeup reference analysis (MakeupBlueprint) | **Done (2026-11-12).** `POST /api/makeup/reference-analysis` returns a versioned `MakeupBlueprint` (1.0). CV measures each zone's color against that face's bare skin and owns color, undertone, intensity and zones. An optional Claude VLM (`claude-opus-5-5`, **off by default**) adds only technique, finish, placement, direction and style. Every field records its source (`cv`/`vlm`). Nothing is persisted. 1129 backend tests. |
| 7.5 | Facial feature classification | **Done (2026-11-12).** FacialProfile 1.1 adds an optional `features` component to `/profile`. It covers eye spacing, tilt and shape; lip fullness, balance and width; brow arch; face shape; and skin depth and undertone. Every label carries its measurements and a confidence, and ambiguous faces return `unknown`. Classification is deterministic, with no training, and calibrated on 598 real faces. Eyelid type is self-reported only (always `unknown` for now). Nothing is persisted. 1261 backend tests. |
| 8 | Makeup adaptation engine (AdaptedMakeupPlan) | Next |
| 9–13 | Rendering, customization, … | Planned (see [docs/ROADMAP.md](docs/ROADMAP.md)) |

**Deferred Phase 3 items (not done).**
- Roll-aligned crop, y-offset fix and the `present` floor.
- A provenance/consent record for the test portrait. This is required before any launch.
- The fine-tuning decision. No custom model is trained without approval.
- Re-measuring latency on AC power.

```
backend/                          FastAPI (Python 3.11+; tested on 3.13)
  app/
    main.py                       app factory, CORS, model loaded once at startup
    config.py                     env-var settings (backend/.env)
    api/facial_analysis.py        HTTP routes only (/landmarks, /topology)
    api/face_segmentation.py      HTTP route for /segmentation
    api/facial_profile.py         HTTP route for /profile
    api/makeup_reference.py       HTTP route for /api/makeup/reference-analysis
    api/uploads.py                shared upload validation (type, bytes, real format, pixels)
    schemas/
      facial_landmarks.py         Landmark, FacialLandmarkResult, FaceMeshTopology (raw data)
      face_measurements.py        BoundingBox, FaceMeasurements, FacialLandmarkResponse (derived)
      facial_profile.py           versioned FacialProfile (components, providers, quality)
      makeup_component.py         versioned MakeupComponent (vocabulary-checked, hex + derived CIELAB)
      makeup_blueprint.py         versioned MakeupBlueprint (components + per-field cv/vlm sources, style, quality)
    services/
      image_loading.py            bytes → RGB numpy (+ EXIF orientation fix)
      facial_analysis/
        landmarks.py              FaceLandmarkService — the only MediaPipe code
        geometry.py               pixel measurements from landmarks (no classification)
      face_regions.py             orchestrator: landmarks -> parser -> zones -> response
      segmentation/               Layer A: face parser (ONNX); the ONLY place model class ids exist
      application_zones/          Layer B: FaceFrame (frame.py), zone catalog + stages (catalog.py),
                                  zone derivation (derive.py)
      facial_profile/             provider protocols + adapters, per-component builder, versions/compat
      facial_features/            deterministic feature classifiers (eye, lip, brow, face shape, skin tone)
                                  and their thresholds
      makeup_vocabulary/          frozen term registry with aliases, resolver, color (hex→CIELAB), versions/compat
      makeup_mapping/             placement→zone table and map_zones (product + placement → zone ids)
      makeup_blueprint/           reference analysis: CV analyzer, VLM protocol + stub + Claude provider,
                                  analysis service (CV + VLM → MakeupBlueprint), versions/compat
  evaluation/                     segmentation eval harness (EasyPortrait IoU, robustness, latency bench);
                                  datasets live outside the repo, reports go to docs/evaluation/
  scripts/download_model.py
  scripts/download_face_parser.py  face-parser weights + SHA-256 check
  tests/                          unit, API and integration tests, per-phase verification suites
                                  (test_phase*_*.py) and golden_batch*.json zone snapshots
frontend/                         Next.js 16 + React 19 + TypeScript
  app/landmark-test/page.tsx      debug page
  components/FaceLandmarkViewer.tsx  canvas overlay
  lib/api.ts                      fetch helpers
  lib/facialProfile.ts            /profile client: version check, typed errors, isComponentAvailable, getFeatures
  types/facialLandmarks.ts        TS mirror of the Pydantic schemas
  types/facialProfile.ts          TS mirror of the FacialProfile schema
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

`POST /api/facial-analysis/profile`: same upload rules. Returns a versioned **FacialProfile**: `schema_version`, `status`, `image`, `components` (`landmarks`, `frame`, `parsing`, `zones`), `quality` (alignment, warnings, `retake_suggested`) and `timings_ms`. Masks are embedded by default. With `?include_masks=false`, the label map and zone masks are dropped, but bbox, polygon and regions are kept. The profile is built per request and is never stored or logged.

- **Components:** each has `meta` (`component_version`, `status` ok/degraded/unavailable, `confidence`, `provider` with id, weights sha256, `algorithm_version` and licence warning, `reason`) and `data`. `data` is `null` exactly when the component is `unavailable`. A failing stage marks only that component, and the components that depend on it, unavailable.
- **Versioning:** a MINOR bump adds optional fields only. A MAJOR bump is a breaking change. The frontend client (`fetchFacialProfile` in `frontend/lib/facialProfile.ts`) rejects unsupported MAJOR versions.
- **Errors:** the same as `/segmentation`. A busy or missing parser gives 503 (with `Retry-After` when busy). The client turns this into `ProfileBusyError`, with the retry delay clamped to 0–60 s. In TypeScript, narrow a component with `isComponentAvailable(c)` or `c.data !== null`, not by checking `meta.status`.

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
cd backend && pytest          # 1261 tests as of Phase 7.5
```
Unit tests use fake services and a stub parser. Integration tests run real MediaPipe and the real face parser on
`tests/fixtures/portrait.jpg` and are skipped if the respective weights are missing. Each Phase 4 batch has a
`golden_batch*.json` snapshot, taken before that batch's changes. It is used to prove that earlier zones stay
bit-identical when new ones are added. The snapshots store hashes only, never masks or image data.

> **The test portrait is not in the repository.** `tests/fixtures/portrait.jpg` and its landmark file are
> git-ignored, because the image shows a real person and no provenance/consent record exists yet. To run the
> integration tests and the `curl` examples above, supply your own photo at that path.

## Face segmentation (Phase 3) and application zones (Phase 4)

There are two layers.
- **Layer A** is the face parser. Its model class IDs exist only inside `services/segmentation/`.
- **Layer B** builds the application zones from the landmarks in a face-aligned frame (`FaceFrame`). It clips them to
  the parser's anatomical regions, so they handle roll and yaw. Each zone reports a `status` (`ok`,
  `low_confidence`, `unavailable`), a reason, a confidence and its coverage.

The 70 zones cover:
- **Skin and face areas:** face skin, forehead (center, perimeter), temples, cheeks (apple, upper, outer, cheekbone,
  contour hollow), jawline (and outer jawline), chin (and chin center).
- **Eyes and brows:** eyes, under-eye (inner and outer halves), inner and outer eye corners, upper and lower lash lines,
  upper and lower eyelids, brows (head, arch, tail), brow bone. The eyelid crease cannot be derived reliably and is
  never added.
- **Nose:** nose, bridge, tip, sides, nostrils. Nostrils are exclusion areas and belong to no makeup stage.
- **Mouth:** full, upper and lower lip, inner and outer lip boundaries, Cupid's bow, lip corners, mouth surround,
  under the lower lip.

The **stage map** assigns zones to 11 makeup steps: primer, color corrector, foundation, concealer, setting
powder, bronzer and contour, blush, highlighter, eyebrow and eye, lips, setting spray. Every layer has zones.
Bronzer was filled in Phase 6.

## Makeup taxonomy and zone mapping (Phase 6)

This is a backend library. There is no endpoint. Phase 7 and 8 code call it directly.

- **Vocabulary:** `services/makeup_vocabulary/`, version 1.1.
  - It is a frozen registry of 12 categories, with product types, finishes, techniques, placements and directions.
  - Aliases are matched regardless of case, separators and apostrophes, so "Cupid's Bow" becomes `cupid_bow`.
  - Each product type names its `STAGES` stage and layer. Special effects have none.
  - To add a term, edit the data and bump the MINOR version.
- **`MakeupComponent`** (`schemas/makeup_component.py`):
  - Only `category` and `product_type` are required.
  - `color_hex` is stored as `#RRGGBB`, and `color_lab` (CIELAB D65) is always derived from it.
  - Raw input is hidden in errors. Log `str(exc)` or `exc.errors(include_input=False)`.
- **Zone mapping:** `map_zones(product_type, placement, target_zones)` and `map_component(c)` return a `ZoneMapping`.
  - The product's layer comes from `STAGES` and is read live.
  - A placement narrows the layer to its intersection with the placement.
  - An unsupported combination returns a typed `none` result instead of raising.
  - Explicit zones outside the layer raise `ZoneOutsideLayerError`, and a missing stage raises `CatalogDriftError`.

```python
from app.services.makeup_mapping import map_zones
map_zones("Cream Blush", "lifted cheek")   # cheek_upper_r, cheek_upper_l
map_zones("bronzer", "temple")             # temple_r, temple_l
```

On `/landmark-test`, click **Segment face**. The panel lets you toggle anatomical regions and application zones
(left and right use different hues), select all zones of a makeup stage, change overlay opacity, and compare with a
wipe slider. Hover shows the region and zones under the cursor. Details, the verified class mapping, measured
performance, limitations and the evaluation protocol: [docs/face_segmentation.md](docs/face_segmentation.md).

## Makeup reference analysis (Phase 7)

`POST /api/makeup/reference-analysis` takes a multipart field `file` and applies the same upload rules as
`/profile`. It returns a `MakeupBlueprint` with `Cache-Control: no-store`.

| Case | HTTP | Body |
|---|---|---|
| Analyzed | 200 | `status: "ok"`, `components`, `style`, `quality` (warnings, `overall_confidence`), `providers`, `timings_ms` |
| No face / multiple faces | 200 | `status: "no_face_detected"` / `"multiple_faces_detected"`, `message`, no components |
| Bad upload | 400 / 413 / 415 | Same as `/profile` |
| Parser missing or busy | 503 | Fixed message; busy adds `Retry-After: 2` |
| Analysis fault | 500 | `{"detail": "Reference analysis failed."}`, never exception text |

How the blueprint is built:
- **CV** (`cv_zone_color`) runs the reference photo through the FacialProfile pipeline. It compares a trimmed-median
  color for each zone against the same face's bare skin.
  - It detects lipstick, blush, eyeshadow, highlighter and contour, and sets their color, undertone, intensity, zones
    and confidence.
  - It does not detect brows, eyeliner or mascara, because natural hair and lashes can't be told apart from product.
  - It skips eyeshadow when glasses are detected.
- **VLM** is a provider protocol.
  - The default is a stub, which adds a `vlm_disabled` warning.
  - The Claude provider receives a re-encoded JPEG (long side at most 1568 px), never the original upload. It must
    answer with enum values only. It fills technique, finish, placement and direction on CV components, adds style
    labels, and may propose up to 4 brow, eyeliner or mascara components with no color or zones.
  - It cannot set or override CV fields.
  - A refusal, SDK error or invalid output degrades to CV-only results, with a `vlm_refused` or `vlm_failed` warning.

Settings (see `backend/.env.example`):

| Variable | Default | |
|---|---|---|
| `MAKEUP_VLM_ENABLED` | `false` | When true, reference photos are sent to the Anthropic API |
| `MAKEUP_VLM_MODEL` | `claude-opus-5-5` | |
| `MAKEUP_VLM_EFFORT` | `medium` | `low` / `medium` / `high` |
| `MAKEUP_VLM_TIMEOUT_S` / `MAKEUP_VLM_MAX_TOKENS` | `30` / `16000` | |
| `MAKEUP_VLM_MAX_RETRIES` | `1` | Each retry resends the face image |

**Credentials** are never read from `.env`. The SDK uses the profile from `ant auth login`, or an
`ANTHROPIC_API_KEY` exported in the shell.

> **Before enabling the VLM outside local development:** the route has no authentication or rate limit, so any
> caller could trigger paid Claude calls with uploaded photos. Every request with the VLM on is one billed API call.
> Refusals may be served by another Anthropic model (server-side fallbacks).

## Facial features (Phase 7.5)

`POST /api/facial-analysis/profile` now returns FacialProfile **1.1**, with an optional `components.features`
component. It has its own provider (`facial-features`) and status. A profile without features omits the key
instead of sending `null`, so 1.0 readers keep working. Skin tone is computed even with `include_masks=false`.
Reference analysis (Phase 7) does not compute features.

| Feature | Labels | Measured from |
|---|---|---|
| Eye spacing / tilt / shape | close · average · wide / downturned · neutral · upturned / almond · round | Landmarks, in the roll-removed face frame |
| Lip fullness / balance / width | thin · medium · full / bottom_heavy · balanced · top_heavy / narrow · average · wide | Landmarks |
| Brow arch | straight · soft_arched · high_arched | Landmarks |
| Brow thickness | always `unknown` | Not measurable from MediaPipe's template brow band |
| Face shape | oval · round · square · heart · oblong · diamond | Nearest calibrated prototype over 4 ratios. Confidence is capped at 0.8; a turned head gives `unknown` |
| Skin depth | very_light · light · intermediate · tan · brown · dark | Photo-measured ITA° on the bare-skin sample. Confidence is capped at 0.7 |
| Skin undertone | cool · neutral · warm · olive | Skin-hue scale. Confidence is capped at 0.6; a strong colour cast gives `unknown` |
| Eyelid type | always `unknown` | **Self-report only, never detected.** The input arrives with the frontend–backend connection |

How classification works:
- Every label is a `Classified` value with `label`, `confidence`, `measurements` and `reason`. A value inside an
  ambiguity margin gives `unknown` rather than a forced label.
- **Confidence is not a probability of being correct.** It measures how clear-cut the measurement is.
- Thresholds are calibrated on 598 cached EasyPortrait faces (offline, aggregates only):
  [face shape](docs/evaluation/face_shape_calibration_2026-11-12.json),
  [eye, lip and brow](docs/evaluation/geometry_calibration_2026-11-12.json) and
  [skin tone](docs/evaluation/skin_tone_distribution_2026-11-12.json). The scripts that produce them are in `backend/evaluation/`.
- Relative descriptors (wide-set, upturned, high-arched) use population bands: the middle 60% is "average".

> **Limits.** No ground-truth labels exist, so **accuracy is unmeasured** for every feature. Treat features as soft,
> confidence-weighted hints, and never gate users on them. Skin tone needs the most care:
> - photo ITA depends on exposure and white balance;
> - the undertone edges are fitted on one dataset;
> - the share of "cool" undertones varies by depth band, which is likely a camera artifact;
> - foundation changes the measured skin.
>
> Skin tone and eyelid type are planned to become user-confirmable once the frontend is connected. Feature values are
> sensitive: they are never logged or persisted.

