# Planner report: WP6-G glasses signal (Phase 3 part), 2026-11-11

Measurement that gates this work: `18_wp6g_glasses_measurement.md`.

### 1. Goal & Target Files
**Goal:** Expose the existing `glasses_detected` warning as an additive soft hint `retake_suggested: ["glasses"]` and show a generic retake prompt in the UI. The threshold, zone statuses, parser and Phase 1/2 code stay unchanged.

**Files (modify only):**
- `backend/app/schemas/face_segmentation.py`: `FaceSegmentationResponse` (~145).
- `backend/app/services/face_regions.py`: `segment_face()` return at 247-256. The early no_face/multiple_faces return at 207-211 stays untouched, so the field defaults to [].
- `backend/tests/test_face_segmentation_api.py`: stubbed parser and landmarker (`_client()`, `_post()`, tests ~50-118).
- `frontend/types/faceSegmentation.ts`: response interface (~91).
- `frontend/components/SegmentationControls.tsx`: warnings rendering (82-90).
- `docs/face_segmentation.md`: eval paragraph (~150) and warnings list (~210-212).

Do not touch `services/segmentation/`, `facial_analysis/`, the uploads/image_loading code, or `GLASSES_MIN_AREA_FRACTION` (line 39).

### 2. Implementation Steps
1. **Schema.** Add `retake_suggested: list[Literal["glasses"]] = Field(default_factory=list, description=...)` to `FaceSegmentationResponse`, next to `warnings`. The description says it is a soft hint, the photo is never rejected, and it is empty unless status is "ok".
2. **Service.** In `segment_face()`, after `warnings = _warnings(...)` (line 231), compute `retake = ["glasses"] if any(w.code == "glasses_detected" for w in warnings) else []`. Pass `retake_suggested=retake` in the final `FaceSegmentationResponse(...)`. No second glasses computation, no change to `_warnings()`. The early-return branch stays as is.
3. **Backend tests.** Reuse the stub helpers; the stub parser must return a label array whose GLASSES pixel count is controllable.
   - A: glasses pixels >= 1 % of `labels.size` gives the `glasses_detected` warning and `retake_suggested == ["glasses"]`.
   - B: glasses pixels < 1 % (e.g. 0.5 %) gives no warning and `retake_suggested == []`.
   - C: no_face and multiple_faces responses have `retake_suggested == []`.
   - D: the hint is present exactly when the warning code is present.
4. **Frontend types.** Add `retake_suggested?: "glasses"[]` to the response interface (optional for older backends). No `any`.
5. **Frontend UI.** After the warnings list, render a prompt only when `retake_suggested?.includes("glasses")`. Text: "To get eye-makeup results, retake the photo without your glasses. You can keep them on, but eye-area results are limited while glasses are on." Existing styling, a `role="note"` element, no blocking behaviour.
6. **Docs.** Update `docs/face_segmentation.md`: the field; the behaviour (soft prompt, photo never rejected, eye and under-eye zones stay unavailable or low_confidence, nothing guessed behind the lenses, generative preview with glasses on is Phase 9 "experimental"); the measurement paragraph (pilot per_image.jsonl, 598 detected images, 2026-11-11): warning on 48 images, on every one 89-100 % of GT eye pixels labelled GLASSES and eye IoU 0.00; false alarms 0; missed glasses 0; glasses area >= 3.1 % on all warned and <= 0.59 % on all others so the 1 % threshold needs no tuning; all 48 among the 93 alignment failures (eye classes empty behind lenses, iris checks fail); other 45 alignment failures mean eye IoU 0.41.

### 3. Security & Performance
None. The hint is derived from an existing warning. No new logging, no coordinates or image data.

### 4. Acceptance Criteria
- Stubbed parser at >= 1 % GLASSES pixels: `glasses_detected` warning present and `retake_suggested == ["glasses"]`.
- Stubbed parser at < 1 % GLASSES pixels: no warning and `retake_suggested == []`.
- `no_face` and `multiple_faces` responses have `retake_suggested == []`.
- The field defaults to `[]`; the existing response-contract test and `test_phase1_independent.py` (275-276) pass unchanged.
- `GLASSES_MIN_AREA_FRACTION` is still 0.01; zone statuses, the parser, `segmentation/` and `facial_analysis/` are unmodified.
- The full backend pytest suite passes offline.
- `npx tsc --noEmit` in `frontend/` passes; no `any`; a response without `retake_suggested` compiles and renders with no prompt.
- The UI shows the exact prompt text only when the hint contains "glasses".
- No new log statements with coordinates or image data.
- The docs contain the field description, the behaviour notes and all the measurement numbers.
