---
# Development Roadmap

| Phase | Name | Status |
|---|---|---|
| 1 | Image Upload & Validation | ✅ COMPLETE |
| 2 | Facial Landmarks & Contouring | ✅ COMPLETE |
| 3 | Makeup-Aware Facial Segmentation | ✅ COMPLETE for the prototype (2026-11-11); some items deferred, see Phase 3 section |
| 4 | Facial Geometry & Makeup Application Zones | 🟡 **CURRENT**: zone engine and 18 initial zones exist; remaining zones in progress |
| 5 | FacialProfile Construction | NOT STARTED |
| 6 | Makeup Taxonomy & Application Mapping | NOT STARTED |
| 7 | Makeup Reference Analysis | NOT STARTED |
| 8 | Makeup Adaptation Engine | NOT STARTED |
| 9 | Generative Makeup Preview | NOT STARTED |
| 10 | Makeup Customization | NOT STARTED |
| 11 | Personalized Tutorial Generation | NOT STARTED |
| 12 | Live AR Makeup Try-On | FUTURE |
| 13 | Makeup Application Feedback | FUTURE |

## Phase 1: Image Upload & Validation (COMPLETE)

- Accept uploads.
- Validate formats and dimensions.
- Handle invalid or corrupted images.
- Validate face presence.
- Prepare images for inference.

Implemented in `backend/app/api/uploads.py` and `backend/app/services/image_loading.py`:
- JPEG/PNG/WebP decoder allow-list based on actual content;
- byte limit and 25 MP pixel cap, both checked before decoding;
- EXIF orientation.

## Phase 2: Facial Landmarks & Contouring (COMPLETE)

MediaPipe Face Landmarker (Tasks API, mediapipe 1.2.2) produces 478 landmarks, including irises.

- Implemented in `backend/app/services/facial_analysis/` and exposed as `POST /api/facial-analysis/landmarks` and `GET /api/facial-analysis/topology`.
- Multiple faces: a second face whose area is at least 20% of the primary face's is rejected with `multiple_faces_detected`.
- Preserve the existing implementation unless there is a demonstrated issue.

## Phase 3: Makeup-Aware Facial Segmentation (COMPLETE for the prototype)

**Responsibilities:**
- Segment anatomical regions: skin, brows, eyes, nose, lips, hair and the other supported classes.
- Keep left and right separate.
- Produce masks aligned with the original image and the landmarks.
- Visualize the results.

**Current implementation:**
- **Model.** BiSeNet with a ResNet18 backbone, trained on the CelebAMask-HQ dataset. CelebAMask-HQ is a *dataset*, not the architecture. Source: `yakhyo/face-parsing` ONNX release.
  - Runs on CPU via onnxruntime.
  - Weights are pinned by SHA-256.
  - The model has 19 classes, mapped to canonical regions in `backend/app/services/segmentation/regions.py`.
- **Endpoint.** `POST /api/facial-analysis/segmentation`.
- **Left and right.** These follow the MediaPipe landmark convention (`services/segmentation/sides.py`).
- **Docs.** `docs/face_segmentation.md`. Plans and agent reports are in `docs/plans/`.
- **⚠️ LICENSE.** CelebAMask-HQ is **non-commercial research only**. These weights are for the prototype only.
  - The product will move to a commercially licensed or self-trained model.
  - **Never** fine-tune the future model from these weights.
  - **Never** use these weights to pre-label training data.
  - Keep model class ids inside `services/segmentation/` so the model stays swappable.

**Anatomical segmentation is not the same as makeup application zones.**
- The segmentation model identifies anatomical regions.
- Makeup zones are *derived* in Phase 4 from masks, landmarks and geometry.
- Do not expect the model to identify blush placement, contour hollows, cheek apples, cheekbones, under-eye concealer zones, the eyelid crease, or Cupid's bow as a class.

**Status (2026-11-11):** closed for the prototype by the user.
- Done: ground-truth evaluation (EasyPortrait v2 test, N=600; IoU skin 0.866, lips 0.787, eye 0.717, brow 0.710), benchmark, robustness suite, WP4a fixes, WP6-G glasses signal, skin-tone stratification (`docs/evaluation/`).
- Deferred (not done): WP6 fixes (roll-aligned crop, y-offset, `present` floor, speck cleanup, crop scale, resolution cap); provenance and consent record for `backend/tests/fixtures/portrait.jpg` (pre-launch privacy item); fine-tuning decision; latency re-measure on AC power (benchmark ran on battery).

Do not train a custom model without approval.

## Phase 4: Facial Geometry & Makeup Application Zones (COMPLETE)

Calculate proportions, derive anatomical subregions and reusable makeup zones, validate them against the segmentation masks, and handle asymmetry. Zones are reusable across makeup categories. Do **not** build a separate segmentation model per zone.

**Already exists** in `backend/app/services/application_zones/`. It was built alongside the segmentation feature and is kept as the start of Phase 4.
- **Engine features:**
  - a face frame that scales with interpupillary distance and is roll-aware;
  - a left/right mirror map;
  - clipping and validation against the masks;
  - honest status and confidence values;
  - an 11-stage map with product layers.
- **18 initial zones:**
  - face skin, forehead, cheeks, nose, chin, jawline;
  - under-eyes;
  - eyes, brows;
  - upper, lower and full lips;
  - inner and outer lip boundaries.
- **Batch 1 (6 zones):** chin center, under the lower lip, mouth surround, Cupid's bow, left and right lip corners.
- **Batch 2 (6 zones):** forehead center, forehead perimeter, left and right temples, left and right outer jawline.
- **Batch 3a (12 zones):** left and right under-eye inner and outer halves, inner and outer eye corners, upper and lower lash lines.
- **Batch 3b (10 zones):** left and right upper eyelid, brow head, brow arch, brow tail and brow bone. The brow bone is the first `highlighter` entry (Batch 4 adds the cheekbones to that list).
- **Batch 4 (10 zones):** left and right cheek apple, upper cheek, outer cheek, cheekbone and contour hollow. Cheekbones join `highlighter`, hollows join `contour`, apples/upper/outer cheeks are `blush` placements; `bronzer` stays empty.
- **Batch 5 (6 zones):** nose bridge, nose tip, left and right nose sides and nostrils. Sides join `contour`, bridge and tip join `highlighter`; nostrils are unstaged exclusion areas.
- **Batch 6 (2 zones):** left and right lower eyelid. Lower-lid band under the lower lash line, nested in under-eye; joins the eye layer after the upper eyelid.

Phase 4 zone list complete (70 zones). The eyelid crease is not derivable and is never added.

The eyelid crease cannot be derived reliably from landmarks or the current classes.

Approved to start by the user on 2026-11-11.

## Phase 5: FacialProfile Construction (COMPLETE)

Combine landmarks, segmentation masks, geometry, zones and confidences into a **versioned** FacialProfile. Individual components must be replaceable without breaking the rest of the application.

**Scope decided by the user (2026-11-11):**
- Assembly only. No persistence.
- A new `POST /api/facial-analysis/profile` endpoint, with masks embedded unless `?include_masks=false` is passed.
- Feature classification is not part of Phase 5: face, eye and lip shape, and the user's own skin tone and undertone. It needs a roadmap home before Phase 8.

Full details are in `docs/DECISIONS.md`.

## Phase 6: Makeup Taxonomy & Application Mapping (COMPLETE)

**Categories:**
1. Skin preparation: primer, moisturizer, SPF.
2. Color correction.
3. Foundation and skin tint.
4. Concealer.
5. Setting powder and spray.
6. Bronzer and contour.
7. Blush.
8. Highlighter.
9. Eyebrows.
10. Eye makeup.
11. Lip makeup.
12. Special effects and decorative makeup.

**Each component has:**
- category;
- product type;
- color;
- undertone;
- finish;
- intensity;
- placement;
- technique;
- target zone;
- direction;
- confidence.

Examples:
- cream blush + peach + dewy + lifted cheek;
- brown eyeshadow + outer V + smoky blending.

Build a flexible structured representation. Do not hardcode every combination.

**Scope decided by the user (2026-11-12):**
- A schema plus a mapping service. There is no endpoint.
- A versioned vocabulary registry.
- Colors are sRGB hex with derived CIELAB and an optional shade name.
- Special effects are included, may target any zones, and get no new stage.

Full details are in `docs/DECISIONS.md`.

## Phase 7: Makeup Reference Analysis (DONE 2026-11-12)

Detect visible makeup, its categories, colors and shades, placement and techniques, intensity, finish and overall style. Output a **MakeupBlueprint**.

- Combine computer vision (OpenCV color and region analysis) with a vision-language model such as Gemini or Qwen-VL.
- Do not rely solely on an LLM for exact colors or spatial coordinates.

**Scope decided by the user (2026-11-12):**
- CV reuses our face pipeline on the reference image, with per-zone color measured against that face's bare skin.
- The VLM is Claude via the Anthropic API. It handles semantics only, and it is off unless enabled in config.
- Reference photos follow the same privacy rules as selfies.
- Deliverables: a versioned MakeupBlueprint and `POST /api/makeup/reference-analysis`.

Full details are in `docs/DECISIONS.md`.

## Phase 7.5: Facial Feature Classification (DONE 2026-11-12)

Classify the user's own facial features so that Phase 8 can adapt a look to them. The output is a `features` component on the FacialProfile.

**Scope decided by the user (2026-11-12):**
- Automatic, deterministic and explainable classification from existing landmarks, zones and parsing. Every label carries the measurements behind it and a confidence. No model training.
  - Eye geometry: spacing, tilt, almond vs. round.
  - Face shape.
  - Lip shape.
  - Brow shape.
  - Skin tone and undertone, on a skin-specific scale. Phase 7's product thresholds do not carry over.
- **Eyelid type is self-reported** (monolid, inner fold, double, hooded, not sure). The input is added later, together with the frontend–backend connection. Until then the field is always `unknown`. It is not detected from the image: the crease is not derivable, and detection would be unreliable and biased.
- Not in scope: crease detection, and eyelid type from the VLM for reference photos (possible later).

Full details are in `docs/DECISIONS.md`.

## Phase 8: Makeup Adaptation Engine (DONE 2026-11-12)

FacialProfile + MakeupBlueprint produces an **AdaptedMakeupPlan**.

- Map the reference placement to the user's zones.
- Adjust for the user's geometry.
- Preserve the intended aesthetic.
- Output target masks, coordinates and instructions.

Start with deterministic, explainable rules. Do not use agents or LLM calls for straightforward geometry.

**Scope decided by the user (2026-11-12):**
- Two images in one call (selfie and reference). The server builds both analyses, and the core is a pure `adapt(profile, blueprint)`.
- Zone selection from the user's 70 zones plus direction hints. No mask warping. Every adjustment is a named rule with a reason.
- Reference colours are kept. Colour changes come in Phase 10.
- Unknown or low-confidence features skip their rule, fall back to a safe default, and record "not adapted".

Full details are in `docs/DECISIONS.md`.

## Phase 9: Generative Makeup Preview (NEXT)

Produce a realistic preview using Diffusers, SD/FLUX editing, inpainting, ControlNet or IP-Adapter, with spatial and mask conditioning.

The preview must:
- preserve identity and facial structure;
- respect the zones;
- leave hair, background and other features unchanged.

Translate the plan's spatial information into conditioning inputs. Do not just pass the plan as a text prompt.

## Phase 10: Makeup Customization

Users can change colors, products, finish, intensity, placement and technique. Changes update the AdaptedMakeupPlan.

Use lightweight mask-based rendering where possible. Regenerate the image only when necessary.

## Phase 11: Personalized Tutorial Generation

Generate instructions from the AdaptedMakeupPlan, in 11 stages:
1. Primer.
2. Color corrector.
3. Foundation / skin tint.
4. Concealer.
5. Setting powder.
6. Bronzer and contour.
7. Blush.
8. Highlighter.
9. Eyebrows and eyes.
10. Lips.
11. Setting spray.

Each stage gives:
- product or type;
- zone;
- shade;
- tool;
- amount and intensity;
- direction;
- blending;
- tips.

Use the user's actual zones, not generic facial instructions.

## Phase 12: Live AR Makeup Try-On (FUTURE)

Real-time landmarks, segmentation and rendering, using MediaPipe, WebGL, Three.js or GLSL, or a commercial SDK such as Banuba. Do **not** run diffusion on every frame.

## Phase 13: Makeup Application Feedback (FUTURE)

Compare a photo of the user's applied makeup against the AdaptedMakeupPlan: region detection, color, placement, blending and coverage, with confidence-based feedback. For example: "Your blush is slightly lower than the target placement. Try blending upward toward your temples."

Do not use naive pixel-by-pixel comparison.

---

# Engineering Principles

1. **Modular architecture.** Isolate inference, preprocessing, geometry, business logic, API routes and schemas.
2. **Reusable services.** Do not duplicate functionality.
3. **Model independence.** Segmentation and vision models must be replaceable.
4. **Structured data.** Use Pydantic schemas and clear interfaces.
5. **Separation of concerns.** Keep CV, LLM reasoning, geometry and image generation in separate components.
6. **Testing.** Write unit and integration tests for each feature.
7. **Evaluation.** Use measurable metrics. Never claim accuracy without evidence.
8. **Privacy.** Facial images and geometry are sensitive. No unnecessary storage, logging or external processing.
9. **Performance.** Load models once, avoid redundant inference, and consider CPU/GPU usage.
10. **Maintainability.** Prefer clear, documented implementations over unnecessary abstraction.

# Development Philosophy

Build the project **one feature at a time**. Do not implement future phases unless explicitly requested.

Every feature should:
- be independently testable;
- have a clear input and output;
- keep business logic separate from API routes and UI;
- use typed schemas;
- be easy to replace or extend later;
- avoid unnecessary abstraction and over-engineering.

Prefer simple, understandable implementations over premature optimization.

**Pre-deployment blockers** (see `docs/face_segmentation.md`):
- concurrent-request memory bound (F1);
- pre-parse body-size cap (F2);
- rate limiting and authentication;
- legal review of biometric data;
- a commercially licensed segmentation model.

---

## Development Workflow

For new features and significant changes, follow the specialized agent workflow defined in `.claude/agents/`. Do not silently skip the configured agents or fall back to a general-purpose agent. If an agent is unavailable, diagnose the loading problem first.

### Required Workflow

1. **Planning**
   - Invoke the Planner to inspect the codebase and create an implementation plan.
   - Present the plan to the user. Do not implement until the user has approved it.
   - Identify security risks, performance considerations and edge cases.

2. **Implementation**
   - Invoke the Coder with the approved implementation plan.
   - Implement only the requested scope.
   - Follow the existing architecture and coding conventions.

3. **Testing**
   - Invoke the Tester to independently test the implementation.
   - Test normal behavior, edge cases, invalid inputs and regressions.
   - The Tester may modify test files but must not modify production code.

4. **Security Review**
   - Invoke the Security Reviewer to independently inspect the implementation.
   - Require evidence for reported vulnerabilities.
   - Do not allow it to modify production code.

5. **Evaluation**
   - Invoke the Evaluator to independently assess the implementation.
   - It must review the original requirements, the actual code, the test results and the security findings.
   - It returns PASS, FAIL or BLOCKED.

6. **Feedback**
   - If evaluation fails, send the findings to the Coder.
   - Repeat the relevant testing and reviews after changes.
   - Stop after three failed implementation iterations.

### Important Rules

- Do not skip agents in the required workflow.
- Do not assume an agent completed work without reviewing its output.
- Never allow multiple agents to edit the same files simultaneously.
- Do not automatically commit or push changes.
- Ask for approval before destructive changes, major architectural changes, expensive infrastructure, or training models.
- For trivial changes, documentation updates and simple bug fixes, use judgment rather than invoking every agent.
- Agent hand-off reports are scratch files: save them in `.scratch/agent_reports/` (git-ignored), not in the repo.
- When a phase, batch or work package closes, add a short entry to `docs/DECISIONS.md` (decisions, approvals, revised criteria, deferrals) and update the README Milestones table.
