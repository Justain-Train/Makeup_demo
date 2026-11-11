# Original User Requirements: Makeup-Aware Facial Segmentation

The user's original feature request, copied verbatim, is the authoritative source for evaluation. The approved implementation plan is `face_segmentation_plan.md`.

Clarifications the user gave later:

- **Model.** The user approved the plan with the CelebAMask-HQ model (BiSeNet ResNet18) for a prototype. The user intends to go commercial later with their own trained model, so the model must be swappable.
- **Plan defaults.** The user accepted all of the plan's defaults.
- **Multiple faces.** If a second significant face is detected, the request is rejected with the prompt "only one face allowed". The threshold is 20% of the main face's area, and smaller background faces are ignored.
- **Phasing.** Phase 1 is the current scope. Phase 2 zones come after.

---

# Feature: Makeup-Aware Facial Segmentation

## 4. Feature Objective
Implement two distinct but connected systems:
**A. Anatomical Facial Segmentation**: identify facial regions at the pixel level using a pretrained facial parsing model.
**B. Makeup Application Zone Derivation**: use anatomical segmentation, existing MediaPipe landmarks and geometric calculations to derive precise regions where individual makeup products can be applied.
These must remain separate architectural layers. For example: segmentation identifies the cheek; geometry derives the cheekbone and cheek apple; the makeup application system uses those derived regions to place blush or contour. Do not attempt to make one segmentation model identify every makeup application technique.

## 5. Pretrained Model Research and Selection
Investigate suitable pretrained facial parsing models: BiSeNet-based face parsing models, CelebAMask-HQ-compatible pretrained models, LaPa-compatible face parsing models, and other suitable lightweight facial parsing architectures.
Evaluate: segmentation accuracy, supported facial classes, availability of pretrained weights, model licensing and usage restrictions, CPU and GPU inference performance, Python compatibility, integration with FastAPI, local inference without relying on an external API, and model size and memory requirements. Recommend the most appropriate approach.
Important: CelebAMask-HQ and LaPa are datasets, not inference models themselves. Verify the actual architecture and pretrained weights being used. Do not assume that a model trained on a dataset supports every class in that dataset. Document the model's actual supported classes. Do not train a model from scratch. Start with pretrained inference.

## 6. Anatomical Segmentation Requirements
Identify as many of the following anatomical regions as the selected model reliably supports.
- General facial regions: facial skin, forehead, left cheek, right cheek, nose, chin, jawline, hair, background.
- Eye regions: left eye, right eye, upper eyelid (where supported), lower eyelid (where supported), inner eye corners, outer eye corners.
- Eyebrows: left eyebrow, right eyebrow.
- Lip regions: upper lip, lower lip, Cupid's bow (where supported or derivable), inner and outer lip boundaries.

Do not fabricate model classes. If the pretrained model cannot distinguish a specific region, document that limitation and determine whether it can be derived from landmarks or geometric calculations. Keep left and right regions separate wherever possible.

## 7. Makeup Application Zones
Our application will eventually support the following 11 makeup stages. Design the derived application-zone system around them.
- **Stage 1, Primer:** full facial skin, forehead, left and right cheeks, nose, chin, jawline, under-eye skin. Support full-face and targeted application masks.
- **Stage 2, Color Corrector:** left and right under-eyes, inner eye corners, outer eye corners, left and right cheeks, nose and nose creases, chin, forehead, candidate discoloration regions. Discoloration detection is a separate future capability. Do not assume that basic segmentation can identify redness, pigmentation or dark circles reliably.
- **Stage 3, Foundation / Skin Tint:** forehead, left cheek, right cheek, nose, chin, jawline, general facial skin. Generate a general facial skin mask while excluding eyes, eyebrows, lips, nostrils and background.
- **Stage 4, Concealer:** left under-eye, right under-eye, inner under-eye corners, outer under-eye corners, forehead center, nose bridge, nose sides, chin center, individual blemish locations if detectable. Use facial landmarks and geometry to derive under-eye and brightening regions. Blemish detection should remain a separate future capability.
- **Stage 5, Setting Powder:** left under-eye, right under-eye, forehead center, nose bridge, nose sides, chin, around the mouth, general facial skin. Support localized and full-face application masks.
- **Stage 6, Bronzer & Contour:**
  - Contour zones: left cheek hollow, right cheek hollow, left and right temples, forehead perimeter, left and right sides of the nose, under the lower lip, jawline.
  - Bronzer zones: left and right upper cheeks, left and right temples, forehead perimeter, outer jawline.
  - Use facial geometry to derive these regions relative to the user's anatomy. Keep bronzer and contour as separate application layers.
- **Stage 7, Blush:** left cheek apple, right cheek apple, left upper cheek, right upper cheek, left outer cheek, right outer cheek. Support future placement styles: apple placement, lifted cheek placement, and draped blush extending toward the temples.
- **Stage 8, Highlighter:** left and right cheekbones, nose bridge, nose tip, Cupid's bow, left and right brow bones, inner eye corners, optional forehead center and chin center. Use landmarks and geometry to derive these regions.
- **Stage 9, Eyebrow & Eye Makeup:**
  - Eyebrows: left and right eyebrow, brow head, brow arch, brow tail, brow bone.
  - Eyes: left and right upper eyelids, lower eyelids, inner corners, outer corners, upper lash lines, lower lash lines and crease regions.
  - Derive more specific regions using landmarks where possible. Do not assume a particular eye shape or eyelid anatomy from segmentation alone.
- **Stage 10, Lips:** upper lip, lower lip, Cupid's bow, left and right lip corners, inner lip boundary, outer lip boundary, surrounding lip skin. Support separate lipstick, lip liner and gloss application masks. Preserve natural lip boundaries.
- **Stage 11, Setting Spray:** full facial skin, forehead, both cheeks, nose, chin, jawline. Support a full-face application mask.

## 8. Application Zone Architecture
Do not build 11 independent segmentation models. Instead, create a reusable pipeline:
1. Run the pretrained facial parsing model.
2. Extract anatomical segmentation masks.
3. Retrieve existing MediaPipe landmarks.
4. Perform geometric calculations.
5. Derive makeup application zones.
6. Validate derived zones against anatomical masks.
7. Return structured results.

Each derived application zone should have a unique identifier, a region name, a corresponding mask or polygon, coordinate dimensions, a source (segmentation, landmarks or derived geometry), and a confidence where meaningful. Keep the system extensible so additional makeup techniques can be introduced without changing the underlying segmentation model. Do not implement actual makeup rendering or product application yet.

## 9. Backend Requirements
Use the existing FastAPI architecture. Maintain separation between API routes, segmentation service, model loading and inference, mask processing, application-zone geometry, and Pydantic schemas.

Requirements:
- Load model weights once rather than on every request.
- Use appropriate preprocessing and postprocessing.
- Preserve image aspect ratios.
- Map segmentation predictions back to original image dimensions.
- Ensure masks align with MediaPipe landmark coordinates.
- Validate image inputs.
- Handle model initialization and inference failures.
- Avoid unnecessary external API dependencies.

Choose an efficient mask representation rather than returning enormous JSON arrays. Evaluate binary masks, PNG-encoded masks, compressed mask representations, and polygons for suitable derived regions. Document the chosen format and reasoning.

## 10. Frontend Requirements
Extend the existing facial analysis test interface. I want to:
1. upload a selfie;
2. run facial segmentation;
3. view the original image;
4. view the segmentation overlay;
5. toggle individual anatomical regions;
6. toggle individual makeup application zones;
7. assign distinguishable colors to regions;
8. adjust overlay opacity;
9. compare the original and segmented image;
10. view the existing facial landmarks alongside segmentation.

The interface should make it easy to identify inaccurate boundaries and misaligned regions. Keep the UI consistent with the existing application. Do not redesign unrelated components.

## 11. Model Evaluation and Future Fine-Tuning
Do not fine-tune or train a custom model during this implementation unless explicitly approved. Design the implementation so that the model can be replaced or fine-tuned later without rewriting the entire application.

Document pretrained model performance, supported anatomical classes, known segmentation limitations, inference latency, model loading time, CPU/GPU requirements, and memory consumption.

Recommend an evaluation approach using manually annotated ground-truth images, considering IoU, Dice coefficient, precision, recall and inference latency. Pay particular attention to small regions such as eyelids, lip boundaries and inner eye corners. If pretrained segmentation performs poorly, identify which regions would benefit most from fine-tuning.

Potential future datasets: CelebAMask-HQ, LaPa, and additional manually annotated makeup-specific facial images. Recognize that general face-parsing datasets do not necessarily contain labels for makeup-specific application zones. Do not claim model accuracy without actual evaluation against appropriate ground-truth annotations.

## 12. Edge Cases and Error Handling
Test and handle:
- unsupported image formats;
- corrupted images;
- no detectable face;
- multiple faces;
- partially visible faces;
- different face angles;
- different lighting conditions;
- different image resolutions;
- unusual aspect ratios;
- failed model inference;
- incorrect or incomplete masks;
- missing MediaPipe landmarks.

Do not silently return misleading segmentation results. Define clear behavior when a region cannot be identified reliably.

## 13. Security and Privacy
Treat facial images and derived facial geometry as sensitive personal data. Consider:
- upload size limits;
- image format validation;
- malformed image handling;
- resource exhaustion;
- temporary file handling;
- unnecessary image storage;
- sensitive data in logs;
- API request limits;
- dependency security.

Do not introduce unnecessary image persistence or external image-processing services.

## 14. Scope Restrictions
Do not implement:
- facial shape classification;
- eye shape classification;
- skin tone classification;
- skin condition diagnosis;
- blemish detection;
- discoloration classification;
- makeup reference analysis;
- makeup adaptation;
- generative image editing;
- makeup rendering;
- AR try-on;
- personalized tutorials;
- database integration;
- FacialProfile generation.

## 15. Acceptance Criteria
- [ ] A suitable pretrained face-parsing model is selected and documented.
- [ ] Model weights are loaded efficiently.
- [ ] Anatomical segmentation produces valid masks.
- [ ] Supported facial regions are correctly mapped.
- [ ] Left and right regions remain distinguishable where supported.
- [ ] Masks align with the original image dimensions.
- [ ] Segmentation integrates with existing MediaPipe landmarks.
- [ ] Derived application zones are implemented for the approved scope.
- [ ] Unsupported zones are documented rather than fabricated.
- [ ] Frontend overlays render correctly.
- [ ] Individual regions can be toggled.
- [ ] Overlay opacity can be adjusted.
- [ ] Invalid inputs and inference failures are handled.
- [ ] Automated tests pass.
- [ ] Existing landmark functionality has not regressed.
- [ ] Model limitations and future fine-tuning opportunities are documented.
- [ ] Security review has no unresolved critical or high-severity findings.
- [ ] Evaluator returns PASS.

## 16. Final Deliverables
1. Working pretrained facial segmentation pipeline.
2. Documented model selection and licensing.
3. Modular backend implementation.
4. Typed segmentation response schemas.
5. Makeup application-zone derivation architecture.
6. Integration with existing MediaPipe landmarks.
7. Frontend segmentation visualization.
8. Automated tests.
9. Performance and limitation documentation.
10. Future model fine-tuning recommendations.
11. Final reports from the existing five-agent workflow.
