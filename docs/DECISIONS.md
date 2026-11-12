# Decision log

This log records decisions, approvals, revised criteria and deferred items that cannot be recovered from the code,
tests or git history. Write one short entry when a phase, batch or work package closes. Day-to-day agent hand-off
reports are scratch files and are not kept in the repo (see `docs/ROADMAP.md`, "Important Rules").

Entries up to 2026-11-11 were condensed from the agent reports that used to live in `docs/plans/reports/` (01–43).

---

## Phase 3 — Face segmentation

### Segmentation deliverable (iterations 1–3, ending 2026-11-10): PASS

The first face-parsing pipeline, together with the initial 18 zones. Final suite: 191 tests.

**Iteration 1 bugs, all fixed:**
- **Mirrored selfies collapsed left and right.**
  - The model's paired classes (eye, brow) are now treated as side-agnostic.
  - The side is assigned from the landmark geometry, using the roll-aware midline (10–152) and `FaceFrame.u`.
  - Iris placement is a **hard** requirement of `alignment_check`.
  - Backstop: an eye or brow mask may not be `ok` if more than 20% of it overlaps the other side, or if it is wider than 1.5× the landmark polygon.
- **Unreadable weights crashed startup.** They now return 503 while `/landmarks` keeps working.
- **A failed startup leaked the landmarker.** `close()` now runs in a `finally`.
- **The compare divider stayed on screen** over error and no-face states. It is now guarded on both the page and the viewer.

**Convention decided.** Left and right follow MediaPipe: subject-right is the side holding landmarks 33/133/468. A mirrored photo cannot be detected, so it follows the image geometry. The README documents this. Do not try to auto-detect mirroring.

**Waived by the user.** The divider fix was not re-run in the browser; it was verified by code trace only. To check it by hand:
1. Segment a photo and tick Compare.
2. Load a blank image. No divider should appear.
3. Load a two-face image. No divider should appear.
4. Segment again. Compare should start unticked, at 0.5.

**Known limitations, accepted:**
- Mirrored photos follow the image geometry.
- Alignment fails at ±35° roll.
- Half-face recall is weak.
- There is a gap of about 5 px at the inner canthus.

### Phase 3 completion: EasyPortrait evaluation (WP1–WP5, 2026-11-10): PASS after 3 iterations

**WP4a.** Parser registration fixed (worst residual 0.002 px). Truncated landmark results now get a status instead of an error. `tests/test_phase3_independent.py` must stay in the suite: it catches registration mutants that the coder's own test missed.

**Pilot.**
- **Data:** EasyPortrait v2 test annotations, N=600 stratified, seed 20261110, 598 faces detected.
- **Agreement (IoU):** skin 0.866, lips 0.787, eye 0.717, brow 0.710.
- **Side swaps:** 0.
- **Detection failures:** 0.33%.
- **Files:** report and sample ids in `docs/evaluation/`.
- **Weighting:** the aggregates are over the stratified sample and are not re-weighted to the 6000-image split.

**Provisional targets (D2), against the pilot:**
- Skin IoU 0.90: **missed**. The cause is precision, not neck pixels.
- Lips 0.75, eye 0.65, brow 0.60: met.
- Boundary F ≥ 0.80 at 0.02·IOD: **missed** for lips (0.683), eye (0.751) and brow (0.653).
- Side swap ≤ 1%: met.
- 1–2 MP p95 latency ≤ 300 ms: **provisional**. It was measured on battery. The benchmark p95 is 176 ms at 2 MP.

**Decision rules:**
- **Rule 1 (acceptable):** not met.
- **Rule 2 (post-processing fix):** triggered. Small regions show a consistent best shift of +1 to +2 px downward. The cause is unexplained; it could be the annotation convention or the pipeline. Landmark polygons sit at (−1, −1), so the two signs disagree. Roll ±30° costs 0.12–0.16 IoU and ±45° costs 0.35–0.54.
- **Rule 3 (replace or fine-tune):** not triggered on the headline numbers. Caveat: eye IoU is 0.00 on the 48 glasses images, and 0.168 on the 93 alignment failures.
- **Rule 4 (detector):** not triggered. Detection loss is 4.5% of skin loss.
- **Rule 5:** the model must be replaced before any commercial release.

**Data handling decided:**
- The dataset is used for evaluation only. Only aggregates and the UUID list are committed. No training, no redistribution.
- The cache lives outside the repo, at `~/.cache/makeup_demo/`.
- EasyPortrait licence: a reworked CC BY-SA 4.0 variant. Attribution is in the report.

**To reproduce,** from `backend/` with the venv:
```
python -m evaluation.cli sample      # deterministic
python -m evaluation.cli fetch --threads 4 --budget-gb 3
python -m evaluation.cli run
python -m evaluation.cli report --run-dir ~/.cache/makeup_demo/easyportrait_runs/<date>
python -m evaluation.cli clean --yes # remove the cache
```

**Deferred findings:**
- **Security F1:** map zlib and struct errors to `FetchError`, and cancel pending futures.
- **Security F3:** add a wall-clock cap per request.
- **Tester F3:** decide whether 479 landmarks should be accepted.
- **Tester F7:** decide whether unknown prediction ids should raise.

### WP6-G glasses signal (2026-11-11): PASS

**Measurement:**
- All 48 images flagged with glasses have eye IoU 0.00. The parser labels the whole eye area as GLASSES.
- 0 false alarms and 0 missed glasses on that proxy.
- The glasses area is 3.1–8.9% on flagged images and at most 0.59% on the rest, so the 1% threshold needs no tuning.

**Built:** the existing warning, plus the hint `retake_suggested: ["glasses"]` and a generic UI prompt. Nothing is guessed behind the lenses. A prompt that depends on the user's makeup choice waits for Phases 6, 10 and 11.

**Revised criterion:** the committed-report byte comparison ignores the volatile "current tree" fingerprint line. This was a test-only change.

**Residual:** glasses the parser does not see at all cannot be counted, because EasyPortrait has no glasses label.

### Phase 3 closed for the prototype (user, 2026-11-11)

Skin-tone testing is done.

**Deferred by the user:**
- WP6 fixes: roll-aligned crop, y-offset, and the `present` floor.
- A provenance and consent record for `backend/tests/fixtures/portrait.jpg`. This is **required before any launch**.
- The fine-tuning decision. No custom model is trained without approval.
- A latency re-measure on AC power.

---

## Phase 4 — Makeup application zones

Every batch kept the earlier zones bit-identical to a `tests/golden_batch*.json` snapshot taken **before** that batch's edits.

### Batch 1 (2026-11-11): PASS after 2 fix loops, 372 tests

**Zones:** `chin_center`, `under_lower_lip`, `cupid_bow`, `lip_corner_l/r`, `mouth_surround`.

**Rejected once** on three points:
- Lip-corner confinement was not demonstrated after the metric change. A 0.3 ipd negative test was added.
- The corner coverage values were not reported.
- The "Phase 1/2 unchanged" claim rested on mtimes only. Bytecode-cache evidence was accepted.

**Note:** the left corner has a strict IoU of 0.168, because the parser lip is thin there. The user confirmed the placement visually.

### Batch 2 (2026-11-11): PASS after 2 fix loops, 392 tests

**Zones:** `forehead_center/perimeter`, `temple_l/r`, `jawline_outer_l/r`.

**Rejected once** on three points:
- `_forehead` equivalence was not proven. It is now proven against the pre-batch code, with goldens pinned.
- The jaw-band proxy was weak. The band is now checked directly.
- Disjointness evidence was missing from the summary. It is now tested for all 6 zones.

### Batch 3a (2026-11-11): PASS after 1 fix loop, 415 tests

**Zones:** under-eye inner and outer halves, inner and outer eye corners, upper and lower lash lines (12 zones).

**Code change:** in `face_regions.py`, only the iris-demotion predicate changed. It now selects eye-group zones by side suffix.

**Notes:**
- The 20% glasses threshold lets a lash line stay `ok` while glasses cover less than that share of it.
- A name-based check for model-only zones will not catch future `eye_`/`brow_` polygon zones.

### Batch 3b (2026-11-11): PASS after 1 fix loop, 439 tests

**Zones:** `eyelid_upper`, brow head, arch and tail, and `brow_bone` (10 zones).

**Approved by the user:** `highlighter = [brow_bone_r, brow_bone_l]`, the first highlighter entries.

**Notes:**
- `brow_bone_l` coverage on the fixture is near its threshold.
- The MEDIUM finding at `test_phase1_independent.py:861` is deferred. That test holds bbox literals measured on the real portrait.

### Batch 4 (2026-11-11): PASS, 463 tests

**Zones:** cheek apple, upper cheek, outer cheek, cheekbone and contour hollow, left and right (10 zones).

**Staging, approved by the user:**
- `blush` = apples, upper cheeks and outer cheeks.
- `highlighter` = brow bone + cheekbone.
- `contour` = jawline + hollows.
- `bronzer` = [], intentionally empty.

**Third fix loop, test-only, approved by the user:**
- The L/R mirror test runs on a yaw-symmetrized fixture, with shared bounds of 0.75–1.33.
- The real fixture is bounded at 0.65–1.5, because it is slightly yawed.

### Batch 5 (2026-11-11): PASS after 1 fix loop, 489 tests

**Zones:** `nose_bridge`, `nose_tip`, `nose_side_l/r`, `nostril_l/r`.

**Staging:** the sides join `contour`, and the bridge and tip join `highlighter`. The nostrils are unstaged exclusion areas.

**Bug fix, approved by the user.** `NOSE_HULL` was ordered so that the polygon crossed itself. It is now `[168,358,327,2,98,129]`. This changed `nose`, `cheek_l` and `mouth_surround` in synthetic scenarios only. Those 15 `golden_batch4/5` entries were re-captured with a dated note.

### Batch 6 (2026-11-11): ACCEPTED by the user, 512 tests

**Zones:** `lower_eyelid_r/l`.

**Geometry.** The zone is a band 0.07 ipd deep below the lower lid line. It is intersected with the geometric `under_eye_{k}` band, excludes the 1 px-dilated eye polygon, and is clipped to skin.

**Staging, approved by the user:** the zones join the `eye` layer after `eyelid_upper`.

**Fix loop 1:**
- At roll ±20°, 1 px fell outside `under_eye`. Fixed by intersecting with the under-eye band.
- The Batch 3a eye-group expected set was extended to include the new zones.
- **Criterion revised:** under glasses, the lower eyelid's reason must equal `eyelid_upper_{k}`'s, instead of containing the word "Glasses".

**Evaluator REJECT ×2, overridden by the user.** The only issue was that "Phase 1/2 unmodified" could not be proven from file contents, because there was no git repo and no pre-batch hashes. The evidence offered:
- No Phase 1/2 file is newer than the pre-edit golden snapshot.
- The edits to `test_phase1_independent.py` only add the two new ids.

**Phase 4 zone list complete: 70 zones.** The eyelid crease cannot be derived and is never added.

**Optional follow-ups:**
- A glasses-specific reason for both eyelids.
- A ratio test on a real photo. The current one uses synthetic labels.

---

## Standing pre-deployment blockers (open)

- **Licence:** the CelebAMask-HQ-derived parser weights are non-commercial only. Replace the model, and never fine-tune from it or pseudo-label with it.
- **Test fixture:** no provenance or consent record exists yet for the test portrait.
- **Hardening:**
  - Bound concurrent inference memory (F1).
  - Cap the request body before parsing (F2).
  - Add rate limiting and authentication.
- **Legal:** a review of biometric data handling.

---

## Phase 5 — FacialProfile construction

### Scope and design decisions (user, 2026-11-11)

- **Persistence: none.** A profile is built per request and returned. Nothing is stored on the server, and nothing is logged.
- **Endpoint:** a new `POST /api/facial-analysis/profile`. `/segmentation` stays unchanged.
- **Masks:** embedded by default, reusing the existing `png_label_map_v1` and `png_bbox_v1` encodings. `?include_masks=false` returns a lightweight profile without them.
- **Scope: assembly only.**
  - Phase 5 combines the existing outputs: landmarks and measurements, the parser label map, zones, stages, confidences, alignment, warnings, and the face frame (ipd, roll).
  - Feature classification is out of scope. That means face, eye and lip shape, and the user's own skin tone and undertone. No roadmap phase owns it yet; it must be scheduled before Phase 8 needs it. Phase 6's "undertone" refers to products.
- **Versioning:** `schema_version` uses `"MAJOR.MINOR"`, starting at `1.0`.
  - A MINOR bump adds optional fields only. A MAJOR bump removes a field or changes its meaning.
  - Each component carries an integer `component_version`, and a provider block with id, kind, weights sha256, `algorithm_version` and licence warning.
  - A new zones algorithm-version constant is introduced.
- **Replaceability:** each component sits behind a provider protocol (landmarks, parsing, zones). If one component fails, it is marked `unavailable` and the others stay valid.
- **Batches:**
  1. Schema and versioning.
  2. Builder, protocols and route.
  3. Frontend types and API client (no UI).

### Batch 1: schema and versioning (2026-11-11): PASS after 2 fix loops, 570 tests

**Files.** All new; no existing file was touched.
- `schemas/facial_profile.py`
- `services/facial_profile/{versions,compat}.py`
- `tests/test_facial_profile_schema.py`, with 58 tests

**Reuse.** Existing types are reused rather than duplicated: `Zone`, `Stage`, `AnatomicalLayer`, `AlignmentCheck` and the landmark/measurement models.

**Versions:**
- `PROFILE_SCHEMA_VERSION` is `"1.0"`.
- `ZONES_ALGORITHM_VERSION` is `"phase4-70zones-1"`.
- `FRAME_ALGORITHM_VERSION` is `"iris-frame-1"`.

**Invariants:**
- A component's `data` is None if and only if its status is `unavailable`.
- Status `ok` requires `components`.
- A status other than `ok` forbids `components` and `retake_suggested`.
- Each slot's `provider.kind` must match the slot. The `frame` slot uses kind `landmarks`.

**Choices made during the batch:**
- `ParsingData.label_map` is `None` when masks are excluded. The region list is always present.
- `midline` is stored in image pixels. Batch 2 converts it from landmark indices.

**Fix loop 1 (security hardening, all MEDIUM):**
- The version mappings are frozen with `MappingProxyType`.
- Version strings are hardened: ASCII only, at most 16 characters. `check()` always raises `UnsupportedProfileVersion`.
- Lists and strings are capped, for example 478 landmarks, 128 zones and 512 characters.
- `weights_sha256` must match a pattern.

**Fix loop 2:** test-only edge cases.

**Evaluator.** It rejected once, on stale "not a git repo" information. Git evidence then showed that no tracked backend file and no golden file had changed.

**Open items, for when profiles ever become client input:**
- `extra="ignore"` is the documented compatibility policy. Revisit it then.
- The reused `face_segmentation` types have no size caps of their own. Add them then.

### Batch 2: providers, builder, `POST /profile` (2026-11-11): PASS after 1 fix loop, 607 tests

**Files.**
- New: `services/facial_profile/{providers,builder}.py`, `api/facial_profile.py`, `tests/test_facial_profile_{api,verification}.py`.
- Changed: `main.py`, for router registration only.

**Design.** The builder does not wrap `segment_face`. That design would hard-wire every stage, and one failing stage would kill the whole response.
- Three Protocols (`LandmarkProvider`, `ParsingProvider`, `ZoneProvider`) each expose `info()`. Default adapters sit behind them.
- `build_profile` runs each stage in its own `try`.
- A failure marks that component `unavailable`, and its reason is the exception type name only. Components that depend on it are marked unavailable too, with the reason "depends on parsing" or "depends on frame".
- The builder imports the `face_regions` helpers, including the underscore ones, rather than copying them. Parity tests check that the profile's zones and regions equal `/segmentation`'s, on synthetic input and on the real fixture.

**Resolved by the orchestrator. The user can revisit these:**
- The landmark provider's `weights_sha256` is computed once from the configured model file. It is not pinned, and an unreadable file gives `None`.
- The zones provider carries the licence warning, because zones derive from the non-commercial parser labels.
- `include_masks=false` drops the masks and the label map, but keeps bbox, polygon and regions.
- The zones component is `degraded` only when alignment fails. Each zone's own status carries the rest.
- A busy parser returns 503 with `Retry-After`, the same as `/segmentation`.

**Fix loop 1.** When the frame was unavailable, `quality.warnings` dropped `face_partially_outside_image`, even though landmarks still reported `degraded`. This is fixed. The warning text was copied, so a drift-guard test checks that it still equals the `/segmentation` text.

**Low follow-ups, not done:**
- Build the landmark provider in `lifespan`. The current lazy `app.state` cache has a benign race.
- Hash the model file in chunks.
- With `include_masks=false`, zone masks are still encoded and then discarded. Fixing this would need a `face_regions` change.

### Batch 3: frontend types and API client (2026-11-12): ACCEPTED by the user, after 2 fix loops

**Files.** Frontend only. No backend file was touched; the backend suite stays at 607 passing.
- New: `frontend/types/facialProfile.ts` and `frontend/lib/facialProfile.ts`.
- Changed: `frontend/lib/api.ts`. The only change is `export` on `API_BASE`.

**Design.**
- The types mirror `schemas/facial_profile.py`. The shared frontend types (`Zone`, `Stage`, `RegionInfo`, `AnatomicalLayer`, `SegmentationWarning`, `Landmark`, `FaceMeasurements`) are imported, not redeclared.
- Defaulted Pydantic fields are typed as nullable, not optional, because they are always serialized.
- `FacialProfile` is a union on `status`. `components` is `null` unless the status is `ok`.
- `fetchFacialProfile(file, {includeMasks, signal})` sends `?include_masks=false` only when it is explicitly requested.
- It mirrors the `compat.py` version check: MAJOR 1, minimum MINOR 0. Unknown fields pass through.
- Typed errors:
  - `ProfileApiError`, with kind `network`, `http`, `unsupported_version` or `invalid_response`.
  - `ProfileBusyError` for a 503. `Retry-After` is read as integer seconds clamped to 0–60, defaulting to 2. An HTTP-date is ignored.
  - `AbortError` is rethrown as-is.
- Nothing is logged.

**Criteria revised by the orchestrator, and accepted by the user:**
- **Narrowing.** TypeScript cannot narrow through the nested `meta.status`, and the wire shape had to stay the same. Callers narrow with `isComponentAvailable(c)` or `c.data !== null`, which give `AvailableComponent<T>` or `UnavailableComponent`.
- **Error kinds.** 4xx and 500 share kind `http` and are told apart by `status`.

**Verification.** There is no frontend test runner. Checks used `tsc` plus scratch-only type tests and `node --experimental-strip-types` behavior scripts (56/56, plus 15/15 for Retry-After). Nothing was added to the repo. `detectLandmarks`, `fetchTopology` and `segmentFace` were checked by diff only.

**Fix loops:**
1. Added the `isComponentAvailable` guard and the named component variants. An over-long digit `Retry-After` used the default of 2; it now clamps to 60.
2. A `Retry-After` with leading zeros (for example `"007"`) became 60. Leading zeros are now stripped before the length check.

**Security.** PASS with no findings. LOW notes, not done:
- The server's `detail` string is shown at any length. Cap it if the UI displays it.
- Only `schema_version` is validated at runtime. Treat the rest as trusted server data, and never render it as HTML.

## Phase 5: CLOSED (2026-11-12)

All three batches were accepted. Still open:
- The Batch 2 low follow-ups listed above.
- Feature classification (face, eye and lip shape, and the user's own skin tone and undertone) still needs a roadmap phase before Phase 8.

---

## Phase 6 — Makeup taxonomy and application mapping

### Scope and design decisions (user, 2026-11-12)

- **Deliverable:** a versioned `MakeupComponent` schema, a controlled vocabulary, validation, and a mapping service that turns a component's placement into zone ids.
  - The mapping reads `STAGES` and the zone catalog as its only source of truth. It never copies them.
  - There is no new endpoint and no frontend work. Phases 7 and 8 use it from code.
- **Vocabulary:** one versioned registry that lists the allowed terms, with aliases, for each category: product types, finishes, techniques, placements and directions.
  - Adding a term is a data change plus a bump of the vocabulary version.
  - Combinations are built from terms and are never listed in advance, as the roadmap requires.
- **Color:**
  - sRGB hex is the canonical value. CIELAB is computed from it during validation, for perceptual comparison in Phase 8.
  - A shade name such as "peach" is optional.
  - Undertone (warm, cool, neutral or olive) is a separate field and describes the product.
- **Category 12 (special effects and decorative makeup):** it is part of the taxonomy.
  - Its components can target any existing zones, or none, with a free-form placement note.
  - It gets no new stage, and the Phase 4 catalog and golden snapshots stay unchanged.

**Batches:**
1. Vocabulary registry and versioning.
2. The `MakeupComponent` schema and color.
3. The placement-to-zones mapping service.

### Batch 1: vocabulary registry (2026-11-12): ACCEPTED by the user, with no fix loops, 695 tests

**Files.** All new, in `services/makeup_vocabulary/`: `__init__`, `versions`, `registry` and `resolve`, plus `tests/test_makeup_vocabulary.py` with 88 tests. No existing file was touched.

**Design.**
- 12 categories, plus product types, finishes, techniques, placements and directions, each with aliases.
- The data is frozen: `MappingProxyType`, tuples and frozen dataclasses. `VOCABULARY_VERSION` is `"1.0"`.
- Each product type stores the key of a `STAGES` stage and one of its layers, never zone ids. Only `special_effects` has none.
- `normalize` applies NFKC and casefold, removes apostrophes, collapses `-`, `_` and whitespace, and caps input at 64 characters.

**Choices made by the orchestrator, accepted by the user:**
- Moisturizer and SPF map to the `primer` stage and layer, because stage 1 already covers the whole skin-prep area.
- A placement key may equal a zone id (`nose_bridge`, `cupid_bow`). That gives natural names, and Batch 3 maps them. Terms of every other kind may never equal a zone id.
- There is no `crease` placement. The eyelid crease cannot be derived.
- Criterion 10 now reads "no tracked **code** file changed". The edits to these docs fall outside it.

**Security.** PASS. LOW notes, folded into Batch 2:
- `validate_registry()` is not run at import.
- The value of `kind` is echoed into an error message without a length bound.
- The length is not checked again after NFKC.
- The exact 64-character limit is not tested.

### Batch 2: MakeupComponent schema and color (2026-11-12): ACCEPTED by the user, after 2 fix loops, 809 tests

**Files.**
- New: `schemas/makeup_component.py`, `services/makeup_vocabulary/{color,compat}.py` and `tests/test_makeup_component.py`.
- Changed in Batch 1's code: `versions.py`, `resolve.py`, `__init__.py` and `test_makeup_vocabulary.py`. These hold the Batch 1 security LOW fixes and `COMPONENT_SCHEMA_VERSION = "1.0"`.

**Design.**
- Only `category` and `product_type` are required. Every other field is optional, because a reference image shows only part of a component. Vocabulary fields are mapped to their canonical terms through the Batch 1 resolver.
- Category consistency is enforced: the product type, and any technique or placement that names `categories`, must belong to the component's category.
- `color_hex` is normalized to `#RRGGBB`. `color_lab` (CIELAB, D65) is always recomputed from it and ignores any value the caller supplies. The color module uses only the standard library.
- `target_zones` lists existing zone ids. They are looked up in `ZONE_SPEC_BY_ID` at validation time, must be unique, and are capped at 128. `special_effects` may leave it empty.
- `extra="ignore"`, as in FacialProfile.

**Changes made by the orchestrator, accepted by the user:**
- `color_hex` is optional, because primer, SPF, clear gel and translucent powder have no color. The plan had made it required.
- The `target_zones` cap is 128. The plan had 16, but the primer stage alone uses 18 zones.

**Fix loops (security MEDIUM: Pydantic errors echoed raw input):**
1. Set `hide_input_in_errors=True`.
2. The test was corrected. Pydantic's `exc.errors()` always contains the input and config cannot suppress it. Callers must therefore use `str(exc)` or `exc.errors(include_input=False)`, which the docstring now states.

**Not verified.** That exactly 128 zones are accepted. The catalog has only 70 zones, and 129 is rejected.

**LOW notes for Phase 7:**
- Send all errors through one helper that drops the input.
- Sanitize free text (`shade_name`, `placement_note`) at the sink.
- `model_construct` and `model_copy` skip the `color_lab` derivation, so never use them on untrusted data.

### Batch 3: placement-to-zones mapping (2026-11-12): ACCEPTED by the user, after 1 fix loop, 901 tests

**Files.** All new, in `services/makeup_mapping/`: `__init__`, `placement_zones` and `mapper`, plus `tests/test_makeup_mapping.py` with 92 tests.

**Design.**
- `map_zones(product_type, placement, target_zones)` and `map_component(c)` return a frozen `ZoneMapping`. It holds `zone_ids` (in layer order), the stage and layer, `source` (layer, placement, explicit or none) and a `reason`.
- The product's layer from `STAGES` gives the candidate zones. `STAGES` is read on every call, never copied.
- A placement narrows those candidates through `PLACEMENT_ZONES`, and the result is the intersection with the layer. A non-special-effects product therefore never leaves its layer.
- An empty intersection gives a typed `none` result and does not raise.
- Explicit `target_zones` take priority, but they must lie inside the layer. Otherwise `ZoneOutsideLayerError` is raised.
- `special_effects` can use any existing zone, or none.
- There is no `side` option; callers filter on `ZONE_SPEC_BY_ID[id].side`.

**Fix loop 1:**
- A stage or layer missing from `STAGES` now raises `CatalogDriftError`. Before, it silently returned no zones.
- `target_zones` must be a list or tuple.
- The sweep is now parametrized over the 69 compatible placement and product pairs.
- A snapshot test pins the 22 pairs that map to `none`. The planner's test-count gate of ≥880 was dropped as a criterion.

**Security LOW notes for Phase 7:**
- Treat `CatalogDriftError` as a server fault, not bad input.
- The `Sequence[str]` annotation is looser than the runtime check.

### Batch 4: catalog gap fixes, scope (user, 2026-11-12)

Batch 3 exposed gaps in the Phase 4 catalog. The user approved editing `catalog.py` `STAGES` and updating the 4 tests that pin the highlighter layer exactly.

**Highlighter layer.** Add these zones:
- `forehead_center`, which the user pointed out
- `chin_center`
- `eye_corner_inner_r/l`
- `cupid_bow`

**Bronzer layer.** It was empty since Phase 4. Fill it with:
- `temple_r/l`
- `forehead_perimeter`
- `cheek_outer_r/l`
- `jawline_outer_r/l`

These follow the classic "3" shape. Contour stays a separate layer.

**Vocabulary.** Add the placements the new zones need, and bump the vocabulary MINOR version.

**Not changed.** The vocabulary mismatches stay as they are and keep mapping to `none`:
- `brow_bone` for brow products
- `under_eye` for eye products
- `outer_lip_edge` for lipstick

### Batch 4: catalog gap fixes (2026-11-12): ACCEPTED by the user, with no fix loops, 930 tests

**Files.**
- Source: `catalog.py` (only `STAGES`), plus the vocabulary `versions.py` and `registry.py` and `placement_zones.py`.
- Tests: 6 earlier test files that pinned the old layers had their pins updated. These are `test_application_zones`, `test_phase1_independent`, `test_phase4_batch2/4/5_verification` and `test_face_segmentation_api`. The two makeup test files were also updated. The orchestrator read every pin diff: only the values changed, and no assertion was dropped.

**Result.**
- The highlighter layer is now the original 6 zones plus `forehead_center`, `chin_center`, `eye_corner_inner_r/l` and `cupid_bow`.
- The bronzer layer is `temple_r/l`, `forehead_perimeter`, `cheek_outer_r/l` and `jawline_outer_r/l`.
- The vocabulary moves to 1.1, with the new placements `forehead`, `chin`, `temples`, `hairline`, `outer_cheek` and `jawline`.
  - The alias "forehead center" was left out, because it normalizes to a zone id.
  - `jawline` also maps to the plain `jawline` zone, so contour still matches.

**What did not change.**
- The golden snapshots. They do not encode `STAGES`. The `/segmentation` and `/profile` `stages` outputs change in layer contents only, and no schema cap is affected.
- The `none` snapshot still holds 22 pairs:
  - 9 were fixed: the bronzer placements, plus inner-corner and cupid's-bow highlight.
  - 9 were added: contour on the bronzer placements, which is expected because contour stays its own layer.
  - No pair returns `layer_has_no_zones` any more.

**Close-out.** The stale `catalog.py` docstring about the lower eyelid was fixed. `docs/face_segmentation.md` no longer describes empty layers.

## Phase 6: CLOSED (2026-11-12)

All 4 batches were accepted, and the suite went from 607 to 930 tests.

**Carried into Phase 7:**
- Build components only through `MakeupComponent` validation.
- Send errors through one helper that uses `include_input=False`.
- Sanitize `shade_name` and `placement_note` at the sink.
- Map `CatalogDriftError` to a server fault.
- Decide how to handle reference zones that fall outside a product's layer, such as blush draped on the temples. Today they raise `ZoneOutsideLayerError`.

**Vocabulary mismatches left as they are.** These map to `none`:
- `brow_bone` for brow products
- `under_eye` for eye products
- `outer_lip_edge` for lipstick

Feature classification (face, eye and lip shape, the user's own skin tone and undertone) still needs a roadmap phase before Phase 8.

---

## Phase 7 — Makeup reference analysis (MakeupBlueprint)

### Scope and design decisions (user, 2026-11-12)

**CV approach.** Reference images go through our own face pipeline: landmarks, parsing and zones, via the FacialProfile builder.
- For each zone, we sample the color and compare it with the bare-skin regions of the same face, to separate makeup from skin.
- Colors, positions and zones come from this CV step only. No new CV model is added.

**VLM.** We use Claude through the Anthropic API.
- It contributes only the semantic attributes: technique, finish, direction and overall style.
- It never decides exact colors or coordinates, as the roadmap requires.
- It sits behind a provider protocol and has a deterministic stub.

**Privacy.** Reference photos get the same treatment as selfies.
- They are not persisted or logged, and uploads go through the same validation.
- Images are sent to Anthropic only when the cloud VLM is enabled in config. That setting is **off by default**. With it off, the blueprint is built from CV alone.

**Interface.**
- A versioned `MakeupBlueprint` schema: a list of `MakeupComponent`s, plus style and confidences.
- An analysis service.
- `POST /api/makeup/reference-analysis`.
- Frontend types come later.

**SDK defaults from the claude-api reference, used unless the user overrides them:**
- Model `claude-opus-5-5`, through the official `anthropic` Python SDK.
- Images are sent as base64 `image` content blocks.
- Output is structured via `client.messages.parse(output_format=<Pydantic model>)`.
- `output_config.effort` is set explicitly, because it defaults to `medium` on Opus 5.5.
- Thinking is adaptive (it cannot be disabled on Opus 5.5).
- `stop_reason == "refusal"` is handled, and server-side `fallbacks` are opted in.
- Typed SDK errors are caught most-specific first.
- Tests never call the real API.

### Step 1 — CV reference analyzer (accepted by the evaluator, 2026-11-12)

- `services/makeup_blueprint/cv_analysis.py`: `ReferenceAnalyzer` protocol and `CvReferenceAnalyzer` (provider id `cv_zone_color`, algorithm 1.0). It is pure, with no logging and no module state.
- Products analyzed: lipstick, powder blush, eyeshadow, highlighter and contour. Each zone's trimmed-median color is compared with that face's bare `face_skin`, excluding the target, under-eye and forehead-perimeter zones.
- **Not analyzed by CV:** brows, eyeliner and mascara. Natural hair and lashes cannot be told apart from product.
- Eyeshadow is skipped when glasses are detected.
- CV populates category, product type, color, undertone, intensity, zones and confidence, all with source `cv`. Finish, placement, technique and direction are left for the VLM.
- Tests: 36 new, full suite 1041.
- **Open:**
  - ~~Undertone cool range~~ **Resolved (user, 2026-11-12):** cool starts at blue. Olive covers yellow-green and green up to 200°. Cool is ≥200° or <25°.
  - ~~Mask PNG decoded before its size is checked (MEDIUM)~~ **Fixed in step 3.** The base64 length cap and the PNG signature, IHDR and dimension check now run before `decode_png`.
  - ~~Fallback color not asserted; non-ok statuses beyond `no_face_detected` untested~~ **Fixed in step 3.**

### Step 2 — VLM provider (accepted by the evaluator, 2026-11-12)

- SDK: `anthropic==2.8.0`, pinned in `backend/requirements.txt`.
- `client.beta.messages.parse` in 2.8.0 accepts `fallbacks`, `output_format`, `thinking` and `betas`. So structured output and server-side refusal fallbacks (`server-side-fallback-2026-07-01`, `fallbacks="default"`) are used together on the beta endpoint.
- Files: `services/makeup_blueprint/vlm.py`, `vlm_prompt.py`, `claude_vlm.py`; five `makeup_vlm_*` settings in `app/config.py`.
  - `VlmProvider` protocol, `StubVlmProvider` and `ClaudeVlmProvider`.
  - `build_vlm_provider` returns the stub unless `makeup_vlm_enabled` is true. With the stub, `anthropic` is never imported.
  - There is no API key setting; the SDK resolves the key from the environment.
- Request: model `claude-opus-5-5`, adaptive thinking, effort `medium`, `max_tokens` 16000, `output_format=VlmResponse`, `fallbacks="default"`, and a static system prompt.
  - The image is re-encoded from the array as JPEG, with the long side at most 1568 px.
- Output: `VlmResponse` is enums only (`extra="forbid"`, bounded lists, no free text) and is re-validated into our schemas.
  - Claude fills only finish, technique, placement and direction on CV components (by index), and only where they are unset.
  - Claude can add style labels.
  - Claude may propose up to 4 brow, eyeliner or mascara components, with no color or zones, all with source `vlm`.
  - A refusal gives `vlm_refused`. Every SDK error, `max_tokens` stop or invalid output gives `vlm_failed` with a fixed message. The provider never raises.
- `httpx2` is the HTTP library that `anthropic` 2.8.0 depends on; it is not a typosquat.
- Tests: 24 new, all with a fake client and no network; full suite 1065.
- **Follow-ups:**
  - ~~Explicit `max_retries`~~, ~~stub fallback on construction failure~~, ~~list-bound overflow rejected whole~~ and ~~cap test~~: all fixed in step 3.
  - Still open: `fallbacks="default"` may serve a request from another Anthropic model while `info().model` reports the configured one (LOW).
- **Credentials (user, 2026-11-12):** develop against the `ant auth login` profile. `anthropic.Anthropic()` resolves it with zero arguments, so no API key is set anywhere in the app or in `.env`. `ANTHROPIC_API_KEY` exported in the shell remains a fallback. The `MAKEUP_VLM_*` settings are documented in `backend/.env.example`.

### Step 3 — Reference analysis service (accepted by the evaluator, 2026-11-12)

- `services/makeup_blueprint/service.py` holds `analyze_reference(image, *, deps, clock)`, which returns a `MakeupBlueprint`. It also holds `ReferenceDeps` and `build_reference_deps(settings, landmark, parsing, zone)`; step 4 reuses the profile route's provider dependencies for these. It has no FastAPI import and no globals.
- **Flow:**
  - Build the FacialProfile with masks.
  - A non-ok status gives an empty blueprint with that status.
  - Otherwise run CV, then the VLM with the CV components, then `attach_annotations`.
  - Warnings are deduped by code, with a cap of 32.
  - Providers are `[cv, vlm]`.
  - `timings_ms` records profile, cv, vlm and total.
  - The profile, masks and landmarks never reach the output.
- **`overall_confidence`:** the mean of the component confidences, falling back to the CV confidence, else None. It is multiplied by 0.9 on `vlm_failed`, `vlm_refused` or `glasses_present`, clamped and rounded to 4 decimal places.
- **Errors:**
  - Any failure in the profile, CV or blueprint build becomes `ReferenceAnalysisError(stage)`, with a fixed message and raised `from None`.
  - A missing parser gives `ParserUnavailableError`.
  - `FaceParserBusyError` propagates.
  - Step 4 maps the first to 500 and the last two to 503.
  - Any VLM failure degrades to CV-only with `vlm_failed`.
- **Step 2 follow-ups folded in:**
  - `makeup_vlm_max_retries` (default 1, 0–3).
  - A client-construction or SDK-import failure gives `FailedVlmProvider`.
  - In `anthropic` 2.8.0, missing credentials surface at request time as `TypeError`, which maps to `vlm_failed`.
  - The SDK gets the lenient `VlmResponseRaw`; lists are truncated to 4/32/8, then strictly validated.
- Tests: full suite 1110. Fix loop 1 wrapped generic CV exceptions and moved the SDK import inside the try.
- ~~Not asserted yet: confidence clamping and rounding, and the 32-warning cap~~ **Covered in step 4.** The cap is tested by patching `MAX_WARNINGS`, because only 7 distinct warning codes exist.

### Step 4 — `POST /api/makeup/reference-analysis` (accepted by the evaluator, 2026-11-12)

- `app/api/makeup_reference.py` holds the route; `main.py` gets only the `include_router`.
  - The route is a sync `def`. It runs Phase 1 `read_validated_image`, then `analyze_reference`, and returns `MakeupBlueprint`.
  - It reuses the profile route's landmark, parsing and zone dependencies.
- **VLM provider:** built once and cached on `app.state.makeup_vlm_provider`, keyed by `Settings` identity (`get_settings` is `lru_cache`d).
- **Error map** (all fixed messages, raised `from None`; logs carry only the type name and stage):
  - Upload errors are unchanged from the profile route: 400, 413, 415.
  - Busy parser: 503 with `Retry-After: 2`.
  - Missing parser: 503 with the profile route's exact text.
  - `ReferenceAnalysisError` or any other exception: 500 "Reference analysis failed.". The stage is never exposed.
  - `no_face_detected` and `multiple_faces_detected` return 200 with that status.
- **`Cache-Control: no-store`** on success. This deliberately differs from the profile route, because the payload derives from a face photo.
- Tests: 16 API tests (including a real-model portrait integration run with sockets blocked) and 3 service gap tests. Full suite 1129.
- **Open (LOW):**
  - `no-store` is not on error responses; their bodies hold no user data.
  - Two concurrent first requests may each build a provider (harmless).
  - **No auth or rate limit on the route. Both must be added before `MAKEUP_VLM_ENABLED=true` is used outside local dev**, because each request can trigger a paid Claude call with a user's photo.
