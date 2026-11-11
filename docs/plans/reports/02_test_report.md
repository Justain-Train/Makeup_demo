# Test Report: Face Segmentation Phase 1 (iteration 1)

Agent: `tester`.

**Overall result: FAIL.** There are 3 defects: 1 high and 2 medium. The orchestrator reproduced BUG-1 independently.

## Summary

- A new test file, `backend/tests/test_phase1_independent.py`, holds 104 tests: 100 pass, and 4 are strict xfails that document real defects.
- The tests use real MediaPipe, the real BiSeNet ResNet18 weights, the portrait fixture, and in-memory synthetic variants.
- No production code was modified.
- Full suite: **164 passed, 4 xfailed**. The original 9 tests are unchanged and pass.
- Frontend typecheck and build pass. The UI was not clicked; the coordinate math was code-reviewed instead.

## Verified passing

| Area | Result |
|---|---|
| License warning (AC1) | On every status, and in the README and docs |
| Weights (AC2) | Hash pinned and matches. One `InferenceSession` across requests |
| Parser unavailable (AC2, AC9) | Missing file, directory, dangling symlink and wrong hash all give 503 on `/segmentation` and 200 on `/landmarks` |
| Class mapping (AC3) | Raw class order and left/right verified |
| Iris alignment (AC3) | Irises land on the correct-side class on the original, EXIF-rotated, ±15° roll, ½× and 5.4× variants |
| Alignment check (AC3) | Recomputed independently and matches |
| Stub geometry (AC4) | Within 1 px, including windows that extend outside the image |
| EXIF coordinates (AC4) | Coordinates agree; max error 0.76 px |
| Mask validity (AC5, AC7) | Labels are canonical or 255; masks are binary; PNG dimensions equal the bbox; face zones ⊆ skin; eye, brow and nose zones equal their class; no eye leakage into face zones |
| Zone contract (AC7, AC8) | Exactly the 18 Phase 1 zones; no Phase 2 names; 11 stages with the specified layers; no model class names in the response |
| `/landmarks` and `/topology` | Backwards compatible; `face_count` is additive; `/topology` unchanged |
| Robustness | 19 variants: ±15° and ±35° roll, cut-off faces, scales down to ⅛, aspect ratios from 1.3 to 1:3.6, grayscale, dark, overexposed, blank. No 500s, and warnings are honest |
| Large images and pixel cap | 25 MP accepted. 25 MP + 1 rejected. Pixel bombs rejected before decoding |
| Multiple faces | Rejected on both endpoints. A small background face is ignored with a warning. The ratio rule was verified on the real detector |
| Invalid input | 12 cases per endpoint return the documented codes. 413 for oversize. Content is checked, not just the declared type |
| Errors and load | Inference failures give a generic 500. A busy parser gives 503 with `Retry-After`. Concurrent requests never return 500 |
| MediaPipe calls | MediaPipe runs exactly once per request |

## Failures

### BUG-1 (High): a horizontally mirrored selfie collapses left and right

Criterion 6 and the "no silent misleading results" rule are violated.

**Reproduce:** mirror `tests/fixtures/portrait.jpg`, save it as PNG, and POST it to `/segmentation`.

| | Iris 468 | Iris 473 | Eye zones | Alignment |
|---|---|---|---|---|
| Original | EYE_R | EYE_L | both `ok` | 1.0 |
| Mirrored | **EYE_L** | EYE_L | `eye_l` is `ok` with a mask **115 px wide covering both eyes**; `eye_r` is `low_confidence` | **still passes at 0.833**, so no warning |

Most of the brow pixels also come back as BROW_L.

**Cause:**

- The parser's left/right output does not flip when the image is mirrored.
- The 0.6 alignment pass fraction tolerates one wrong iris.
- Eye validation (IoU ≥ 0.30 against the landmark polygon) accepts a mask that covers both eyes.

**Note:** MediaPipe labels landmarks by which side of the image they fall on, so for mirrored photos the "subject's own side" is ambiguous. Whichever convention is chosen, the two eyes must not share one class.

**Tests:** `test_ac6_mirrored_selfie_eyes_get_different_classes_xfail`, `test_ac6_mirrored_selfie_ok_eye_zone_does_not_cover_the_other_eye_xfail`.

### BUG-2 (Medium): unreadable weights crash startup instead of degrading to 503

Criterion 2 is not met for this case.

**Reproduce:** `chmod 000` the weights file, then start the app. Startup raises `PermissionError`.

**Cause:** `OnnxFaceParser.__init__` calls `read_bytes()` without catching `OSError`, and the lifespan only catches `FaceParserError`.

**Test:** `test_unreadable_weights_do_not_prevent_startup`.

### BUG-3 (Medium): a failed startup leaks a MediaPipe landmarker, and its garbage collection deadlocks

**Reproduce:** start once with valid weights, then with unreadable weights so startup raises, then call `gc.collect()`. It hangs every time: `FaceLandmarker.__del__` blocks on a serial-dispatcher future. Calling `close()` explicitly returns normally.

**Cause:** in `main.py`, the landmark service is created before the parser, but `service.close()` is only in a `finally` around `yield`.

**Test:** `test_failed_startup_does_not_leave_a_landmarker_that_hangs_on_gc` (runs in a subprocess with a 90 s timeout).

## Observations (not failures)

1. The 0.6 alignment pass fraction is the mechanism behind BUG-1.
2. The README line "left/right = subject's own side" holds only for unmirrored photos, and the mirrored-photo limitation is not documented.
3. MediaPipe returns `no_face_detected` for a 45% half-face crop and for 2.4:1 and 1:3.6 padded canvases. That is honest, but it shows limited recall.
4. The parser's eye mask stops 5.0–5.6 px short of the inner canthus (landmarks 133 and 155). This is a coarse-mask limitation.
5. Stage map details:
   - `lipstick` also lists `lip_upper`, `lip_lower` and `lip_inner_boundary`.
   - `chin` appears in setting powder; the requirement supports that, the plan table does not.
6. The `lip_full` polygon is the outer contour, while its mask covers only the lip bands.
7. A truncated 300-point landmark result produces a generic 500. That is acceptable, but not specific.
8. Headers above about 179 MP hit Pillow's own guard first, which returns 400 with the generic decode message.

## Coverage gaps

- No real glasses, beard, earring or necklace photos. Classes 9 and 15 are unverified.
- Only one real face. Everything else is synthetic.
- No ground truth, so accuracy is not measured, as intended.
- The UI was not clicked: left/right colours, the wipe compare and message display are unverified in a browser.
- Memory with two concurrent 25 MP requests was not tested. The Security Reviewer measured concurrency separately.
- The ResNet34 path is not tested.

## Performance (observed, no thresholds)

- **Fixture, `segment_face`:** median 145 ms; inference 84–89 ms.
- **Fixture over HTTP:** median about 155 ms; payload about 75 kB, of which the label-map PNG is about 5.6 kB.
- **24.5 MP image:** HTTP median 804 ms; zones step 370 ms; payload 209 kB; peak RSS 1.49 GB.
- **Load times:** MediaPipe 56–149 ms; ONNX parser 157–205 ms.

## Commands

```bash
cd backend
.venv/bin/pytest -q -rxX -p no:warnings          # 164 passed, 4 xfailed
.venv/bin/pytest tests/test_phase1_independent.py -q -p no:warnings --runxfail \
    -k "xfail or unreadable or leaves_a_landmarker or mirrored"   # shows the 4 bugs failing
cd ../frontend && npm run typecheck && npm run build
```
