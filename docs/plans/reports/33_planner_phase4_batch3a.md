# Planner report: Phase 4 Batch 3a, eye-area zones (2026-11-11)

Batch 3 was split. 3a covers 12 zones:
- `under_eye_inner/outer_l/r`
- `eye_corner_inner/outer_l/r`
- `lash_line_upper/lower_l/r`

3b covers `eyelid_upper_l/r`, `brow_head/arch/tail_l/r` and `brow_bone_l/r`, plus the eyebrow and brow-bone
highlighter decision. The eyelid crease is excluded.

How each zone is validated:
- Under-eye halves: the parent's geometric area is split at the eye centre. Validation uses the coverage rule.
- Eye corners: tolerance coverage within 0.05·ipd of the parser eye mask, minimum 0.6.
- Lash lines: a `geometric_only` stroke.

Glasses and occlusion:
- `_eye_area_guard` may only demote.
- A zone is demoted when GLASSES covers at least 20% of its geometric area, or when the parent `eye_x` is not `ok`.
- Nothing is inferred behind lenses.

`face_regions.py`: the iris demotion is widened to cover all eye-group zones on the failing side. It is
behaviour-identical for existing zones, because the eye group held only `eye` and `under_eye`.

Goldens were captured before any edit. Stages: the halves go in concealer and color_corrector; the corners and lash
lines go in the stage 9 eye layer. Highlighter stays empty, with a TODO for Batch 4: the inner eye corner.

Acceptance criteria:
1. The suite passes with nothing loosened.
2. Golden equality.
3. The under-eye halves partition `under_eye`.
4. Containment.
5. Sides at ±20° roll, and inner closer to the midline than outer.
6. Guarded negative tests.
7. Iris demotion covers the new zones.
8. Stages.
9. Protected files.
10. Fixture results, payload and timings.
