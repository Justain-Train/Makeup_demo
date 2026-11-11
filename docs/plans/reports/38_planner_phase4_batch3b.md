# Planner report: Phase 4 Batch 3b, brow and eyelid zones (2026-11-11)

Zones (10): `eyelid_upper_l/r`, `brow_head_l/r`, `brow_arch_l/r`, `brow_tail_l/r`, `brow_bone_l/r`. There is no
eyelid crease and no lower eyelid.

Brow parts partition the landmark brow polygon. They do not partition the parser mask, because intersecting with the
parser mask would make the tolerance check vacuous. Each pixel is assigned by its projection along the head→tail axis
(landmarks 107→70): head below 0.35, arch from 0.35 to 0.70, tail from 0.70. These splits are untuned. Validation is
tolerance coverage within 0.05·ipd of the parser brow, with a minimum of 0.6.

Eyelid: the band above EYE_UPPER, minus the eye and the brow. Brow bone: the band under the brow tail and arch, minus
the brow, the eye and the eyelid geometry, so it is disjoint from the eyelid by construction. Both use the skin
coverage rule.

`_eye_area_guard` gets an optional `parent` argument, and the default is unchanged. The eyelid's parent is the eye.
The brow parts and the brow bone have the brow as parent. The eyelid is in group "eye", so it is iris-demoted. The
brow zones are not, because the iris says nothing about brows.

Stages:
- eyebrow: + head/arch/tail
- eye: + eyelid_upper
- highlighter = [brow_bone_r, brow_bone_l] (approved by the user; Batch 4 extends this)
