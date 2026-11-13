"""Named thresholds for geometric feature bands.

Every value is heuristic and uncalibrated: a rule of thumb, not fitted to data.
For each band: *_EDGES are ascending boundaries; *_MARGIN is the half-width around a
boundary inside which the label is "unknown"; *_SCALE is the distance from the nearest
boundary at which confidence reaches 1.0 (confidence = clamp(distance / scale, 0, 1)).
"""

# Smallest usable length in pixels; below this a width/height is treated as degenerate.
MIN_LENGTH_PX = 1e-3  # heuristic, uncalibrated: guards divisions only

# Policy for relative descriptors (eye spacing, eye tilt, brow arch): the middle ~60% of the
# population (p20-p80) is "average"/"neutral"/"soft_arched"; the outer 20% on each side gets the
# extreme label. Population-relative bands, not clinical definitions.

# Eye spacing = inner-canthal distance / mean eye width.
EYE_SPACING_EDGES = (1.19, 1.37)  # calibrated 2026-11-12 on docs/evaluation/geometry_calibration_2026-11-12.json (p20/p80 of 598 faces); population-relative, not validated against ground truth
EYE_SPACING_MARGIN = 0.02  # heuristic, uncalibrated: ~2% of eye width is landmark jitter
EYE_SPACING_SCALE = 0.10  # heuristic, uncalibrated: a further 0.10 from the edge is clear-cut
EYE_WIDTH_ASYMMETRY_SCALE = 0.30  # heuristic, uncalibrated: left/right width gap that zeroes confidence

# Eye tilt = canthal angle in degrees, outer corner higher than inner is positive.
EYE_TILT_EDGES_DEG = (-0.38, 6.0)  # calibrated 2026-11-12 on docs/evaluation/geometry_calibration_2026-11-12.json (p20/p80 of 598 faces); population-relative, not validated against ground truth
EYE_TILT_MARGIN_DEG = 1.0  # heuristic, uncalibrated: about one degree of landmark noise
EYE_TILT_SCALE_DEG = 4.0  # heuristic, uncalibrated: 4 degrees beyond the edge is clear-cut
EYE_TILT_DISAGREE_SCALE_DEG = 10.0  # heuristic, uncalibrated: left/right gap that zeroes confidence

# Eye shape = aperture height / width.
EYE_SHAPE_EDGES = (0.34,)  # heuristic, uncalibrated: taller openings read as round
EYE_SHAPE_MARGIN = 0.02  # heuristic, uncalibrated: eyelid landmark jitter
EYE_SHAPE_SCALE = 0.08  # heuristic, uncalibrated
EYE_SHAPE_DISAGREE_SCALE = 0.10  # heuristic, uncalibrated: left/right ratio gap that zeroes confidence

# Lip fullness = (upper + lower lip thickness) / IPD.
# IPD is used (not face height) because it comes from the irises, which are stable and
# independent of jaw landmark noise.
LIP_FULLNESS_EDGES = (0.22, 0.34)  # heuristic, uncalibrated: ~14 mm and ~21 mm over a ~63 mm IPD
LIP_FULLNESS_MARGIN = 0.02  # heuristic, uncalibrated
LIP_FULLNESS_SCALE = 0.08  # heuristic, uncalibrated

# Lip balance = upper thickness / lower thickness (ascending: bottom_heavy, balanced, top_heavy).
# heuristic, uncalibrated. Makeup-guide convention (orchestrator decision, 2026-11-12): "balanced" means upper and lower
# lips are roughly equal (ratio ~0.8-1.25). Most faces have a fuller lower lip (~1:1.5), so they read as bottom_heavy.
LIP_BALANCE_EDGES = (0.8, 1.25)
LIP_BALANCE_MARGIN = 0.03  # heuristic, uncalibrated
LIP_BALANCE_SCALE = 0.15  # heuristic, uncalibrated

# Lip width = mouth width / IPD.
LIP_WIDTH_EDGES = (0.72, 0.88)  # heuristic, uncalibrated: mouth is typically ~0.8 IPD
LIP_WIDTH_MARGIN = 0.02  # heuristic, uncalibrated
LIP_WIDTH_SCALE = 0.08  # heuristic, uncalibrated

# Brow arch = max perpendicular rise of the upper brow chain above the head-tail line / eye width.
BROW_ARCH_EDGES = (0.226, 0.287)  # calibrated 2026-11-12 on docs/evaluation/geometry_calibration_2026-11-12.json (p20/p80 of 598 faces); population-relative, not validated against ground truth
BROW_ARCH_MARGIN = 0.008  # calibrated 2026-11-12 on docs/evaluation/geometry_calibration_2026-11-12.json (p20/p80 of 598 faces); population-relative, not validated against ground truth
BROW_ARCH_SCALE = 0.03  # calibrated 2026-11-12 on docs/evaluation/geometry_calibration_2026-11-12.json (p20/p80 of 598 faces); population-relative, not validated against ground truth
BROW_DISAGREE_SCALE = 0.15  # heuristic, uncalibrated: left/right gap that zeroes confidence

# Brow thickness = mean paired upper-to-lower brow distance / eye width.
# UNUSED until a parsing-mask thickness measure exists: MediaPipe brow points are a template band
# (p10-p90 of thickness_ratio is only 0.179-0.232) and do not track real brow thickness.
BROW_THICKNESS_EDGES = (0.18, 0.30)  # heuristic, uncalibrated: slim vs bold brow band
BROW_THICKNESS_MARGIN = 0.015  # heuristic, uncalibrated
BROW_THICKNESS_SCALE = 0.06  # heuristic, uncalibrated

# Face shape: nearest prototype in a 4-vector (landmarks, MediaPipe 478):
# (length_ratio, forehead_ratio, jaw_ratio, chin_taper)
#   length_ratio   = (v152 - v10) / cheek width (234/454)
#   forehead_ratio = width(54/284) / cheek width
#   jaw_ratio      = width(172/397) / cheek width
#   chin_taper     = width(176/400) / width(172/397)
# Population mean and SD measured in landmark space: docs/evaluation/face_shape_calibration_2026-11-12.json,
# "frontal" block, n=334 frontal faces (yaw_asymmetry <= 0.10), EasyPortrait v2 test sample, seed 20261110.
# Heuristic; centred on measured landmark-space distribution, shape labels not validated against ground truth (no labelled data).
FACE_SHAPE_POP_MEAN = (1.2127, 0.8712, 0.8006, 0.4282)
FACE_SHAPE_DIM_SCALE = (0.0903, 0.0330, 0.0275, 0.0224)

# Prototypes as SD offsets from the population mean. Heuristic; shape labels not validated against ground truth.
FACE_SHAPE_PROTOTYPES_Z: dict[str, tuple[float, float, float, float]] = {
    "oval": (0.0, 0.0, 0.0, 0.0),  # balanced, population centre
    "round": (-1.2, 0.0, 0.6, 0.8),  # short, full lower face, broad chin
    "square": (-0.6, 0.4, 1.3, 1.0),  # wide jaw and chin
    "heart": (0.2, 1.2, -1.0, -1.0),  # wide forehead, narrow jaw, pointed chin
    "oblong": (1.4, 0.0, 0.0, 0.0),  # long
    "diamond": (0.3, -1.3, -0.8, -0.3),  # narrow forehead and jaw relative to cheekbones
}
FACE_SHAPE_PROTOTYPES: dict[str, tuple[float, float, float, float]] = {
    name: tuple(m + z * s for m, z, s in zip(FACE_SHAPE_POP_MEAN, zs, FACE_SHAPE_DIM_SCALE))  # type: ignore[misc]
    for name, zs in FACE_SHAPE_PROTOTYPES_Z.items()
}
# All below are in z units. Heuristic; shape labels not validated against ground truth (no labelled data).
FACE_SHAPE_MARGIN = 0.25  # heuristic: nearest-vs-runner-up gap below which the label is unknown
FACE_SHAPE_GAP_SCALE = 1.0  # heuristic: gap at which the gap factor reaches 1.0
FACE_SHAPE_DIST_SCALE = 4.0  # heuristic: distance at which the fit factor reaches 0
FACE_SHAPE_CONF_CAP = 0.8  # heuristic: face shape is never reported with more than this confidence
FACE_SHAPE_YAW_SOFT = 0.10  # heuristic: yaw asymmetry where confidence starts to fall
FACE_SHAPE_YAW_UNKNOWN = 0.30  # heuristic: yaw asymmetry above which the head is treated as turned

# Skin depth from ITA = atan2(L* - 50, b*) in degrees (very_light, light, intermediate, tan, brown, dark).
SKIN_ITA_EDGES = (-30.0, 10.0, 28.0, 41.0, 55.0)  # standard bands (Chardon 1991 / Del Bino 2013), ascending
SKIN_ITA_LABELS = ("dark", "brown", "tan", "intermediate", "light", "very_light")  # matches ascending edges
SKIN_ITA_MARGIN = 2.0  # heuristic, uncalibrated: ITA noise from lighting and median color
SKIN_ITA_SCALE = 10.0  # heuristic, uncalibrated: degrees beyond an edge that count as clear-cut
SKIN_DEPTH_CONF_CAP = 0.7  # heuristic: photo L* is exposure-dependent, unlike colorimeter ITA (see DECISIONS)

# Undertone from Lab hue angle (deg) of the median skin color.
# calibrated 2026-11-12, see CALIBRATED notes
# CALIBRATED on docs/evaluation/skin_tone_distribution_2026-11-12.json (598 EasyPortrait faces, photo-measured):
# skin hue p25/p50/p75/p90 = 37.9/46.6/53.2/59.6 deg. Edges split that range so each label is reachable;
# this assumes undertones are spread across the sample (no ground-truth labels exist).
SKIN_HUE_COOL_BELOW = 41.0  # heuristic, calibrated: pinker than this reads cool (~p30)
SKIN_HUE_WARM_ABOVE = 52.0  # heuristic, calibrated: yellower than this reads warm (~p70)
SKIN_HUE_OLIVE_ABOVE = 60.0  # heuristic, calibrated: greenish-yellow reads olive (~p90)
SKIN_OLIVE_LOW_CHROMA = 11.0  # heuristic, calibrated: muted chroma (~p10 of skin chroma 10.9)
SKIN_OLIVE_LOW_CHROMA_HUE = 52.0  # heuristic, calibrated: olive if chroma is low and hue is in the warm range
SKIN_UNDERTONE_MARGIN = 2.0  # heuristic, uncalibrated: degrees around a hue edge that stay unknown
SKIN_UNDERTONE_SCALE = 10.0  # heuristic, uncalibrated: degrees beyond an edge that count as clear-cut
SKIN_MIN_CHROMA = 4.0  # heuristic, uncalibrated: below this hue is meaningless
SKIN_UNDERTONE_CONF_CAP = 0.6  # heuristic, uncalibrated: undertone is the least reliable photo read

# Lighting checks: clipping on skin pixels and background color cast.
SKIN_CRUSHED_MAX = 12  # heuristic, uncalibrated; clipping-based so deep skin in good light is not penalised
SKIN_BLOWN_MIN = 245  # heuristic, uncalibrated; clipping-based so deep skin in good light is not penalised
SKIN_CLIP_CAUTION = 0.05  # heuristic, uncalibrated; clipping-based so deep skin in good light is not penalised
SKIN_CLIP_POOR = 0.20  # heuristic, uncalibrated; clipping-based so deep skin in good light is not penalised
SKIN_CLIP_CONF_FACTOR = 0.7  # heuristic, uncalibrated: confidence factor when clipping is above caution
# calibrated 2026-11-12, see CALIBRATED notes
# CALIBRATED: background chroma p50/p75/p90 = 10.2/14.6/20.9 (a coloured wall is not always a colour cast,
# so only the top quartile is flagged).
SKIN_CAST_CHROMA_CAUTION = 15.0  # heuristic, calibrated: ~p75 of background chroma
SKIN_CAST_CHROMA_POOR = 21.0  # heuristic, calibrated: ~p90 of background chroma
SKIN_CAST_CONF_FACTOR = 0.7  # heuristic, uncalibrated: confidence factor under caution
SKIN_GLASSES_CONF_FACTOR = 0.9  # heuristic, uncalibrated: glasses may tint nearby skin
SKIN_BG_DILATE_PX = 10  # heuristic, uncalibrated: background sampled outside the bbox padded by this
SKIN_BG_MIN_PX = 500  # heuristic, uncalibrated: fewer background pixels means no cast estimate
SKIN_BG_MAX_SIDE = 256  # heuristic: background is downsampled to about this side for the estimate
