"""Feature rule engine (step 2). Pure; no logging; geometry-only features; the eyelid field is never read (self-report only).

Rules never touch component colour, finish or intensity; they only add adjustments and may change zones.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

from app.schemas.adapted_makeup_plan import Adjustment, Direction, RuleEvaluation, RuleFeature, RuleReason
from app.schemas.facial_profile import FeaturesData
from app.schemas.makeup_component import MakeupComponent
from app.services.application_zones import catalog
from app.services.makeup_mapping import ZoneOutsideLayerError, map_zones

T_DIRECTION = 0.3
T_ZONE = 0.4
T_LIP_SHAPE = 0.5
RULE11_MAX_RATIO = 0.55

EYELINER = frozenset({"eyeliner"})
EYESHADOW = frozenset({"eyeshadow"})
CONTOUR = frozenset({"contour_powder", "contour_cream"})
BLUSH = frozenset({"cream_blush", "powder_blush", "liquid_blush"})
BRONZER = frozenset({"bronzer"})
BROW = frozenset({"brow_pencil", "brow_powder", "brow_gel"})
LIP_LINER = frozenset({"lip_liner"})
LIPSTICK = frozenset({"lipstick"})


def _prefix(prefix: str) -> tuple[str, ...]:
    return tuple(z for z in catalog.ZONE_SPEC_BY_ID if z.startswith(prefix))


@dataclass(frozen=True)
class Change:
    """products: product types this change applies to. zones_mode: 'set' replaces, 'add' extends."""

    products: frozenset[str]
    zones: tuple[str, ...] = ()
    zones_mode: Literal["set", "add"] = "set"
    direction: Direction | None = None
    intensity_multiplier: float | None = None
    emphasis: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rule:
    rule_id: str
    reason: RuleReason
    feature: RuleFeature
    trigger_label: str
    threshold: float
    changes: tuple[Change, ...]
    min_ratio_key: str | None = None  # measurement that must be <= RULE11_MAX_RATIO

    @property
    def applies_to(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for c in self.changes:
            out |= c.products
        return out


RULES: tuple[Rule, ...] = (
    Rule("eye_tilt_down_lift", "lift_downturned_eyes", "eye_tilt", "downturned", T_DIRECTION,
         (Change(EYELINER | EYESHADOW, direction="upward_outward"),)),
    Rule("eye_tilt_up_balance", "balance_upturned_eyes", "eye_tilt", "upturned", T_DIRECTION,
         (Change(EYELINER, direction="horizontal"),)),
    Rule("eye_close_set_outer", "emphasize_outer_corner_close_set", "eye_spacing", "close", T_ZONE,
         (Change(EYESHADOW, zones=_prefix("eyelid_upper_") + _prefix("eye_corner_outer_")),)),
    Rule("eye_wide_set_inner", "emphasize_inner_corner_wide_set", "eye_spacing", "wide", T_ZONE,
         (Change(EYESHADOW, zones=_prefix("eye_corner_inner_"), zones_mode="add"),)),
    Rule("eye_round_elongate", "elongate_round_eyes", "eye_shape", "round", T_DIRECTION,
         (Change(EYELINER, direction="outward"),)),
    Rule("face_round_contour", "contour_round_face", "face_shape", "round", T_ZONE,
         (Change(CONTOUR, zones=_prefix("contour_hollow_"), direction="diagonal_up"),
          Change(BLUSH, zones=_prefix("cheek_upper_")))),
    Rule("face_oblong_horizontal", "horizontal_blush_oblong_face", "face_shape", "oblong", T_ZONE,
         (Change(BLUSH, zones=_prefix("cheek_apple_"), direction="horizontal"),
          Change(CONTOUR, intensity_multiplier=0.7))),
    Rule("face_square_soften", "soften_square_jaw", "face_shape", "square", T_ZONE,
         (Change(CONTOUR, zones=("jawline",), zones_mode="add"),)),
    Rule("face_heart_balance", "balance_heart_face", "face_shape", "heart", T_ZONE,
         (Change(BLUSH, zones=_prefix("cheek_apple_")),
          Change(BRONZER, zones=_prefix("temple_"), zones_mode="add"))),
    Rule("lip_thin_full", "add_lip_outer_boundary_thin_lips", "lip_fullness", "thin", T_LIP_SHAPE,
         (Change(LIP_LINER, zones=("lip_outer_boundary",), zones_mode="add"),)),
    Rule("lip_bottom_heavy_balance", "emphasize_cupid_bow_bottom_heavy", "lip_balance", "bottom_heavy", T_ZONE,
         (Change(LIPSTICK | LIP_LINER, emphasis=("cupid_bow", "lip_upper")),), min_ratio_key="upper_to_lower"),
    Rule("brow_straight_keep", "keep_straight_brows_horizontal", "brow_arch", "straight", T_DIRECTION,
         (Change(BROW, direction="horizontal"),)),
)


def _eval(rule: Rule, label: str, conf: float | None, outcome: str) -> RuleEvaluation:
    return RuleEvaluation(
        rule_id=rule.rule_id, feature=rule.feature, label=label[:32], confidence=conf,
        threshold=rule.threshold, outcome=outcome,  # type: ignore[arg-type]
    )


def _kinds(ch: Change) -> set[str]:
    kinds: set[str] = set()
    if ch.zones:
        kinds.add("zones")
    if ch.direction is not None:
        kinds.add("direction")
    if ch.intensity_multiplier is not None:
        kinds.add("intensity")
    if ch.emphasis:
        kinds.add("emphasis")
    return kinds


def evaluate(
    features: FeaturesData | None,
    component: MakeupComponent,
    kept_zones: Sequence[str],
    available_zone_ids: Collection[str],
) -> tuple[list[RuleEvaluation], list[Adjustment], tuple[str, ...] | None]:
    if features is None:
        return [], [], None
    evaluations: list[RuleEvaluation] = []
    adjustments: list[Adjustment] = []
    new_zones: tuple[str, ...] | None = None
    taken: set[str] = set()
    available = set(available_zone_ids)
    pt = component.product_type

    for rule in RULES:
        cls = getattr(features, rule.feature, None)
        label = "unknown" if cls is None else str(cls.label)
        conf = None if cls is None else float(cls.confidence)
        change = next((c for c in rule.changes if pt in c.products), None)
        if change is None:
            evaluations.append(_eval(rule, label, conf, "skipped_not_applicable"))
            continue
        if cls is None or label == "unknown" or conf is None or conf < rule.threshold:
            evaluations.append(_eval(rule, label, conf, "skipped_uncertain"))
            continue
        ratio_ok = True
        if rule.min_ratio_key is not None:
            ratio = cls.measurements.get(rule.min_ratio_key)
            if ratio is None:
                evaluations.append(_eval(rule, label, conf, "skipped_uncertain"))
                continue
            ratio_ok = ratio <= RULE11_MAX_RATIO
        if label != rule.trigger_label or not ratio_ok:
            evaluations.append(_eval(rule, label, conf, "skipped_not_triggered"))
            continue

        kinds = _kinds(change)
        if kinds & taken:
            evaluations.append(_eval(rule, label, conf, "skipped_conflict"))
            continue

        result_zones: tuple[str, ...] | None = None
        emphasis: list[str] = []
        if change.zones:
            target = tuple(change.zones) if change.zones_mode == "set" else tuple(dict.fromkeys([*kept_zones, *change.zones]))
            try:
                mapped = map_zones(pt, component.placement, target).zone_ids
            except ZoneOutsideLayerError:
                evaluations.append(_eval(rule, label, conf, "skipped_zone_unavailable"))
                continue
            filtered = tuple(z for z in mapped if z in available)
            gained = [z for z in filtered if z in change.zones]
            if not gained:
                evaluations.append(_eval(rule, label, conf, "skipped_zone_unavailable"))
                continue
            result_zones = filtered
        if change.emphasis:
            layer = set(map_zones(pt, None, ()).zone_ids)
            emphasis = [z for z in change.emphasis if z in layer and z in available]
            if not emphasis:
                evaluations.append(_eval(rule, label, conf, "skipped_zone_unavailable"))
                continue

        taken |= kinds
        evaluations.append(_eval(rule, label, conf, "applied"))
        if result_zones is not None:
            new_zones = result_zones
            adjustments.append(Adjustment(rule_id=rule.rule_id, reason=rule.reason, effect="zones"))
        if change.direction is not None:
            adjustments.append(
                Adjustment(rule_id=rule.rule_id, reason=rule.reason, effect="direction", direction=change.direction)
            )
        if change.intensity_multiplier is not None:
            adjustments.append(
                Adjustment(
                    rule_id=rule.rule_id, reason=rule.reason, effect="intensity",
                    intensity_multiplier=change.intensity_multiplier,
                )
            )
        if emphasis:
            adjustments.append(
                Adjustment(rule_id=rule.rule_id, reason=rule.reason, effect="emphasis", emphasis_zones=emphasis)
            )
    return evaluations, adjustments, new_zones
