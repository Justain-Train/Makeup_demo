"""Robustness suite (WP3): perturb image and GT together, re-score, plus GT-free consistency scores.

Perturbations (positive roll = counter-clockwise): mirror (GT sides swapped), roll +-15/30/45 (canvas expanded;
rotation fill is NOT scored), scale 0.5, JPEG q30, Gaussian blur, gamma 0.5/2.0, grayscale, and a vertical flip
sanity check. Per-image rows go outside the repo; the report holds aggregates only.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from app.services.image_loading import load_rgb_image
from app.services.segmentation.regions import NOT_PARSED, Region
from evaluation import labels as lb
from evaluation import metrics as mt
from evaluation.dataset import SampleEntry, annotation_path, decode_annotation, image_path, refuse_inside_repo
from evaluation.predict import predict_full
from evaluation.report import BOOT_N, BOOT_SEED

logger = logging.getLogger("evaluation")

SUBSET_SEED = 20261110
SUBSET_N = 100
ROLLS = (15, -15, 30, -30, 45, -45)
PERTURBATIONS = (
    "mirror", *(f"roll{a:+d}" for a in ROLLS), "scale0.5", "jpeg30", "blur", "gamma0.5", "gamma2.0", "gray", "vflip",
)
HEADLINE_GROUPS = ("skin", "lips", "eye", "brow")
EQUIV_GROUPS: dict[str, tuple[Region, ...]] = {
    "skin": (Region.SKIN, Region.NOSE), "lips": (Region.LIP_UPPER, Region.LIP_LOWER),
    "eye": (Region.EYE_R, Region.EYE_L), "brow": (Region.BROW_R, Region.BROW_L),
    "eye_R": (Region.EYE_R,), "eye_L": (Region.EYE_L,), "brow_R": (Region.BROW_R,), "brow_L": (Region.BROW_L,),
}
GT_SIDE_SWAP = {lb.EP_LEFT_BROW: lb.EP_RIGHT_BROW, lb.EP_RIGHT_BROW: lb.EP_LEFT_BROW,
                lb.EP_LEFT_EYE: lb.EP_RIGHT_EYE, lb.EP_RIGHT_EYE: lb.EP_LEFT_EYE}
PRED_SIDE_SWAP = {Region.EYE_L: Region.EYE_R, Region.EYE_R: Region.EYE_L, Region.BROW_L: Region.BROW_R,
                  Region.BROW_R: Region.BROW_L, Region.EAR_L: Region.EAR_R, Region.EAR_R: Region.EAR_L}


# --- sample subset -------------------------------------------------------------------------------

def pick_subset(entries: list[SampleEntry], n: int = SUBSET_N, seed: int = SUBSET_SEED, split: str = "dev") -> list[SampleEntry]:
    pool = sorted((e for e in entries if e.split == split), key=lambda e: e.image_id)
    if n > len(pool):
        raise ValueError("subset larger than the split")
    rng = np.random.default_rng(seed)
    return [pool[i] for i in sorted(rng.choice(len(pool), size=n, replace=False))]


# --- transforms (image and GT together) ---------------------------------------------------------------

def swap_labels(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    lut = np.arange(256, dtype=np.uint8)
    for a, b in mapping.items():
        lut[int(a)] = int(b)
    return lut[arr]


def rotate_expand(arr: np.ndarray, angle: float, interp: int, fill: int | tuple[int, ...] = 0) -> np.ndarray:
    """Rotate about the centre by `angle` degrees (positive = counter-clockwise) on an expanded canvas."""
    h, w = arr.shape[:2]
    rad = math.radians(angle)
    nw = int(math.ceil(abs(w * math.cos(rad)) + abs(h * math.sin(rad))))
    nh = int(math.ceil(abs(w * math.sin(rad)) + abs(h * math.cos(rad))))
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    m[0, 2] += (nw - w) / 2.0
    m[1, 2] += (nh - h) / 2.0
    return cv2.warpAffine(arr, m, (nw, nh), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def apply_perturbation(name: str, rgb: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(image, GT, valid). `valid` is None or a bool mask; False pixels (rotation fill) are not scored."""
    if name == "mirror":
        return rgb[:, ::-1].copy(), swap_labels(gt[:, ::-1], GT_SIDE_SWAP).copy(), None
    if name == "vflip":  # no left/right change: x is untouched
        return rgb[::-1].copy(), gt[::-1].copy(), None
    if name.startswith("roll"):
        angle = float(name[4:])
        img = rotate_expand(rgb, angle, cv2.INTER_LINEAR, 0)
        g = rotate_expand(gt, angle, cv2.INTER_NEAREST, 0)
        valid = rotate_expand(np.ones(gt.shape, dtype=np.uint8), angle, cv2.INTER_NEAREST, 0).astype(bool)
        return img, g, valid
    if name == "scale0.5":
        h, w = gt.shape
        size = (max(1, w // 2), max(1, h // 2))
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST), None
    if name == "jpeg30":
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=30)
        return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8), gt, None
    if name == "blur":
        return cv2.GaussianBlur(rgb, (0, 0), 3.0), gt, None
    if name.startswith("gamma"):
        g = float(name[5:])
        lut = (255.0 * (np.arange(256) / 255.0) ** g).round().astype(np.uint8)
        return lut[rgb], gt, None
    if name == "gray":
        y = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).round().astype(np.uint8)
        return np.repeat(y[:, :, None], 3, axis=2), gt, None
    raise ValueError(f"unknown perturbation {name}")


def transform_prediction(name: str, pred: np.ndarray) -> np.ndarray | None:
    """Apply a geometric perturbation to a canonical label map (sides swapped for the mirror); 255 = unparsed/fill."""
    if name == "mirror":
        return swap_labels(pred[:, ::-1], {int(a): int(b) for a, b in PRED_SIDE_SWAP.items()}).copy()
    if name == "vflip":
        return pred[::-1].copy()
    if name.startswith("roll"):
        return rotate_expand(pred, float(name[4:]), cv2.INTER_NEAREST, NOT_PARSED)
    return None


# --- GT-free consistency ---------------------------------------------------------------------------------

def equivariance_counts(pred_perturbed: np.ndarray, pred_transformed: np.ndarray, valid: np.ndarray | None = None) -> dict[str, list[int]]:
    """tp/fp/fn between two label maps over pixels parsed in both (and valid): [tp, fp, fn] per class group."""
    both = (pred_perturbed != NOT_PARSED) & (pred_transformed != NOT_PARSED)
    if valid is not None:
        both &= valid
    out = {}
    for name, regions in EQUIV_GROUPS.items():
        a = lb.pred_mask(pred_perturbed, regions) & both
        b = lb.pred_mask(pred_transformed, regions) & both
        c = mt.counts_from_masks(b, a)  # reference = transformed original prediction
        out[name] = [c.tp, c.fp, c.fn]
    return out


# --- the run -----------------------------------------------------------------------------------------------

def _shifts(gt: np.ndarray, pred: np.ndarray) -> dict[str, list[float] | None]:
    out: dict[str, list[float] | None] = {}
    for name in ("eye", "lips", "brow"):
        comp = lb.COMPARISON_BY_NAME[name]
        g, p = lb.ep_mask(gt, comp.ep_ids), lb.pred_mask(pred, comp.canonical)
        s = mt.best_shift(g, p) if g.sum() >= mt.MIN_PIXELS and p.sum() >= mt.MIN_PIXELS else None
        out[name] = list(s) if s else None
    return out


def evaluate_perturbation(
    name: str, rgb: np.ndarray, gt: np.ndarray, services: tuple[Any, Any], base_pred: np.ndarray | None
) -> tuple[dict[str, Any], np.ndarray | None]:
    """(row, prediction). `base_pred` is the unperturbed prediction, used for the equivariance counts."""
    if name == "baseline":
        img, g, valid = rgb, gt, None
    else:
        img, g, valid = apply_perturbation(name, rgb, gt)
    outcome, pred, _ = predict_full(img, *services)
    row: dict[str, Any] = {"outcome": outcome}
    if pred is None:
        conf = mt.not_predicted_confusion(g if valid is None else g[valid])
        row["conf"] = conf.tolist()
        return row, None
    row["conf"] = mt.confusion_matrix(g, pred, valid=valid).tolist()
    row["shift"] = _shifts(g, pred if valid is None else np.where(valid, pred, NOT_PARSED).astype(np.uint8))
    if base_pred is not None and (name in ("mirror", "vflip") or name.startswith("roll")):
        t = transform_prediction(name, base_pred)
        if t is not None and t.shape == pred.shape:
            row["equiv"] = equivariance_counts(pred, t, valid)
    return row, pred


def run_suite(
    entries: list[SampleEntry], cache: Path, out_dir: Path, services: tuple[Any, Any],
    perturbations: tuple[str, ...] = PERTURBATIONS, progress: Callable[[int, int], None] | None = None,
) -> Path:
    out_dir = refuse_inside_repo(out_dir, "output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    path = out_dir / "perturb_rows.jsonl"
    t0 = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for k, e in enumerate(entries, 1):
            rgb = load_rgb_image(image_path(cache, e.image_id).read_bytes())
            gt = decode_annotation(annotation_path(cache, e.image_id).read_bytes(), rgb.shape[:2])
            base, base_pred = evaluate_perturbation("baseline", rgb, gt, services, None)
            rows = {"baseline": base}
            for name in perturbations:
                rows[name] = evaluate_perturbation(name, rgb, gt, services, base_pred)[0]
            f.write(json.dumps({"id": e.image_id, "rows": rows}, separators=(",", ":")) + "\n")
            f.flush()
            if progress:
                progress(k, len(entries))
            logger.info("perturb %d/%d %.0fs", k, len(entries), time.perf_counter() - t0)
    return path


# --- aggregation -----------------------------------------------------------------------------------------------

def load_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            for v in r["rows"].values():
                v["conf"] = np.array(v["conf"], dtype=np.int64)
            out.append(r)
    return out


def _arr(rows: list[dict[str, Any]], name: str, group: str, only_ok: bool) -> np.ndarray:
    comp = lb.COMPARISON_BY_NAME[group]
    cols = lb.column_ids(comp.canonical)
    vals = []
    for r in rows:
        v = r["rows"][name]
        if only_ok and v["outcome"] != "ok":
            continue
        c = mt.counts_from_confusion(v["conf"], comp.ep_ids, cols)
        vals.append([c.tp, c.fp, c.fn])
    return np.array(vals, dtype=np.int64).reshape(-1, 3)


def _iou(a: np.ndarray) -> float | None:
    return mt.scores(*[float(x) for x in a.sum(axis=0)])["iou"] if len(a) else None


def paired_delta_ci(base: np.ndarray, pert: np.ndarray) -> tuple[float | None, tuple[float | None, float | None]]:
    """Delta of dataset-level IoU (perturbed - baseline) over the same images, paired bootstrap over images."""
    if len(base) == 0 or base.shape != pert.shape:
        return None, (None, None)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(base)
    deltas = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        a, b = _iou(base[idx]), _iou(pert[idx])
        if a is not None and b is not None:
            deltas.append(b - a)
    point = (_iou(pert) or 0) - (_iou(base) or 0)
    if not deltas:
        return point, (None, None)
    return point, (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def median_shift(rows: list[dict[str, Any]], name: str, region: str) -> dict[str, Any]:
    vals = [r["rows"][name]["shift"][region] for r in rows if r["rows"][name]["outcome"] == "ok" and r["rows"][name].get("shift", {}).get(region)]
    if not vals:
        return {"n": 0, "median_dx": None, "median_dy": None}
    a = np.array(vals, dtype=float)
    return {"n": len(vals), "median_dx": float(np.median(a[:, 0])), "median_dy": float(np.median(a[:, 1]))}


def aggregate(rows: list[dict[str, Any]], perturbations: tuple[str, ...] = PERTURBATIONS) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"n_images": n, "perturbations": {}}
    for name in ("baseline", *perturbations):
        ok = sum(1 for r in rows if r["rows"][name]["outcome"] == "ok")
        entry: dict[str, Any] = {"n_ok": ok, "detection_rate": ok / n if n else None, "failure_rate": 1 - ok / n if n else None,
                                 "outcomes": {o: sum(1 for r in rows if r["rows"][name]["outcome"] == o) for o in ("ok", "no_face", "multiple_faces", "pipeline_error")}}
        for g in HEADLINE_GROUPS:
            b_all = _arr(rows, name, g, False)
            entry[f"{g}_B"] = _iou(b_all)
            entry[f"{g}_A"] = _iou(_arr(rows, name, g, True))
            if name != "baseline":
                d, ci = paired_delta_ci(_arr(rows, "baseline", g, False), b_all)
                entry[f"{g}_B_delta"], entry[f"{g}_B_delta_ci95"] = d, list(ci)
        entry["shift"] = {reg: median_shift(rows, name, reg) for reg in ("eye", "lips", "brow")}
        eq = [r["rows"][name]["equiv"] for r in rows if r["rows"][name].get("equiv")]
        if eq:
            entry["equivariance"] = {
                g: {"n_images": len(eq), **mt.scores(*[float(sum(e[g][i] for e in eq)) for i in range(3)])} for g in EQUIV_GROUPS
            }
        out["perturbations"][name] = entry
    return out


def _f(x: Any) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_markdown(agg: dict[str, Any], header: dict[str, Any]) -> str:
    P = agg["perturbations"]
    L = ["# Robustness suite on EasyPortrait dev images (pilot)", "",
         f"Date: {header['date']}. {agg['n_images']} images, seeded subset of the **dev** split (seed {SUBSET_SEED}); "
         "perturbations are applied to image and annotation together. Numbers are *agreement with EasyPortrait annotations* "
         "(dataset-level IoU, full image; rotation fill is not scored), not an accuracy figure. Positive roll = counter-clockwise.", "",
         f"Source fingerprint `{header['fingerprint']}`; crop scale 2.0; one machine ({header['cpu']}).", "",
         "## Detection and agreement per perturbation", "",
         "B = all images (failures are empty predictions); delta = paired difference to the unperturbed baseline with a paired 95% bootstrap CI.", "",
         "| perturbation | detected | failure rate | skin B | lips B | eye B | brow B | eye B delta [CI] | brow B delta [CI] | lips B delta [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in P.items():
        def d(g: str) -> str:
            if f"{g}_B_delta" not in e or e[f"{g}_B_delta"] is None:
                return "-"
            ci = e[f"{g}_B_delta_ci95"]
            return f"{e[f'{g}_B_delta']:+.3f} [{_f(ci[0])}, {_f(ci[1])}]"
        L.append(f"| {name} | {e['n_ok']}/{agg['n_images']} | {e['failure_rate']:.1%} | {_f(e['skin_B'])} | {_f(e['lips_B'])} | {_f(e['eye_B'])} | {_f(e['brow_B'])} | {d('eye')} | {d('brow')} | {d('lips')} |")
    L += ["", "Denominator A (detected images only) is in the JSON. Failures are `no_face` / `multiple_faces` outcomes: " +
          "; ".join(f"{n}: {e['outcomes']}" for n, e in P.items() if e["failure_rate"]), ""]
    L += ["## GT-free consistency (equivariance)", "",
          "Mirror: IoU between the prediction on the mirrored image and the mirrored prediction on the original (sides swapped back). "
          "Roll: between the prediction on the rotated image and the rotated original prediction. Pixels parsed in both and not rotation fill only; images detected in both.", "",
          "| perturbation | images | " + " | ".join(EQUIV_GROUPS) + " |", "|---|---|" + "---|" * len(EQUIV_GROUPS)]
    for name, e in P.items():
        if "equivariance" in e:
            eq = e["equivariance"]
            L.append(f"| {name} | {eq['skin']['n_images']} | " + " | ".join(_f(eq[g]["iou"]) for g in EQUIV_GROUPS) + " |")
    L += ["", "## Roll: where the failures start", "",
          "| roll | detected | eye B | brow B | lips B |", "|---|---|---|---|---|"]
    for name in ("baseline", *[n for n in P if n.startswith("roll")]):
        e = P[name]
        L.append(f"| {name} | {e['n_ok']}/{agg['n_images']} | {_f(e['eye_B'])} | {_f(e['brow_B'])} | {_f(e['lips_B'])} |")
    base, vf = P["baseline"], P["vflip"]
    L += ["", "## Vertical-flip sanity check for the residual y offset", "",
          "Median best integer shift of the parser's prediction to the GT within +-3 px (+dy = prediction must move down in that image). "
          "A convention bias of the model (labels sit lower on the face than the annotators draw them) moves with the face, so its dy flips sign when the face is upside down; "
          "a pipeline coordinate bug is tied to the image axes and keeps its sign.", "",
          "| region | upright dx, dy (n) | vertically flipped dx, dy (n) |", "|---|---|---|"]
    for reg in ("eye", "lips", "brow"):
        a, b = base["shift"][reg], vf["shift"][reg]
        L.append(f"| {reg} | {_f(a['median_dx'])}, {_f(a['median_dy'])} ({a['n']}) | {_f(b['median_dx'])}, {_f(b['median_dy'])} ({b['n']}) |")
    L += ["", f"Detection rate on upside-down faces: {vf['detection_rate']:.1%} ({vf['n_ok']} of {agg['n_images']}). "
          + ("Too few detect for a conclusion." if vf["n_ok"] < 20 else "Enough detect for a comparison, but only the detected (right-way-up-compatible) subset enters it."), ""]
    L += ["## Mirror: median best shifts", "", "| region | upright | mirrored |", "|---|---|---|"]
    m = P["mirror"]
    for reg in ("eye", "lips", "brow"):
        L.append(f"| {reg} | {_f(base['shift'][reg]['median_dx'])}, {_f(base['shift'][reg]['median_dy'])} | {_f(m['shift'][reg]['median_dx'])}, {_f(m['shift'][reg]['median_dy'])} |")
    L += ["", "## Limitations", "",
          "- 100 images from the dev split only; per-perturbation intervals are wide. Not a test-split decision.",
          "- Blur sigma 3 px and rotation fill value 0 are single settings. Gamma and grayscale change the image only.",
          "- Rotation canvases are expanded, so large rolls enlarge the image; fill is excluded from scoring and from the equivariance counts.",
          "- GT is rotated with nearest-neighbour interpolation, which adds a boundary error of up to about 1 px.", "",
          "Data: EasyPortrait (Kapitanov, Kvanchiani, Kirillova, arXiv:2304.13509), https://github.com/hukenovs/easyportrait ; licence variant of CC BY-SA 4.0 (see the main evaluation report). Evaluation only, aggregates only.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import date

    from evaluation import dataset as ds
    from evaluation.cli import DEFAULT_SAMPLE
    from evaluation.runner import _cpu_name, build_services, source_fingerprint

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default="~/.cache/makeup_demo/easyportrait_runs/robustness")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--reuse", action="store_true", help="aggregate an existing perturb_rows.jsonl without re-running")
    args = ap.parse_args(argv)
    cache = ds.ensure_cache(ds.resolve_cache_dir(args.cache_dir))
    out_dir = refuse_inside_repo(Path(args.out_dir), "output directory")
    subset = pick_subset(ds.parse_sample(Path(args.sample).read_text()))
    if not args.reuse:
        landmarks, parser = build_services(2.0)
        try:
            run_suite(subset, cache, out_dir, (landmarks, parser))
        finally:
            landmarks.close()
    agg = aggregate(load_rows(out_dir / "perturb_rows.jsonl"))
    header = {"date": args.date or date.today().isoformat(), "fingerprint": source_fingerprint(), "cpu": _cpu_name()}
    prefix = Path(args.out_prefix) if args.out_prefix else ds.REPO_ROOT / "docs" / "evaluation" / f"robustness_{header['date']}"
    prefix.with_suffix(".md").write_text(render_markdown(agg, header))
    prefix.with_suffix(".json").write_text(json.dumps(agg, indent=1, sort_keys=True) + "\n")
    print(f"wrote {prefix}.md")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
