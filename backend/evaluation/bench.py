"""Reproducible benchmark (WP2) and the offline label-map resolution-cap experiment.

* Stage latency: the portrait fixture upscaled to about 0.8, 2, 10 and 25 MP (SYNTHETIC scale: the same face
  enlarged, not real photos), 1 discarded + 30 timed requests, p50/p95/p99 per stage.
* Concurrency 1, 2, 4 with peak RSS per configuration, each in a fresh subprocess (`--worker`).
* Cap experiment: agreement with EasyPortrait annotations on the dev split when the label map is downscaled
  (nearest) to a long side of 2048 or 1024 and back. Production code is not changed.

`ru_maxrss` is bytes on macOS and kilobytes on Linux (`normalise_maxrss`).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import resource
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.image_loading import load_rgb_image
from evaluation import labels as lb
from evaluation import metrics as mt
from evaluation.dataset import annotation_path, decode_annotation, image_path, parse_sample
from evaluation.predict import predict_full

logger = logging.getLogger("evaluation")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "portrait.jpg"
SIZES_MP = (0.8, 2.0, 10.0, 25.0)
MAX_PIXELS = 25_000_000
WARM_RUNS = 30
STAGES = ("landmarks", "preprocess", "inference", "postprocess", "zones", "encode", "total")
CAPS = (None, 2048, 1024)


def normalise_maxrss(value: float, system: str | None = None) -> float:
    """ru_maxrss -> megabytes (1 MB = 1e6 bytes). Bytes on macOS (Darwin), kilobytes elsewhere (Linux)."""
    system = system or platform.system()
    return value / 1e6 if system == "Darwin" else value * 1024 / 1e6


def parse_worker_output(stdout: str) -> dict[str, Any]:
    """The worker prints log noise and one final JSON object line; return that object."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("worker produced no JSON result")


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    a = np.array(values)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def upscaled_fixture(megapixels: float) -> np.ndarray:
    img = load_rgb_image(FIXTURE.read_bytes())
    h, w = img.shape[:2]
    target = min(megapixels * 1e6, MAX_PIXELS - 1000)
    s = (target / (h * w)) ** 0.5
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def bench_stages(services: tuple[Any, Any], sizes: tuple[float, ...] = SIZES_MP, runs: int = WARM_RUNS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mp in sizes:
        img = upscaled_fixture(mp)
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        predict_full(img, *services)  # discarded
        per: dict[str, list[float]] = {s: [] for s in STAGES}
        decode, wall = [], []
        for _ in range(runs):
            t = time.perf_counter()
            load_rgb_image(data)
            decode.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            outcome, _, timings = predict_full(img, *services)
            wall.append((time.perf_counter() - t) * 1000)
            if outcome != "ok":
                raise RuntimeError("benchmark image was not segmented")
            for s in STAGES:
                if s in timings:
                    per[s].append(timings[s])
        out[str(mp)] = {
            "pixels": int(img.shape[0] * img.shape[1]), "shape": [int(img.shape[0]), int(img.shape[1])], "jpeg_bytes": len(data),
            "decode_ms": percentiles(decode), "wall_ms": percentiles(wall), **{s: percentiles(per[s]) for s in STAGES},
        }
        logger.info("bench %.1f MP done", mp)
    return out


def worker(megapixels: float, concurrency: int, per_thread: int) -> dict[str, Any]:
    """Run in a fresh process: `concurrency` threads, `per_thread` timed requests each after one warm-up."""
    from evaluation.runner import build_services

    landmarks, parser = build_services(2.0)
    # waiting requests must not time out here: this measures latency and memory, not the 503 policy
    parser._queue_timeout_s = 120.0
    img = upscaled_fixture(megapixels)
    predict_full(img, landmarks, parser)
    lat: list[float] = []
    lock = threading.Lock()
    failures = []

    def run() -> None:
        for _ in range(per_thread):
            t = time.perf_counter()
            outcome, _, _ = predict_full(img, landmarks, parser)
            dt = (time.perf_counter() - t) * 1000
            with lock:
                lat.append(dt)
                if outcome != "ok":
                    failures.append(outcome)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=run) for _ in range(concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    elapsed = time.perf_counter() - t0
    result = {
        "megapixels": megapixels, "concurrency": concurrency, "parser_max_concurrency": 2, "requests": len(lat),
        "failures": len(failures), "elapsed_s": elapsed, "throughput_rps": len(lat) / elapsed,
        "latency_ms": percentiles(lat), "peak_rss_mb": normalise_maxrss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    landmarks.close()
    return result


def bench_concurrency(sizes: tuple[float, ...], levels: tuple[int, ...] = (1, 2, 4), per_thread: int = 6) -> list[dict[str, Any]]:
    out = []
    for mp in sizes:
        for c in levels:
            proc = subprocess.run(
                [sys.executable, "-m", "evaluation.bench", "--worker", "--mp", str(mp), "--concurrency", str(c), "--per-thread", str(per_thread)],
                capture_output=True, text=True, timeout=1800, cwd=str(Path(__file__).resolve().parents[1]),
            )
            if proc.returncode != 0:
                raise RuntimeError(f"worker failed (exit {proc.returncode})")
            out.append(parse_worker_output(proc.stdout))
            logger.info("concurrency %s MP x%d done", mp, c)
    return out


# --- label-map resolution cap (offline experiment) -------------------------------------------------------------

def cap_label_map(pred: np.ndarray, cap: int | None) -> np.ndarray:
    """Downscale the parsed rectangle of a label map (nearest) to a long side <= cap and back to full size."""
    if cap is None:
        return pred
    ys, xs = np.nonzero(pred != 255)
    if xs.size == 0:
        return pred
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = pred[y0:y1, x0:x1]
    long_side = max(crop.shape)
    if long_side <= cap:
        return pred
    s = cap / long_side
    small = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
    back = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = pred.copy()
    out[y0:y1, x0:x1] = back
    return out


def cap_experiment(entries: list, cache: Path, services: tuple[Any, Any], caps: tuple[int | None, ...] = CAPS) -> dict[str, Any]:
    conf = {str(c): np.zeros((9, lb.CONF_COLS), dtype=np.int64) for c in caps}
    per_image: dict[str, list[np.ndarray]] = {str(c): [] for c in caps}
    affected = {str(c): 0 for c in caps}
    n_ok = 0
    for k, e in enumerate(entries, 1):
        rgb = load_rgb_image(image_path(cache, e.image_id).read_bytes())
        gt = decode_annotation(annotation_path(cache, e.image_id).read_bytes(), rgb.shape[:2])
        outcome, pred, _ = predict_full(rgb, *services)
        if outcome != "ok":
            continue
        n_ok += 1
        for c in caps:
            capped = cap_label_map(pred, c)
            affected[str(c)] += int(not np.array_equal(capped, pred))
            m = mt.confusion_matrix(gt, capped)
            conf[str(c)] += m
            per_image[str(c)].append(m)
        if k % 25 == 0:
            logger.info("cap experiment %d/%d", k, len(entries))
    out: dict[str, Any] = {"n_images": n_ok, "caps": {}}
    for c in caps:
        key = str(c)
        entry: dict[str, Any] = {"images_changed": affected[key]}
        for name in ("skin", "lips", "eye", "brow"):
            comp = lb.COMPARISON_BY_NAME[name]
            cols = lb.column_ids(comp.canonical)
            arr = np.array([[x.tp, x.fp, x.fn] for x in (mt.counts_from_confusion(m, comp.ep_ids, cols) for m in per_image[key])]).reshape(-1, 3)
            entry[name] = {**mt.scores(*[float(v) for v in arr.sum(axis=0)]), "ci95": mt.bootstrap_scores(arr, 1000, 20261110)["iou"]}
        out["caps"][key] = entry
    return out


# --- environment and report ------------------------------------------------------------------------------------------

def power_state() -> str:
    """macOS `pmset -g batt` first lines (AC or battery); 'unknown' elsewhere. No identifiers are recorded."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        first = out.splitlines()[0] if out else ""
        return first.replace("Now drawing from", "drawing from").strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def sleep_check(start: datetime, end: datetime) -> str:
    """Count macOS sleep events (pmset -g log) between two local times; 'unknown' where pmset is unavailable."""
    try:
        out = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return "unknown (pmset unavailable)"
    events = []
    for line in out.splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\S+\s+(Sleep|Wake|DarkWake)\b", line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if start <= t <= end:
                events.append((m.group(2), m.group(1)))
    sleeps = [e for e in events if e[0] == "Sleep"]
    return (f"{len(sleeps)} sleep event(s) between {start:%H:%M:%S} and {end:%H:%M:%S}"
            + (": " + ", ".join(t for _, t in sleeps) if sleeps else ""))


def environment() -> dict[str, Any]:
    from evaluation.runner import _cpu_name, source_fingerprint
    from importlib import metadata

    return {
        "python": platform.python_version(), "os": platform.platform(), "cpu": _cpu_name(),
        "packages": {p: metadata.version(p) for p in ("numpy", "onnxruntime", "mediapipe", "Pillow")},
        "power": power_state(), "opencv_threads": cv2.getNumThreads(), "source_fingerprint_sha256": source_fingerprint(),
        "ru_maxrss_unit": "bytes (Darwin)" if platform.system() == "Darwin" else "kilobytes (Linux)",
    }


def render_markdown(res: dict[str, Any]) -> str:
    def p(d: dict[str, Any], k: str) -> str:
        return "n/a" if d[k] is None else f"{d[k]:.1f}"

    env = res["environment"]
    L = [f"# Benchmark (pilot), {res['date']}", "",
         "One machine, CPU only. **Synthetic scale:** the large images are the 820x1024 portrait fixture enlarged, not real photos. "
         "Latency is `segment_face` end to end excluding HTTP; decode is `load_rgb_image` on a q90 JPEG.", "",
         f"- Python {env['python']}, packages {env['packages']}", f"- {env['os']}, {env['cpu']}, OpenCV threads {env['opencv_threads']}",
         f"- Power at start: {env['power']}. Sleep inside the window: {res.get('sleep_check', 'not checked')}.",
         f"- Source fingerprint `{env['source_fingerprint_sha256']}`; ru_maxrss unit on this machine: {env['ru_maxrss_unit']} (converted to MB)", "",
         f"## Stage latency (1 discarded + {res['runs']} timed runs; ms, p50 / p95 / p99)", "",
         "| size | pixels | decode | " + " | ".join(STAGES) + " |", "|---|---|---|" + "---|" * len(STAGES)]
    for mp, r in res["stages"].items():
        cells = [f"{p(r[s], 'p50')} / {p(r[s], 'p95')} / {p(r[s], 'p99')}" for s in STAGES]
        L.append(f"| {mp} MP | {r['pixels']:,} | {p(r['decode_ms'], 'p50')} / {p(r['decode_ms'], 'p95')} / {p(r['decode_ms'], 'p99')} | " + " | ".join(cells) + " |")
    L += ["", "## Concurrency and peak memory (each row a fresh process; parser max concurrency 2 as in the app)", "",
          "Peak RSS is of the whole process (includes the MediaPipe landmarker, about 200 MB). Latency is per request including time waiting for a parser slot.", "",
          "| size | threads | requests | failures | throughput (req/s) | latency p50 / p95 (ms) | peak RSS (MB) |", "|---|---|---|---|---|---|---|"]
    for r in res["concurrency"]:
        L.append(f"| {r['megapixels']} MP | {r['concurrency']} | {r['requests']} | {r['failures']} | {r['throughput_rps']:.2f} | {r['latency_ms']['p50']:.0f} / {r['latency_ms']['p95']:.0f} | {r['peak_rss_mb']:.0f} |")
    cap = res["cap"]
    L += ["", f"## Label-map resolution cap vs agreement (dev split, {cap['n_images']} detected images)", "",
          "The label map is downscaled (nearest) to the given long side and scaled back before scoring against EasyPortrait annotations (dataset-level IoU, full image). "
          "This only tests the output resolution; the production parser already computes logits at a working size of at most 1024. EasyPortrait images are mostly 1.4-1.9k px, so "
          "a 2048 cap changes nothing for them (it matters for large phone photos, not tested here).", "",
          "| cap (long side) | images changed | skin | lips | eye | brow |", "|---|---|---|---|---|---|"]
    for key, e in cap["caps"].items():
        L.append(f"| {'none' if key == 'None' else key} | {e['images_changed']} | " + " | ".join(f"{e[n]['iou']:.4f}" for n in ("skin", "lips", "eye", "brow")) + " |")
    L += ["", "## Limitations", "", "- Single machine and a single face upscaled; real large photos may differ.",
          "- Concurrent runs use 6 requests per thread; peak RSS is a maximum, not a distribution.",
          "- The cap experiment cannot show the memory or latency benefit of a real cap, only its agreement cost.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mp", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--per-thread", type=int, default=6)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--date", default=None)
    args = ap.parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args.mp, args.concurrency, args.per_thread)))
        return 0
    from evaluation import dataset as ds
    from evaluation.cli import DEFAULT_SAMPLE
    from evaluation.runner import build_services

    cache = ds.ensure_cache(ds.resolve_cache_dir(args.cache_dir))
    entries = [e for e in parse_sample(Path(args.sample or DEFAULT_SAMPLE).read_text()) if e.split == "dev"]
    started = datetime.now()
    landmarks, parser = build_services(2.0)
    try:
        res: dict[str, Any] = {"date": args.date or date.today().isoformat(), "runs": WARM_RUNS, "environment": environment()}
        res["stages"] = bench_stages((landmarks, parser))
        res["cap"] = cap_experiment(entries, cache, (landmarks, parser))
    finally:
        landmarks.close()
    res["concurrency"] = bench_concurrency((2.0, 25.0))
    res["sleep_check"] = sleep_check(started, datetime.now())
    prefix = Path(args.out_prefix) if args.out_prefix else ds.REPO_ROOT / "docs" / "evaluation" / f"benchmark_{res['date']}"
    prefix.with_suffix(".md").write_text(render_markdown(res))
    prefix.with_suffix(".json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print(f"wrote {prefix}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
