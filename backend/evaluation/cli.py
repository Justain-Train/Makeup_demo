"""Command line: python -m evaluation.cli {sample,fetch,run,report,clean} (run from backend/)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from evaluation import dataset as ds
from evaluation.remote_zip import FetchError

logger = logging.getLogger("evaluation")
REPO_ROOT = ds.REPO_ROOT
DEFAULT_SAMPLE = REPO_ROOT / "docs" / "evaluation" / f"easyportrait_test_sample_seed{ds.DEFAULT_SEED}_n{ds.DEFAULT_N}.txt"
DEFAULT_RUNS_DIR = Path("~/.cache/makeup_demo/easyportrait_runs")
DEFAULT_MAKEUP_RUNS_DIR = Path("~/.cache/makeup_demo/makeup_runs")
DEFAULT_MAKEUP_EVAL_RUNS_DIR = Path("~/.cache/makeup_demo/makeup_runs/eval")
DEFAULT_MAKEUP_SEED = 20261113
DEFAULT_MAKEUP_SAMPLE = REPO_ROOT / "docs" / "evaluation" / f"makeup_sample_seed{DEFAULT_MAKEUP_SEED}_n200.txt"


def _cache(args: argparse.Namespace) -> Path:
    return ds.resolve_cache_dir(args.cache_dir)


def cmd_sample(args: argparse.Namespace) -> int:
    cache = ds.ensure_cache(_cache(args))
    rows = ds.load_or_fetch_meta(cache)
    entries = ds.draw_sample(rows, args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.format_sample(entries, args.seed, args.n))
    cells = {(e.brightness_quartile, e.resolution_group) for e in entries}
    print(f"wrote {len(entries)} ids ({sum(e.split == 'dev' for e in entries)} dev) over {len(cells)} strata to {out}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    cache = ds.ensure_cache(_cache(args))
    entries = ds.parse_sample(Path(args.sample).read_text())
    s = ds.fetch_samples(entries, cache, budget_bytes=int(args.budget_gb * 1024**3), threads=args.threads)
    print(
        f"fetched {s.n_downloaded} (skipped {s.n_skipped} already cached, failed {s.n_failed}) of {s.n_requested}; "
        f"{s.bytes_received:,} bytes ({s.bytes_received / 1e6:.1f} MB) in {s.requests} requests, {s.seconds:.1f} s, {s.throughput_mb_s:.2f} MB/s"
    )
    return 1 if s.n_failed else 0


def cmd_run(args: argparse.Namespace) -> int:
    from evaluation.runner import run_evaluation

    cache = ds.ensure_cache(_cache(args))
    entries = ds.parse_sample(Path(args.sample).read_text())
    out_dir = ds.refuse_inside_repo(Path(args.out_dir) if args.out_dir else DEFAULT_RUNS_DIR / date.today().isoformat(), "output directory")
    summary = run_evaluation(entries, cache, out_dir, crop_scale=args.crop_scale, warmups=args.warmups, limit=args.limit)
    print(json.dumps(summary.__dict__, indent=1))
    return 2 if summary.stopped_early else 0


def cmd_report(args: argparse.Namespace) -> int:
    from evaluation import report as rp

    run_dir = ds.refuse_inside_repo(Path(args.run_dir), "run directory")
    entries = ds.parse_sample(Path(args.sample).read_text())
    rows = rp.load_rows(run_dir / "per_image.jsonl")
    run_meta = json.loads((run_dir / "run_meta.json").read_text())
    sample_rel = str(Path(args.sample).resolve().relative_to(REPO_ROOT)) if REPO_ROOT in Path(args.sample).resolve().parents else Path(args.sample).name
    rep = rp.build_report(rows, entries, run_meta, sample_rel, args.date)
    prefix = Path(args.out_prefix) if args.out_prefix else REPO_ROOT / "docs" / "evaluation" / f"easyportrait_test_{rep['date']}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    md, js = rp.write_report(rep, prefix)
    print(f"wrote {md} and {js}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    cache = _cache(args)
    if not args.yes:
        print(f"would delete the cache at {cache} (needs its marker file); pass --yes to do it")
        return 0
    print(f"removed {ds.clean_cache(cache)} files")
    return 0


def cmd_makeup_fetch(args: argparse.Namespace) -> int:
    from evaluation import makeup_dataset as md
    from evaluation.hf_resolve import ResolvingFetcher
    from evaluation.remote_zip import BudgetedFetcher

    cache = md.ensure_cache(md.resolve_cache_dir(args.cache_dir))
    bf = BudgetedFetcher(ResolvingFetcher(md.URL), int(args.budget_mb * 1024 * 1024), md.PINNED_LENGTHS)
    ids, table = md.load_or_build_index(cache, bf)
    samples = md.draw_samples(ids, args.seed, args.n, args.n_cross)
    out = Path(args.sample_out or REPO_ROOT / "docs" / "evaluation" / f"makeup_sample_seed{args.seed}_n{args.n}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md.format_samples(samples, args.seed, args.n))
    s = md.fetch_makeup(cache, md.required_files(samples), args.budget_mb, args.workers, bf=bf, table=table)
    print(
        f"{len(ids)} complete identities; sample written to {out}; downloaded {s.downloaded}, skipped {s.skipped}, "
        f"failed {s.failed}, {s.bytes_received:,} bytes{' (budget reached)' if s.budget_reached else ''}"
    )
    return 1 if s.failed or s.budget_reached else 0


def cmd_makeup_sheet(args: argparse.Namespace) -> int:
    from evaluation import makeup_dataset as md
    from evaluation.makeup_pipeline import PairRunner
    from evaluation.makeup_sheet import build_sheet

    cache = md.ensure_cache(md.resolve_cache_dir(args.cache_dir))
    samples = md.parse_samples(Path(args.sample).read_text())
    runs = ds.refuse_inside_repo(Path(args.runs_dir) if args.runs_dir else DEFAULT_MAKEUP_RUNS_DIR / date.today().isoformat(), "output directory")
    runner = PairRunner()
    try:
        path = build_sheet(cache, samples, runner, runs / "contact_sheet.png")
    finally:
        runner.close()
    print(path)
    return 0


def cmd_makeup_eval(args: argparse.Namespace) -> int:
    from evaluation import makeup_dataset as md
    from evaluation import makeup_eval as me

    cache = md.ensure_cache(md.resolve_cache_dir(args.cache_dir))
    text = Path(args.sample).read_text()
    samples = md.parse_samples(text)
    runs = ds.refuse_inside_repo(Path(args.runs_dir) if args.runs_dir else DEFAULT_MAKEUP_EVAL_RUNS_DIR, "output directory")
    ep_cache = ds.resolve_cache_dir(args.ep_cache_dir)
    ep_ids: list[str] = []
    if ep_cache.is_dir() and Path(args.ep_sample).is_file():
        ep_ids = [e.image_id for e in ds.parse_sample(Path(args.ep_sample).read_text())]
    rep = me.run(
        cache, samples, runs, Path(args.out_dir),
        ep_cache=ep_cache if ep_ids else None, ep_ids=ep_ids, seed=me.parse_seed(text),
        limit=args.limit, label_margin=args.label_margin, audit_sheet=args.audit_sheet,
    )
    print(json.dumps(rep["_run"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="evaluation", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--cache-dir", default=None, help=f"default: ${ds.CACHE_ENV} or {ds.DEFAULT_CACHE_DIR} (never inside the repo)")

    sp = sub.add_parser("sample", help="draw the stratified sample (downloads the 1.9 MB meta table)")
    common(sp)
    sp.add_argument("--n", type=int, default=ds.DEFAULT_N)
    sp.add_argument("--seed", type=int, default=ds.DEFAULT_SEED)
    sp.add_argument("--out", default=str(DEFAULT_SAMPLE))
    sp.set_defaults(fn=cmd_sample)

    sp = sub.add_parser("fetch", help="download only the sampled images and annotations (HTTP Range)")
    common(sp)
    sp.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    sp.add_argument("--budget-gb", type=float, default=3.0)
    sp.add_argument("--threads", type=int, default=4)
    sp.set_defaults(fn=cmd_fetch)

    sp = sub.add_parser("run", help="offline: run the pipeline over the cached samples")
    common(sp)
    sp.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    sp.add_argument("--out-dir", default=None, help="outside the repo; default ~/.cache/makeup_demo/easyportrait_runs/<date>")
    sp.add_argument("--crop-scale", type=float, default=2.0)
    sp.add_argument("--warmups", type=int, default=3)
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("report", help="aggregate a run into docs/evaluation/*.md and *.json")
    sp.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--out-prefix", default=None)
    sp.add_argument("--date", default=None)
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("clean", help="delete the cache (requires its marker file)")
    common(sp)
    sp.add_argument("--yes", action="store_true")
    sp.set_defaults(fn=cmd_clean)

    sp = sub.add_parser("makeup-fetch", help="index FFHQ-Makeup, draw the sample and fetch it (HTTP Range)")
    sp.add_argument("--cache-dir", default=None, help="default: ~/.cache/makeup_demo/ffhq_makeup (never inside the repo)")
    sp.add_argument("--seed", type=int, default=DEFAULT_MAKEUP_SEED)
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--n-cross", type=int, default=60)
    sp.add_argument("--budget-mb", type=float, default=300)
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--sample-out", default=None)
    sp.set_defaults(fn=cmd_makeup_fetch)

    sp = sub.add_parser("makeup-sheet", help="offline: contact sheet of bare | reference | preview | truth")
    sp.add_argument("--cache-dir", default=None)
    sp.add_argument("--sample", default=str(DEFAULT_MAKEUP_SAMPLE))
    sp.add_argument("--runs-dir", default=None, help="outside the repo; default ~/.cache/makeup_demo/makeup_runs/<date>")
    sp.set_defaults(fn=cmd_makeup_sheet)

    sp = sub.add_parser("makeup-eval", help="offline: Phase 7 detection diagnosis and gap-closed metrics on FFHQ-Makeup")
    sp.add_argument("--cache-dir", default=None)
    sp.add_argument("--sample", default=str(DEFAULT_MAKEUP_SAMPLE))
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "evaluation"), help="default: ../docs/evaluation")
    sp.add_argument("--runs-dir", default=None, help=f"outside the repo; default {DEFAULT_MAKEUP_EVAL_RUNS_DIR}")
    sp.add_argument("--label-margin", type=float, default=0.5)
    sp.add_argument("--audit-sheet", action="store_true")
    sp.add_argument("--ep-cache-dir", default=None, help="EasyPortrait cache for the false-positive rate (skipped if absent)")
    sp.add_argument("--ep-sample", default=str(DEFAULT_SAMPLE))
    sp.set_defaults(fn=cmd_makeup_eval)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ds.CacheError, FetchError, ValueError, OSError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
