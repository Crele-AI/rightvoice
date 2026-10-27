"""The ``rightvoice`` command: sources, build, items, timings, score, report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, sources
from .subsets import quick_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rightvoice", description=__doc__)
    parser.add_argument("--version", action="version", version=f"rightvoice {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    src = sub.add_parser("sources", help="list or fetch the pinned sources")
    src_sub = src.add_subparsers(dest="action", required=True)
    p = src_sub.add_parser("plan", help="print every download with its size and URL; fetches nothing")
    p.add_argument("groups", nargs="*", default=None, help=f"one of {sources.GROUPS} or 'all'")
    f = src_sub.add_parser("fetch", help="download and verify")
    f.add_argument("groups", nargs="+", help=f"one of {sources.GROUPS} or 'all'")
    f.add_argument("--pin", action="store_true", help="allow a source with no sha256 yet, and record what arrived")
    f.add_argument("--parallel", action="store_true", help="several ranged connections per file, for slow servers")

    b = sub.add_parser("build", help="plan, select and render the items")
    b.add_argument("--seed", type=int, default=20261022)
    b.add_argument("--out", type=Path, default=Path("data/build/v1"))
    b.add_argument("--count", type=int, default=1500, help="30 s clips to select")
    b.add_argument("--long-count", type=int, default=30, help="long files")
    b.add_argument("--candidates", type=int, default=None,
                   help="clip candidates to plan (default 4 times --count)")
    b.add_argument("--min-per-value", type=int, default=40,
                   help="selection quota per bucket value")
    b.add_argument("--min-absent", type=int, default=60, help="selection quota of target-absent items")
    b.add_argument("--rooms", type=int, default=600, help="simulated rooms in the bank")
    b.add_argument("--only-plan", action="store_true", help="write selection.json and stop")
    b.add_argument("--limit", type=int, default=None, help="render only the first N selected items")

    lp = sub.add_parser("listen", help="write a static listening page for a build")
    lp.add_argument("--build", type=Path, required=True)
    lp.add_argument("--count", type=int, default=20)
    lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--out", type=Path, default=None, help="default <build>/listen.html")

    sc = sub.add_parser("score", help="score one system's outputs on a build")
    sc.add_argument("--build", type=Path, required=True)
    sc.add_argument("--system", required=True,
                    help="system name; scores go to <build>/scores/<system>.jsonl")
    sc.add_argument("--outputs", type=Path, default=None,
                    help="folder of <item id>.wav outputs (default <build>/outputs/<system>)")
    sc.add_argument("--asr-model", default=None,
                    help="recogniser checkpoint (default: the recognition adapter's own)")
    sc.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                    help="device for the recogniser and the speaker embedder")
    sc.add_argument("--asr-cache", type=Path, default=None,
                    help="folder that keeps each recognition result, so a rescore skips it")
    sc.add_argument("--no-quality", action="store_true", help="leave the quality metric empty")
    sc.add_argument("--quick", action="store_true",
                    help="score only the quick subset, into <build>/scores/quick/<system>.jsonl")
    sc.add_argument("--allow-partial", action="store_true",
                    help="score even when under 90%% of items have an output or a logged "
                         "abstention; the rest count as empty output")

    it = sub.add_parser("items", help="print the item ids of a build, one per line")
    it.add_argument("--build", type=Path, required=True)
    it.add_argument("--quick", action="store_true", help="only the quick subset")

    tp = sub.add_parser("timings", help="write word_timings.json for the utterances a build places")
    tp.add_argument("--build", type=Path, required=True)
    tp.add_argument("--asr-model", default=None,
                    help="recogniser checkpoint (default: the recognition adapter's own)")
    tp.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                    help="device for the recogniser")
    tp.add_argument("--asr-cache", type=Path, default=None,
                    help="folder that keeps each recognition result")
    tp.add_argument("--limit", type=int, default=None,
                    help="recognise at most N utterances not yet in the file")

    rp = sub.add_parser("report", help="write report.md and report.json for a scored build")
    rp.add_argument("--build", type=Path, required=True)
    rp.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    rp.add_argument("--seed", type=int, default=20261022, help="bootstrap seed")
    rp.add_argument("--quick", action="store_true",
                    help="report on the quick subset, into report-quick.md and report-quick.json")

    args = parser.parse_args(argv)
    if args.command == "build":
        return _build(args)
    if args.command == "score":
        return _score(args)
    if args.command == "timings":
        return _timings(args)
    if args.command == "items":
        return _items(args)
    if args.command == "report":
        return _report(args)
    if args.command == "listen":
        from .listen import write_page

        print(write_page(args.build, args.out, count=args.count, seed=args.seed))
        return 0
    if args.command == "sources":
        if args.action == "plan":
            sources.plan(args.groups or None)
            return 0
        for source in sources.select(args.groups):
            path = sources.fetch(source, pin=args.pin, parallel=args.parallel)
            print(f"{source.name}: {path}")
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


def _build(args: argparse.Namespace) -> int:
    from .build import build

    info = build(args.out, seed=args.seed, count=args.count, long_count=args.long_count,
                 candidates=args.candidates, min_per_value=args.min_per_value,
                 min_absent=args.min_absent, n_rooms=args.rooms, only_plan=args.only_plan,
                 limit=args.limit)
    counts, timing = info["counts"], info["timing"]
    print(f"planned {counts['planned']} of {counts['candidates']} candidates "
          f"({counts['plan_failed']} failed), selected {counts['selected']} clips "
          f"and {counts['long']} long items")
    print(f"rendered {counts['rendered']} items ({counts['render_failed']} failed) "
          f"into {args.out}")
    for key in ("load_s", "plan_s", "plan_long_s", "room_bank_s", "render_s", "clip_mean_s",
                "long_mean_s", "total_s"):
        if timing.get(key) is not None:
            print(f"  {key}: {timing[key]:.2f}")
    for failure in info["render_failures"]:
        print(f"  failed {failure['id']}: {failure['error']}", file=sys.stderr)
    return 1 if counts["render_failed"] else 0


def _score(args: argparse.Namespace) -> int:
    # The adapters import their packages only when first used, so --help stays light.
    from . import score
    from .report import QUICK_SCORES, SCORES, load_word_timings

    name = args.system
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        print(f"rightvoice score: {name!r} is not a plain system name", file=sys.stderr)
        return 2
    rows = score.read_manifest(args.build / "manifest.jsonl")
    ids = quick_ids(rows) if args.quick else [str(r["id"]) for r in rows]
    outputs = args.outputs if args.outputs is not None else args.build / "outputs" / name
    unanswered = score.unanswered(outputs, ids)
    found = len(ids) - len(unanswered)
    # The silence control returns nothing for every item by design.
    if (name != "control_silence" and not args.allow_partial
            and found < score.MIN_COVERAGE * len(ids)):
        print(f"rightvoice score: only {found} of {len(ids)} items have an output or a logged "
              f"abstention in {outputs}. A finished run answers at least "
              f"{score.MIN_COVERAGE:.0%}. To return nothing for an item, write an empty WAV "
              "for it. Rerun the items that failed, or pass --allow-partial to score anyway "
              "with the missing items counted as empty output.", file=sys.stderr)
        return 2
    asr_kwargs = {"device": args.device, "cache_dir": args.asr_cache}
    if args.asr_model:
        asr_kwargs["model"] = args.asr_model
    mos = None
    if not args.no_quality:
        cache = Path(args.asr_cache).parent / "dnsmos" if args.asr_cache else None
        mos = score.DnsmosP808(cache_dir=cache)
    timings = load_word_timings(args.build)
    if timings is None:
        print("rightvoice score: no word_timings.json in the build; an utterance without "
              "timings in its manifest row has its words spread over its placement",
              file=sys.stderr)
    dest = args.build / (QUICK_SCORES if args.quick else SCORES) / f"{name}.jsonl"
    scores = score.score_system(args.build, name, asr=score.WhisperASR(**asr_kwargs),
                                vad=score.SileroVAD(), embed=score.EcapaEmbedder(device=args.device),
                                word_timings=timings, outputs=args.outputs, mos=mos,
                                out_path=dest, items=ids if args.quick else None)
    empty = sum(bool(s["empty_output"]) for s in scores)
    print(f"scored {len(scores)} items for {name} ({empty} empty output) into {dest}")
    if unanswered and name != "control_silence":
        shown = ", ".join(unanswered[:10]) + (", ..." if len(unanswered) > 10 else "")
        print(f"warning: {len(unanswered)} items have no output and no logged abstention and "
              f"were scored as empty output: {shown}", file=sys.stderr)
    return 0


def _items(args: argparse.Namespace) -> int:
    from .score import read_manifest

    rows = read_manifest(args.build / "manifest.jsonl")
    for item_id in quick_ids(rows) if args.quick else [str(r["id"]) for r in rows]:
        print(item_id)
    return 0


def _timings(args: argparse.Namespace) -> int:
    import json

    from . import score
    from .timings import WORD_TIMINGS, report_path, word_timings

    asr_kwargs = {"device": args.device, "cache_dir": args.asr_cache}
    if args.asr_model:
        asr_kwargs["model"] = args.asr_model
    out = args.build / WORD_TIMINGS
    word_timings(args.build, score.WhisperASR(**asr_kwargs), out_path=out, limit=args.limit)
    rep = json.loads(report_path(out).read_text(encoding="utf-8"))
    share = rep["within_one_word"]
    print(f"timed {rep['utterances_done']} of {rep['utterances_used']} utterances "
          f"({rep['utterances_left']} left, {rep['utterances_no_words']} with no words), "
          f"{rep['words']} words; "
          + (f"{share:.1%} within one word of the corpus text" if share is not None
             else "nothing to compare"))
    print(out)
    print(report_path(out))
    if rep["not_in_catalogue"]:
        print(f"rightvoice timings: {len(rep['not_in_catalogue'])} used utterances are not in "
              "the catalogue", file=sys.stderr)
    return 0


def _report(args: argparse.Namespace) -> int:
    from .report import write_report

    report = write_report(args.build, n_boot=args.boot, seed=args.seed, quick=args.quick)
    print(report["files"]["markdown"])
    print(report["files"]["json"])
    for warning in report["controls"]["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
