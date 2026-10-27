"""The report: headline numbers with paired intervals, pairwise verdicts and bucket tables.

``write_report`` reads ``manifest.jsonl`` and every ``scores/<system>.jsonl`` of a build and
writes ``report.md`` and ``report.json`` into the build folder.

Every rate is a ratio of sums over items, never a mean of per-item ratios: target words lost is
the summed edits over the summed target words, a per-minute rate is summed words over summed
minutes, and activity F1 is ``2 tp / (2 tp + fp + fn)`` on summed frame counts. Voice match,
quality and SI-SDR improvement are means over the items where they are defined.

An item a system has no score line for counts as that system's empty output. Its line is
``score.empty_output_scores`` on the reference built from the manifest row, with
``word_timings.json`` from the build when present, so every target word is deleted, nothing
is said over silence and nothing is a wrong voice.

Intervals come from one paired bootstrap: ``N_BOOT`` multinomial resamples of the item ids,
drawn once from ``numpy.random.default_rng(seed)`` and applied to every system and every
metric, so a difference between two systems is taken on the same items in every resample. An
interval is the 2.5 and 97.5 percentiles of the resampled values. Resamples where a metric is
undefined (a zero denominator) are left out of that metric's percentiles.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import score
from .subsets import QUICK_EVERY, quick_ids

SEED = 20261022
N_BOOT = 1000
PERCENTILES = (2.5, 97.5)
CLEAR_RELATIVE = 0.10        # "clearly better" needs at least this share of the worse value
MIN_BUCKET_ITEMS = 10        # bucket values with fewer items are listed but not scored
REPORT_MD = "report.md"
REPORT_JSON = "report.json"
QUICK_MD = "report-quick.md"
QUICK_JSON = "report-quick.json"
SCORES = "scores"
QUICK_SCORES = "scores/quick"
WORD_TIMINGS = "word_timings.json"
SI_SDR_NOTE = ("SI-SDR improvement is in report.json only: it measures waveform agreement with "
               "the clean stem rather than what a recogniser or a listener gets from the output.")

CONTROLS = ("control_clean", "control_mixture", "control_silence")
CONTROL_NOTES = {
    "control_clean": ("returns the clean target stem, so its target words lost should sit near "
                      "the recogniser's floor."),
    "control_mixture": "returns the mixture unchanged, so it should be high in wrong-voice words.",
    "control_silence": ("returns silence, so it should show zero words per minute on "
                        "target-absent items."),
}
CONTROL_REASONS = {
    "control_clean": ("The clean target should sit near the recogniser's floor, so the "
                      "references, the word timings or the recogniser may be off."),
    "control_mixture": ("The mixture should carry many words from other voices, so the "
                        "references or the wrong-voice matching may be off."),
    "control_silence": ("Silence should give no words at all, so the handling of empty output "
                        "may be off."),
}
# (metric, limit, "max" or "min"): the control is out of range past the limit.
CONTROL_LIMITS = {
    "control_clean": ("target_wer", 0.15, "max"),
    "control_mixture": ("wrong_voice_per_100_words", 5.0, "min"),
    "control_silence": ("absent_words_per_minute", 0.0, "max"),
}

BUCKET_KEYS = ("overlap", "reentry", "tir_db", "bystanders", "same_sex", "capture", "sample_s",
               "sample_device", "target_absent", "long")
# Display order of the labelled values, as in rightvoice.build.
_VALUE_ORDER: dict[str, tuple[str, ...]] = {
    "overlap": ("<5", "5-25", "25-50", ">=50"),
    "reentry": ("none", "<2", "2-5", "5-10", ">=10"),
    "tir_db": ("<=0", ">0"),
    "capture": ("near_clean", "room", "room_reverberant", "call", "laptop_opus",
                "phone_amr_wb", "phone_amr_nb", "phone_g711"),
}

Table = dict[str, np.ndarray]


# Metrics

@dataclass(frozen=True)
class Metric:
    name: str
    label: str              # in sentences: "on <label>: x against y"
    header: str             # table column
    lower_is_better: bool
    decimals: int
    scale: float = 1.0      # display only
    unit: str = ""          # printed after each number in sentences
    headline: bool = True   # False keeps it out of the Markdown


def _rate(weights: Any, num: np.ndarray, den: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """``scale * (w @ num) / (w @ den)`` for each row of ``weights``; nan where ``w @ den`` is 0."""
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    top = w @ num
    bottom = w @ den
    out = np.full(top.shape, np.nan)
    ok = bottom > 0
    out[ok] = scale * top[ok] / bottom[ok]
    return out


METRIC_FNS: dict[str, Callable[[Table, Any], np.ndarray]] = {
    "target_wer": lambda t, w: _rate(w, t["errors"], t["n_ref"]),
    "wrong_voice_per_100_words": lambda t, w: _rate(w, t["wrong_voice_present"], t["n_ref"], 100.0),
    "wrong_voice_per_minute": lambda t, w: _rate(w, t["wrong_voice"], t["minutes"]),
    "silence_words_per_minute": lambda t, w: _rate(w, t["silence_words"], t["silent_minutes"]),
    "absent_words_per_minute": lambda t, w: _rate(w, t["absent_words"], t["absent_minutes"]),
    "activity_f1": lambda t, w: _rate(w, 2.0 * t["tp"], 2.0 * t["tp"] + t["fp"] + t["fn"]),
    "voice_match": lambda t, w: _rate(w, t["voice_match"], t["voice_match_n"]),
    "quality": lambda t, w: _rate(w, t["quality"], t["quality_n"]),
    "si_sdr_i": lambda t, w: _rate(w, t["si_sdr_i"], t["si_sdr_i_n"]),
}

METRICS: dict[str, Metric] = {m.name: m for m in (
    Metric("target_wer", "target words lost", "Target words lost (%)", True, 1, 100.0, "%"),
    Metric("wrong_voice_per_100_words", "wrong-voice words as a share of target words",
           "Wrong-voice words (% of target words)", True, 1, 1.0, "%"),
    Metric("wrong_voice_per_minute", "wrong-voice words per minute",
           "Wrong-voice words per minute", True, 2),
    Metric("silence_words_per_minute", "words per minute while the target is silent",
           "Words per minute, target silent", True, 2),
    Metric("absent_words_per_minute", "words per minute on target-absent items",
           "Words per minute, target absent", True, 2),
    Metric("activity_f1", "target activity F1", "Activity F1", False, 3),
    Metric("voice_match", "voice match", "Voice match", False, 3),
    Metric("quality", "quality", "Quality", False, 2),
    Metric("si_sdr_i", "SI-SDR improvement", "SI-SDR improvement (dB)", False, 2,
           headline=False),
)}
HEADLINE = tuple(name for name, m in METRICS.items() if m.headline)
LOWER_IS_BETTER = frozenset(name for name, m in METRICS.items() if m.lower_is_better)


def _num(x: Any) -> float:
    """``x`` as a float; None and anything unreadable become nan."""
    if x is None or isinstance(x, (str, bytes)):
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _field(line: Mapping, *path: str) -> Any:
    cur: Any = line
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            raise ValueError(f"score line {line.get('id', '?')!r} of system "
                             f"{line.get('system', '?')!r} has no {'.'.join(path)}")
        cur = cur[key]
    return cur


def item_table(lines: Sequence[Mapping], absent: Any = None, minutes: Any = None) -> Table:
    """Per-item columns of one system's score lines, in the order given.

    ``absent`` (target-absent flags) and ``minutes`` (item lengths in minutes) default to the
    lines' own ``target_absent`` and ``length_s``; the report passes them from the manifest so
    that every system shares them. Words and edits of target-absent items are left out of the
    columns that feed per-target-word rates, and frame counts of items without target activity
    are left out of the activity columns.
    """
    n = len(lines)
    if absent is None:
        absent = [bool(line.get("target_absent", False)) for line in lines]
    if minutes is None:
        minutes = [_num(_field(line, "length_s")) / 60.0 for line in lines]
    absent = np.asarray(absent, dtype=bool).reshape(n)
    minutes = np.asarray(minutes, dtype=np.float64).reshape(n)
    present = ~absent

    def col(*path: str) -> np.ndarray:
        return np.array([_num(_field(line, *path)) for line in lines], dtype=np.float64)

    def optional(key: str) -> np.ndarray:
        return np.array([_num(line.get(key)) for line in lines], dtype=np.float64)

    errors = (col("target_words_lost", "sub") + col("target_words_lost", "del")
              + col("target_words_lost", "ins"))
    n_ref = col("target_words_lost", "n_ref")
    wrong = col("wrong_voice_words", "count")
    words = col("talk_over_silence", "words")
    silent_min = col("talk_over_silence", "silent_s") / 60.0
    tp, fp, fn = (col("activity_f1", k) for k in ("tp", "fp", "fn"))
    active = (tp + fn) > 0
    table: Table = {
        "absent": absent.astype(np.float64),
        "minutes": minutes,
        "errors": np.where(present, errors, 0.0),
        "n_ref": np.where(present, n_ref, 0.0),
        "wrong_voice_present": np.where(present, wrong, 0.0),
        "wrong_voice": wrong,
        "silence_words": words,
        "silent_minutes": silent_min,
        "absent_words": np.where(absent, words, 0.0),
        "absent_minutes": np.where(absent, silent_min, 0.0),
        "tp": np.where(active, tp, 0.0),
        "fp": np.where(active, fp, 0.0),
        "fn": np.where(active, fn, 0.0),
        "empty": np.array([bool(line.get("empty_output")) or bool(line.get("missing"))
                           for line in lines], dtype=np.float64),
    }
    for name, key in (("voice_match", "voice_match"), ("quality", "quality"),
                      ("si_sdr_i", "si_sdr_improvement")):
        values = optional(key)
        ok = np.isfinite(values)
        table[name] = np.where(ok, values, 0.0)
        table[f"{name}_n"] = ok.astype(np.float64)
    return table


def metric_points(table: Table, weights: Any = None) -> dict[str, float]:
    """Every metric of one table, with item ``weights`` (default all 1)."""
    w = np.ones(len(table["minutes"])) if weights is None else np.asarray(weights, np.float64)
    return {name: float(fn(table, w)[0]) for name, fn in METRIC_FNS.items()}


# Bootstrap

def resample_counts(n_items: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """``n`` multinomial resamples of ``n_items`` item ids: how often each id is drawn.

    One ``numpy.random.default_rng(seed)`` makes every draw. Shape ``(n, n_items)``.
    """
    if n_items <= 0:
        return np.zeros((n, 0))
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=n).astype(np.float64)


def _interval(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"lo": math.nan, "hi": math.nan}
    lo, hi = np.percentile(v, PERCENTILES)
    return {"lo": float(lo), "hi": float(hi)}


def paired_bootstrap(items_by_system: Mapping[str, Any],
                     metric_fn: Callable[[Table, np.ndarray], np.ndarray],
                     n: int = N_BOOT, seed: int = SEED, *,
                     counts: np.ndarray | None = None) -> dict:
    """Point estimates and paired percentile intervals of one metric for every system.

    ``items_by_system`` maps a system to its items, all systems in the same item order: either
    a list of score lines or a table from ``item_table``. ``metric_fn(table, weights)`` takes a
    2-D array of item weights (one row per resample) and returns one value per row. The
    resamples come from ``resample_counts(n_items, n, seed)`` unless ``counts`` is given, and
    the same rows go to every system, so each difference is paired.

    Returns ``{"n", "seed", "items", "systems": {s: {"point", "lo", "hi"}},
    "differences": {a: {b: {"point", "lo", "hi"}}}}`` where a difference is ``a`` minus ``b``,
    for every ordered pair of distinct systems.
    """
    tables = {s: v if isinstance(v, Mapping) else item_table(list(v))
              for s, v in items_by_system.items()}
    sizes = {len(t["minutes"]) for t in tables.values()}
    if len(sizes) > 1:
        raise ValueError(f"systems cover different numbers of items: {sorted(sizes)}")
    n_items = sizes.pop() if sizes else 0
    if counts is None:
        counts = resample_counts(n_items, n, seed)
    if counts.shape[1:] != (n_items,):
        raise ValueError(f"resamples cover {counts.shape[1:]} items, the systems {n_items}")
    ones = np.ones((1, n_items))
    points = {s: float(np.asarray(metric_fn(t, ones))[0]) for s, t in tables.items()}
    draws = {s: np.asarray(metric_fn(t, counts), dtype=np.float64) for s, t in tables.items()}
    return {
        "n": int(counts.shape[0]),
        "seed": seed,
        "items": n_items,
        "systems": {s: {"point": points[s], **_interval(draws[s])} for s in tables},
        "differences": {a: {b: {"point": points[a] - points[b],
                                **_interval(draws[a] - draws[b])}
                            for b in tables if b != a}
                        for a in tables},
    }


def clearly_better(a: float, b: float, metric: str, interval: Sequence[float]) -> bool:
    """Whether a system at ``a`` is clearly better than one at ``b`` on ``metric``.

    ``interval`` is the (lo, hi) percentile interval of the paired difference ``a - b``. True
    when ``a`` is better in the metric's direction, the difference is at least
    ``CLEAR_RELATIVE`` of the worse value, and the interval lies wholly on the better side of
    zero. Anything undefined gives False.
    """
    a, b = _num(a), _num(b)
    lo, hi = (_num(x) for x in interval)
    if not all(math.isfinite(x) for x in (a, b, lo, hi)):
        return False
    if metric in LOWER_IS_BETTER:
        better, excludes = a < b, hi < 0.0
    else:
        better, excludes = a > b, lo > 0.0
    # When a is better, b is the worse value.
    big = abs(a - b) >= CLEAR_RELATIVE * abs(b) - 1e-12
    return bool(better and big and excludes)


# Reading a build

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_word_timings(build_dir: str | Path) -> dict | None:
    """``word_timings.json`` of a build (utterance id to ``[start, end, word]`` rows), or None."""
    path = Path(build_dir) / WORD_TIMINGS
    blob = _read_json(path) if path.is_file() else None
    return blob if isinstance(blob, dict) else None


def load_scores(build_dir: str | Path, folder: str = SCORES) -> dict[str, list[dict]]:
    """Every ``<folder>/<system>.jsonl`` of a build as ``{system: lines}``, the file stem naming
    the system. Partly written files (``.jsonl.tmp<pid>``) are not read."""
    folder = Path(build_dir) / folder
    out: dict[str, list[dict]] = {}
    if folder.is_dir():
        for path in sorted(folder.glob("*.jsonl")):
            text = path.read_text(encoding="utf-8")
            out[path.stem] = [json.loads(line) for line in text.splitlines() if line.strip()]
    return out


def system_order(names: Sequence[str]) -> list[str]:
    """The controls first, in ``CONTROLS`` order, then the other systems by name."""
    names = list(dict.fromkeys(names))
    return [c for c in CONTROLS if c in names] + sorted(n for n in names if n not in CONTROLS)


def _row_id(row: Mapping) -> str:
    return str(row.get("id", row.get("item_id", "")))


def bucket_label(row: Mapping, key: str) -> str:
    """The label of ``row`` in bucket ``key`` as text; ``"unknown"`` when it has none."""
    buckets = row.get("buckets")
    if isinstance(buckets, Mapping) and key in buckets:
        value = buckets[key]
    elif key in ("target_absent", "long"):
        value = bool(row.get(key, False))
    elif key == "capture" and isinstance(row.get("capture"), Mapping):
        value = row["capture"].get("branch")
    else:
        value = None
    return "unknown" if value is None else str(value)


def _value_order(key: str, values: Sequence[str]) -> list[str]:
    known = _VALUE_ORDER.get(key, ())

    def sort_key(v: str) -> tuple:
        if v in known:
            return (0, known.index(v), 0.0, "")
        if v == "unknown":
            return (3, 0, 0.0, "")
        try:
            return (1, 0, float(v), "")
        except ValueError:
            return (2, 0, 0.0, v)

    return sorted(set(values), key=sort_key)


def _has(k: int) -> str:
    return "has" if k == 1 else "have"


def _lines(k: int) -> str:
    return "line" if k == 1 else "lines"


def _items(k: int) -> str:
    return "item has" if k == 1 else "items have"


def _empty_line(row: Mapping, timings: Mapping | None) -> dict:
    ref = score.reference_from_manifest(row, timings)
    return {"id": ref.item_id, "length_s": ref.length_s, "long": ref.long,
            "target_absent": ref.target_absent, "empty_output": True, "missing": True,
            **score.empty_output_scores(ref)}


# Reweighting

def natural_weights(rows: Sequence[Mapping], selection: Any) -> tuple[np.ndarray, dict]:
    """Item weights that give the reported items the capture mix of the candidate pool.

    The pool is every planned candidate before the quotas. Its capture counts come from
    ``selection["pool"]["capture"]``, else from the ``buckets`` of a ``candidates`` list. An
    item weighs (pool share of its capture value) / (share of that value among ``rows``); a
    value missing from the pool weighs 0. Without a pool every weight is 1 and the returned
    info says reweighting is unavailable.
    """
    n = len(rows)
    labels = [bucket_label(r, "capture") for r in rows]
    info: dict[str, Any] = {"available": False, "dimension": "capture", "source": None,
                            "pool_items": 0, "values": {}, "notes": []}
    pool: dict[str, int] | None = None
    if isinstance(selection, Mapping):
        counted = selection.get("pool")
        if isinstance(counted, Mapping) and isinstance(counted.get("capture"), Mapping):
            pool = {str(k): int(v) for k, v in counted["capture"].items()}
            info["source"] = "the capture counts of every planned candidate in selection.json"
        elif isinstance(selection.get("candidates"), list):
            caps = [c["buckets"]["capture"] for c in selection["candidates"]
                    if isinstance(c, Mapping) and isinstance(c.get("buckets"), Mapping)
                    and c["buckets"].get("capture") is not None]
            if caps:
                pool = dict(Counter(str(c) for c in caps))
                info["source"] = "the bucket labels of the candidates in selection.json"
    if not pool or sum(pool.values()) <= 0:
        info["reason"] = ("selection.json is missing or holds no capture labels for the "
                          "candidate pool")
        return np.ones(n), info
    selected = Counter(labels)
    if not any(pool.get(v, 0) > 0 for v in selected):
        info["reason"] = "no reported item has a capture value found in the candidate pool"
        return np.ones(n), info
    total = sum(pool.values())
    weight: dict[str, float] = {}
    for v in _value_order("capture", list(pool) + list(selected)):
        pool_share = pool.get(v, 0) / total
        sel_share = selected.get(v, 0) / n if n else 0.0
        # A value no reported item holds has no weight to give.
        w = pool_share / sel_share if sel_share > 0 else math.nan
        weight[v] = w
        info["values"][v] = {"pool": pool.get(v, 0), "pool_share": pool_share,
                             "selected": selected.get(v, 0), "selected_share": sel_share,
                             "weight": w}
        if sel_share > 0 and pool.get(v, 0) == 0:
            info["notes"].append(f"capture value {v} is not in the candidate pool, so its "
                                 f"{selected[v]} reported items weigh 0.")
        if sel_share == 0 and pool.get(v, 0) > 0:
            info["notes"].append(f"capture value {v} holds {pool_share:.1%} of the pool but no "
                                 "reported item, so the weighted mix leaves it out.")
    info.update(available=True, pool_items=total)
    return np.array([weight[v] for v in labels], dtype=np.float64), info


# The report

def build_report(build_dir: str | Path, n_boot: int = N_BOOT, seed: int = SEED, *,
                 date: str | None = None, quick: bool = False) -> dict:
    """Everything the report holds, as a dict with nan for undefined numbers.

    With ``quick`` the report covers only the quick subset (``subsets.quick_ids``). It reads the
    full ``scores/`` files restricted to those items and the ``scores/quick/`` files, which take
    precedence for a system that has both, so a quick run is compared item by item with every
    system scored on the full set.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    build = Path(build_dir)
    rows = score.read_manifest(build / "manifest.jsonl")
    ids = [_row_id(r) for r in rows]
    if any(not i for i in ids):
        raise ValueError("a manifest row has no id")
    if len(set(ids)) != len(ids):
        raise ValueError("the manifest repeats an item id")
    in_manifest = set(ids)
    total = len(rows)
    if quick:
        keep = set(quick_ids(rows))
        rows = [r for r in rows if _row_id(r) in keep]
        ids = [_row_id(r) for r in rows]
    n = len(rows)
    absent = np.array([bool(r.get("target_absent", False)) for r in rows], dtype=bool)
    long = np.array([bucket_label(r, "long") == "True" for r in rows], dtype=bool)
    minutes = np.array([score._item_length(r) / 60.0 for r in rows], dtype=np.float64)
    selection = _read_json(build / "selection.json")
    build_info = _read_json(build / "build_info.json")
    timings = load_word_timings(build)

    raw = load_scores(build)
    if quick:
        raw.update(load_scores(build, QUICK_SCORES))
    order = system_order(list(raw))
    known = set(ids)
    empty_lines: dict[str, dict] = {}
    notes: list[str] = []
    if quick:
        notes.append(f"Quick subset: {n} of {total} items, every {QUICK_EVERY}th item of each "
                     "stratum. Compare these numbers only with other quick-subset numbers.")
    tables: dict[str, Table] = {}
    systems: dict[str, dict] = {}
    for s in order:
        by_id: dict[str, dict] = {}
        extra = duplicate = 0
        for line in raw[s]:
            i = str(line.get("id", ""))
            if i not in known:
                extra += i not in in_manifest
                continue
            duplicate += i in by_id
            by_id[i] = line
        missing = [i for i in ids if i not in by_id]
        lines = []
        for row, i in zip(rows, ids):
            if i in by_id:
                lines.append(by_id[i])
            else:
                if i not in empty_lines:
                    empty_lines[i] = _empty_line(row, timings)
                lines.append(empty_lines[i])
        tables[s] = item_table(lines, absent, minutes)
        empty = int(tables[s]["empty"].sum())
        systems[s] = {"control": s in CONTROLS, "lines": len(by_id), "missing_lines": len(missing),
                      "missing_ids": missing,
                      "empty_outputs": {"count": empty, "share": empty / n if n else math.nan},
                      "metrics": {}, "reweighted": {}}
        if missing:
            notes.append(f"{s}: {len(missing)} of {n} items {_has(len(missing))} no score "
                         "line and count as empty output.")
        if extra:
            notes.append(f"{s}: {extra} score {_lines(extra)} name items that are not in the "
                         "manifest and are left out.")
        if duplicate:
            notes.append(f"{s}: {duplicate} {_items(duplicate)} more than one score line; "
                         "the last one is used.")

    counts = resample_counts(n, n_boot, seed)
    differences: dict[str, dict] = {}
    for name, fn in METRIC_FNS.items():
        boot = paired_bootstrap(tables, fn, n_boot, seed, counts=counts)
        for s in order:
            systems[s]["metrics"][name] = boot["systems"][s]
        differences[name] = boot["differences"]

    plain = [s for s in order if s not in CONTROLS]
    verdicts = []
    for name in HEADLINE:
        for a in plain:
            for b in plain:
                if a == b:
                    continue
                pa = systems[a]["metrics"][name]["point"]
                pb = systems[b]["metrics"][name]["point"]
                d = differences[name][a][b]
                if clearly_better(pa, pb, name, (d["lo"], d["hi"])):
                    verdicts.append({"better": a, "worse": b, "metric": name,
                                     "better_value": pa, "worse_value": pb, "difference": d})

    weights, reweighting = natural_weights(rows, selection)
    for s in order:
        systems[s]["reweighted"] = metric_points(tables[s], weights)
    notes.extend(reweighting["notes"])

    buckets: dict[str, list[dict]] = {}
    for key in BUCKET_KEYS:
        labels = np.array([bucket_label(r, key) for r in rows], dtype=object)
        entries = []
        for value in _value_order(key, list(labels)):
            mask = (labels == value).astype(np.float64)
            k = int(mask.sum())
            entry: dict[str, Any] = {"value": value, "items": k,
                                     "skipped": k < MIN_BUCKET_ITEMS}
            if k >= MIN_BUCKET_ITEMS:
                entry["systems"] = {s: {**metric_points(tables[s], mask),
                                        "empty_outputs": int(tables[s]["empty"] @ mask)}
                                    for s in order}
            entries.append(entry)
        buckets[key] = entries

    seed_of_build = None
    for blob in (build_info, selection):
        if isinstance(blob, Mapping) and blob.get("seed") is not None:
            seed_of_build = int(blob["seed"])
            break
    if seed_of_build is None:
        seeds = {r.get("seed") for r in rows if r.get("seed") is not None}
        seed_of_build = int(seeds.pop()) if len(seeds) == 1 else None

    report: dict[str, Any] = {
        "title": "RightVoice report, quick subset" if quick else "RightVoice report",
        "subset": {"name": "quick", "items": n, "of": total} if quick else None,
        "date": date or datetime.date.today().isoformat(),
        "build": {"dir": str(build_dir), "seed": seed_of_build, "items": n,
                  "clips": int(n - long.sum()), "long": int(long.sum()),
                  "target_absent": int(absent.sum()),
                  "minutes": float(minutes.sum()),
                  "word_timings": timings is not None},
        "bootstrap": {"resamples": int(n_boot), "seed": int(seed),
                      "percentiles": list(PERCENTILES), "paired": True},
        "rule": {"relative": CLEAR_RELATIVE, "interval_excludes_zero": True},
        "metrics": {name: {"label": m.label, "header": m.header,
                           "lower_is_better": m.lower_is_better, "headline": m.headline}
                    for name, m in METRICS.items()},
        "order": order,
        "systems": systems,
        "differences": differences,
        "clearly_better": verdicts,
        "reweighting": reweighting,
        "buckets": buckets,
        "min_bucket_items": MIN_BUCKET_ITEMS,
        "notes": notes,
    }
    report["controls"] = {
        "present": [c for c in CONTROLS if c in systems],
        "missing": [c for c in CONTROLS if c not in systems],
        "expected": {c: {"metric": m, "limit": lim, "bound": bound}
                     for c, (m, lim, bound) in CONTROL_LIMITS.items()},
        "warnings": controls_pass(report),
    }
    return report


def controls_pass(report: Mapping) -> list[str]:
    """Warnings for every scored control outside its expected range; empty when all pass.

    The clean target should lose at most 15% of target words, the mixture should carry at
    least 5 wrong-voice words per 100 target words, and silence should say nothing on
    target-absent items. An undefined value is a warning too, since the check cannot run.
    Controls that were not scored give no warning.
    """
    warnings = []
    systems = report.get("systems", {})
    for name, (metric, limit, bound) in CONTROL_LIMITS.items():
        if name not in systems:
            continue
        m = METRICS[metric]
        value = _num(systems[name]["metrics"][metric]["point"])
        if not math.isfinite(value):
            warnings.append(f"{name}: {m.label} is undefined on this build, so the check "
                            "cannot run.")
            continue
        out = value > limit if bound == "max" else value < limit
        if out:
            side = "above" if bound == "max" else "under"
            warnings.append(f"{name}: {m.label} is {_fmt(m, value)}{m.unit}, {side} "
                            f"{_fmt(m, limit)}{m.unit}. {CONTROL_REASONS[name]}")
    return warnings


# Markdown

def _fmt(metric: Metric, value: Any) -> str:
    v = _num(value)
    if not math.isfinite(v):
        return "n/a"
    return _fixed(v * metric.scale, metric.decimals)


def _fixed(value: Any, decimals: int) -> str:
    v = _num(value)
    return f"{round(v, decimals) + 0.0:.{decimals}f}" if math.isfinite(v) else "n/a"


def _with_interval(metric: Metric, cell: Mapping) -> str:
    point = _fmt(metric, cell.get("point"))
    lo, hi = _fmt(metric, cell.get("lo")), _fmt(metric, cell.get("hi"))
    if point == "n/a" or lo == "n/a" or hi == "n/a":
        return point
    return f"{point} ({lo} to {hi})"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], left: int = 1) -> list[str]:
    align = ["---"] * left + ["---:"] * (len(header) - left)
    out = ["| " + " | ".join(_cell(h) for h in header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def _empty_cell(empty: Mapping) -> str:
    share = _num(empty.get("share"))
    pct = f"{100.0 * share:.1f}%" if math.isfinite(share) else "n/a"
    return f"{int(empty.get('count', 0))} ({pct})"


def _headline_rows(report: Mapping, names: Sequence[str]) -> list[list[str]]:
    rows = []
    for s in names:
        sysd = report["systems"][s]
        rows.append([s] + [_with_interval(METRICS[m], sysd["metrics"][m]) for m in HEADLINE]
                    + [_empty_cell(sysd["empty_outputs"])])
    return rows


def _items_phrase(b: Mapping) -> str:
    return (f"{b['items']} items ({b['clips']} clips and {b['long']} long; "
            f"{b['target_absent']} target-absent)")


def render_markdown(report: Mapping) -> str:
    """``report.md`` from a report dict. Plain Markdown, no HTML."""
    b = report["build"]
    boot = report["bootstrap"]
    order = list(report["order"])
    controls = [s for s in order if s in CONTROLS]
    plain = [s for s in order if s not in CONTROLS]
    header = ["System"] + [METRICS[m].header for m in HEADLINE] + ["Empty outputs"]
    out: list[str] = [f"# {report['title']}", ""]
    out.append(f"Dated {report['date']}. Build `{b['dir']}`: {_items_phrase(b)}, "
               f"{len(order)} systems scored.")
    out.append("")
    out.append(f"Intervals are 95% percentile intervals from a paired bootstrap over items: "
               f"{boot['resamples']} resamples of the item ids (seed {boot['seed']}), the same "
               "resamples for every system. Every rate is a ratio of sums over items. An item "
               "without a score line counts as empty output.")
    out.append("")

    out += ["## Positive controls", ""]
    out.append("The controls check the scorer and the references rather than a system.")
    out.append("")
    out += [f"- {c} {CONTROL_NOTES[c]}" for c in CONTROLS]
    out.append("")
    if controls:
        out += _table(header, _headline_rows(report, controls))
        out.append("")
    missing = report["controls"]["missing"]
    if missing:
        out.append("Not scored: " + ", ".join(missing) + ".")
        out.append("")
    warnings = report["controls"]["warnings"]
    if warnings:
        out.append("Control warnings:")
        out.append("")
        out += [f"- {w}" for w in warnings]
    elif controls:
        out.append("Every scored control is in its expected range.")
    out.append("")

    out += [f"## Results, {report['date']}", ""]
    if plain:
        out += _table(header, _headline_rows(report, plain))
    else:
        out.append("No systems besides the controls were scored.")
    out.append("")

    out += ["## Clearly better", ""]
    rule = report["rule"]
    out.append(f"A system is clearly better than another on a metric when the difference is at "
               f"least {rule['relative']:.0%} of the worse value and its interval excludes zero. "
               "Controls are left out of these comparisons.")
    out.append("")
    verdicts = report["clearly_better"]
    if verdicts:
        for v in verdicts:
            m = METRICS[v["metric"]]
            out.append(f"- {v['better']} is clearly better than {v['worse']} on {m.label}: "
                       f"{_fmt(m, v['better_value'])}{m.unit} against "
                       f"{_fmt(m, v['worse_value'])}{m.unit}.")
    else:
        out.append("No pair of systems is clearly apart on any metric.")
    out.append("")

    out += ["## Reweighted to the natural pool", ""]
    rw = report["reweighting"]
    if rw["available"]:
        out.append("The items were selected with quotas per bucket value, so some capture "
                   "values are over-represented. Here each item weighs the pool share of its "
                   "capture value over that value's share among the reported items, which "
                   f"restores the capture mix of the {rw['pool_items']} planned candidates. "
                   "Other quota dimensions keep their selected mix. Point estimates only.")
        out.append("")
        out += _table(["Capture", "Pool share", "Reported share", "Weight"],
                      [[v, f"{100.0 * _num(d['pool_share']):.1f}%",
                        f"{100.0 * _num(d['selected_share']):.1f}%", _fixed(d["weight"], 3)]
                       for v, d in rw["values"].items()])
        out.append("")
        out += _table(["System"] + [METRICS[m].header for m in HEADLINE],
                      [[s] + [_fmt(METRICS[m], report["systems"][s]["reweighted"][m])
                              for m in HEADLINE] for s in order])
    else:
        out.append(f"Reweighting unavailable: {rw.get('reason', 'no candidate pool')}. The "
                   "numbers above are over the selected items as they are.")
    out.append("")

    out += ["## By bucket", ""]
    out.append(f"Point estimates over the items holding each bucket value. Values with fewer "
               f"than {report['min_bucket_items']} items are listed with their count and not "
               "scored.")
    out.append("")
    bucket_header = (["Value", "Items", "System"] + [METRICS[m].header for m in HEADLINE]
                     + ["Empty outputs"])
    for key, entries in report["buckets"].items():
        out += [f"### {key}", ""]
        rows = []
        for e in entries:
            if e["skipped"]:
                continue
            for s in order:
                p = e["systems"][s]
                rows.append([e["value"], str(e["items"]), s]
                            + [_fmt(METRICS[m], p[m]) for m in HEADLINE]
                            + [str(p["empty_outputs"])])
        if rows:
            out += _table(bucket_header, rows, left=3)
            out.append("")
        small = [f"{e['value']} ({e['items']} items)" for e in entries if e["skipped"]]
        if small:
            out.append(f"Fewer than {report['min_bucket_items']} items, not scored: "
                       + ", ".join(small) + ".")
            out.append("")

    out += ["## Notes", ""]
    notes = [*report["notes"], SI_SDR_NOTE]
    out += [f"- {note}" for note in notes]
    out.append("")

    out += ["---", ""]
    seed = b["seed"] if b["seed"] is not None else "unknown"
    out.append(f"Build seed {seed}. {_items_phrase(b)}. Report dated {report['date']}, "
               f"{boot['resamples']} bootstrap resamples with seed {boot['seed']}.")
    return "\n".join(out) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_report(build_dir: str | Path, out_md: str | Path | None = None,
                 out_json: str | Path | None = None, n_boot: int = N_BOOT, seed: int = SEED,
                 *, date: str | None = None, quick: bool = False) -> dict:
    """Write ``report.md`` and ``report.json`` (default into the build folder), or
    ``report-quick.md`` and ``report-quick.json`` for the quick subset.

    Returns the report as written to the JSON file: undefined numbers are None.
    """
    build = Path(build_dir)
    md_path = Path(out_md) if out_md is not None else build / (QUICK_MD if quick else REPORT_MD)
    json_path = (Path(out_json) if out_json is not None
                 else build / (QUICK_JSON if quick else REPORT_JSON))
    report = build_report(build, n_boot=n_boot, seed=seed, date=date, quick=quick)
    report["files"] = {"markdown": str(md_path), "json": str(json_path)}
    safe = score.json_safe(report)
    _write_text(md_path, render_markdown(report))
    _write_text(json_path, json.dumps(safe, indent=1, allow_nan=False) + "\n")
    return safe


__all__ = [
    "BUCKET_KEYS",
    "CONTROLS",
    "HEADLINE",
    "LOWER_IS_BETTER",
    "METRICS",
    "METRIC_FNS",
    "Metric",
    "bucket_label",
    "build_report",
    "clearly_better",
    "controls_pass",
    "item_table",
    "load_scores",
    "load_word_timings",
    "metric_points",
    "natural_weights",
    "paired_bootstrap",
    "render_markdown",
    "resample_counts",
    "system_order",
    "write_report",
]
