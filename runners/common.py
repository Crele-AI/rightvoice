"""Shared pieces of the reference runners: items in, one 16 kHz output per item out.

A runner walks a build's manifest, hands each item's mixture and voice sample to its system and
writes what comes back to ``<build>/outputs/<system>/<item id>.wav``, 16 kHz mono 16-bit, cut or
padded to the mixture's length. A system that returns None for an item abstains: no file is
written, and the scorer treats a missing output as empty output. Every item a run processes
adds one JSON line to ``<build>/outputs/<system>/runs.jsonl`` with its status (done, abstained
or error), the seconds the system took and the error text; each run first adds a line naming
the system, its package version and its weights.

``--quick`` runs only the quick subset that ``rightvoice score --quick`` scores.

Items run one at a time. An output is written through a temporary file, so a
stopped run never leaves a partial wav behind, and a resumed run skips items that have one.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

RATE = 16000
ONLY = ("clips", "long")
DEVICES = ("cpu", "cuda", "mps")
RUNS_LOG = "runs.jsonl"
QUICK_EVERY = 8  # the quick subset; the same rule as rightvoice/subsets.py
PEAK = 0.99  # an output louder than full scale is scaled down to this peak, never clipped

#: ``process(mixture, sample, row)`` returns the output at 16 kHz, or None to abstain.
Process = Callable[[np.ndarray, np.ndarray, dict], "np.ndarray | None"]


# Items

def read_manifest(build_dir: str | Path) -> list[dict]:
    text = (Path(build_dir) / "manifest.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def quick_ids(rows: list[dict], every: int = QUICK_EVERY) -> list[str]:
    """The quick subset: every ``every``-th clip of each stratum in manifest order, the strata
    being the target-absent clips and the other clips by capture condition; no long files.
    The same rule as ``rightvoice.subsets.quick_ids``, which this file cannot import."""
    seen: dict[str, int] = {}
    ids = []
    for row in rows:
        if row.get("long"):
            continue
        if row.get("target_absent"):
            key = "target_absent"
        else:
            key = f"clip:{(row.get('buckets') or {}).get('capture')}"
        k = seen.get(key, 0)
        seen[key] = k + 1
        if k % every == 0:
            ids.append(str(row["id"]))
    return ids


def iter_items(build_dir: str | Path, only: str | None = None, limit: int | None = None,
               quick: bool = False) -> Iterator[tuple[str, Path, Path, dict]]:
    """``(item id, mixture path, sample path, manifest row)`` in manifest order.

    ``quick`` keeps the quick subset; ``only`` keeps the 30 s clips (``"clips"``) or the long
    files (``"long"``); ``limit`` keeps the first N items left after that.
    """
    if only not in (None, *ONLY):
        raise ValueError(f"only must be one of {ONLY} or None, not {only!r}")
    build = Path(build_dir)
    rows = read_manifest(build)
    keep = set(quick_ids(rows)) if quick else None
    n = 0
    for row in rows:
        if limit is not None and n >= limit:
            return
        if keep is not None and str(row["id"]) not in keep:
            continue
        if only is not None and bool(row.get("long", False)) != (only == "long"):
            continue
        item_id = str(row["id"])
        files = row.get("files") or {}
        mixture = build / files.get("mixture", f"items/{item_id}/mixture.wav")
        sample = build / files.get("sample", f"items/{item_id}/sample.wav")
        yield item_id, mixture, sample, row
        n += 1


# Audio

def resample(x: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """``x`` as float32 at ``rate_out``."""
    x = np.asarray(x, dtype=np.float32)
    if rate_in != rate_out and len(x):
        x = soxr.resample(x, int(rate_in), int(rate_out), quality="VHQ")
    return np.ascontiguousarray(x, dtype=np.float32)


def read_wav_16k(path: str | Path) -> np.ndarray:
    """A wav as mono float32 at 16 kHz: channels averaged, other rates resampled."""
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    return resample(x.mean(axis=1, dtype=np.float32), int(rate), RATE)


def write_wav_16k(path: str | Path, audio: np.ndarray) -> None:
    """Mono 16 kHz PCM_16, written to a temporary file and renamed into place.

    Non-finite samples become 0 and samples past full scale are clipped.
    """
    path = Path(path)
    x = np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1)), -1.0, 1.0)
    tmp = path.with_name(f"{path.name}.part")
    sf.write(tmp, x, RATE, subtype="PCM_16", format="WAV")
    tmp.replace(path)


def as_output(audio: Any, n: int) -> tuple[np.ndarray, float]:
    """A system's output as mono float32 of exactly ``n`` samples, and the gain in dB applied.

    Leading or trailing axes of size one are dropped; anything else that is not one channel is
    an error. Non-finite samples become 0. The output is cut or zero-padded to ``n``. When its
    peak is above full scale it is scaled down to ``PEAK``, so a system whose output level is
    arbitrary is not clipped.
    """
    x = np.asarray(audio, dtype=np.float32)
    x = x.squeeze() if x.ndim > 1 else x.reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {np.shape(audio)}")
    x = np.nan_to_num(x)
    x = x[:n] if len(x) >= n else np.concatenate([x, np.zeros(n - len(x), dtype=np.float32)])
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    gain_db = 0.0
    if peak > 1.0:
        x = x * np.float32(PEAK / peak)
        gain_db = 20.0 * float(np.log10(PEAK / peak))
    return x, gain_db


def in_chunks(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, sr: int,
              chunk_s: float, overlap_s: float) -> np.ndarray:
    """``fn`` over windows of ``chunk_s`` seconds that overlap by ``overlap_s``, joined with a
    linear crossfade over each overlap. Input no longer than one window goes through whole.

    For systems that cannot hold a long file in memory at once. Each window's output is cut or
    padded to the window's length.
    """
    size = round(chunk_s * sr)
    overlap = min(round(overlap_s * sr), size // 2)
    if len(x) <= size:
        return np.asarray(fn(x), dtype=np.float32).reshape(-1)
    hop = size - overlap
    starts = list(range(0, len(x) - overlap, hop))
    out = np.zeros(len(x), dtype=np.float64)
    weight = np.zeros(len(x), dtype=np.float64)
    for k, s in enumerate(starts):
        seg = x[s:s + size]
        y = np.asarray(fn(seg), dtype=np.float64).reshape(-1)
        y = y[:len(seg)] if len(y) >= len(seg) else np.pad(y, (0, len(seg) - len(y)))
        w = np.ones(len(seg))
        if k > 0 and overlap:
            w[:overlap] = np.linspace(0.0, 1.0, overlap + 2)[1:-1]
        if k < len(starts) - 1 and overlap:
            w[-overlap:] = np.linspace(1.0, 0.0, overlap + 2)[1:-1]
        out[s:s + len(seg)] += y * w
        weight[s:s + len(seg)] += w
    return (out / np.maximum(weight, 1e-9)).astype(np.float32)


# Runs

def output_dir(build_dir: str | Path, system: str) -> Path:
    """``<build>/outputs/<system>``, created."""
    if not system or system.startswith(".") or "/" in system or "\\" in system:
        raise ValueError(f"{system!r} is not a plain system name")
    folder = Path(build_dir) / "outputs" / system
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def run_system(build_dir: str | Path, system: str, process: Process, *, only: str | None = None,
               limit: int | None = None, resume: bool = True, log: bool = True,
               info: Mapping[str, Any] | None = None, quick: bool = False) -> dict[str, int]:
    """Run ``process`` over the build's items and write the outputs.

    ``info`` (package, version, weights) goes into the run's first line in ``runs.jsonl``.
    With ``resume`` an item that already has an output is skipped and gets no line; without it
    an existing output is removed before the item runs, so a failed item never keeps an older
    run's file. An exception inside one item is logged as an error and the run goes on. Returns
    counts of ``done``, ``abstained``, ``error`` and ``skipped``.
    """
    build = Path(build_dir)
    folder = output_dir(build, system)
    counts = {"done": 0, "abstained": 0, "error": 0, "skipped": 0}
    items = list(iter_items(build, only, limit, quick))
    with (folder / RUNS_LOG).open("a", encoding="utf-8") as fh:
        head = {"run": "start", "system": system, "time": _now(), "build": str(build),
                "only": only, "limit": limit, "quick": quick, "resume": resume,
                "items": len(items),
                **dict(info or {})}
        fh.write(json.dumps(head, default=str) + "\n")
        fh.flush()
        for k, (item_id, mixture_path, sample_path, row) in enumerate(items, start=1):
            dest = folder / f"{item_id}.wav"
            if resume and dest.exists():
                counts["skipped"] += 1
                continue
            dest.unlink(missing_ok=True)
            line: dict[str, Any] = {"id": item_id, "status": "error", "seconds": None,
                                    "error": None}
            t0 = time.perf_counter()
            try:
                mixture = read_wav_16k(mixture_path)
                sample = read_wav_16k(sample_path)
                line["audio_s"] = round(len(mixture) / RATE, 3)
                t0 = time.perf_counter()
                out = process(mixture, sample, row)
                line["seconds"] = round(time.perf_counter() - t0, 3)
                if out is None:
                    line["status"] = "abstained"
                else:
                    y, gain_db = as_output(out, len(mixture))
                    write_wav_16k(dest, y)
                    line["status"] = "done"
                    if gain_db:
                        line["gain_db"] = round(gain_db, 2)
            # Model code can raise anything, and one item's failure must not stop the run.
            except Exception as exc:  # noqa: BLE001
                line["seconds"] = round(time.perf_counter() - t0, 3)
                line["error"] = f"{type(exc).__name__}: {exc}"
                if log:
                    traceback.print_exc(file=sys.stderr)
            counts[line["status"]] += 1
            fh.write(json.dumps(line) + "\n")
            fh.flush()
            if log:
                print(f"[{k}/{len(items)}] {item_id} {line['status']} {line['seconds']} s",
                      file=sys.stderr)
    return counts


# Command lines

def package_version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def add_arguments(parser: argparse.ArgumentParser, build_required: bool = True) -> None:
    """The arguments every runner takes."""
    parser.add_argument("--build", type=Path, required=build_required, help="the build folder")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N items")
    parser.add_argument("--only", choices=ONLY, default=None, help="only clips or only long files")
    parser.add_argument("--quick", action="store_true",
                        help="only the quick subset that rightvoice score --quick scores")
    parser.add_argument("--device", choices=DEVICES, default=None, help="where the model runs")
    parser.add_argument("--no-resume", action="store_true",
                        help="run every item again instead of skipping items with an output")


def finish(system: str, counts: Mapping[str, int], build: Path) -> int:
    """Print the counts and return the exit code: 1 when any item failed."""
    print(f"{system}: {counts['done']} done, {counts['abstained']} abstained, "
          f"{counts['error']} errors, {counts['skipped']} skipped; "
          f"log in {build / 'outputs' / system / RUNS_LOG}")
    return 1 if counts["error"] else 0
