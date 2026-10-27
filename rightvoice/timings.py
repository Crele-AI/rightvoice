"""Word timings for the utterances a build places, from the recogniser.

``word_timings.json`` in a build maps an utterance id to ``[start, end, word]`` rows in seconds
from the start of the utterance. That is the format ``report.load_word_timings`` reads and
``score.reference_from_manifest`` uses. Each utterance the manifest references (placements, the
voice sample's pieces, the bystanders' pieces) is read from the LibriTTS-R catalogue, resampled
from its own rate to 16 kHz and recognised on its own. The words that come back with both times
are kept as recognised, with surrounding spaces stripped; the scorer normalizes them.

The file grows as the run goes: it is written through a temporary file renamed into place every
``SAVE_EVERY`` utterances and at the end, so a stopped run resumes where it stopped and ids
already in the file are skipped. ``word_timings_report.json`` beside it counts what the file
holds and compares, per utterance, the number of recognised words with the number of words in
the corpus text, both counted after ``score.normalize``.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm

from . import libritts, score
from .libritts import Utterance
from .report import WORD_TIMINGS

SAVE_EVERY = 200
RATE = 16000


# The utterances a build uses

def _ids_of(block: Any) -> set[str]:
    """Utterance ids in a sample or bystander block: its pieces, ``utterance_ids`` and
    ``utterance_id``."""
    ids: set[str] = set()
    if not isinstance(block, Mapping):
        return ids
    for piece in block.get("pieces") or []:
        if isinstance(piece, Mapping) and piece.get("utterance_id"):
            ids.add(str(piece["utterance_id"]))
    listed = block.get("utterance_ids") or []
    if isinstance(listed, str):
        listed = [listed]
    ids.update(str(u) for u in listed if u)
    if block.get("utterance_id"):
        ids.add(str(block["utterance_id"]))
    return ids


def _placements(row: Mapping) -> list:
    """The row's placements, at the top level or under ``conversation`` as the scorer allows."""
    if "placements" in row:
        return list(row["placements"] or [])
    nested = row.get("conversation")
    if isinstance(nested, Mapping):
        return list(nested.get("placements") or [])
    return []


def used_utterance_ids(build_dir: str | Path) -> set[str]:
    """Every utterance id ``manifest.jsonl`` references: placements, sample pieces and
    bystander pieces. Placements of the target on target-absent items count too."""
    ids: set[str] = set()
    for row in score.read_manifest(Path(build_dir) / "manifest.jsonl"):
        for p in _placements(row):
            if isinstance(p, Mapping) and p.get("utterance_id"):
                ids.add(str(p["utterance_id"]))
        ids |= _ids_of(row.get("sample"))
        for b in row.get("bystanders") or []:
            ids |= _ids_of(b)
    return ids


# Files

def report_path(out_path: str | Path) -> Path:
    """The report beside a timings file: ``word_timings.json`` gives
    ``word_timings_report.json``."""
    p = Path(out_path)
    return p.with_name(f"{p.stem}_report.json")


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _catalogue(catalogue: Iterable | Mapping | None, root: Path | None) -> dict[str, Utterance]:
    if catalogue is None:
        catalogue = libritts.load(root)
    rows = catalogue.values() if isinstance(catalogue, Mapping) else catalogue
    out: dict[str, Utterance] = {}
    for u in rows:
        u = u if isinstance(u, Utterance) else Utterance(**u)
        out[u.id] = u
    return out


def _read_16k(path: Path) -> np.ndarray:
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32)
    if rate != RATE and len(x):
        x = soxr.resample(x, rate, RATE, quality="VHQ")
    return np.ascontiguousarray(x, dtype=np.float32)


def _rows(result: Any) -> list[list]:
    """``[start, end, word]`` rows of the recognised words that have both times."""
    rows = []
    for w in score.asr_words(result):
        word = w["word"].strip()
        if word:
            rows.append([round(w["start"], 3), round(w["end"], 3), word])
    return rows


def _label(asr: Any) -> dict:
    return {"name": str(getattr(asr, "name", type(asr).__name__)),
            "version": str(getattr(asr, "version", ""))}


# The report

def timings_report(timings: Mapping[str, list], catalogue: Mapping[str, Utterance],
                   used: Iterable[str], asr: Any = None) -> dict:
    """Counts over a timings mapping and the word-count comparison with the corpus text.

    ``within_one_word`` is the share of timed utterances in the catalogue whose recognised word
    count is within one word of their text's, both after ``score.normalize``.
    """
    used = set(used)
    per: dict[str, dict[str, int]] = {}
    within = 0
    for uid in sorted(timings):
        u = catalogue.get(uid)
        if u is None:
            continue
        got = len(score.normalize(" ".join(str(w) for _, _, w in timings[uid])))
        want = len(score.normalize(u.text))
        per[uid] = {"recognised": got, "text": want}
        within += abs(got - want) <= 1
    return {
        "utterances_used": len(used),
        "utterances_done": len(timings),
        "utterances_left": len(used - set(timings)),
        "words": sum(len(rows) for rows in timings.values()),
        "utterances_no_words": sum(1 for rows in timings.values() if not rows),
        "not_in_catalogue": sorted(u for u in used if u not in catalogue),
        "compared": len(per),
        "within_one_word": within / len(per) if per else None,
        "normalizer": score.normalizer_name(),
        "asr": _label(asr) if asr is not None else None,
        "per_utterance": per,
    }


# The run

def word_timings(build_dir: str | Path, asr: Any, catalogue: Iterable | Mapping | None = None,
                 corpus_root: str | Path | None = None, out_path: str | Path | None = None,
                 limit: int | None = None, progress: bool = True) -> dict:
    """Recognise every utterance the build uses and write ``word_timings.json``.

    ``asr`` is anything with ``transcribe(audio, sr)`` returning timed words, such as
    ``score.WhisperASR``. ``catalogue`` is a list of ``Utterance`` (or a mapping of id to one)
    and defaults to ``libritts.load(corpus_root)``; wav paths are relative to
    ``libritts.corpus_root(corpus_root)``. Ids already in ``out_path`` (default
    ``<build>/word_timings.json``) are skipped, and ``limit`` caps how many new utterances this
    call recognises. Used ids missing from the catalogue are listed in the report and skipped.
    Writes the report beside the file and returns the whole mapping, old rows included.
    """
    build = Path(build_dir)
    out = Path(out_path) if out_path is not None else build / WORD_TIMINGS
    root = Path(corpus_root) if corpus_root is not None else None
    by_id = _catalogue(catalogue, root)
    wav_root = libritts.corpus_root(root)
    used = sorted(used_utterance_ids(build))
    timings: dict[str, list] = dict(_read_json(out) or {})
    todo = [u for u in used if u not in timings and u in by_id]
    if limit is not None:
        todo = todo[:max(0, int(limit))]

    missing = [u for u in used if u not in by_id]
    if missing:
        print(f"word timings: {len(missing)} used utterances are not in the catalogue and are "
              f"skipped, the first being {missing[0]}", file=sys.stderr)
    before = _read_json(report_path(out)) or {}
    if todo and before.get("asr") and before["asr"] != _label(asr):
        print(f"word timings: resuming a file made with {before['asr']} using {_label(asr)}",
              file=sys.stderr)

    unsaved = 0
    try:
        for uid in tqdm(todo, desc="word timings", unit="utt", disable=not progress):
            audio = _read_16k(wav_root / by_id[uid].path)
            timings[uid] = _rows(asr.transcribe(audio, RATE))
            unsaved += 1
            if unsaved >= SAVE_EVERY:
                _save(out, timings)
                unsaved = 0
    finally:
        # Runs on an error too, so whatever was recognised is kept for the resume.
        _save(out, timings)
    _save(report_path(out), timings_report(timings, by_id, used, asr))
    return timings


__all__ = [
    "RATE",
    "SAVE_EVERY",
    "WORD_TIMINGS",
    "report_path",
    "timings_report",
    "used_utterance_ids",
    "word_timings",
]
