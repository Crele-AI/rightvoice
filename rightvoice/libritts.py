"""A catalogue of the LibriTTS-R test and dev utterances RightVoice may place.

The corpus root holds ``<split>/<speaker>/<chapter>/<speaker>_<chapter>_<a>_<b>.wav`` with a
``.normalized.txt`` beside each wav and a ``<speaker>_<chapter>.trans.tsv`` per chapter. The
catalogue keeps one row per utterance with its duration read from the wav header, filtered to
the fixed reader list in ``sources.SPEAKERS``. Scanning is cached as JSON because it reads
every header once (about 12,000 files).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf

from .sources import SPEAKERS, data_root

#: A reader with less speech than this cannot supply turns and a voice sample from another
#: chapter, so it is never a target or partner. It may still be a bystander.
MIN_SPEAKER_SECONDS = 180.0
#: A reader with less speech than this is not placed at all.
MIN_BYSTANDER_SECONDS = 30.0


@dataclass(frozen=True)
class Utterance:
    id: str
    speaker: str
    chapter: str
    split: str
    path: str          # relative to the corpus root
    text: str          # the normalized text
    seconds: float
    sample_rate: int


def corpus_root(root: Path | None = None) -> Path:
    return root or data_root() / "sources" / "libritts_r" / "LibriTTS_R"


def scan(root: Path | None = None, speakers: frozenset[str] = SPEAKERS) -> list[Utterance]:
    """Every utterance under ``root`` whose reader is in ``speakers``, sorted by id."""
    root = corpus_root(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no corpus at {root}")
    rows: list[Utterance] = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for spk_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if spk_dir.name not in speakers:
                continue
            for wav in sorted(spk_dir.rglob("*.wav")):
                text_path = wav.with_suffix("").with_suffix(".normalized.txt")
                if not text_path.exists():
                    continue
                info = sf.info(wav)
                rows.append(Utterance(
                    id=wav.stem,
                    speaker=spk_dir.name,
                    chapter=wav.parent.name,
                    split=split_dir.name,
                    path=str(wav.relative_to(root)),
                    text=text_path.read_text(encoding="utf-8").strip(),
                    seconds=float(info.frames) / info.samplerate,
                    sample_rate=info.samplerate,
                ))
    rows.sort(key=lambda u: u.id)
    return rows


def cache_path(root: Path | None = None) -> Path:
    return data_root() / "cache" / "libritts_r_catalogue.json" if root is None else root / "catalogue.json"


def load(root: Path | None = None, *, speakers: frozenset[str] = SPEAKERS,
         rebuild: bool = False) -> list[Utterance]:
    """The catalogue, from its JSON cache when present."""
    path = cache_path(root)
    if path.exists() and not rebuild:
        return [Utterance(**row) for row in json.loads(path.read_text())]
    rows = scan(root, speakers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=0))
    return rows


def reader_table(root: Path | None = None) -> dict[str, dict[str, str]]:
    """``speakers.tsv`` at the corpus root: reader id to ``{"sex", "subset", "name"}``.

    The file comes from the corpus's doc archive. Rows are tab separated: reader, sex (F or M),
    subset, name. The header line is skipped.
    """
    path = corpus_root(root) / "speakers.tsv"
    table: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3 or not fields[0].strip().isdigit():
            continue
        table[fields[0].strip()] = {
            "sex": fields[1].strip().upper(),
            "subset": fields[2].strip(),
            "name": fields[3].strip() if len(fields) > 3 else "",
        }
    return table


def seconds_by_speaker(rows: list[Utterance]) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for r in rows:
        total[r.speaker] += r.seconds
    return dict(total)


def eligible_speakers(rows: list[Utterance]) -> dict[str, list[str]]:
    """Readers sorted into ``conversation`` (target or partner) and ``bystander`` pools.

    A reader needs ``MIN_SPEAKER_SECONDS`` of speech over at least two chapters to hold a
    conversation, because the voice sample comes from another chapter or another stretch.
    """
    seconds = seconds_by_speaker(rows)
    chapters: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        chapters[r.speaker].add(r.chapter)
    conversation = sorted(
        s for s, sec in seconds.items() if sec >= MIN_SPEAKER_SECONDS and len(chapters[s]) >= 2)
    bystander = sorted(
        s for s, sec in seconds.items() if s not in conversation and sec >= MIN_BYSTANDER_SECONDS)
    return {"conversation": conversation, "bystander": bystander}


def summary(rows: list[Utterance], readers: dict[str, dict[str, str]] | None = None) -> dict:
    seconds = seconds_by_speaker(rows)
    pools = eligible_speakers(rows)
    sex = {}
    if readers:
        for role, ids in pools.items():
            sex[role] = {k: sum(readers.get(i, {}).get("sex") == k for i in ids) for k in ("F", "M")}
    by_split: dict[str, dict] = {}
    for r in rows:
        b = by_split.setdefault(r.split, {"speakers": set(), "utterances": 0, "seconds": 0.0})
        b["speakers"].add(r.speaker)
        b["utterances"] += 1
        b["seconds"] += r.seconds
    return {
        "speakers": len(seconds),
        "utterances": len(rows),
        "hours": round(sum(seconds.values()) / 3600, 2),
        "conversation_speakers": len(pools["conversation"]),
        "bystander_speakers": len(pools["bystander"]),
        "sex_by_pool": sex,
        "splits": {k: {"speakers": len(v["speakers"]), "utterances": v["utterances"],
                       "hours": round(v["seconds"] / 3600, 2)} for k, v in sorted(by_split.items())},
    }
