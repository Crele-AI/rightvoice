"""The FSD50K clips RightVoice may use as background noise and music.

Only the eval set is used. A clip is kept when its licence is CC0 or CC BY (read per clip from
``eval_clips_info_FSD50K.json``), and sorted into a pool by its ground-truth labels, which in
FSD50K already include every ancestor label: anything under ``Human_voice`` or
``Human_group_actions`` is excluded so no stray voice enters as "noise"; ``Music`` goes to the
music pool; everything else to the noise pool. ``attribution_rows`` lists every kept clip with
its uploader and licence for the build's ``attribution.csv``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .sources import data_root, fsd50k_allowed_clips

EXCLUDED_LABELS: frozenset[str] = frozenset({"Human_voice", "Human_group_actions"})
MUSIC_LABEL = "Music"


@dataclass(frozen=True)
class Clip:
    id: str
    labels: tuple[str, ...]
    licence: str
    uploader: str
    title: str
    pool: str            # "noise", "music", or "excluded"
    reason: str = ""     # why excluded: "licence" or "voice"

    @property
    def path(self) -> str:
        return f"FSD50K.eval_audio/{self.id}.wav"


def fsd50k_root(root: Path | None = None) -> Path:
    return root or data_root() / "sources" / "fsd50k"


def pool_for(labels: tuple[str, ...]) -> str:
    if EXCLUDED_LABELS.intersection(labels):
        return "excluded"
    return "music" if MUSIC_LABEL in labels else "noise"


def catalogue(root: Path | None = None) -> list[Clip]:
    """Every eval clip with its pool, in id order; excluded clips stay listed with a reason."""
    root = fsd50k_root(root)
    info = json.loads((root / "FSD50K.metadata" / "eval_clips_info_FSD50K.json").read_text())
    allowed = fsd50k_allowed_clips(info)
    clips: list[Clip] = []
    with (root / "FSD50K.ground_truth" / "eval.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            cid = row["fname"]
            labels = tuple(row["labels"].split(","))
            meta = info.get(cid, {})
            if cid not in allowed:
                pool, reason = "excluded", "licence"
            else:
                pool = pool_for(labels)
                reason = "voice" if pool == "excluded" else ""
            clips.append(Clip(
                id=cid, labels=labels, licence=str(meta.get("license", "")),
                uploader=str(meta.get("uploader", "")), title=str(meta.get("title", "")),
                pool=pool, reason=reason))
    clips.sort(key=lambda c: int(c.id))
    return clips


def kept(clips: list[Clip], pool: str) -> list[Clip]:
    return [c for c in clips if c.pool == pool]


def summary(clips: list[Clip]) -> dict:
    out = {"total": len(clips), "noise": 0, "music": 0, "excluded_licence": 0, "excluded_voice": 0}
    for c in clips:
        if c.pool == "excluded":
            out[f"excluded_{c.reason}"] += 1
        else:
            out[c.pool] += 1
    return out


def attribution_rows(clips: list[Clip]) -> list[dict[str, str]]:
    """One row per kept clip: id, Freesound URL, uploader, licence URL, title."""
    return [
        {"clip_id": c.id, "url": f"https://freesound.org/s/{c.id}/", "uploader": c.uploader,
         "licence": c.licence, "title": c.title}
        for c in clips if c.pool != "excluded"
    ]


def write_attribution(clips: list[Clip], path: Path) -> None:
    rows = attribution_rows(clips)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["clip_id"])
        writer.writeheader()
        writer.writerows(rows)


def to_json(clips: list[Clip], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in clips], indent=0))
