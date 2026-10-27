"""The item builder: plan, select and render RightVoice items through the capture simulation.

An item is a two-reader conversation on an AMI timeline, captured through the capture simulation.
Every item is a pure function of ``(seed, index)``: each stage draws from
``rng.stream(seed, index, stage)``. Planning makes every draw that decides what an item holds
(window, readers, utterances, voice sample, levels, bystanders, room, background, whether the
target is absent) and reads no audio, so thousands of candidates can be planned cheaply.
Rendering plans the item again and keeps drawing from the same streams for the draws made
inside the capture functions (noise crop offsets, playback and device parameters).

Each item folder holds five 16 kHz mono 16-bit files:

- ``mixture.wav``: what a system receives.
- ``sample.wav``: the voice sample of the target, from a chapter the conversation does not use.
- ``target.wav``: the clean target stem, after the room if any, at the mixture's level.
  Silence on target-absent items.
- ``partner.wav``: the partner stem, the same way.
- ``others.wav``: the mixture minus the target and partner stems. That is the bystanders, noise
  and music after their processing, plus whatever the device chain changed in the whole
  capture, so ``mixture = target + partner + others`` holds on every item.
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import capture, libritts, noise, rooms
from . import rng as rng_mod
from .capture import SR
from .conversation import (
    Conversation,
    NotEnoughSpeech,
    Window,
    candidate_windows,
    load_meetings,
    plan_conversation,
    render_stems,
    to_dict,
    voice_sample_candidates,
)
from .libritts import Utterance
from .noise import Clip
from .rooms import Response
from .sources import data_root

DEFAULT_SEED = 20261022
ID_PREFIX = "rv1"

# Windows
CLIP_S = 30.0
HOP_S = 15.0
LONG_LENGTHS_S: tuple[float, ...] = (300.0, 360.0, 420.0, 480.0, 540.0, 600.0)
# Long item k draws from stream index LONG_STREAM_BASE + k, so it never shares draws with clip k.
LONG_STREAM_BASE = 1_000_000

# Readers and the voice sample
SAME_SEX_P = 0.5
READER_REDRAWS = 5
SAMPLE_BUCKETS_S: tuple[float, ...] = (3.2, 5.0, 10.0)
SAMPLE_RANGE_S = (3.2, 10.0)      # a sample chapter holds at least one utterance this long
SAMPLE_DEVICES = ("same", "other")
GAP_S = 0.3                       # silence between two joined utterances
MIN_PIECE_S = 0.5                 # shortest second piece when two utterances are joined
FADE_S = 0.01                     # raised-cosine fade where an utterance is cut
TARGET_DBFS = capture.SOURCE_LEVEL_DBFS

# Bystanders
BYSTANDER_COUNTS = (0, 1, 2)
BYSTANDER_COUNT_P = (0.6, 0.25, 0.15)
BYSTANDER_S = (2.0, 15.0)
PLAYBACK_P = 0.5

# Background and absence
MUSIC_P = 0.2
ABSENT_P = 0.08
BACKGROUND_TRIES = 5

# Capture values and their share of items. Drawn on the "rooms" stream.
CAPTURE_SHARES: dict[str, float] = {
    "near_clean": 0.15,
    "room": 0.30,
    "room_reverberant": 0.05,
    "call": 0.20,
    "laptop_opus": 0.10,
    "phone_amr_wb": 0.07,
    "phone_amr_nb": 0.07,
    "phone_g711": 0.06,
}
# Room per capture value: a simulated shoebox, a measured response, or none.
ROOM_KIND: dict[str, str | None] = {
    "near_clean": None,
    "room": "simulated",
    "room_reverberant": "measured",
    "call": None,
    "laptop_opus": "simulated",
    "phone_amr_wb": "simulated",
    "phone_amr_nb": "simulated",
    "phone_g711": "simulated",
}
# The SNR branch (and device chain check) each capture value uses.
SNR_BRANCH: dict[str, str] = {
    "near_clean": "near_clean",
    "room": "room",
    "room_reverberant": "room",
    "call": "call",
    "laptop_opus": "room",
    "phone_amr_wb": "room",
    "phone_amr_nb": "room",
    "phone_g711": "room",
}
PHONE_BRANCHES: dict[str, str] = {
    "phone_amr_wb": "amr_wb",
    "phone_amr_nb": "amr_nb",
    "phone_g711": "g711",
}

# Rooms
N_ROOMS = 600
ROOM_SOURCES = 3
MEASURED_DISTANCE_M = (0.3, 3.0)
# The capture is scaled down before the device chain when its peak is above this, so a later
# gain stage cannot push the codec input past full scale.
PRE_CHAIN_PEAK = 0.7

# Selection
MIN_PER_VALUE = 40
MIN_ABSENT = 60
CANDIDATES_PER_ITEM = 4
QUOTA_DIMS = ("overlap", "reentry", "bystanders", "capture", "sample_s", "sample_device")
BUCKET_VALUES: dict[str, tuple] = {
    "overlap": ("<5", "5-25", "25-50", ">=50"),
    "reentry": ("none", "<2", "2-5", "5-10", ">=10"),
    "bystanders": BYSTANDER_COUNTS,
    "capture": tuple(CAPTURE_SHARES),
    "sample_s": SAMPLE_BUCKETS_S,
    "sample_device": SAMPLE_DEVICES,
}

STEMS = ("mixture", "sample", "target", "partner", "others")


class PlanFailed(RuntimeError):
    """An item cannot be planned from this world; the candidate is skipped."""


# Buckets


def overlap_bucket(share: float) -> str:
    pct = 100.0 * share
    if pct < 5.0:
        return "<5"
    if pct < 25.0:
        return "5-25"
    if pct < 50.0:
        return "25-50"
    return ">=50"


def reentry_bucket(seconds: float) -> str:
    if seconds <= 0.0:
        return "none"
    if seconds < 2.0:
        return "<2"
    if seconds < 5.0:
        return "2-5"
    if seconds < 10.0:
        return "5-10"
    return ">=10"


def tir_bucket(db: float) -> str:
    return "<=0" if db <= 0.0 else ">0"


def draw_capture(rng: np.random.Generator) -> str:
    """One capture value, drawn by the shares in ``CAPTURE_SHARES``."""
    names = tuple(CAPTURE_SHARES)
    p = np.array([CAPTURE_SHARES[k] for k in names], dtype=np.float64)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def item_id(index: int, long: bool = False) -> str:
    return f"{ID_PREFIX}-long-{index:03d}" if long else f"{ID_PREFIX}-{index:05d}"


def stream_index(index: int, long: bool = False) -> int:
    return LONG_STREAM_BASE + index if long else index


# The world an item is drawn from


@dataclass
class World:
    """Everything items are drawn from, loaded once per build.

    The simulated room bank is built with ``seed``, so items planned with another seed still
    draw their shoebox rooms from this world's bank.
    """

    seed: int
    catalogue: list[Utterance]
    readers: dict[str, dict[str, str]]
    conversation_pool: list[str]
    bystander_pool: list[str]
    meetings: dict
    clip_windows: list[Window]
    long_windows: dict[float, list[Window]]
    noise_clips: list[Clip]
    music_clips: list[Clip]
    measured: list[Response]
    n_rooms: int
    corpus_root: Path
    fsd_root: Path
    but_root: Path
    bank_dir: Path
    amr_wb_available: bool
    by_id: dict[str, Utterance] = field(init=False, repr=False)
    by_reader: dict[str, list[Utterance]] = field(init=False, repr=False)
    by_chapter: dict[tuple[str, str], list[Utterance]] = field(init=False, repr=False)
    sample_chapters: dict[str, list[str]] = field(init=False, repr=False)
    target_pool: list[str] = field(init=False, repr=False)
    clips_by_id: dict[str, Clip] = field(init=False, repr=False)
    measured_rooms: list[str] = field(init=False, repr=False)
    _bank: dict[str, list[Response]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_id = {u.id: u for u in self.catalogue}
        self.by_reader = {}
        self.by_chapter = {}
        for u in sorted(self.catalogue, key=lambda u: u.id):
            self.by_reader.setdefault(u.speaker, []).append(u)
            self.by_chapter.setdefault((u.speaker, u.chapter), []).append(u)
        self.sample_chapters = {}
        for reader in self.conversation_pool:
            chapters = sorted({u.chapter for u in self.by_reader.get(reader, [])})
            self.sample_chapters[reader] = [
                ch for ch in chapters
                if voice_sample_candidates(self.by_chapter[(reader, ch)], reader, ch,
                                           *SAMPLE_RANGE_S)
            ]
        self.target_pool = [r for r in self.conversation_pool if self.sample_chapters.get(r)]
        self.clips_by_id = {c.id: c for c in [*self.noise_clips, *self.music_clips]}
        self.measured_rooms = sorted({r.room for r in self.measured})

    def sex(self, reader: str) -> str | None:
        return self.readers.get(reader, {}).get("sex") or None

    def bank(self) -> dict[str, list[Response]]:
        """Simulated responses by shoebox room, built on first use under ``bank_dir``."""
        if self._bank is None:
            responses = rooms.build_bank(self.bank_dir, seed=self.seed, n_rooms=self.n_rooms,
                                         n_sources=ROOM_SOURCES)
            bank: dict[str, list[Response]] = {}
            for resp in sorted(responses, key=lambda r: r.id):
                bank.setdefault(resp.room, []).append(resp)
            self._bank = bank
        return self._bank


def default_bank_dir() -> Path:
    return data_root() / "cache" / "rooms_bank"


def load_world(seed: int = DEFAULT_SEED, *, n_rooms: int = N_ROOMS,
               long_lengths: tuple[float, ...] = LONG_LENGTHS_S) -> World:
    """Load the catalogues, timelines and candidate windows from ``data_root()``.

    Long windows are listed for each length in ``long_lengths``; pass ``()`` when no long item
    is needed. The measured room catalogue is optional: without BUT ReverbDB the
    ``room_reverberant`` items fail to plan and are skipped.
    """
    catalogue = libritts.load()
    readers = libritts.reader_table()
    pools = libritts.eligible_speakers(catalogue)
    meetings = load_meetings()
    clip_windows = candidate_windows(meetings, CLIP_S, HOP_S)
    long_windows = {float(length): candidate_windows(meetings, float(length), HOP_S)
                    for length in long_lengths}
    clips = noise.catalogue()
    try:
        measured_all = rooms.load_measured_catalogue()
    except FileNotFoundError:
        measured_all = []
    measured = rooms.measured_within(measured_all, distance=MEASURED_DISTANCE_M)
    return World(
        seed=int(seed),
        catalogue=catalogue,
        readers=readers,
        conversation_pool=list(pools["conversation"]),
        bystander_pool=list(pools["bystander"]),
        meetings=meetings,
        clip_windows=clip_windows,
        long_windows=long_windows,
        noise_clips=noise.kept(clips, "noise"),
        music_clips=noise.kept(clips, "music"),
        measured=measured,
        n_rooms=int(n_rooms),
        corpus_root=libritts.corpus_root(),
        fsd_root=noise.fsd50k_root(),
        but_root=rooms.resolve_but_root(),
        bank_dir=default_bank_dir(),
        amr_wb_available=capture.codec_available("amr_wb"),
    )


# Planning


def _pick(rng: np.random.Generator, items: list):
    return items[int(rng.integers(len(items)))]


def _draw_window(g: np.random.Generator, world: World, long: bool) -> Window:
    if long:
        lengths = [length for length in sorted(world.long_windows) if world.long_windows[length]]
        if not lengths:
            raise PlanFailed("no long candidate windows")
        windows = world.long_windows[_pick(g, lengths)]
    else:
        windows = world.clip_windows
    if not windows:
        raise PlanFailed("no candidate windows")
    return _pick(g, windows)


def _draw_readers(g: np.random.Generator, world: World) -> tuple[str, str, bool]:
    """Target, partner and whether the partner was drawn from the target's sex."""
    if not world.target_pool:
        raise PlanFailed("no reader can be a target")
    target = _pick(g, world.target_pool)
    same = bool(g.random() < SAME_SEX_P)
    others = [r for r in world.conversation_pool if r != target]
    if not others:
        raise PlanFailed("no second conversation reader")
    if same:
        sex = world.sex(target)
        alike = [r for r in others if sex is not None and world.sex(r) == sex]
        others = alike or others
    return target, _pick(g, others), same


def _piece(u: Utterance, seconds: float) -> dict:
    return {"utterance_id": u.id, "seconds": round(float(seconds), 6),
            "trimmed": bool(seconds < u.seconds - 1e-6), "text": u.text}


def _pieces(g: np.random.Generator, utts: list[Utterance], length_s: float) -> list[dict]:
    """Utterances that fill ``length_s``: one cut to length, else two joined by ``GAP_S``.

    One utterance at least ``length_s`` long is drawn and cut. When none is, an ordered pair
    whose second piece supplies at least ``MIN_PIECE_S`` is drawn and the second piece is cut.
    When no pair reaches the length, utterances are joined in a drawn order until it is
    reached.
    """
    if not utts:
        raise PlanFailed("no utterances to cut from")
    long_enough = [u for u in utts if u.seconds >= length_s]
    if long_enough:
        return [_piece(_pick(g, long_enough), length_s)]
    secs = np.array([u.seconds for u in utts], dtype=np.float64)
    need = length_s - secs - GAP_S
    ok = (need[:, None] >= MIN_PIECE_S) & (secs[None, :] >= need[:, None])
    np.fill_diagonal(ok, False)
    first, second = np.nonzero(ok)
    if first.size:
        k = int(g.integers(first.size))
        a, b = utts[int(first[k])], utts[int(second[k])]
        return [_piece(a, a.seconds), _piece(b, length_s - a.seconds - GAP_S)]
    pieces: list[dict] = []
    total = 0.0
    for i in g.permutation(len(utts)).tolist():
        u = utts[i]
        start = total + (GAP_S if pieces else 0.0)
        room = length_s - start
        if room <= 0.0:
            return pieces
        take = min(u.seconds, room)
        pieces.append(_piece(u, take))
        total = start + take
        if total >= length_s - 1e-9:
            return pieces
    raise PlanFailed(f"{len(utts)} utterances hold less than {length_s:.1f} s")


def _plan_room(g: np.random.Generator, world: World, branch: str, voices: list[str]) -> dict | None:
    kind = ROOM_KIND[branch]
    if kind is None:
        return None
    if kind == "simulated":
        index = int(g.integers(world.n_rooms))
        return {"source": "simulated", "index": index, "id": f"shoebox_{index:05d}",
                "positions": {v: k % ROOM_SOURCES for k, v in enumerate(voices)}}
    if not world.measured_rooms:
        raise PlanFailed("no measured room responses")
    name = _pick(g, world.measured_rooms)
    options = [r for r in world.measured if r.room == name]
    order = g.permutation(len(options)).tolist()
    picks = {v: options[order[k % len(options)]] for k, v in enumerate(voices)}
    return {"source": "measured", "id": name,
            "responses": {v: r.id for v, r in picks.items()},
            "rt60_s": {v: r.rt60_s for v, r in picks.items()},
            "distances_m": {v: r.distance_m for v, r in picks.items()}}


def _plan_background(g: dict[str, np.random.Generator], world: World, branch: str) -> list[dict]:
    channels = ("target", "partner") if branch == "call" else ("mixture",)
    out = []
    for channel in channels:
        music = bool(g["music"].random() < MUSIC_P)
        pool = world.music_clips if music else world.noise_clips
        if not pool:
            raise PlanFailed(f"the {'music' if music else 'noise'} pool is empty")
        clip = _pick(g["music"] if music else g["noise"], pool)
        snr = capture.draw_snr_db(SNR_BRANCH[branch], g["noise"])
        out.append({"channel": channel, "clip_id": clip.id, "pool": clip.pool, "snr_db": snr})
    return out


def _plan(seed: int, index: int, world: World,
          long: bool) -> tuple[dict, dict[str, np.random.Generator], Conversation]:
    """The plan of one item, the streams after planning, and its conversation."""
    sidx = stream_index(index, long)
    g = {stage: rng_mod.stream(seed, sidx, stage) for stage in rng_mod.STAGES
         if stage != "selection"}

    # 1, 2. conversation and utterances: window, readers, sample chapter, the filled plan
    gc = g["conversation"]
    window = _draw_window(gc, world, long)
    segments = world.meetings[window.meeting]
    conv = None
    last: Exception | None = None
    redraws = 0
    for attempt in range(READER_REDRAWS + 1):
        target, partner, same_draw = _draw_readers(gc, world)
        chapter = _pick(gc, world.sample_chapters[target])
        try:
            conv = plan_conversation(window, segments, world.catalogue, target, partner,
                                     g["utterances"], sample_chapter=chapter)
            break
        except NotEnoughSpeech as exc:
            last = exc
            redraws = attempt + 1
    if conv is None:
        raise PlanFailed(f"not enough speech after {READER_REDRAWS} reader redraws: {last}")

    # 3. sample
    gs = g["sample"]
    sample_s = float(_pick(gs, list(SAMPLE_BUCKETS_S)))
    placed = {p.utterance_id for p in conv.placements}
    unplaced = [u for u in world.by_chapter[(target, chapter)] if u.id not in placed]
    sample_pieces = _pieces(gs, unplaced, sample_s)
    sample_device = "same" if gs.random() < 0.5 else "other"

    # 4. levels
    gl = g["levels"]
    partner_gain = capture.partner_gain_db(gl)
    mixture_dbfs = capture.mixture_level(gl)

    # 5. bystanders
    gb = g["bystanders"]
    count = int(gb.choice(BYSTANDER_COUNTS, p=BYSTANDER_COUNT_P))
    pool = [r for r in world.bystander_pool if r not in (target, partner)]
    count = min(count, len(pool))
    canvas = round(window.length * SR)
    bystanders = []
    for k in gb.choice(len(pool), size=count, replace=False).tolist() if count else []:
        reader = pool[k]
        length_s = float(gb.uniform(*BYSTANDER_S))
        pieces = _pieces(gb, world.by_reader[reader], length_s)
        samples = round(length_s * SR)
        start = capture.place_bystander(canvas, samples, gb)
        bystanders.append({
            "reader": reader,
            "sex": world.sex(reader),
            "utterance_ids": [p["utterance_id"] for p in pieces],
            "pieces": pieces,
            "start_s": round(start / SR, 6),
            "start_sample": start,
            "samples": samples,
            "seconds": round(length_s, 6),
            "gain_db": capture.bystander_gain_db(gb),
            "playback": bool(gb.random() < PLAYBACK_P),
        })

    # 6. rooms: the capture value first, then the room
    gr = g["rooms"]
    drawn = draw_capture(gr)
    branch = drawn
    if drawn == "phone_amr_wb" and not world.amr_wb_available:
        branch = "phone_amr_nb"
    voices = ["target", "partner", *[f"bystander{k}" for k in range(len(bystanders))]]
    room = _plan_room(gr, world, branch, voices)

    # 7. noise and music
    background = _plan_background(g, world, branch)

    # 9. absent (8, the chain, draws only while rendering)
    absent = bool(g["absent"].random() < ABSENT_P)

    f = window.features
    plan = {
        "id": item_id(index, long),
        "index": index,
        "stream_index": sidx,
        "seed": int(seed),
        "long": bool(long),
        "window": {"meeting": window.meeting, "target": window.target,
                   "partner": window.partner, "start": window.start,
                   "length": window.length, "features": dict(f)},
        "readers": {"target": target, "partner": partner,
                    "target_sex": world.sex(target), "partner_sex": world.sex(partner),
                    "same_sex_draw": same_draw, "redraws": redraws},
        "placements": to_dict(conv)["placements"],
        "sample": {"chapter": chapter, "length_s": sample_s,
                   "utterance_ids": [p["utterance_id"] for p in sample_pieces],
                   "pieces": sample_pieces,
                   "internal_gaps_s": [GAP_S] * (len(sample_pieces) - 1),
                   "device": sample_device},
        "levels": {"target_dbfs": TARGET_DBFS, "partner_gain_db": partner_gain,
                   "mixture_dbfs": mixture_dbfs},
        "bystanders": bystanders,
        "capture": {"branch": branch, "drawn_branch": drawn, "room": room,
                    "background": background},
        "target_absent": absent,
        "buckets": {
            "overlap": overlap_bucket(f["overlap_share"]),
            "reentry": reentry_bucket(f["reentry_s"]),
            "tir_db": None,          # set when rendered; it needs the bystander levels
            "bystanders": len(bystanders),
            "same_sex": (None if world.sex(target) is None or world.sex(partner) is None
                         else world.sex(target) == world.sex(partner)),
            "capture": branch,
            "sample_s": sample_s,
            "sample_device": sample_device,
            "target_absent": absent,
            "long": bool(long),
        },
    }
    return plan, g, conv


def plan_item(seed: int, index: int, world: World, *, long: bool = False) -> dict:
    """The plan of item ``index`` as a JSON-friendly dict. Reads no audio.

    Raises ``PlanFailed`` when the world cannot supply the item.
    """
    return _plan(seed, index, world, long)[0]


# Rendering


def _fade_out(x: np.ndarray, n: int) -> None:
    n = min(n, len(x))
    if n > 0:
        x[-n:] *= (0.5 * (1.0 + np.cos(np.pi * np.arange(1, n + 1) / n))).astype(x.dtype)


def _read_utterance(world: World, utterance_id: str) -> np.ndarray:
    u = world.by_id[utterance_id]
    x, rate = sf.read(world.corpus_root / u.path, dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32)
    if rate != SR:
        x = soxr.resample(x, rate, SR, quality="VHQ")
    return np.array(x, dtype=np.float32)


def _render_pieces(world: World, pieces: list[dict], n: int) -> np.ndarray:
    """Pieces joined by ``GAP_S`` of silence, exactly ``n`` samples."""
    fade = round(FADE_S * SR)
    parts: list[np.ndarray] = []
    for k, p in enumerate(pieces):
        if k:
            parts.append(np.zeros(round(GAP_S * SR), dtype=np.float32))
        x = _read_utterance(world, p["utterance_id"])[:round(p["seconds"] * SR)].copy()
        if p["trimmed"]:
            _fade_out(x, fade)
        parts.append(x)
    y = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if y.size >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - y.size, dtype=np.float32)])


@functools.lru_cache(maxsize=32)
def _read_clip(path: str) -> np.ndarray:
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32)
    if rate != SR:
        x = soxr.resample(x, rate, SR, quality="VHQ")
    return np.array(x, dtype=np.float32)


def _background(world: World, spec: dict, reference: np.ndarray,
                g: np.random.Generator) -> np.ndarray:
    """The background clip scaled to ``spec["snr_db"]`` against ``reference``."""
    clip = world.clips_by_id[spec["clip_id"]]
    audio = _read_clip(str(world.fsd_root / clip.path))
    last: Exception | None = None
    for _ in range(BACKGROUND_TRIES):
        try:
            _, scaled = capture.add_at_snr(reference, audio, spec["snr_db"], g)
            return scaled.astype(np.float64)
        except ValueError as exc:     # a crop that is all silence: crop again
            if "silent" not in str(exc):
                raise
            last = exc
    raise RuntimeError(f"background clip {clip.id} stays silent: {last}")


def _skipped(name: str) -> dict:
    return {"stage": "codec", "codec": name, "skipped": "codec unavailable"}


def _run_stages(x: np.ndarray, g: np.random.Generator,
                stages: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Run named device stages in order. A codec stage may carry its codec and bitrate."""
    y = np.asarray(x, dtype=np.float32)
    out: list[dict] = []
    for stage in stages:
        name = stage["stage"]
        if name == "agc":
            y, params = capture.agc_drift(y, g)
        elif name == "clip":
            y, params = capture.clip_fraction(y, g)
        elif name == "packet_loss":
            y, params = capture.packet_loss(y, g)
        elif name == "codec":
            codec = stage.get("codec", capture.DEVICE_CODEC)
            if capture.codec_available(codec):
                y, params = capture.codec(y, codec, g, bitrate_kbps=stage.get("bitrate_kbps"))
            else:
                params = _skipped(codec)
        else:
            raise ValueError(f"unknown device stage {name!r}")
        out.append(params)
    return y, out


def _laptop_chain(x: np.ndarray, g: np.random.Generator, branch: str,
                  force_codec: bool = False) -> tuple[np.ndarray, list[dict]]:
    """``capture.device_chain``; with ``force_codec`` the Opus stage always runs.

    When ffmpeg has no Opus encoder the drawn stages run without the codec, which is
    recorded as skipped.
    """
    if not force_codec and capture.codec_available(capture.DEVICE_CODEC):
        return capture.device_chain(x, g, SNR_BRANCH[branch])
    probs = dict(capture.DEVICE_STAGE_PROBS)
    if force_codec:
        probs["codec"] = 1.0
    kept, _ = capture.draw_device_stages(g, probs)
    return _run_stages(x, g, [{"stage": s} for s in kept])


def _phone_chain(x: np.ndarray, g: np.random.Generator, tier: str) -> tuple[np.ndarray, list[dict]]:
    if not capture.codec_available(tier):
        return np.asarray(x, dtype=np.float32), [_skipped(tier)]
    y, params = capture.phone_tier(x, tier, g)
    return y, [params]


def _apply_chain(x: np.ndarray, g: np.random.Generator,
                 branch: str) -> tuple[np.ndarray, list[dict]]:
    """The device processing of one capture value."""
    if branch == "near_clean":
        return np.asarray(x, dtype=np.float32), []
    if branch in PHONE_BRANCHES:
        return _phone_chain(x, g, PHONE_BRANCHES[branch])
    return _laptop_chain(x, g, branch, force_codec=branch == "laptop_opus")


def _sample_chain(x: np.ndarray, g: np.random.Generator, world: World, branch: str,
                  device: str, mixture_chain: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """The voice sample's device processing, drawn after the mixture's.

    ``same`` runs the mixture's stages again, with the same codec and bitrate. ``other`` draws
    its own: a phone tier for phone captures, a device chain for the rest.
    """
    if device == "same":
        stages = [{k: s[k] for k in ("stage", "codec", "bitrate_kbps") if k in s}
                  for s in mixture_chain]
        return _run_stages(x, g, stages)
    if branch in PHONE_BRANCHES:
        drawn = _pick(g, list(capture.PHONE_TIERS))
        tier = "amr_nb" if drawn == "amr_wb" and not world.amr_wb_available else drawn
        y, params = _phone_chain(x, g, tier)
        if tier != drawn:
            params[0]["redirected_from"] = drawn
        return y, params
    return _laptop_chain(x, g, branch)


def _room_responses(world: World, room: dict, voices: list[str]) -> dict[str, tuple]:
    """Each voice's ``(response, rir, direct_index)``."""
    out = {}
    if room["source"] == "simulated":
        bank = world.bank()
        responses = bank.get(room["id"])
        if not responses:
            raise RuntimeError(f"room {room['id']} is not in the bank at {world.bank_dir}")
        for v in voices:
            resp = responses[room["positions"][v] % len(responses)]
            rir, direct = rooms.load_response(resp, world.bank_dir)
            out[v] = (resp, rir, direct)
    else:
        by_id = {r.id: r for r in world.measured}
        for v in voices:
            resp = by_id[room["responses"][v]]
            rir, direct = rooms.load_response(resp, world.but_root)
            out[v] = (resp, rir, direct)
    return out


def _db(gain: float) -> float:
    return 20.0 * math.log10(gain) if gain > 0.0 else -math.inf


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _level_gain(x: np.ndarray, target_dbfs: float) -> float:
    """The gain that brings the active-speech RMS of ``x`` to ``target_dbfs``.

    Refined like ``capture.level_to``, because a gain moves quiet frames across the activity
    threshold.
    """
    gain = 1.0
    for _ in range(4):
        level = capture.active_rms_dbfs(x * gain)
        if not math.isfinite(level) or abs(level - target_dbfs) < 1e-3:
            break
        gain *= 10.0 ** ((target_dbfs - level) / 20.0)
    return gain


def _write_wav(path: Path, x: np.ndarray) -> None:
    sf.write(path, np.asarray(x, dtype=np.float32), SR, subtype="PCM_16")


def render_item(seed: int, index: int, world: World, out_dir: Path | str, *,
                long: bool = False) -> dict:
    """Render item ``index`` into ``<out_dir>/items/<id>/`` and return its manifest row."""
    plan, g, conv = _plan(seed, index, world, long)
    branch = plan["capture"]["branch"]
    absent = plan["target_absent"]
    n = round(conv.window.length * SR)

    # Clean stems at their levels: target at -25 dBFS, partner at the drawn gain.
    stems = render_stems(conv, world.by_id, world.corpus_root)
    target = capture.level_to(stems["target"], TARGET_DBFS).astype(np.float64)
    partner = (capture.level_to(stems["partner"], TARGET_DBFS).astype(np.float64)
               * 10.0 ** (plan["levels"]["partner_gain_db"] / 20.0))
    conversation_db = capture.active_rms_dbfs(target + partner)

    # Bystanders, relative to the conversation's active RMS.
    by_tracks: list[np.ndarray] = []
    by_rows: list[dict] = []
    for b in plan["bystanders"]:
        x = _render_pieces(world, b["pieces"], b["samples"])
        playback = None
        if b["playback"]:
            x, playback = capture.playback_chain(x, g["bystanders"])
        x = capture.level_to(x, conversation_db + b["gain_db"])
        track = np.zeros(n, dtype=np.float64)
        track[b["start_sample"]:b["start_sample"] + x.size] = x
        by_tracks.append(track)
        row = {k: b[k] for k in ("reader", "sex", "utterance_ids", "pieces", "start_s",
                                 "seconds", "gain_db")}
        row["dbfs"] = _finite(capture.active_rms_dbfs(x))
        row["playback"] = playback
        by_rows.append(row)

    # Target over the louder of partner and bystanders, after gains, before the room.
    target_db = capture.active_rms_dbfs(target)
    louder = max([capture.active_rms_dbfs(partner)]
                 + [capture.active_rms_dbfs(t) for t in by_tracks])
    tir_db = target_db - louder

    # Room: each voice through its own response.
    voices = ["target", "partner", *[f"bystander{k}" for k in range(len(by_tracks))]]
    dry = dict(zip(voices, [target, partner, *by_tracks]))
    wet = dict(dry)
    room_row = None
    room = plan["capture"]["room"]
    if room is not None:
        responses = _room_responses(world, room, voices)
        for v in voices:
            _, rir, direct = responses[v]
            wet[v] = capture.reverberate(dry[v], rir, direct).astype(np.float64)
        room_row = {
            "source": room["source"],
            "id": room["id"],
            "rt60_s": responses["target"][0].rt60_s,
            "distances": {v: responses[v][0].distance_m for v in voices},
            "responses": [{"voice": v, "id": responses[v][0].id,
                           "rt60_s": responses[v][0].rt60_s,
                           "distance_m": responses[v][0].distance_m} for v in voices],
        }
    bystanders_wet = sum((wet[v] for v in voices[2:]), np.zeros(n, dtype=np.float64))

    # Background, the device chain, and the mixture.
    background_rows = []
    if branch != "call":
        spec = plan["capture"]["background"][0]
        bg = _background(world, spec, (wet["target"] + wet["partner"]).astype(np.float32),
                         g["noise"])
        background_rows.append({k: spec[k] for k in ("clip_id", "pool", "snr_db")})
        target_out = np.zeros(n) if absent else wet["target"]
        partner_out = wet["partner"]
        acoustic = target_out + partner_out + bystanders_wet + bg
        pre = capture.peak_gain(acoustic, PRE_CHAIN_PEAK)
        target_out, partner_out, acoustic = target_out * pre, partner_out * pre, acoustic * pre
        mixture, chain_rows = _apply_chain(acoustic.astype(np.float32), g["chain"], branch)
        mixture = mixture.astype(np.float64)
        clean = acoustic
        sample_reference_chain = chain_rows
    else:
        specs = plan["capture"]["background"]
        bg_t = _background(world, specs[0], target.astype(np.float32), g["noise"])
        bg_p = _background(world, specs[1], partner.astype(np.float32), g["noise"])
        background_rows = [{k: s[k] for k in ("channel", "clip_id", "pool", "snr_db")}
                           for s in specs]
        target_out = np.zeros(n) if absent else target
        partner_out = partner
        ch_t = target_out + bg_t
        ch_p = partner_out + bystanders_wet + bg_p
        pre = min(capture.peak_gain(ch_t, PRE_CHAIN_PEAK), capture.peak_gain(ch_p, PRE_CHAIN_PEAK))
        target_out, partner_out, ch_t, ch_p = (a * pre for a in (target_out, partner_out, ch_t, ch_p))
        y_t, chain_t = _apply_chain(ch_t.astype(np.float32), g["chain"], branch)
        y_p, chain_p = _apply_chain(ch_p.astype(np.float32), g["chain"], branch)
        mixture = y_t.astype(np.float64) + y_p.astype(np.float64)
        clean = ch_t + ch_p
        chain_rows = ([{"channel": "target", **s} for s in chain_t]
                      + [{"channel": "partner", **s} for s in chain_p])
        sample_reference_chain = chain_t

    # The voice sample: levelled to -25 dBFS, then its own device processing.
    sp = plan["sample"]
    sample = capture.level_to(_render_pieces(world, sp["pieces"], round(sp["length_s"] * SR)),
                              TARGET_DBFS)
    sample_dbfs = capture.active_rms_dbfs(sample)
    sample, sample_chain = _sample_chain(sample, g["chain"], world, branch, sp["device"],
                                         sample_reference_chain)
    sample_peak = capture.peak_gain(sample)
    sample = sample.astype(np.float64) * sample_peak

    # Final levels: one gain to the drawn mixture level, then one peak gain, on every stem.
    residual = mixture - clean
    level_gain = _level_gain(mixture, plan["levels"]["mixture_dbfs"])
    mixture, target_out, partner_out = (a * level_gain for a in (mixture, target_out, partner_out))
    others = mixture - target_out - partner_out
    peak = min(capture.peak_gain(a) for a in (mixture, target_out, partner_out, others))
    mixture, target_out, partner_out, others = (a * peak for a in
                                                (mixture, target_out, partner_out, others))

    out = Path(out_dir)
    folder = out / "items" / plan["id"]
    folder.mkdir(parents=True, exist_ok=True)
    audio = {"mixture": mixture, "sample": sample, "target": target_out, "partner": partner_out,
             "others": others}
    files = {}
    for name in STEMS:
        _write_wav(folder / f"{name}.wav", audio[name])
        files[name] = f"items/{plan['id']}/{name}.wav"

    buckets = dict(plan["buckets"])
    buckets["tir_db"] = tir_bucket(tir_db)
    ran = any("skipped" not in s for s in chain_rows)
    residual_db = capture.rms_db(residual) - capture.rms_db(clean) if ran else -math.inf
    sample_row = {k: sp[k] for k in ("utterance_ids", "chapter", "length_s", "device",
                                     "pieces", "internal_gaps_s")}
    sample_row.update(chain=sample_chain, dbfs_before_chain=_finite(sample_dbfs),
                      peak_gain=sample_peak)
    return {
        "id": plan["id"],
        "index": plan["index"],
        "stream_index": plan["stream_index"],
        "seed": plan["seed"],
        "long": plan["long"],
        "window": plan["window"],
        "readers": plan["readers"],
        "placements": plan["placements"],
        "sample": sample_row,
        "bystanders": by_rows,
        "capture": {
            "branch": branch,
            "drawn_branch": plan["capture"]["drawn_branch"],
            "room": room_row,
            "background": background_rows if branch == "call" else background_rows[0],
        },
        "chain": chain_rows,
        "levels": {
            "target_dbfs": TARGET_DBFS,
            "partner_gain_db": plan["levels"]["partner_gain_db"],
            "conversation_dbfs": _finite(conversation_db),
            "tir_db": _finite(tir_db),
            "pre_chain_gain": pre,
            "mixture_dbfs": plan["levels"]["mixture_dbfs"],
            "level_gain_db": _db(level_gain),
            "peak_gain": peak,
            "mixture_dbfs_final": _finite(capture.active_rms_dbfs(mixture)),
            "device_change_db": _finite(residual_db),
        },
        "target_absent": absent,
        "buckets": buckets,
        "files": files,
    }


# Selection


def _key(value) -> str:
    return str(value)


def selection_order(seed: int, indices) -> list[int]:
    """Candidates sorted by one draw each from ``rng.stream(seed, index, "selection")``."""
    keys = {i: float(rng_mod.stream(seed, i, "selection").random()) for i in indices}
    return sorted(keys, key=lambda i: (keys[i], i))


def select(buckets: dict[int, dict], seed: int, count: int, min_per_value: int = MIN_PER_VALUE,
           min_absent: int = MIN_ABSENT) -> dict:
    """Quota then fill, in selection-stream order.

    The first pass takes every candidate that holds a value of ``QUOTA_DIMS`` still below
    ``min_per_value``, or that is target-absent while fewer than ``min_absent`` are taken. The
    second pass fills up to ``count`` in the same order. Each taken item lists the quotas it
    counted towards, or ``["fill"]``.
    """
    order = selection_order(seed, sorted(buckets))
    counts = {dim: Counter() for dim in QUOTA_DIMS}
    absent = 0
    taken: list[int] = []
    reasons: dict[int, list[str]] = {}

    def take(i: int, why: list[str]) -> None:
        nonlocal absent
        taken.append(i)
        reasons[i] = why
        for dim in QUOTA_DIMS:
            counts[dim][_key(buckets[i][dim])] += 1
        absent += bool(buckets[i]["target_absent"])

    for i in order:
        if len(taken) >= count:
            break
        b = buckets[i]
        why = [f"{dim}={_key(b[dim])}" for dim in QUOTA_DIMS
               if counts[dim][_key(b[dim])] < min_per_value]
        if b["target_absent"] and absent < min_absent:
            why.append("target_absent")
        if why:
            take(i, why)
    for i in order:
        if len(taken) >= count:
            break
        if i not in reasons:
            take(i, ["fill"])

    pool = {dim: dict(sorted(Counter(_key(b[dim]) for b in buckets.values()).items()))
            for dim in QUOTA_DIMS}
    pool["target_absent"] = sum(bool(b["target_absent"]) for b in buckets.values())
    chosen = {dim: dict(sorted(c.items())) for dim, c in counts.items()}
    chosen["target_absent"] = absent
    return {"order": order, "selected": [{"index": i, "quotas": reasons[i]} for i in taken],
            "pool": pool, "counts": chosen}


# The build


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _dumps(obj, indent: int | None = None) -> str:
    return json.dumps(obj, default=_json_default, indent=indent, allow_nan=False)


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("numpy", "scipy", "soundfile", "soxr", "pyroomacoustics"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _pins() -> dict | None:
    path = data_root() / "sources" / "pins.json"
    return json.loads(path.read_text()) if path.exists() else None


def build(out_dir: Path | str, *, seed: int = DEFAULT_SEED, count: int = 1500,
          long_count: int = 30, candidates: int | None = None,
          min_per_value: int = MIN_PER_VALUE, min_absent: int = MIN_ABSENT,
          n_rooms: int = N_ROOMS, only_plan: bool = False, limit: int | None = None,
          long_lengths: tuple[float, ...] = LONG_LENGTHS_S, world: World | None = None,
          progress: bool = True) -> dict:
    """Plan candidates, select clips by quota then fill, add long items, and render them.

    Writes ``selection.json`` and ``build_info.json``, and unless ``only_plan`` the item
    folders, ``manifest.jsonl`` and ``attribution.csv``. ``limit`` renders only the first
    ``limit`` items (selected clips in selection order, then long items). Returns the build
    info.
    """
    started = time.perf_counter()
    timing: dict[str, float] = {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if world is None:
        world = load_world(seed, n_rooms=n_rooms, long_lengths=long_lengths if long_count else ())
    timing["load_s"] = time.perf_counter() - started

    # Plan the clip candidates. Only their buckets are kept.
    t0 = time.perf_counter()
    n_candidates = CANDIDATES_PER_ITEM * count if candidates is None else int(candidates)
    buckets: dict[int, dict] = {}
    failed: list[dict] = []
    redirected = 0
    for i in range(n_candidates):
        try:
            plan = plan_item(seed, i, world)
        except PlanFailed as exc:
            failed.append({"index": i, "reason": str(exc)})
            continue
        buckets[i] = plan["buckets"]
        redirected += plan["capture"]["drawn_branch"] != plan["capture"]["branch"]
    timing["plan_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    selection = select(buckets, seed, count, min_per_value, min_absent)
    timing["select_s"] = time.perf_counter() - t0

    # Long items: indices 0, 1, ... of the long list, skipping any that cannot be planned.
    t0 = time.perf_counter()
    long_items: list[dict] = []
    long_failed: list[dict] = []
    k = 0
    while len(long_items) < long_count and k < 10 * long_count + 10:
        try:
            plan = plan_item(seed, k, world, long=True)
            long_items.append({"index": k, "id": plan["id"], "stream_index": plan["stream_index"],
                               "buckets": plan["buckets"]})
        except PlanFailed as exc:
            long_failed.append({"index": k, "reason": str(exc)})
        k += 1
    timing["plan_long_s"] = time.perf_counter() - t0

    for item in selection["selected"]:
        item["id"] = item_id(item["index"])
        item["buckets"] = buckets[item["index"]]
    selection_blob = {
        "seed": int(seed),
        "count": count,
        "candidates": n_candidates,
        "min_per_value": min_per_value,
        "min_absent": min_absent,
        "planned": len(buckets),
        "failed": failed,
        "order": selection["order"],
        "selected": selection["selected"],
        "pool": selection["pool"],
        "counts": selection["counts"],
        "long": long_items,
        "long_failed": long_failed,
    }
    (out / "selection.json").write_text(_dumps(selection_blob, indent=1) + "\n")

    jobs = [(item["index"], False) for item in selection["selected"]]
    jobs += [(item["index"], True) for item in long_items]
    if limit is not None:
        jobs = jobs[:max(0, int(limit))]
    rows: list[dict] = []
    render_failed: list[dict] = []
    per_item: dict[str, list[float]] = {"clip": [], "long": []}
    item_seconds: dict[str, float] = {}
    plans_by_job = {(i, False): buckets[i] for i in buckets}
    plans_by_job.update({(item["index"], True): item["buckets"] for item in long_items})
    if not only_plan and any(ROOM_KIND[plans_by_job[job]["capture"]] == "simulated"
                             for job in jobs):
        t0 = time.perf_counter()
        world.bank()        # built once here, so item timings leave it out
        timing["room_bank_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if not only_plan:
        from tqdm import tqdm

        manifest = out / "manifest.jsonl"
        with manifest.open("w") as fh:
            for index, long in tqdm(jobs, desc="render", unit="item", disable=not progress):
                t_item = time.perf_counter()
                try:
                    row = render_item(seed, index, world, out, long=long)
                    line = _dumps(row)
                except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
                    # recorded in build_info.json; one bad item does not stop a build
                    render_failed.append({"id": item_id(index, long), "index": index,
                                          "long": long, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                per_item["long" if long else "clip"].append(time.perf_counter() - t_item)
                item_seconds[row["id"]] = round(time.perf_counter() - t_item, 3)
                rows.append(row)
                fh.write(line + "\n")
                fh.flush()
        rows.sort(key=lambda r: r["id"])
        manifest.write_text("".join(_dumps(r) + "\n" for r in rows))
        used = sorted({b["clip_id"] for r in rows for b in
                       (r["capture"]["background"] if isinstance(r["capture"]["background"], list)
                        else [r["capture"]["background"]])}, key=int)
        noise.write_attribution([world.clips_by_id[c] for c in used], out / "attribution.csv")
    timing["render_s"] = time.perf_counter() - t0
    timing["clip_mean_s"] = float(np.mean(per_item["clip"])) if per_item["clip"] else None
    timing["long_mean_s"] = float(np.mean(per_item["long"])) if per_item["long"] else None
    timing["total_s"] = time.perf_counter() - started
    timing["per_item_s"] = dict(sorted(item_seconds.items()))

    info = {
        "seed": int(seed),
        "settings": {"count": count, "long_count": long_count, "candidates": n_candidates,
                     "min_per_value": min_per_value, "min_absent": min_absent,
                     "n_rooms": world.n_rooms, "long_lengths_s": list(world.long_windows),
                     "only_plan": only_plan, "limit": limit},
        "counts": {
            "candidates": n_candidates,
            "planned": len(buckets),
            "plan_failed": len(failed),
            "selected": len(selection["selected"]),
            "long": len(long_items),
            "long_failed": len(long_failed),
            "rendered": len(rows),
            "render_failed": len(render_failed),
            "capture": dict(sorted(Counter(r["capture"]["branch"] for r in rows).items())),
            "target_absent": sum(r["target_absent"] for r in rows),
        },
        "amr_wb_available": world.amr_wb_available,
        "amr_wb_redirected": {
            "candidates": int(redirected),
            "rendered": sum(r["capture"]["drawn_branch"] != r["capture"]["branch"] for r in rows),
        },
        "pins": _pins(),
        "ffmpeg": {k: v for k, v in capture.ffmpeg_info().items() if k != "path"},
        "versions": _versions(),
        "python": sys.version,
        "timing": timing,
        "render_failures": render_failed,
    }
    (out / "build_info.json").write_text(_dumps(info, indent=1) + "\n")
    return info


__all__ = [
    "CAPTURE_SHARES",
    "PlanFailed",
    "World",
    "build",
    "draw_capture",
    "item_id",
    "load_world",
    "overlap_bucket",
    "plan_item",
    "reentry_bucket",
    "render_item",
    "select",
    "selection_order",
    "tir_bucket",
]
