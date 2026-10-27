"""Two LibriTTS-R readers talking on a turn-taking timeline taken from the AMI meetings.

The AMI manual annotations give, for every meeting participant (agent ``A`` to ``E``), the
segments they spoke in and word times inside them. A window of one meeting, seen through an
ordered pair of agents (the target and the partner), is a real turn-taking pattern: gaps,
overlaps, back-channels and long silences before the target speaks again. Each role's speech in
that window becomes a list of slots, and each slot is filled with read utterances of one reader
whose durations fit it. The result is a plan (which utterance plays when, for each role) that can
be rendered into two clean 16 kHz stems.

Activity is measured on a 50 Hz frame grid: frame ``i`` of a window starting at ``start`` covers
``[start + i * frame, start + (i + 1) * frame)`` and is active when its centre lies inside one of
the agent's segments.
"""
from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .libritts import Utterance
from .libritts import corpus_root as libritts_corpus_root
from .sources import data_root

NITE = "http://nite.sourceforge.net/"
NITE_ID = f"{{{NITE}}}id"
NITE_CHILD = f"{{{NITE}}}child"
AMI_ENCODING = "ISO-8859-1"

FRAME_S = 0.02            # 50 Hz activity frames
MIN_SLOT_S = 0.6          # shorter slots are not filled
MERGE_S = 0.15            # same-role slots closer than this become one slot
BACKCHANNEL_MAX_S = 1.0   # partner segments shorter than this, under target speech
GAP_S = 0.3               # silence between two utterances placed in one slot
TOLERANCE = 0.15          # relative duration tolerance when matching an utterance to a slot
FADE_S = 0.01             # raised-cosine fade where an utterance is cut
SAMPLE_RATE = 16000
CACHE_VERSION = 1
ROLES: tuple[str, str] = ("target", "partner")


class NotEnoughSpeech(RuntimeError):
    """A reader has no utterance left for a slot of the plan."""


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Window:
    meeting: str
    target: str               # AMI agent letter whose turns the target reader takes
    partner: str              # AMI agent letter whose turns the partner reader takes
    start: float              # seconds into the meeting
    length: float             # seconds
    features: dict


@dataclass(frozen=True)
class Slot:
    role: str                 # "target" or "partner"
    start: float              # seconds from the window start
    end: float


@dataclass(frozen=True)
class Placement:
    role: str
    utterance_id: str
    start: float              # seconds from the window start
    seconds: float            # how long it plays; shorter than the file when trimmed
    text: str                 # the utterance's full normalized text, also when trimmed
    trimmed: bool = False


@dataclass(frozen=True)
class Conversation:
    window: Window
    target_reader: str
    partner_reader: str
    placements: list[Placement]
    sample_chapter: str | None = None


# AMI annotations

def default_ami_root() -> Path:
    return data_root() / "sources" / "ami_annotations"


def list_meetings(ami_root: Path) -> list[str]:
    """Meeting ids that have at least one segments file, sorted."""
    names = {p.name.split(".", 1)[0] for p in (Path(ami_root) / "segments").glob("*.segments.xml")}
    return sorted(names)


def _read_xml(path: Path) -> ET.Element:
    # The explicit encoding wins over the declaration, so Latin-1 bytes decode the same way
    # whether or not a file declares them.
    parser = ET.XMLParser(encoding=AMI_ENCODING)
    return ET.parse(path, parser=parser).getroot()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_words(path: Path) -> tuple[dict[str, int], list[Word | None]]:
    """Every element of a words file in order: id to position, and the timed word or None.

    Punctuation, vocal sounds, gaps, disfluency markers and words without both times are None,
    so they keep their position in the id ranges the segments refer to.
    """
    if not path.exists():
        return {}, []
    index: dict[str, int] = {}
    tokens: list[Word | None] = []
    for el in _read_xml(path):
        ident = el.get(NITE_ID)
        if ident is not None:
            index[ident] = len(tokens)
        word = None
        start, end = el.get("starttime"), el.get("endtime")
        if (_local(el.tag) == "w" and start is not None and end is not None
                and el.get("punc") != "true"):
            text = (el.text or "").strip()
            if text:
                word = Word(float(start), float(end), text)
        tokens.append(word)
    return index, tokens


_HREF = re.compile(r"^(?P<file>[^#]*)#id\((?P<a>[^)]+)\)(?:\.\.id\((?P<b>[^)]+)\))?$")


def _segment_words(segment: ET.Element, words_file: str, index: dict[str, int],
                   tokens: list[Word | None], start: float, end: float) -> tuple[Word, ...]:
    """The words a segment points to through its ``nite:child`` id range.

    When the segment has no usable link, the words whose midpoint lies inside it are taken.
    """
    picked: list[Word] = []
    linked = False
    for child in segment.iter(NITE_CHILD):
        match = _HREF.match(child.get("href", ""))
        if not match or Path(match["file"]).name != words_file:
            continue
        a = index.get(match["a"])
        b = index.get(match["b"] or match["a"])
        if a is None or b is None:
            continue
        linked = True
        picked.extend(w for w in tokens[min(a, b):max(a, b) + 1] if w is not None)
    if not linked:
        picked = [w for w in tokens if w is not None and start <= (w.start + w.end) / 2 <= end]
    return tuple(picked)


def parse_meeting(ami_root: Path, meeting: str) -> dict[str, list[Segment]]:
    """Agent letter to its segments, sorted by start.

    Segment times are the transcriber's. Segments without words or with ``end <= start`` are
    dropped, and so is an agent left without segments.
    """
    root = Path(ami_root)
    out: dict[str, list[Segment]] = {}
    for seg_path in sorted((root / "segments").glob(f"{meeting}.*.segments.xml")):
        parts = seg_path.name.split(".")
        if len(parts) != 4 or parts[0] != meeting:
            continue
        agent = parts[1]
        words_file = f"{meeting}.{agent}.words.xml"
        index, tokens = _read_words(root / "words" / words_file)
        segments: list[Segment] = []
        for el in _read_xml(seg_path):
            if _local(el.tag) != "segment":
                continue
            s, e = el.get("transcriber_start"), el.get("transcriber_end")
            if s is None or e is None:
                continue
            start, end = float(s), float(e)
            if end <= start:
                continue
            words = _segment_words(el, words_file, index, tokens, start, end)
            if words:
                segments.append(Segment(start, end, words))
        if segments:
            segments.sort(key=lambda g: (g.start, g.end))
            out[agent] = segments
    return out


def timeline_cache_path() -> Path:
    return data_root() / "cache" / "ami_timelines.json"


def _meetings_to_json(meetings: Mapping[str, Mapping[str, list[Segment]]]) -> dict:
    return {
        meeting: {
            agent: [[g.start, g.end, [[w.start, w.end, w.text] for w in g.words]] for g in segs]
            for agent, segs in agents.items()
        }
        for meeting, agents in meetings.items()
    }


def _meetings_from_json(blob: dict) -> dict[str, dict[str, list[Segment]]]:
    return {
        meeting: {
            agent: [Segment(s, e, tuple(Word(ws, we, t) for ws, we, t in words))
                    for s, e, words in segs]
            for agent, segs in agents.items()
        }
        for meeting, agents in blob.items()
    }


def load_meetings(ami_root: Path | None = None,
                  rebuild: bool = False) -> dict[str, dict[str, list[Segment]]]:
    """Every meeting's segments by agent, from the JSON cache when it was built from ``ami_root``.

    The cache lives under ``data_root()/cache``, never under the annotations directory.
    """
    root = Path(ami_root) if ami_root is not None else default_ami_root()
    path = timeline_cache_path()
    source = str(root.resolve())
    if path.exists() and not rebuild:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("version") == CACHE_VERSION and blob.get("ami_root") == source:
            return _meetings_from_json(blob["meetings"])
    if not (root / "segments").is_dir():
        raise FileNotFoundError(f"no AMI segments under {root}")
    meetings = {m: parse_meeting(root, m) for m in list_meetings(root)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"version": CACHE_VERSION, "ami_root": source,
                               "meetings": _meetings_to_json(meetings)}), encoding="utf-8")
    tmp.replace(path)
    return meetings


# Activity and window features

def _frame_span(start: float, end: float, frame: float) -> tuple[int, int]:
    """Frames whose centre lies in ``[start, end)``, with times relative to frame 0."""
    i0 = math.ceil(start / frame - 0.5 - 1e-9)
    i1 = math.ceil(end / frame - 0.5 - 1e-9)
    return i0, i1


def activity(segments: Iterable[Segment], start: float, length: float,
             frame: float = FRAME_S) -> np.ndarray:
    """Boolean frame activity of one agent in ``[start, start + length)``."""
    n = round(length / frame)
    act = np.zeros(n, dtype=bool)
    for seg in segments:
        i0, i1 = _frame_span(seg.start - start, seg.end - start, frame)
        i0, i1 = max(i0, 0), min(i1, n)
        if i1 > i0:
            act[i0:i1] = True
    return act


def _bounds(segments: Sequence[Segment]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.fromiter((g.start for g in segments), dtype=np.float64, count=len(segments))
    ends = np.fromiter((g.end for g in segments), dtype=np.float64, count=len(segments))
    return starts, ends


def _window_features(act_t: np.ndarray, act_p: np.ndarray,
                     bounds_t: tuple[np.ndarray, np.ndarray],
                     bounds_p: tuple[np.ndarray, np.ndarray],
                     start: float, length: float, frame: float) -> dict:
    n = len(act_t)
    end = start + length
    t_frames = int(act_t.sum())
    both = int((act_t & act_p).sum())
    on = np.flatnonzero(act_t)
    # Longest silence that the target ends by speaking again; it must also start after target
    # speech inside the window, so a silence running in from before the window does not count.
    longest = int((np.diff(on) - 1).max()) if len(on) > 1 else 0
    t_start, t_end = bounds_t
    p_start, p_end = bounds_p
    target_turns = int(((t_start < end) & (t_end > start)).sum())
    partner_turns = int(((p_start < end) & (p_end > start)).sum())
    backchannels = 0
    short = (p_end - p_start < BACKCHANNEL_MAX_S) & (p_start >= start) & (p_end <= end)
    for s, e in zip(p_start[short], p_end[short]):
        i0, i1 = _frame_span(s - start, e - start, frame)
        if i1 <= i0:  # shorter than a frame: use the frame holding its midpoint
            i0 = int(((s + e) / 2 - start) // frame)
            i1 = i0 + 1
        i0, i1 = max(i0, 0), min(i1, n)
        if i1 > i0 and act_t[i0:i1].all():
            backchannels += 1
    return {
        "target_share": round(t_frames / n, 4) if n else 0.0,
        "partner_share": round(int(act_p.sum()) / n, 4) if n else 0.0,
        "overlap_share": round(both / t_frames, 4) if t_frames else 0.0,
        "reentry_s": round(longest * frame, 2),
        "target_turns": target_turns,
        "partner_turns": partner_turns,
        "backchannels": backchannels,
    }


def features(seg_t: Sequence[Segment], seg_p: Sequence[Segment], start: float, length: float,
             frame: float = FRAME_S) -> dict:
    """Turn-taking features of a target and a partner agent inside one window.

    ``target_share`` and ``partner_share`` are the fractions of frames each is active,
    ``overlap_share`` the fraction of target-active frames where the partner is active too,
    ``reentry_s`` the longest target silence inside the window that ends with the target speaking
    again (0 if none), ``target_turns`` and ``partner_turns`` the segments overlapping the window,
    and ``backchannels`` the partner segments under 1 s that lie inside the window and wholly
    under target speech.
    """
    act_t = activity(seg_t, start, length, frame)
    act_p = activity(seg_p, start, length, frame)
    return _window_features(act_t, act_p, _bounds(seg_t), _bounds(seg_p), start, length, frame)


def _passes(f: dict, min_target_share: float, min_partner_share: float, min_turns: int) -> bool:
    return (f["target_share"] >= min_target_share and f["partner_share"] >= min_partner_share
            and f["target_turns"] >= min_turns and f["partner_turns"] >= min_turns)


def candidate_windows(meetings: Mapping[str, Mapping[str, Sequence[Segment]]], length_s: float,
                      hop_s: float, min_target_share: float = 0.25,
                      min_partner_share: float = 0.10, min_turns: int = 2,
                      frame: float = FRAME_S) -> list[Window]:
    """Windows of every meeting and ordered agent pair that hold a two-person exchange.

    Window starts lie on a grid ``0, hop_s, 2 * hop_s, ...`` and every window ends at or before
    the meeting's last segment end. A window is kept when the target and the partner reach their
    activity shares and each has at least ``min_turns`` segments in it. Deterministic: meetings,
    agents and starts are visited in sorted order.
    """
    if length_s <= 0 or hop_s <= 0:
        raise ValueError("length_s and hop_s must be positive")
    n = round(length_s / frame)
    slack = 1e-3  # the prefilter below is loose; the kept decision uses the rounded features
    out: list[Window] = []
    for meeting in sorted(meetings):
        agents = {a: list(s) for a, s in meetings[meeting].items() if s}
        if len(agents) < 2:
            continue
        meeting_end = max(g.end for segs in agents.values() for g in segs)
        if meeting_end + 1e-9 < length_s:
            continue
        count = math.floor((meeting_end - length_s) / hop_s + 1e-9) + 1
        starts = [round(k * hop_s, 6) for k in range(count)]
        offsets = np.array([round(s / frame) for s in starts])
        total = int(offsets[-1]) + n
        acts = {a: activity(segs, 0.0, total * frame, frame) for a, segs in agents.items()}
        shares = {}
        for a, act in acts.items():
            cum = np.concatenate(([0], np.cumsum(act, dtype=np.int64)))
            shares[a] = (cum[offsets + n] - cum[offsets]) / n
        bounds = {a: _bounds(segs) for a, segs in agents.items()}
        for target in sorted(agents):
            for partner in sorted(agents):
                if partner == target:
                    continue
                keep = ((shares[target] >= min_target_share - slack)
                        & (shares[partner] >= min_partner_share - slack))
                for k in np.flatnonzero(keep):
                    i0 = int(offsets[k])
                    f = _window_features(acts[target][i0:i0 + n], acts[partner][i0:i0 + n],
                                         bounds[target], bounds[partner], starts[k], length_s,
                                         frame)
                    if _passes(f, min_target_share, min_partner_share, min_turns):
                        out.append(Window(meeting, target, partner, starts[k], float(length_s), f))
    return out


# Slots and filling

def _role_slots(role: str, segments: Iterable[Segment], start: float, length: float,
                min_slot_s: float, merge_s: float) -> list[Slot]:
    spans = sorted(
        (max(g.start, start) - start, min(g.end, start + length) - start)
        for g in segments if g.end > start and g.start < start + length
    )
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] < merge_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [Slot(role, round(s, 6), round(e, 6)) for s, e in merged if e - s >= min_slot_s]


def slots_for(window: Window, segments_by_agent: Mapping[str, Sequence[Segment]],
              min_slot_s: float = MIN_SLOT_S, merge_s: float = MERGE_S) -> list[Slot]:
    """Each role's segments clipped to the window, as slots relative to the window start.

    Same-role slots that overlap or are separated by less than ``merge_s`` are merged first;
    slots shorter than ``min_slot_s`` are then dropped. Sorted by start.
    """
    slots = []
    for role, agent in (("target", window.target), ("partner", window.partner)):
        slots += _role_slots(role, segments_by_agent.get(agent, ()), window.start, window.length,
                             min_slot_s, merge_s)
    slots.sort(key=lambda s: (s.start, ROLES.index(s.role)))
    return slots


def _overrun_limit(slots: Sequence[Slot], k: int, gap_s: float, end_s: float | None) -> float:
    """How far an utterance placed in slot ``k`` may run past the slot's end.

    It stops ``gap_s`` before the same role's next slot, so one reader never overlaps itself,
    and at ``end_s`` when given, so nothing runs past the window.
    """
    limit = math.inf if end_s is None else end_s
    role = slots[k].role
    for later in slots[k + 1:]:
        if later.role == role:
            limit = min(limit, later.start - gap_s)
            break
    return limit


def fill(slots: Sequence[Slot], target_utts: Sequence[Utterance],
         partner_utts: Sequence[Utterance], rng: np.random.Generator, gap_s: float = GAP_S,
         tolerance: float = TOLERANCE, min_slot_s: float = MIN_SLOT_S,
         end_s: float | None = None) -> list[Placement]:
    """Fill each slot with utterances of its role's reader.

    For the time left in a slot, an utterance whose duration is within ``tolerance`` (relative)
    of it is drawn uniformly with ``rng`` and ends the slot. Without one, the longest utterance
    shorter than the time left is placed and filling continues ``gap_s`` after it while at least
    ``min_slot_s`` remains. When every utterance left is longer, the shortest is placed and
    trimmed at the slot end. A matched utterance may run past the slot end only up to
    ``gap_s`` before the same role's next slot and never past ``end_s``. Each utterance is used
    once. Raises ``NotEnoughSpeech`` when a role has nothing left for a slot.
    """
    pools = {"target": sorted(target_utts, key=lambda u: u.id),
             "partner": sorted(partner_utts, key=lambda u: u.id)}
    ordered = sorted(slots, key=lambda s: (s.start, ROLES.index(s.role)))
    placements: list[Placement] = []
    for k, slot in enumerate(ordered):
        pool = pools[slot.role]
        reach_end = max(slot.end, _overrun_limit(ordered, k, gap_s, end_s))
        cursor = slot.start
        while True:
            if not pool:
                raise NotEnoughSpeech(
                    f"no {slot.role} utterance left for the slot at {slot.start:.2f} s")
            room = slot.end - cursor
            reach = reach_end - cursor
            close = [i for i, u in enumerate(pool)
                     if abs(u.seconds - room) <= tolerance * room and u.seconds <= reach]
            if close:
                u = pool.pop(close[int(rng.integers(len(close)))])
                placements.append(Placement(slot.role, u.id, round(cursor, 6), u.seconds, u.text))
                break
            shorter = [i for i, u in enumerate(pool) if u.seconds < room]
            if shorter:
                u = pool.pop(max(shorter, key=lambda i: pool[i].seconds))
                placements.append(Placement(slot.role, u.id, round(cursor, 6), u.seconds, u.text))
                cursor += u.seconds + gap_s
                if slot.end - cursor < min_slot_s:
                    break
                continue
            u = pool.pop(min(range(len(pool)), key=lambda i: pool[i].seconds))
            placements.append(Placement(slot.role, u.id, round(cursor, 6), round(room, 6), u.text,
                                        trimmed=True))
            break
    return placements


def plan_conversation(window: Window, segments_by_agent: Mapping[str, Sequence[Segment]],
                      catalogue: Sequence[Utterance], target_reader: str, partner_reader: str,
                      rng: np.random.Generator, sample_chapter: str | None = None) -> Conversation:
    """The target reader takes the target agent's turns and the partner reader the partner's.

    Utterances of ``sample_chapter`` (a chapter of the target reader) are kept out of the fill so
    the voice sample can be taken from it.
    """
    if target_reader == partner_reader:
        raise ValueError("target and partner must be different readers")
    target_utts = [u for u in catalogue if u.speaker == target_reader]
    if sample_chapter is not None:
        if not any(u.chapter == sample_chapter for u in target_utts):
            raise ValueError(f"reader {target_reader} has no chapter {sample_chapter}")
        target_utts = [u for u in target_utts if u.chapter != sample_chapter]
    partner_utts = [u for u in catalogue if u.speaker == partner_reader]
    slots = slots_for(window, segments_by_agent)
    placements = fill(slots, target_utts, partner_utts, rng, end_s=window.length)
    return Conversation(window, target_reader, partner_reader, placements, sample_chapter)


def voice_sample_candidates(catalogue: Iterable[Utterance], reader: str, chapter: str,
                            min_s: float, max_s: float) -> list[Utterance]:
    """Utterances of ``reader`` in ``chapter`` lasting between ``min_s`` and ``max_s``, by id."""
    return sorted((u for u in catalogue
                   if u.speaker == reader and u.chapter == chapter and min_s <= u.seconds <= max_s),
                  key=lambda u: u.id)


# Rendering

def _fade_out(x: np.ndarray, n: int) -> None:
    n = min(n, len(x))
    if n > 0:
        x[-n:] *= (0.5 * (1.0 + np.cos(np.pi * np.arange(1, n + 1) / n))).astype(x.dtype)


def render_stems(conv: Conversation, catalogue_by_id: Mapping[str, Utterance],
                 corpus_root: Path | None, sr: int = SAMPLE_RATE) -> dict[str, np.ndarray]:
    """The clean target and partner stems, float32, exactly ``round(window.length * sr)`` long.

    Each placed wav is resampled to ``sr``, cut to ``seconds`` when trimmed and placed at its
    start. Anything past the window end is dropped. A 10 ms raised-cosine fade-out is applied
    where an utterance is cut, and nowhere else.
    """
    root = libritts_corpus_root(Path(corpus_root) if corpus_root is not None else None)
    n = round(conv.window.length * sr)
    stems = {role: np.zeros(n, dtype=np.float32) for role in ROLES}
    fade = round(FADE_S * sr)
    for p in conv.placements:
        utt = catalogue_by_id[p.utterance_id]
        x, rate = sf.read(root / utt.path, dtype="float32", always_2d=True)
        x = x.mean(axis=1, dtype=np.float32)
        if rate != sr:
            x = soxr.resample(x, rate, sr, quality="VHQ")
        x = np.array(x, dtype=np.float32)
        if p.trimmed:
            x = x[:round(p.seconds * sr)]
            _fade_out(x, fade)
        i0 = round(p.start * sr)
        if i0 >= n or len(x) == 0:
            continue
        if i0 + len(x) > n:
            x = x[:n - i0]
            _fade_out(x, fade)
        stems[p.role][i0:i0 + len(x)] += x
    return stems


# JSON

def to_dict(conv: Conversation) -> dict:
    return asdict(conv)


def from_dict(d: Mapping) -> Conversation:
    w = d["window"]
    window = Window(w["meeting"], w["target"], w["partner"], float(w["start"]),
                    float(w["length"]), dict(w["features"]))
    placements = [Placement(p["role"], p["utterance_id"], float(p["start"]), float(p["seconds"]),
                            p["text"], bool(p.get("trimmed", False)))
                  for p in d["placements"]]
    return Conversation(window, d["target_reader"], d["partner_reader"], placements,
                        d.get("sample_chapter"))


__all__ = [
    "BACKCHANNEL_MAX_S",
    "FRAME_S",
    "MIN_SLOT_S",
    "SAMPLE_RATE",
    "Conversation",
    "NotEnoughSpeech",
    "Placement",
    "Segment",
    "Slot",
    "Window",
    "Word",
    "activity",
    "candidate_windows",
    "default_ami_root",
    "features",
    "fill",
    "from_dict",
    "list_meetings",
    "load_meetings",
    "parse_meeting",
    "plan_conversation",
    "render_stems",
    "slots_for",
    "timeline_cache_path",
    "to_dict",
    "voice_sample_candidates",
]
