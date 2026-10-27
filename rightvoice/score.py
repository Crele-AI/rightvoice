"""Scoring: every metric from a system's output audio and the item's references.

A system is scored on its output audio alone, so any file-in, file-out system can be scored.
The reference for an item comes from its manifest row and the word timings of the utterances
placed in it. The hypothesis comes from speech recognition and voice activity detection run on
the output. The recogniser, the activity detector and the speaker embedder are thin adapters
behind small interfaces (``ASR``, ``VAD``, ``Embedder``) whose packages are imported only when
they are used, so the metric functions run without them and tests can pass fakes.

Times are seconds from the start of the item. Activity is on the 50 Hz grid of
``rightvoice.conversation``: frame ``i`` covers ``[i * 0.02, (i + 1) * 0.02)`` and is active when
its centre lies inside a span. Text on both sides goes through ``normalize`` before any
comparison. Metric functions return plain ints and floats; an undefined ratio is ``nan`` in
memory and ``null`` in the scores file.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
import soxr

FRAME_S = 0.02            # 50 Hz activity frames, as in rightvoice.conversation
FRAME_RATE = 50
SAMPLE_RATE = 16000
MIN_OUTPUT_S = 0.01       # shorter output is empty output
MIN_VOICE_S = 1.0         # voice match needs this much target-active audio
WRONG_VOICE_WINDOW_S = 1.0
SI_SDR_LIMIT_DB = 120.0   # SI-SDR is clipped to +/- this, so it stays a finite number
ADAPTER_RATE = 16000      # rate the adapters feed to their packages
SCORE_EXTRA = "uv sync --extra score"

Op = tuple[str, int | None, int | None]


# Text normalisation

_APOSTROPHES = re.compile("[‘’ʼ]")
_NOT_WORD = re.compile(r"[^\w\s']|_")
_LOOSE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")


def basic_normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation except apostrophes inside words, split on whitespace.

    Nothing is expanded. Curly apostrophes count as apostrophes; other punctuation becomes a
    space, so ``rock-n-roll`` gives three words.
    """
    t = _APOSTROPHES.sub("'", (text or "").lower())
    t = _NOT_WORD.sub(" ", t)
    t = _LOOSE_APOSTROPHE.sub(" ", t)
    return t.split()


@functools.cache
def _normalizer() -> tuple[Callable[[str], list[str]], str]:
    try:
        from whisper.normalizers import EnglishTextNormalizer
    except ImportError:
        return basic_normalize, "basic"
    english = EnglishTextNormalizer()
    return (lambda text: english(text or "").split()), "whisper-english"


def normalizer_name() -> str:
    """Which normalizer ``normalize`` uses: ``"whisper-english"`` or ``"basic"``."""
    return _normalizer()[1]


def normalize(text: str) -> list[str]:
    """Normalized words of ``text``.

    Uses the English normalizer of the ``openai-whisper`` package when it can be imported and
    ``basic_normalize`` otherwise. ``NORMALIZER_NAME`` says which.
    """
    return _normalizer()[0](text or "")


def __getattr__(name: str) -> Any:
    # NORMALIZER_NAME is resolved on first access so that importing this module never imports
    # the recognition package.
    if name == "NORMALIZER_NAME":
        return normalizer_name()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Alignment and word error rate

def _distances(r: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Levenshtein table: ``d[i, j]`` is the distance between ``r[:i]`` and ``h[:j]``."""
    n, m = len(r), len(h)
    cols = np.arange(m + 1, dtype=np.int32)
    d = np.empty((n + 1, m + 1), dtype=np.int32)
    d[0] = cols
    for i in range(1, n + 1):
        row = np.empty(m + 1, dtype=np.int32)
        row[0] = i
        if m:
            cost = (h != r[i - 1]).astype(np.int32)
            row[1:] = np.minimum(d[i - 1, :-1] + cost, d[i - 1, 1:] + 1)
        # row[j] = min(row[j], row[j - 1] + 1) along the row, as a running minimum of row - j
        d[i] = np.minimum.accumulate(row - cols) + cols
    return d


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[Op]:
    """Edit operations turning ``ref`` into ``hyp``, in order.

    Each operation is ``(kind, ref_index, hyp_index)`` with kind ``"equal"``, ``"sub"``,
    ``"del"`` (``hyp_index`` None) or ``"ins"`` (``ref_index`` None). The backtrace from the end
    of both lists prefers a substitution or match, then a deletion, then an insertion.
    """
    vocab: dict[str, int] = {}
    r = np.array([vocab.setdefault(w, len(vocab)) for w in ref], dtype=np.int64)
    h = np.array([vocab.setdefault(w, len(vocab)) for w in hyp], dtype=np.int64)
    d = _distances(r, h).tolist()
    ops: list[Op] = []
    i, j = len(r), len(h)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = r[i - 1] == h[j - 1]
            if d[i][j] == d[i - 1][j - 1] + (0 if same else 1):
                ops.append(("equal" if same else "sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
            continue
        ops.append(("ins", None, j - 1))
        j -= 1
    ops.reverse()
    return ops


def edit_counts(ops: Iterable[Op]) -> dict:
    """Counts of each operation kind, and ``n_ref`` (sub + del + equal)."""
    counts = {"sub": 0, "ins": 0, "del": 0, "equal": 0}
    for kind, _, _ in ops:
        counts[kind] += 1
    counts["n_ref"] = counts["sub"] + counts["del"] + counts["equal"]
    return counts


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else math.nan


def corpus_wer(counts: Iterable[Mapping]) -> float:
    """(sub + del + ins) / n_ref, each summed over items; nan when the summed n_ref is 0.

    Words against an empty reference are insertions and count in the numerator.
    """
    errors = n_ref = 0
    for c in counts:
        errors += int(c["sub"]) + int(c["del"]) + int(c["ins"])
        n_ref += int(c["n_ref"])
    return _ratio(errors, n_ref)


# Timed words

@dataclass(frozen=True)
class TimedWord:
    word: str
    start: float              # seconds into the item
    end: float
    voice: str = ""           # "target", "partner", "bystander", or "" for output words

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)


def _spread(tokens: Sequence[str], start: float, end: float) -> list[tuple[str, float, float]]:
    """``tokens`` sharing ``[start, end]`` in equal parts, in order."""
    if not tokens:
        return []
    step = (end - start) / len(tokens)
    return [(t, start + q * step, start + (q + 1) * step) for q, t in enumerate(tokens)]


def _retime(pieces: list[tuple[str, float, float]],
            sentence: list[str]) -> list[tuple[str, float, float]]:
    """Times for ``sentence`` tokens from word-by-word ``pieces`` that normalized differently."""
    ops = align([p[0] for p in pieces], sentence)
    spans: list[list[float] | None] = [None] * len(sentence)
    subs: set[int] = set()
    for kind, i, j in ops:
        if kind in ("equal", "sub"):
            spans[j] = [pieces[i][1], pieces[i][2]]
            if kind == "sub":
                subs.add(j)
    # A piece with no token of its own (number words merged into one, for example) widens the
    # substituted token next to it, the following one first.
    before: list[int | None] = []
    last = None
    for _, _, j in ops:
        before.append(last)
        if j is not None:
            last = j
    after: list[int | None] = [None] * len(ops)
    last = None
    for k in range(len(ops) - 1, -1, -1):
        after[k] = last
        if ops[k][2] is not None:
            last = ops[k][2]
    for k, (kind, i, _) in enumerate(ops):
        if kind != "del":
            continue
        for j in (after[k], before[k]):
            if j is not None and j in subs:
                spans[j] = [min(spans[j][0], pieces[i][1]), max(spans[j][1], pieces[i][2])]
                break
    # A token with no piece shares the gap between its timed neighbours.
    lo = pieces[0][1] if pieces else 0.0
    hi = pieces[-1][2] if pieces else lo
    k = 0
    while k < len(spans):
        if spans[k] is not None:
            k += 1
            continue
        r = k
        while r < len(spans) and spans[r] is None:
            r += 1
        left = spans[k - 1][1] if k > 0 else lo
        right = spans[r][0] if r < len(spans) else hi
        for q, (_, s, e) in enumerate(_spread(sentence[k:r], left, max(left, right))):
            spans[k + q] = [s, e]
        k = r
    return [(w, s, e) for w, (s, e) in zip(sentence, spans)]


def normalize_timed(words: Iterable[Sequence], *, offset: float = 0.0,
                    voice: str = "") -> list[TimedWord]:
    """Normalize timed raw words as one text, keeping a time span for every output word.

    ``words`` are ``(start, end, word)``. The joined text is normalized as a whole, which is what
    ``normalize`` gives for the sentence. Each raw word is also normalized on its own and its
    span shared evenly among the pieces it gives. When the two token lists are the same the
    pieces are the result. Otherwise they are aligned: a token takes the span of the piece it
    aligns with, widened over a neighbouring piece left without a token, and a token without a
    piece shares the gap between its neighbours. ``offset`` is added to every time.
    """
    raw = [(float(s), float(e), str(w)) for s, e, w in words]
    pieces: list[tuple[str, float, float]] = []
    for s, e, w in raw:
        pieces.extend(_spread(normalize(w), s, max(s, e)))
    sentence = normalize(" ".join(w for _, _, w in raw))
    timed = pieces if [p[0] for p in pieces] == sentence else _retime(pieces, sentence)
    return [TimedWord(w, s + offset, e + offset, voice) for w, s, e in timed]


# Frames and samples

def n_frames(length_s: float) -> int:
    """Number of 50 Hz frames in ``length_s`` seconds."""
    return max(0, round(length_s * FRAME_RATE))


def _frame_span(start: float, end: float) -> tuple[int, int]:
    """Frames whose centre lies in ``[start, end)``."""
    i0 = math.ceil(start / FRAME_S - 0.5 - 1e-9)
    i1 = math.ceil(end / FRAME_S - 0.5 - 1e-9)
    return i0, i1


def _mark(act: np.ndarray, start: float, end: float) -> None:
    i0, i1 = _frame_span(start, end)
    i0, i1 = max(i0, 0), min(i1, len(act))
    if i1 > i0:
        act[i0:i1] = True


def _frame_of(t: float, n: int) -> int:
    """The frame holding time ``t``, clipped to ``[0, n - 1]``."""
    return min(max(math.floor(t / FRAME_S + 1e-9), 0), n - 1)


def _fit_frames(act: Any, n: int) -> np.ndarray:
    a = np.asarray(act, dtype=bool).reshape(-1)
    if len(a) >= n:
        return a[:n].copy()
    return np.concatenate([a, np.zeros(n - len(a), dtype=bool)])


def _mono(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError(f"expected a mono 1-D array, got shape {a.shape}")
    return a


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """``x`` cut or zero-padded to ``n`` samples."""
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def active_samples(activity: Any, n: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A sample mask of length ``n`` from 50 Hz frames: sample ``t`` takes frame ``t * 50 // sr``.

    Samples past the last frame are inactive.
    """
    act = np.asarray(activity, dtype=bool).reshape(-1)
    idx = np.arange(n, dtype=np.int64) * FRAME_RATE // int(sr)
    mask = np.zeros(n, dtype=bool)
    ok = idx < len(act)
    mask[ok] = act[idx[ok]]
    return mask


# References

@dataclass(eq=False)
class Reference:
    target_words: list[TimedWord]
    other_words: list[TimedWord]      # partner and bystander words
    target_activity: np.ndarray       # bool, 50 Hz, n_frames(length_s) long
    length_s: float
    target_absent: bool = False
    timings_estimated: bool = False   # some placed utterance had no word timings
    item_id: str = ""
    long: bool = False


def _get(row: Mapping, key: str, default: Any = None) -> Any:
    """``row[key]``, else the same key under ``row["conversation"]``, else ``default``."""
    if key in row:
        return row[key]
    nested = row.get("conversation")
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    return default


def _item_length(row: Mapping) -> float:
    window = _get(row, "window")
    if isinstance(window, Mapping) and window.get("length") is not None:
        return float(window["length"])
    if row.get("length_s") is not None:
        return float(row["length_s"])
    spans = list(_get(row, "placements") or []) + list(row.get("bystanders") or [])
    return max((float(p.get("start", 0.0)) + float(p.get("seconds", 0.0)) for p in spans),
               default=0.0)


def _timed_words(timed: Sequence[Sequence], start: float, seconds: float,
                 voice: str) -> list[TimedWord]:
    """Utterance-local timings placed at ``start``; words past ``seconds`` were cut off."""
    local = [(float(s), float(e), str(w)) for s, e, w in timed]
    if seconds > 0:
        local = [(s, min(e, seconds), w) for s, e, w in local if (s + e) / 2 < seconds]
    return normalize_timed(local, offset=start, voice=voice)


def _spread_words(text: str, start: float, seconds: float, voice: str) -> list[TimedWord]:
    return [TimedWord(w, s, e, voice)
            for w, s, e in _spread(normalize(text), start, start + max(seconds, 0.0))]


def _placed_words(uid: str | None, text: str, start: float, seconds: float,
                  timings: Mapping, voice: str) -> tuple[list[TimedWord], bool]:
    timed = timings.get(uid) if uid is not None else None
    if timed:
        return _timed_words(timed, start, seconds, voice), False
    words = _spread_words(text, start, seconds, voice)
    return words, bool(words)


PIECE_GAP_S = 0.3  # silence the builder puts between two joined utterances


def _bystander_words(b: Mapping, timings: Mapping) -> tuple[list[TimedWord], bool]:
    """Words of one bystander clip.

    The builder writes a clip as ``pieces`` (utterance id, seconds played, text), joined by
    ``PIECE_GAP_S`` of silence from ``start_s``. Each piece takes its utterance's timings when
    present, cut to the seconds played; otherwise its text is spread over its own span. Rows
    without ``pieces`` fall back to ``utterance_ids`` with optional ``offsets``, or to ``text``.
    """
    start = float(b.get("start_s", b.get("start", 0.0)))
    seconds = float(b.get("seconds", 0.0))
    pieces = b.get("pieces")
    if isinstance(pieces, Sequence) and pieces:
        words: list[TimedWord] = []
        estimated = False
        offset = start
        for piece in pieces:
            u = piece.get("utterance_id")
            played = float(piece.get("seconds", 0.0))
            if timings.get(u):
                timed = _timed_words(timings[u], offset, 0.0, "bystander")
                words += [w for w in timed if w.mid < offset + played]
            else:
                words += _spread_words(str(piece.get("text", "")), offset, played, "bystander")
                estimated = True
            offset += played + PIECE_GAP_S
        return words, estimated
    ids = b.get("utterance_ids")
    if ids is None:
        ids = [b["utterance_id"]] if b.get("utterance_id") else []
    elif isinstance(ids, str):
        ids = [ids]
    offsets = b.get("offsets")
    if len(ids) > 1 and not (isinstance(offsets, Sequence) and len(offsets) == len(ids)):
        offsets = None
    if ids and all(timings.get(u) for u in ids) and (len(ids) == 1 or offsets is not None):
        words = []
        for u, off in zip(ids, offsets or [0.0]):
            words += _timed_words(timings[u], start + float(off), 0.0, "bystander")
        if seconds > 0:
            words = [w for w in words if w.mid < start + seconds]
        return words, False
    text = b.get("text")
    if text is None:
        text = " ".join(str(w) for u in ids for _, _, w in (timings.get(u) or []))
    elif not isinstance(text, str):
        text = " ".join(str(t) for t in text)
    words = _spread_words(text, start, seconds, "bystander")
    return words, bool(words)


def _heard(words: list[TimedWord], length_s: float) -> list[TimedWord]:
    """Words whose midpoint lies inside the item, ends clipped to it, sorted by time."""
    kept = [TimedWord(w.word, w.start, min(w.end, length_s), w.voice)
            for w in words if 0.0 <= w.mid < length_s]
    return sorted(kept, key=lambda w: (w.start, w.end))


def reference_from_manifest(row: Mapping, word_timings: Mapping | None = None) -> Reference:
    """The reference of one item from its manifest row.

    ``word_timings`` maps an utterance id to ``(start, end, word)`` in utterance-local seconds;
    a ``word_timings`` mapping inside the row is used for ids it lacks. An utterance without
    timings has its normalized text spread evenly over its placement (``timings_estimated``).
    Timed words past the end of a trimmed placement are dropped; estimated words of a trimmed
    placement are all kept, spread over what plays. Words whose midpoint lies past the item end
    are dropped. On a target-absent item there are no target words and no target activity.
    Missing keys mean empty lists.
    """
    timings: dict = {}
    if isinstance(row.get("word_timings"), Mapping):
        timings.update(row["word_timings"])
    if word_timings:
        timings.update(word_timings)
    length = _item_length(row)
    absent = bool(row.get("target_absent", False))
    activity = np.zeros(n_frames(length), dtype=bool)
    target: list[TimedWord] = []
    other: list[TimedWord] = []
    estimated = False
    for p in _get(row, "placements") or []:
        role = str(p.get("role", "target"))
        if role == "target" and absent:
            continue
        start = float(p.get("start", 0.0))
        seconds = float(p.get("seconds", 0.0))
        words, est = _placed_words(p.get("utterance_id"), p.get("text", ""), start, seconds,
                                   timings, role)
        estimated |= est
        if role == "target":
            target += words
            _mark(activity, start, start + seconds)
        else:
            other += words
    for b in row.get("bystanders") or []:
        words, est = _bystander_words(b, timings)
        estimated |= est
        other += words
    item_id = row.get("id", row.get("item_id", ""))
    return Reference(target_words=_heard(target, length), other_words=_heard(other, length),
                     target_activity=activity, length_s=length, target_absent=absent,
                     timings_estimated=estimated, item_id=str(item_id),
                     long=bool(row.get("long", False)))


# Hypothesis

@dataclass(eq=False)
class Hypothesis:
    words: list[TimedWord]            # normalized output words
    activity: np.ndarray              # bool, 50 Hz


def asr_words(result: Any) -> list[dict]:
    """Flat ``{"word", "start", "end"}`` dicts from a recognition result.

    Takes a result dict with ``segments``, a list of segments with ``words``, or a list of word
    dicts. Words without both times are skipped.
    """
    items = result.get("segments", []) if isinstance(result, Mapping) else list(result or [])
    flat: list[Mapping] = []
    for item in items:
        if "word" not in item and "words" in item:
            flat.extend(item.get("words") or [])
        else:
            flat.append(item)
    return [{"word": str(w.get("word", "")), "start": float(w["start"]), "end": float(w["end"])}
            for w in flat if w.get("start") is not None and w.get("end") is not None]


def hypothesis_from_asr(segments_or_words: Any, length_s: float,
                        activity: Any = None) -> Hypothesis:
    """The hypothesis of one output from its recognised words, in transcript order.

    Times are clipped to the item and words starting past its end are dropped. ``activity`` is
    the 50 Hz detector output, cut or padded to the item; without it, frames whose centre lies
    inside a recognised word are active.
    """
    raw = []
    for w in asr_words(segments_or_words):
        if w["start"] >= length_s:
            continue
        s = min(max(w["start"], 0.0), length_s)
        raw.append((s, min(max(w["end"], s), length_s), w["word"]))
    words = normalize_timed(raw)
    n = n_frames(length_s)
    if activity is None:
        act = np.zeros(n, dtype=bool)
        for s, e, _ in raw:
            _mark(act, s, e)
    else:
        act = _fit_frames(activity, n)
    return Hypothesis(words, act)


# Metrics

def target_words_lost(ref: Reference, hyp: Hypothesis) -> dict:
    """Edit counts of the output words against the target words, and the item's ``wer``."""
    counts = edit_counts(align([w.word for w in ref.target_words], [w.word for w in hyp.words]))
    counts["wer"] = _ratio(counts["sub"] + counts["del"] + counts["ins"], counts["n_ref"])
    return counts


def wrong_voice_words(ref: Reference, hyp: Hypothesis,
                      window_s: float = WRONG_VOICE_WINDOW_S) -> dict:
    """Output words that another voice in the recording said.

    The candidates are the substituted and inserted output words of the alignment against the
    target words. A candidate is a wrong-voice word when an other-voice reference word has the
    same text and a midpoint within ``window_s`` of its own. Pairs are matched one to one,
    closest in time first, so one reference word explains at most one output word.
    ``by_voice`` splits the count by the reference word's voice.
    """
    ops = align([w.word for w in ref.target_words], [w.word for w in hyp.words])
    candidates = [j for kind, _, j in ops if kind in ("sub", "ins")]
    by_text: dict[str, list[int]] = {}
    for k, o in enumerate(ref.other_words):
        by_text.setdefault(o.word, []).append(k)
    pairs = []
    for j in candidates:
        h = hyp.words[j]
        for k in by_text.get(h.word, ()):
            gap = abs(ref.other_words[k].mid - h.mid)
            if gap <= window_s + 1e-9:
                pairs.append((gap, j, k))
    pairs.sort()
    used_hyp: set[int] = set()
    used_ref: set[int] = set()
    by_voice: dict[str, int] = {}
    for _, j, k in pairs:
        if j in used_hyp or k in used_ref:
            continue
        used_hyp.add(j)
        used_ref.add(k)
        voice = ref.other_words[k].voice or "other"
        by_voice[voice] = by_voice.get(voice, 0) + 1
    count = len(used_hyp)
    return {"count": count, "candidates": len(candidates),
            "per_target_word": _ratio(count, len(ref.target_words)),
            "per_minute": _ratio(count, ref.length_s / 60.0), "by_voice": by_voice}


def talk_over_silence(ref: Reference, hyp: Hypothesis) -> dict:
    """Output speech while the target is silent.

    ``words`` counts output words whose midpoint falls in a target-silent frame, ``per_minute``
    divides them by the target-silent minutes, and ``speech_share`` is the share of
    target-silent frames where the output is active. On a target-absent item every frame is
    target-silent. With no target-silent time there is nothing to talk over and both rates
    are 0.
    """
    n = len(ref.target_activity)
    silent = np.ones(n, dtype=bool) if ref.target_absent else ~ref.target_activity
    speech = _fit_frames(hyp.activity, n) & silent
    words = sum(1 for w in hyp.words if n and silent[_frame_of(w.mid, n)])
    silent_frames = int(silent.sum())
    silent_s = silent_frames / FRAME_RATE
    return {"words": words,
            "per_minute": words / (silent_s / 60.0) if silent_frames else 0.0,
            "speech_share": int(speech.sum()) / silent_frames if silent_frames else 0.0,
            "silent_s": silent_s, "speech_s": int(speech.sum()) / FRAME_RATE}


def activity_f1(ref: Reference, hyp: Hypothesis) -> dict:
    """Frame precision, recall and F1 of output activity against target activity.

    F1 is ``2 tp / (2 tp + fp + fn)``: 0 when only one side has active frames, and nan when
    neither has any, because an item with nothing to detect and nothing detected says nothing
    about detection (target-absent items are covered by ``talk_over_silence``). Precision is
    nan without output frames and recall nan without target frames.
    """
    r = np.asarray(ref.target_activity, dtype=bool)
    h = _fit_frames(hyp.activity, len(r))
    tp = int((r & h).sum())
    fp = int((~r & h).sum())
    fn = int((r & ~h).sum())
    return {"precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (na > 0 and nb > 0) or len(a) != len(b):
        return math.nan
    return float(np.dot(a, b) / (na * nb))


def voice_match(embed: Callable[[np.ndarray, int], Any], out_audio: Any, target_audio: Any,
                activity: Any, sr: int = SAMPLE_RATE) -> float:
    """Cosine between speaker embeddings of the output and of the clean target.

    Both are restricted to target-active samples (the 50 Hz ``activity`` taken to samples), the
    active stretches concatenated. ``embed(x, sr)`` returns a 1-D vector. The output is cut or
    padded to the target's length first. nan with under ``MIN_VOICE_S`` of active audio.
    """
    target = _mono(target_audio)
    out = _fit(_mono(out_audio), len(target))
    mask = active_samples(activity, len(target), sr)
    if int(mask.sum()) < MIN_VOICE_S * sr:
        return math.nan
    a = np.asarray(embed(out[mask], sr), dtype=np.float64).reshape(-1)
    b = np.asarray(embed(target[mask], sr), dtype=np.float64).reshape(-1)
    return _cosine(a, b)


def si_sdr(est: Any, ref: Any) -> float:
    """Scale-invariant signal-to-distortion ratio of ``est`` against ``ref``, in dB.

    Both are made zero-mean and ``est`` is cut or padded to ``ref``. nan when the reference or
    the estimate is silent. Clipped to +/- ``SI_SDR_LIMIT_DB``.
    """
    r = _mono(ref).astype(np.float64)
    if len(r) == 0:
        return math.nan
    e = _fit(_mono(est), len(r)).astype(np.float64)
    r = r - r.mean()
    e = e - e.mean()
    rr = float(r @ r)
    if not (rr > 0 and math.isfinite(rr)) or not float(e @ e) > 0:
        return math.nan
    projected = (float(e @ r) / rr) * r
    residual = e - projected
    tt, nn = float(projected @ projected), float(residual @ residual)
    if nn <= 0:
        return SI_SDR_LIMIT_DB
    if tt <= 0:
        return -SI_SDR_LIMIT_DB
    return float(np.clip(10.0 * math.log10(tt / nn), -SI_SDR_LIMIT_DB, SI_SDR_LIMIT_DB))


def si_sdr_improvement(est: Any, mix: Any, ref: Any) -> float:
    """SI-SDR of ``est`` minus SI-SDR of the mixture, both against ``ref``, in dB."""
    return si_sdr(est, ref) - si_sdr(mix, ref)


def empty_output_scores(ref: Reference) -> dict:
    """The metrics of a system that returns nothing for ``ref``.

    Every target word is deleted (wer 1.0, nan on a target-absent item), nothing is a wrong
    voice or talks over silence, activity F1 is 0 (nan on a target-absent item), and voice
    match, SI-SDR and quality are nan.
    """
    hyp = Hypothesis([], np.zeros(len(ref.target_activity), dtype=bool))
    return {
        "target_words_lost": target_words_lost(ref, hyp),
        "wrong_voice_words": wrong_voice_words(ref, hyp),
        "talk_over_silence": talk_over_silence(ref, hyp),
        "activity_f1": activity_f1(ref, hyp),
        "voice_match": math.nan,
        "si_sdr": math.nan,
        "si_sdr_improvement": math.nan,
        "quality": math.nan,
    }


# Adapters

class ASR(Protocol):
    name: str
    version: str

    def transcribe(self, audio: np.ndarray, sr: int) -> list[dict]: ...


class VAD(Protocol):
    name: str
    version: str

    def activity(self, audio: np.ndarray, sr: int) -> np.ndarray: ...


#: ``embed(audio, sr)`` returns a 1-D speaker embedding.
Embedder = Callable[[np.ndarray, int], np.ndarray]
#: ``mos(audio, sr)`` returns a predicted quality score.
QualityPredictor = Callable[[np.ndarray, int], float]


def _import(module: str, package: str, user: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"{user} needs the '{package}' package, which could not be imported "
                           f"({exc}); install the score extra with: {SCORE_EXTRA}") from exc


def _package_version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _to_rate(x: np.ndarray, sr: int, rate: int) -> np.ndarray:
    if sr != rate and len(x):
        x = soxr.resample(x, sr, rate, quality="VHQ")
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    tmp.replace(path)


class WhisperASR:
    """Word-timed English recognition with the ``openai-whisper`` package.

    Decoding uses beam search (beam 5) at temperature 0 with word timestamps, without
    conditioning on previous text. With ``cache_dir`` each result is stored as JSON under the
    sha256 of the 16 kHz float32 audio bytes and the model name, and a stored result is returned
    without loading the package.
    """

    package = "openai-whisper"

    def __init__(self, model: str = "large-v3", device: str | None = None,
                 cache_dir: str | Path | None = None):
        self.model_name = model
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.name = f"whisper-{model}"
        self.version = _package_version(self.package)
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            whisper = _import("whisper", self.package, "WhisperASR")
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def cache_key(self, audio16k: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(audio16k, dtype=np.float32).tobytes())
        h.update(self.model_name.encode("utf-8"))
        return h.hexdigest()

    def transcribe(self, audio: np.ndarray, sr: int) -> list[dict]:
        x = _to_rate(_mono(audio), sr, ADAPTER_RATE)
        path = None
        if self.cache_dir is not None:
            path = self.cache_dir / f"{self.cache_key(x)}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        result = self._load().transcribe(x, language="en", temperature=0.0, beam_size=5,
                                         word_timestamps=True, condition_on_previous_text=False)
        words = asr_words(result)
        if path is not None:
            _write_json(path, words)
        return words


class SileroVAD:
    """50 Hz speech activity from the ``silero-vad`` package's speech timestamps."""

    package = "silero-vad"

    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 100,
                 min_silence_ms: int = 100, pad_ms: int = 30):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.pad_ms = pad_ms
        self.name = "silero-vad"
        self.version = _package_version(self.package)
        self._model = None

    def activity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        x = _mono(audio)
        act = np.zeros(n_frames(len(x) / sr), dtype=bool)
        silero = _import("silero_vad", self.package, "SileroVAD")
        torch = _import("torch", "torch", "SileroVAD")
        if self._model is None:
            self._model = silero.load_silero_vad()
        x = _to_rate(x, sr, ADAPTER_RATE)
        if not len(x):
            return act
        spans = silero.get_speech_timestamps(
            torch.from_numpy(x), self._model, threshold=self.threshold,
            sampling_rate=ADAPTER_RATE, min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms, speech_pad_ms=self.pad_ms)
        for span in spans:
            _mark(act, span["start"] / ADAPTER_RATE, span["end"] / ADAPTER_RATE)
        return act


class EcapaEmbedder:
    """Speaker embeddings from a speaker encoder loaded with the ``speechbrain`` package."""

    package = "speechbrain"
    source = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, device: str | None = None, savedir: str | Path | None = None):
        self.device = device
        self.savedir = savedir
        self.name = self.source
        self.version = _package_version(self.package)
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            speaker = _import("speechbrain.inference.speaker", self.package, "EcapaEmbedder")
            kwargs: dict[str, Any] = {"source": self.source}
            if self.savedir is not None:
                kwargs["savedir"] = str(self.savedir)
            if self.device is not None:
                kwargs["run_opts"] = {"device": self.device}
            self._model = speaker.EncoderClassifier.from_hparams(**kwargs)
        return self._model

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        x = _to_rate(_mono(audio), sr, ADAPTER_RATE)
        model = self._load()
        torch = _import("torch", "torch", "EcapaEmbedder")
        with torch.no_grad():
            emb = model.encode_batch(torch.from_numpy(x).unsqueeze(0))
        return np.asarray(emb.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


class DnsmosP808:
    """DNSMOS P.808 overall quality from the DNS Challenge's ``model_v8.onnx``.

    Reproduces the reference script: 16 kHz audio, 9.01 s segments every second (a clip
    shorter than that is tiled), a 120-band mel spectrogram (FFT 321, hop 160) in dB relative
    to the maximum then scaled as (x + 40) / 40, fed as ``input_1``, and the mean over segments.
    The model file is read from ``model_path`` or fetched once into ``cache_dir`` from the DNS
    Challenge repository, whose content licence is CC BY 4.0; the file is never redistributed.
    """

    package = "onnxruntime"
    url = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx"
    segment_s = 9.01
    hop_s = 1.0

    def __init__(self, model_path: str | Path | None = None, cache_dir: str | Path | None = None,
                 session: Any | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.name = "dnsmos-p808"
        self.version = _package_version(self.package)
        self._session = session

    def _file(self) -> Path:
        if self.model_path is not None:
            return self.model_path
        folder = self.cache_dir or Path("data") / "cache" / "dnsmos"
        path = folder / "model_v8.onnx"
        if not path.exists():
            import urllib.request
            folder.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(self.url, headers={"User-Agent": "rightvoice-score/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        return path

    def _load(self) -> Any:
        if self._session is None:
            ort = _import("onnxruntime", self.package, "DnsmosP808")
            self._session = ort.InferenceSession(str(self._file()),
                                                 providers=["CPUExecutionProvider"])
        return self._session

    @staticmethod
    def _melspec(audio: np.ndarray, sr: int) -> np.ndarray:
        librosa = _import("librosa", "librosa", "DnsmosP808")
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=321, hop_length=160, n_mels=120)
        mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        return mel.T

    def __call__(self, audio: np.ndarray, sr: int) -> float:
        x = _to_rate(_mono(audio), sr, ADAPTER_RATE).astype(np.float32)
        fs = ADAPTER_RATE
        need = int(self.segment_s * fs)
        while x.size < need:
            x = np.concatenate([x, x])
        session = self._load()
        hops = int(np.floor(x.size / fs) - self.segment_s) + 1
        scores: list[float] = []
        for k in range(max(hops, 1)):
            start = int(k * self.hop_s * fs)
            seg = x[start:start + need]
            if seg.size < need:
                continue
            feats = self._melspec(seg[:-160], fs).astype(np.float32)[None, :, :]
            out = session.run(None, {"input_1": feats})
            scores.append(float(np.asarray(out[0], dtype=np.float64).reshape(-1)[0]))
        return float(np.mean(scores)) if scores else float("nan")


def _label(adapter: Any) -> dict | None:
    if adapter is None:
        return None
    name = getattr(adapter, "name", None) or getattr(adapter, "__name__", None)
    return {"name": str(name or type(adapter).__name__),
            "version": str(getattr(adapter, "version", ""))}


# Scoring

def load_output(path: str | Path, length_s: float | None,
                sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """A wav as float32 mono at ``sr``, cut to ``length_s`` (never padded).

    Channels are averaged and other rates resampled. None when the file is missing or cannot be
    read. Not padding keeps a too-short file visible to ``score_item``.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        x, rate = sf.read(p, dtype="float32", always_2d=True)
    except (sf.SoundFileError, OSError, RuntimeError, ValueError):
        return None
    y = _to_rate(x.mean(axis=1, dtype=np.float32), int(rate), sr)
    if length_s is not None:
        y = y[:round(length_s * sr)]
    return y


def _is_empty(x: np.ndarray | None, sr: int) -> bool:
    return x is None or len(x) < round(MIN_OUTPUT_S * sr) or not np.any(x)


def score_item(ref: Reference, output_audio: np.ndarray | None, *, asr: ASR, vad: VAD,
               embed: Embedder | None = None, target_audio: np.ndarray | None = None,
               mixture_audio: np.ndarray | None = None, sr: int = SAMPLE_RATE,
               mos: QualityPredictor | None = None) -> dict:
    """Every metric of one output against its reference, as one JSON-friendly dict.

    ``output_audio`` None, all zero or shorter than ``MIN_OUTPUT_S`` is empty output and gets
    ``empty_output_scores``. Otherwise non-finite samples become 0 and the output is cut or
    padded to the item length before recognition, activity detection, voice match (needs
    ``embed`` and ``target_audio``), SI-SDR (needs ``target_audio``, and ``mixture_audio`` for
    the improvement) and quality (needs ``mos``). Missing inputs give nan.
    """
    x = None if output_audio is None else np.nan_to_num(_mono(output_audio))
    empty = _is_empty(x, sr)
    result: dict[str, Any] = {
        "id": ref.item_id,
        "length_s": ref.length_s,
        "long": ref.long,
        "target_absent": ref.target_absent,
        "timings_estimated": ref.timings_estimated,
        "empty_output": empty,
    }
    if empty:
        result.update(empty_output_scores(ref))
    else:
        n = round(ref.length_s * sr)
        out = _fit(x, n)
        hyp = hypothesis_from_asr(asr.transcribe(out, sr), ref.length_s,
                                  activity=vad.activity(out, sr))
        target = None if target_audio is None else _fit(_mono(target_audio), n)
        mixture = None if mixture_audio is None else _fit(_mono(mixture_audio), n)
        result.update({
            "target_words_lost": target_words_lost(ref, hyp),
            "wrong_voice_words": wrong_voice_words(ref, hyp),
            "talk_over_silence": talk_over_silence(ref, hyp),
            "activity_f1": activity_f1(ref, hyp),
            "voice_match": (voice_match(embed, out, target, ref.target_activity, sr)
                            if embed is not None and target is not None else math.nan),
            "si_sdr": si_sdr(out, target) if target is not None else math.nan,
            "si_sdr_improvement": (si_sdr_improvement(out, mixture, target)
                                   if target is not None and mixture is not None else math.nan),
            "quality": float(mos(out, sr)) if mos is not None else math.nan,
        })
    result["adapters"] = {"asr": _label(asr), "vad": _label(vad), "embed": _label(embed),
                          "mos": _label(mos), "normalizer": normalizer_name()}
    return result


def json_safe(obj: Any) -> Any:
    """``obj`` with numpy scalars as Python ones and nan or infinite floats as None."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def read_manifest(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def score_system(build_dir: str | Path, system: str, *, asr: ASR, vad: VAD,
                 embed: Embedder | None = None, word_timings: Mapping | None = None,
                 out_path: str | Path | None = None, outputs: str | Path | None = None,
                 mos: QualityPredictor | None = None, sr: int = SAMPLE_RATE,
                 progress: bool = True, items: Collection[str] | None = None) -> list[dict]:
    """Score every item of a build for one system, or only the ``items`` given.

    Reads ``manifest.jsonl``, the clean ``target.wav`` and ``mixture.wav`` of each item from
    ``items/<id>/``, and the system's ``<id>.wav`` from ``outputs`` (default
    ``outputs/<system>/`` under the build). Writes one JSON line per item, in manifest order, to
    ``out_path`` (default ``scores/<system>.jsonl`` under the build), with nan written as null,
    and returns the scores with nan kept.
    """
    build = Path(build_dir)
    rows = read_manifest(build / "manifest.jsonl")
    if items is not None:
        keep = {str(i) for i in items}
        rows = [r for r in rows if str(r.get("id", "")) in keep]
    out_dir = Path(outputs) if outputs is not None else build / "outputs" / system
    dest = Path(out_path) if out_path is not None else build / "scores" / f"{system}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    scores: list[dict] = []
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for k, row in enumerate(rows, start=1):
                ref = reference_from_manifest(row, word_timings)
                if not ref.item_id:
                    raise ValueError(f"manifest row {k} has no id")
                item = build / "items" / ref.item_id
                score = {"system": system, **score_item(
                    ref, load_output(out_dir / f"{ref.item_id}.wav", ref.length_s, sr),
                    asr=asr, vad=vad, embed=embed, sr=sr, mos=mos,
                    target_audio=load_output(item / "target.wav", ref.length_s, sr),
                    mixture_audio=load_output(item / "mixture.wav", ref.length_s, sr))}
                fh.write(json.dumps(json_safe(score), allow_nan=False) + "\n")
                scores.append(score)
                if progress:
                    status = "empty" if score["empty_output"] else "scored"
                    print(f"[{k}/{len(rows)}] {ref.item_id} {status}", file=sys.stderr)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return scores


# A run that finished answers nearly every item: an output file, or an abstention its runner
# logged. Scoring refuses a system below this share unless told otherwise, so that a run that
# stopped halfway is not scored as a system that returned nothing.
MIN_COVERAGE = 0.9
RUNS_LOG = "runs.jsonl"


def unanswered(outputs: str | Path, ids: Iterable[str]) -> list[str]:
    """The ``ids`` with neither an ``<id>.wav`` in ``outputs`` nor a logged abstention there.

    An abstention is a line with ``"status": "abstained"`` in ``outputs/runs.jsonl``, as the
    runners in ``runners/`` write it. An empty WAV is an output, so writing one is the other
    way to return nothing for an item.
    """
    folder = Path(outputs)
    found = {p.stem for p in folder.glob("*.wav")} if folder.is_dir() else set()
    log = folder / RUNS_LOG
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("status") == "abstained" and row.get("id"):
                found.add(str(row["id"]))
    return [str(i) for i in ids if str(i) not in found]


__all__ = [
    "ASR",
    "FRAME_S",
    "MIN_COVERAGE",
    "MIN_OUTPUT_S",
    "MIN_VOICE_S",
    "SAMPLE_RATE",
    "SI_SDR_LIMIT_DB",
    "VAD",
    "WRONG_VOICE_WINDOW_S",
    "DnsmosP808",
    "EcapaEmbedder",
    "Embedder",
    "Hypothesis",
    "QualityPredictor",
    "Reference",
    "SileroVAD",
    "TimedWord",
    "WhisperASR",
    "active_samples",
    "activity_f1",
    "align",
    "asr_words",
    "basic_normalize",
    "corpus_wer",
    "edit_counts",
    "empty_output_scores",
    "hypothesis_from_asr",
    "json_safe",
    "load_output",
    "n_frames",
    "normalize",
    "normalize_timed",
    "normalizer_name",
    "read_manifest",
    "reference_from_manifest",
    "score_item",
    "score_system",
    "si_sdr",
    "si_sdr_improvement",
    "talk_over_silence",
    "target_words_lost",
    "unanswered",
    "voice_match",
    "wrong_voice_words",
]
