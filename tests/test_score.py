import hashlib
import importlib.util
import json
import math

import numpy as np
import pytest
import soundfile as sf

from rightvoice import score
from rightvoice.score import Hypothesis, Reference, TimedWord

SR = 16000

# Words every normalizer leaves as they are: no numbers, fillers or contractions.
TEN = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today"]


class FakeASR:
    name = "fake-asr"
    version = "1"

    def __init__(self, words):
        self.words = [dict(w) for w in words]
        self.calls = 0

    def transcribe(self, audio, sr):
        self.calls += 1
        return [dict(w) for w in self.words]


class FakeVAD:
    name = "fake-vad"
    version = "1"

    def __init__(self, frames):
        self.frames = np.asarray(frames, dtype=bool)

    def activity(self, audio, sr):
        return self.frames.copy()


class FakeEmbed:
    """A fixed vector per distinct input, handed out in order of first appearance."""

    name = "fake-embed"
    version = "1"

    def __init__(self):
        self.vectors = [np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])]
        self.calls = []
        self.seen = {}

    def __call__(self, x, sr):
        x = np.asarray(x)
        self.calls.append(x.copy())
        key = hashlib.sha256(x.tobytes()).hexdigest()
        if key not in self.seen:
            self.seen[key] = self.vectors[len(self.seen)]
        return self.seen[key]


def _missing(module):
    return importlib.util.find_spec(module) is None


def _tw(word, mid, voice="", half=0.15):
    return TimedWord(word, mid - half, mid + half, voice)


def _ref(target=(), other=(), length_s=10.0, active=None, absent=False):
    n = score.n_frames(length_s)
    act = np.zeros(n, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    return Reference(target_words=list(target), other_words=list(other), target_activity=act,
                     length_s=length_s, target_absent=absent)


def _hyp(words, length_s=10.0, active=None):
    n = score.n_frames(length_s)
    act = np.zeros(n, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    return Hypothesis([TimedWord(w.word, w.start, w.end) for w in words], act)


def _row(item_id="item0001"):
    return {
        "id": item_id,
        "window": {"meeting": "TS9999z", "start": 0.0, "length": 10.0},
        "long": False,
        "target_absent": False,
        "placements": [
            {"role": "target", "utterance_id": "t1", "start": 1.0, "seconds": 2.0,
             "text": "The quick brown fox.", "trimmed": False},
            {"role": "partner", "utterance_id": "p1", "start": 3.5, "seconds": 1.0,
             "text": "Hello there.", "trimmed": False},
            {"role": "target", "utterance_id": "t2", "start": 5.0, "seconds": 1.5,
             "text": "Jumps over the dog!", "trimmed": False},
        ],
        "bystanders": [{"reader": "9999", "utterance_ids": ["b1"], "start": 7.0, "seconds": 2.0,
                        "text": "Breaking news tonight"}],
    }


TIMINGS = {"t1": [(0.1, 0.4, "the"), (0.4, 0.8, "quick"), (0.8, 1.2, "brown"),
                  (1.2, 1.8, "fox")]}


def _stems(ref, seed=0):
    """A target stem that sounds only where the target is active, and a mixture with noise."""
    n = round(ref.length_s * SR)
    t = np.arange(n) / SR
    mask = score.active_samples(ref.target_activity, n, SR)
    target = (0.3 * np.sin(2 * np.pi * 220 * t) * mask).astype(np.float32)
    noise = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    return target, (target + 0.1 * noise).astype(np.float32)


# Normalisation

def test_basic_normalize_keeps_inner_apostrophes_only():
    text = "Hello, World! It's 'quoted' rock-n-roll. Don’t"
    assert score.basic_normalize(text) == ["hello", "world", "it's", "quoted", "rock", "n",
                                           "roll", "don't"]
    assert score.basic_normalize("") == []


@pytest.mark.skipif(not _missing("whisper"), reason="the recognition package is installed")
def test_fallback_normalizer_is_active_without_the_package():
    assert score.NORMALIZER_NAME == "basic"
    assert score.normalizer_name() == "basic"
    assert score.normalize("A, b.") == ["a", "b"]


def test_normalize_timed_keeps_times_when_words_merge(monkeypatch):
    def merging(text):
        toks = score.basic_normalize(text)
        out, k = [], 0
        while k < len(toks):
            if toks[k:k + 2] == ["twenty", "one"]:
                out.append("21")
                k += 2
            else:
                out.append({"twenty": "20", "one": "1"}.get(toks[k], toks[k]))
                k += 1
        return out

    monkeypatch.setattr(score, "normalize", merging)
    raw = [(0.0, 0.5, "I"), (0.5, 1.0, "have"), (1.0, 1.4, "twenty"), (1.4, 1.8, "one"),
           (1.8, 2.2, "cats.")]
    words = score.normalize_timed(raw, offset=10.0)
    assert [w.word for w in words] == ["i", "have", "21", "cats"]
    assert (words[2].start, words[2].end) == pytest.approx((11.0, 11.8))
    assert words[3].mid == pytest.approx(12.0)


# Alignment

def test_align_identical_lists_are_all_equal():
    ops = score.align(TEN, TEN)
    assert [op[0] for op in ops] == ["equal"] * 10
    assert ops[3] == ("equal", 3, 3)
    counts = score.edit_counts(ops)
    assert counts == {"sub": 0, "ins": 0, "del": 0, "equal": 10, "n_ref": 10}
    assert score.corpus_wer([counts]) == 0.0


def test_one_substitution_in_ten_words_is_wer_point_one():
    hyp = list(TEN)
    hyp[4] = "leaps"
    ops = score.align(TEN, hyp)
    assert ("sub", 4, 4) in ops
    counts = score.edit_counts(ops)
    assert (counts["sub"], counts["ins"], counts["del"], counts["n_ref"]) == (1, 0, 0, 10)
    assert score.corpus_wer([counts]) == pytest.approx(0.1)


def test_empty_reference_counts_insertions():
    counts = score.edit_counts(score.align([], ["a", "b", "c"]))
    assert counts == {"sub": 0, "ins": 3, "del": 0, "equal": 0, "n_ref": 0}
    assert math.isnan(score.corpus_wer([counts]))
    assert math.isnan(score.corpus_wer([]))


def test_align_breaks_ties_sub_then_del_then_ins():
    assert score.align(["a", "b"], ["c"]) == [("del", 0, None), ("sub", 1, 0)]
    assert score.align(["a"], ["b", "c"]) == [("ins", None, 0), ("sub", 0, 1)]
    assert score.align(["a"], []) == [("del", 0, None)]
    assert score.align([], []) == []


def test_align_is_a_minimal_edit_script():
    def distance(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1))
            prev = cur
        return prev[-1]

    rng = np.random.default_rng(7)
    for _ in range(200):
        a = list(rng.choice(list("abcd"), size=rng.integers(0, 9)))
        b = list(rng.choice(list("abcd"), size=rng.integers(0, 9)))
        ops = score.align(a, b)
        assert sum(op[0] != "equal" for op in ops) == distance(a, b)
        assert [b[j] for _, _, j in ops if j is not None] == b
        assert [a[i] for _, i, _ in ops if i is not None] == a
        assert all(a[i] == b[j] for kind, i, j in ops if kind == "equal")


def test_corpus_wer_is_a_ratio_of_sums():
    one = list(TEN)
    one[0] = "a"
    three = list(TEN)
    three[2] = "green"           # one substitution
    del three[5]                 # one deletion
    three.append("again")        # one insertion
    counts = [score.edit_counts(score.align(TEN, h)) for h in (one, three)]
    assert [c["n_ref"] for c in counts] == [10, 10]
    assert [c["sub"] + c["del"] + c["ins"] for c in counts] == [1, 3]
    assert score.corpus_wer(counts) == pytest.approx(0.2)


# References and hypotheses

def test_reference_from_manifest_words_midpoints_and_activity():
    ref = score.reference_from_manifest(_row(), TIMINGS)
    assert ref.item_id == "item0001" and ref.length_s == 10.0
    assert not ref.target_absent and not ref.long
    assert ref.timings_estimated  # t2, p1 and the bystander have no timings
    assert [w.word for w in ref.target_words] == ["the", "quick", "brown", "fox", "jumps",
                                                  "over", "the", "dog"]
    timed = [1.25, 1.6, 2.0, 2.5]                    # t1 timings shifted by its start
    spread = [5.0 + 0.375 * (k + 0.5) for k in range(4)]  # 1.5 s shared by four words
    assert [w.mid for w in ref.target_words] == pytest.approx(timed + spread)
    assert {w.voice for w in ref.target_words} == {"target"}

    assert [(w.word, w.voice) for w in ref.other_words] == [
        ("hello", "partner"), ("there", "partner"),
        ("breaking", "bystander"), ("news", "bystander"), ("tonight", "bystander")]
    assert [w.mid for w in ref.other_words] == pytest.approx(
        [3.75, 4.25, 7.0 + 1 / 3, 8.0, 9.0 - 1 / 3])

    act = ref.target_activity
    assert act.dtype == bool and len(act) == 500
    assert int(act.sum()) == 100 + 75                # 2.0 s and 1.5 s at 50 Hz
    assert act[50:150].all() and act[250:325].all()
    assert not act[:50].any() and not act[150:250].any() and not act[325:].any()


def test_reference_with_all_timings_is_not_estimated():
    row = {"id": "x", "window": {"length": 4.0},
           "placements": [{"role": "target", "utterance_id": "t1", "start": 1.0,
                           "seconds": 2.0, "text": "The quick brown fox."}]}
    ref = score.reference_from_manifest(row, TIMINGS)
    assert not ref.timings_estimated and len(ref.target_words) == 4


def test_trimmed_placement_drops_timed_words_past_the_cut():
    row = _row()
    row["placements"][0]["seconds"] = 1.1   # t1 cut after "quick" ends at 0.8 s
    row["placements"][0]["trimmed"] = True
    ref = score.reference_from_manifest(row, TIMINGS)
    assert [w.word for w in ref.target_words][:3] == ["the", "quick", "brown"]
    assert ref.target_words[2].end == pytest.approx(2.1)  # brown clipped at the cut
    assert ref.target_words[3].word == "jumps"


def test_target_absent_and_missing_keys():
    row = _row()
    row["target_absent"] = True
    ref = score.reference_from_manifest(row, TIMINGS)
    assert ref.target_absent and ref.target_words == [] and not ref.target_activity.any()
    assert len(ref.other_words) == 5

    bare = score.reference_from_manifest({"id": "y", "window": {"length": 2.0}})
    assert bare.target_words == [] and bare.other_words == []
    assert len(bare.target_activity) == 100 and not bare.timings_estimated


def test_bystander_with_one_timed_utterance_uses_its_timings():
    row = {"id": "z", "window": {"length": 5.0},
           "bystanders": [{"utterance_id": "b1", "start": 2.0, "seconds": 1.0}]}
    ref = score.reference_from_manifest(row, {"b1": [(0.1, 0.3, "news"), (1.5, 1.7, "late")]})
    assert [(w.word, w.voice) for w in ref.other_words] == [("news", "bystander")]
    assert ref.other_words[0].mid == pytest.approx(2.2)
    assert not ref.timings_estimated


def test_hypothesis_from_asr_flattens_segments_and_clips():
    result = {"segments": [
        {"words": [{"word": " Hello,", "start": 0.1, "end": 0.4},
                   {"word": " world.", "start": 0.5, "end": 0.9}]},
        {"words": [{"word": " late", "start": 2.5, "end": 2.8}]}]}
    hyp = score.hypothesis_from_asr(result, 2.0)
    assert [w.word for w in hyp.words] == ["hello", "world"]
    assert len(hyp.activity) == 100 and int(hyp.activity.sum()) == 15 + 20
    flat = score.hypothesis_from_asr(result["segments"][0]["words"], 2.0,
                                     activity=np.ones(50, dtype=bool))
    assert [w.word for w in flat.words] == ["hello", "world"]
    assert int(flat.activity.sum()) == 50 and len(flat.activity) == 100


# Metrics

def test_target_words_lost_reports_counts_and_wer():
    ref = _ref([_tw(w, 0.5 + k) for k, w in enumerate(TEN[:4])])
    hyp = _hyp([_tw("the", 0.5), _tw("quick", 1.5), _tw("brown", 2.5)])
    r = score.target_words_lost(ref, hyp)
    assert (r["del"], r["n_ref"], r["wer"]) == (1, 4, pytest.approx(0.25))


def test_wrong_voice_partner_word_at_its_time_counts():
    target = [_tw("good", 1.0, "target"), _tw("morning", 1.5, "target")]
    ref = _ref(target, [_tw("hello", 3.0, "partner")])
    r = score.wrong_voice_words(ref, _hyp([*target, _tw("hello", 3.2)]))
    assert (r["count"], r["candidates"], r["by_voice"]) == (1, 1, {"partner": 1})
    assert r["per_target_word"] == pytest.approx(0.5)
    assert r["per_minute"] == pytest.approx(6.0)


def test_wrong_voice_substitution_counts_too():
    target = [_tw("good", 1.0, "target"), _tw("morning", 1.5, "target")]
    ref = _ref(target, [_tw("hello", 1.6, "partner")])
    r = score.wrong_voice_words(ref, _hyp([target[0], _tw("hello", 1.5)]))
    assert r["count"] == 1


def test_wrong_voice_same_word_two_seconds_away_does_not_count():
    target = [_tw("good", 1.0, "target"), _tw("morning", 1.5, "target")]
    ref = _ref(target, [_tw("hello", 3.0, "partner")])
    r = score.wrong_voice_words(ref, _hyp([*target, _tw("hello", 5.0)]))
    assert (r["count"], r["candidates"]) == (0, 1)


def test_wrong_voice_reference_word_matches_once():
    target = [_tw("good", 1.0, "target")]
    once = _ref(target, [_tw("hello", 3.0, "partner")])
    hyp = _hyp([*target, _tw("hello", 2.9), _tw("hello", 3.3)])
    assert score.wrong_voice_words(once, hyp)["count"] == 1
    twice = _ref(target, [_tw("hello", 3.0, "partner"), _tw("hello", 3.4, "bystander")])
    r = score.wrong_voice_words(twice, hyp)
    assert r["count"] == 2 and r["by_voice"] == {"partner": 1, "bystander": 1}


def test_talk_over_silence_on_a_target_absent_item():
    row = {"id": "a", "window": {"length": 60.0}, "target_absent": True,
           "placements": [{"role": "target", "utterance_id": "t", "start": 0.0,
                           "seconds": 60.0, "text": "gone"}]}
    ref = score.reference_from_manifest(row, {})
    words = [{"word": "word", "start": k + 0.2, "end": k + 0.6} for k in range(60)]
    active = np.zeros(3000, dtype=bool)
    active[:750] = True
    hyp = score.hypothesis_from_asr(words, 60.0, activity=active)
    r = score.talk_over_silence(ref, hyp)
    assert r["words"] == 60
    assert r["per_minute"] == pytest.approx(60.0)
    assert r["speech_share"] == pytest.approx(0.25)
    assert r["silent_s"] == pytest.approx(60.0) and r["speech_s"] == pytest.approx(15.0)


def test_talk_over_silence_is_zero_when_the_target_speaks_throughout():
    row = {"id": "b", "window": {"length": 60.0},
           "placements": [{"role": "target", "utterance_id": "t", "start": 0.0,
                           "seconds": 60.0, "text": "word"}]}
    ref = score.reference_from_manifest(row, {})
    assert ref.target_activity.all()
    words = [{"word": "word", "start": k + 0.2, "end": k + 0.6} for k in range(60)]
    hyp = score.hypothesis_from_asr(words, 60.0, activity=np.ones(3000, dtype=bool))
    r = score.talk_over_silence(ref, hyp)
    assert (r["words"], r["per_minute"], r["speech_share"], r["silent_s"]) == (0, 0.0, 0.0, 0.0)


def test_activity_f1():
    act = np.zeros(500, dtype=bool)
    act[50:150] = True
    act[250:325] = True
    ref = _ref(active=act)
    same = score.activity_f1(ref, _hyp([], active=act))
    assert (same["precision"], same["recall"], same["f1"]) == (1.0, 1.0, 1.0)
    assert score.activity_f1(ref, _hyp([], active=~act))["f1"] == 0.0
    assert score.activity_f1(ref, _hyp([]))["f1"] == 0.0
    assert score.activity_f1(_ref(), _hyp([], active=act))["f1"] == 0.0
    nothing = score.activity_f1(_ref(), _hyp([]))
    assert math.isnan(nothing["f1"]) and math.isnan(nothing["recall"])


def test_si_sdr():
    t = np.arange(SR) / SR
    sine = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert score.si_sdr(0.3 * sine, sine) > 60
    noise = (0.5 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)
    assert score.si_sdr(noise, sine) < 1.0
    mix = sine + noise
    est = sine + 0.1 * noise
    imp = score.si_sdr_improvement(est, mix, sine)
    assert imp == pytest.approx(score.si_sdr(est, sine) - score.si_sdr(mix, sine))
    assert imp > 15
    assert math.isnan(score.si_sdr(sine, np.zeros(SR, dtype=np.float32)))
    assert math.isnan(score.si_sdr_improvement(est, mix, np.zeros(SR, dtype=np.float32)))


def test_voice_match_uses_only_target_active_samples():
    n = 4 * SR
    target = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    out = target[::-1].copy()
    act = np.zeros(200, dtype=bool)
    act[25:75] = True      # 0.5 s to 1.5 s
    act[125:175] = True    # 2.5 s to 3.5 s
    embed = FakeEmbed()
    value = score.voice_match(embed, out, target, act, SR)
    picked = np.r_[8000:24000, 40000:56000]
    assert [len(c) for c in embed.calls] == [32000, 32000]
    assert np.array_equal(embed.calls[0], out[picked])
    assert np.array_equal(embed.calls[1], target[picked])
    assert value == pytest.approx(1 / math.sqrt(2))

    assert score.voice_match(FakeEmbed(), target, target, act, SR) == pytest.approx(1.0)
    short = np.zeros(200, dtype=bool)
    short[:40] = True      # 0.8 s
    lazy = FakeEmbed()
    assert math.isnan(score.voice_match(lazy, out, target, short, SR))
    assert lazy.calls == []


# Per-item scoring

def _metrics(result):
    keys = score.empty_output_scores(_ref()).keys()
    return {k: result[k] for k in keys}


def test_empty_output_scores_and_score_item_none_agree():
    ref = score.reference_from_manifest(_row(), TIMINGS)
    target, mixture = _stems(ref)
    asr, embed = FakeASR([{"word": "x", "start": 1.0, "end": 1.2}]), FakeEmbed()
    empty = score.empty_output_scores(ref)
    result = score.score_item(ref, None, asr=asr, vad=FakeVAD(ref.target_activity),
                              embed=embed, target_audio=target, mixture_audio=mixture)
    assert result["empty_output"] is True and asr.calls == 0 and embed.calls == []
    assert json.dumps(_metrics(result), sort_keys=True) == json.dumps(empty, sort_keys=True)
    assert result["adapters"]["asr"] == {"name": "fake-asr", "version": "1"}

    assert empty["target_words_lost"]["wer"] == 1.0
    assert empty["target_words_lost"]["del"] == len(ref.target_words) == 8
    assert empty["wrong_voice_words"]["count"] == 0
    assert empty["talk_over_silence"]["words"] == 0
    assert empty["talk_over_silence"]["speech_share"] == 0.0
    assert empty["activity_f1"]["f1"] == 0.0
    assert math.isnan(empty["voice_match"]) and math.isnan(empty["si_sdr"])

    row = _row()
    row["target_absent"] = True
    absent = score.empty_output_scores(score.reference_from_manifest(row, TIMINGS))
    assert math.isnan(absent["target_words_lost"]["wer"])
    assert math.isnan(absent["activity_f1"]["f1"])


@pytest.mark.parametrize("audio", [np.zeros(SR, dtype=np.float32),
                                   np.full(SR // 200, 0.1, dtype=np.float32),
                                   np.full(SR, np.nan, dtype=np.float32),
                                   np.zeros(0, dtype=np.float32)])
def test_silent_short_or_broken_output_is_empty(audio):
    ref = score.reference_from_manifest(_row(), TIMINGS)
    asr = FakeASR([])
    result = score.score_item(ref, audio, asr=asr, vad=FakeVAD(ref.target_activity),
                              embed=None)
    assert result["empty_output"] is True and asr.calls == 0


def test_score_item_on_a_system_that_returns_the_target():
    ref = score.reference_from_manifest(_row(), TIMINGS)
    target, mixture = _stems(ref)
    words = [{"word": w.word, "start": w.start, "end": w.end} for w in ref.target_words]
    embed = FakeEmbed()
    result = score.score_item(ref, target, asr=FakeASR(words), vad=FakeVAD(ref.target_activity),
                              embed=embed, target_audio=target, mixture_audio=mixture)
    assert result["empty_output"] is False and result["id"] == "item0001"
    assert result["target_words_lost"]["wer"] == 0.0
    assert result["wrong_voice_words"]["count"] == 0
    assert result["talk_over_silence"]["words"] == 0
    assert result["activity_f1"]["f1"] == 1.0
    assert result["voice_match"] == pytest.approx(1.0)
    assert [len(c) for c in embed.calls] == [175 * 320] * 2
    assert result["si_sdr"] > 60 and result["si_sdr_improvement"] > 0
    assert math.isnan(result["quality"])
    assert result["adapters"]["vad"]["name"] == "fake-vad"
    assert result["adapters"]["embed"]["name"] == "fake-embed"
    assert result["adapters"]["normalizer"] == score.normalizer_name()
    json.dumps(score.json_safe(result), allow_nan=False)


def test_score_item_pads_short_output_to_the_item():
    ref = score.reference_from_manifest(_row(), TIMINGS)
    target, _ = _stems(ref)
    seen = []

    class Recorder(FakeASR):
        def transcribe(self, audio, sr):
            seen.append(len(audio))
            return super().transcribe(audio, sr)

    score.score_item(ref, target[: 3 * SR], asr=Recorder([]), vad=FakeVAD([]))
    score.score_item(ref, np.concatenate([target, target]), asr=Recorder([]), vad=FakeVAD([]))
    assert seen == [10 * SR, 10 * SR]


def test_load_output(tmp_path):
    assert score.load_output(tmp_path / "missing.wav", 1.0) is None
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    assert score.load_output(bad, 1.0) is None

    t = np.arange(44100) / 44100
    left = 0.4 * np.sin(2 * np.pi * 300 * t)
    path = tmp_path / "out.wav"
    sf.write(path, np.stack([left, 0.5 * left], axis=1), 44100, subtype="FLOAT")
    x = score.load_output(path, 2.0)
    assert x.dtype == np.float32 and x.ndim == 1 and len(x) == SR
    assert np.sqrt(np.mean(x[1000:-1000] ** 2)) == pytest.approx(0.75 * 0.4 / math.sqrt(2),
                                                                 rel=0.02)
    assert len(score.load_output(path, 0.5)) == SR // 2


def test_score_system_end_to_end(tmp_path):
    build = tmp_path / "build"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    rows = []
    for item_id in ("a", "b"):
        row = {"id": item_id, "window": {"length": 2.0}, "target_absent": False, "long": False,
               "placements": [{"role": "target", "utterance_id": f"t_{item_id}", "start": 0.5,
                               "seconds": 1.0, "text": "Alpha bravo."}],
               "bystanders": []}
        rows.append(row)
        ref = score.reference_from_manifest(row)
        target, mixture = _stems(ref)
        item = build / "items" / item_id
        item.mkdir(parents=True)
        sf.write(item / "target.wav", target, SR, subtype="FLOAT")
        sf.write(item / "mixture.wav", mixture, SR, subtype="FLOAT")
        if item_id == "a":
            sf.write(outputs / "a.wav", target, SR, subtype="FLOAT")
    (build / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    frames = np.zeros(100, dtype=bool)
    frames[25:75] = True
    asr = FakeASR([{"word": " Alpha", "start": 0.6, "end": 0.9},
                   {"word": " bravo.", "start": 1.0, "end": 1.3}])
    scores = score.score_system(build, "sys", asr=asr, vad=FakeVAD(frames), embed=FakeEmbed(),
                                outputs=outputs, progress=False)
    assert len(scores) == 2 and asr.calls == 1

    path = build / "scores" / "sys.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 2

    def strict(token):
        raise ValueError(f"non-standard JSON constant {token}")

    parsed = [json.loads(line, parse_constant=strict) for line in lines]
    assert [p["id"] for p in parsed] == ["a", "b"]
    assert [p["system"] for p in parsed] == ["sys", "sys"]
    assert [p["empty_output"] for p in parsed] == [False, True]
    assert parsed[0]["target_words_lost"]["wer"] == 0.0
    assert parsed[0]["voice_match"] == pytest.approx(1.0)
    assert parsed[1]["voice_match"] is None and math.isnan(scores[1]["voice_match"])
    assert not list(path.parent.glob("*.tmp*"))

    elsewhere = tmp_path / "elsewhere.jsonl"
    score.score_system(build, "sys", asr=asr, vad=FakeVAD(frames), embed=None,
                       outputs=outputs, out_path=elsewhere, progress=False)
    assert len(elsewhere.read_text().splitlines()) == 2

    one = tmp_path / "one.jsonl"
    only_b = score.score_system(build, "sys", asr=asr, vad=FakeVAD(frames), embed=None,
                                outputs=outputs, out_path=one, progress=False, items=["b"])
    assert [x["id"] for x in only_b] == ["b"]
    assert [json.loads(x)["id"] for x in one.read_text().splitlines()] == ["b"]


def test_unanswered_skips_outputs_and_logged_abstentions(tmp_path):
    out = tmp_path / "out"
    assert score.unanswered(out, ["a"]) == ["a"]
    out.mkdir()
    (out / "a.wav").write_bytes(b"")
    (out / "runs.jsonl").write_text('{"run": "start"}\n{"id": "b", "status": "abstained"}\n'
                                     '{"id": "c", "status": "error"}\nnot json\n')
    assert score.unanswered(out, ["a", "b", "c", "d"]) == ["c", "d"]


# Adapters

def test_adapters_have_a_name_and_a_version():
    for adapter in (score.WhisperASR(), score.SileroVAD(), score.EcapaEmbedder()):
        assert isinstance(adapter.name, str) and adapter.name
        assert isinstance(adapter.version, str) and adapter.version


def test_recognition_cache_hit_needs_no_package(tmp_path):
    asr = score.WhisperASR(cache_dir=tmp_path)
    audio = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    words = [{"word": " hi", "start": 0.0, "end": 0.05}]
    (tmp_path / f"{asr.cache_key(audio)}.json").write_text(json.dumps(words))
    assert asr.transcribe(audio, SR) == words
    assert score.WhisperASR(model="other", cache_dir=tmp_path).cache_key(audio) != \
        asr.cache_key(audio)


@pytest.mark.skipif(not _missing("whisper"), reason="the recognition package is installed")
def test_recognition_adapter_names_its_missing_package(tmp_path):
    with pytest.raises(RuntimeError, match="openai-whisper"):
        score.WhisperASR(cache_dir=tmp_path).transcribe(np.full(1600, 0.1, np.float32), SR)


@pytest.mark.skipif(not _missing("silero_vad"), reason="the activity package is installed")
def test_activity_adapter_names_its_missing_package():
    with pytest.raises(RuntimeError, match="silero-vad"):
        score.SileroVAD().activity(np.full(1600, 0.1, np.float32), SR)


@pytest.mark.skipif(not _missing("speechbrain"), reason="the embedding package is installed")
def test_embedding_adapter_names_its_missing_package():
    with pytest.raises(RuntimeError, match="speechbrain"):
        score.EcapaEmbedder()(np.full(1600, 0.1, np.float32), SR)


def test_bystander_pieces_shape():
    """The builder's bystander row: start_s and pieces joined by 0.3 s gaps."""
    row = {
        "id": "x", "window": {"length": 30.0}, "placements": [
            {"role": "target", "utterance_id": "t1", "start": 0.0, "seconds": 2.0, "text": "one two"}],
        "bystanders": [{"reader": "9", "utterance_ids": ["b1", "b2"], "start_s": 10.0, "seconds": 4.3,
                        "pieces": [{"utterance_id": "b1", "seconds": 2.0, "text": "alpha beta"},
                                   {"utterance_id": "b2", "seconds": 2.0, "text": "gamma delta"}]}],
    }
    timings = {"b1": [(0.0, 1.0, "alpha"), (1.0, 2.0, "beta")]}
    ref = score.reference_from_manifest(row, timings)
    by = [w for w in ref.other_words if w.voice == "bystander"]
    assert [w.word for w in by] == ["alpha", "beta", "gamma", "delta"]
    assert abs(by[0].start - 10.0) < 1e-6 and abs(by[1].end - 12.0) < 1e-6
    assert 12.3 <= by[2].start < 13.3 and by[3].end <= 14.3 + 1e-6
    assert ref.timings_estimated is True


def test_dnsmos_p808_segments_and_averages(monkeypatch):
    calls = []

    class FakeSession:
        def run(self, _outputs, feeds):
            feats = feeds["input_1"]
            calls.append(feats.shape)
            return [np.array([[3.0 + 0.5 * len(calls)]], dtype=np.float32)]

    def fake_mel(audio, sr):
        assert sr == 16000 and audio.size == int(9.01 * 16000) - 160
        return np.zeros((901, 120), dtype=np.float32)

    monkeypatch.setattr(score.DnsmosP808, "_melspec", staticmethod(fake_mel))
    mos = score.DnsmosP808(session=FakeSession())
    # 30 s clip: hops = floor(30) - 9.01 + 1 -> 21 segments
    value = mos(np.random.default_rng(0).standard_normal(30 * 16000).astype(np.float32) * 0.1, 16000)
    assert len(calls) == 21 and calls[0] == (1, 901, 120)
    assert abs(value - (3.0 + 0.5 * 11)) < 1e-6  # mean of 3.5 .. 13.5
    # a 1 s clip is tiled by doubling to 16 s, as the reference script does, giving 7 hops
    calls.clear()
    short = mos(np.ones(16000, dtype=np.float32) * 0.1, 16000)
    assert len(calls) == 7 and abs(short - (3.0 + 0.5 * 4)) < 1e-6
