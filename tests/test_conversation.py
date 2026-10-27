import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rightvoice import conversation as conv
from rightvoice.conversation import (
    Conversation,
    NotEnoughSpeech,
    Placement,
    Segment,
    Slot,
    Window,
    Word,
)
from rightvoice.libritts import Utterance
from rightvoice.rng import stream

ROOT = Path(__file__).resolve().parents[1]
REAL_AMI = ROOT / "data" / "sources" / "ami_annotations"
SEED = 20261022

HEAD = ('<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>\n'
        '<nite:root nite:id="{root_id}" xmlns:nite="http://nite.sourceforge.net/">\n')
TAIL = "</nite:root>\n"


def _seg_xml(meeting, agent, rows):
    """rows: (start, end, first word index, last word index or None for no link)."""
    lines = [HEAD.format(root_id=f"{meeting}.{agent}.segs")]
    for k, (s, e, a, b) in enumerate(rows):
        lines.append(f'   <segment nite:id="{meeting}.sync.{k}" channel="0" '
                     f'transcriber_start="{s}" transcriber_end="{e}">\n')
        if a is not None:
            ref = f"id({meeting}.{agent}.words{a})"
            if b is not None and b != a:
                ref += f"..id({meeting}.{agent}.words{b})"
            lines.append(f'      <nite:child href="{meeting}.{agent}.words.xml#{ref}"/>\n')
        lines.append("   </segment>\n")
    lines.append(TAIL)
    return "".join(lines)


def _words_xml(meeting, agent, body):
    return HEAD.format(root_id=f"{meeting}.{agent}.words") + body + TAIL


def _write_ami(root, meeting="TS9999z"):
    """Two agents with three usable segments each, written as ISO-8859-1 bytes."""
    m = meeting
    a_words = (
        f'   <w nite:id="{m}.A.words0" starttime="1.0" endtime="1.4">Hello</w>\n'
        f'   <w nite:id="{m}.A.words1" starttime="1.4" endtime="1.4" punc="true">,</w>\n'
        f'   <w nite:id="{m}.A.words2" starttime="1.4" endtime="1.9">café</w>\n'
        f'   <vocalsound nite:id="{m}.A.words3" starttime="3.0" endtime="3.2" type="laugh"/>\n'
        f'   <w nite:id="{m}.A.words4" starttime="3.2" endtime="3.5">yes</w>\n'
        f'   <w nite:id="{m}.A.words5">Mm</w>\n'
        f'   <w nite:id="{m}.A.words6" starttime="6.0" endtime="6.3">right</w>\n'
        f'   <w nite:id="{m}.A.words7" starttime="6.3" endtime="6.8">then</w>\n'
    )
    b_words = (
        f'   <w nite:id="{m}.B.words0" starttime="2.0" endtime="2.4">okay</w>\n'
        f'   <w nite:id="{m}.B.words1" starttime="2.4" endtime="2.4" punc="true">.</w>\n'
        f'   <disfmarker nite:id="{m}.B.words2" starttime="4.0" endtime="4.1"/>\n'
        f'   <w nite:id="{m}.B.words3" starttime="4.1" endtime="4.3">so</w>\n'
        f'   <w nite:id="{m}.B.words4" starttime="4.3" endtime="4.8">über</w>\n'
        f'   <w nite:id="{m}.B.words5" starttime="7.1" endtime="7.5">sure</w>\n'
        f'   <vocalsound nite:id="{m}.B.words6" starttime="8.5" endtime="8.9" type="cough"/>\n'
        f'   <gap/>\n'
        f'   <w nite:id="{m}.B.words7" starttime="9.5" endtime="9.6">no</w>\n'
    )
    a_segs = [(0.9, 2.0, 0, 2), (2.9, 3.8, 3, 5), (5.9, 7.0, 6, 7)]
    b_segs = [
        (1.95, 2.5, 0, 1),
        (4.0, 5.0, 2, 4),
        (7.0, 8.0, None, None),   # no link: words are taken by time
        (8.5, 9.0, 6, 6),         # only a vocal sound: dropped
        (9.5, 9.5, 7, 7),         # end <= start: dropped
    ]
    (root / "segments").mkdir(parents=True, exist_ok=True)
    (root / "words").mkdir(parents=True, exist_ok=True)
    for agent, segs, words in (("A", a_segs, a_words), ("B", b_segs, b_words)):
        (root / "segments" / f"{m}.{agent}.segments.xml").write_bytes(
            _seg_xml(m, agent, segs).encode("iso-8859-1"))
        (root / "words" / f"{m}.{agent}.words.xml").write_bytes(
            _words_xml(m, agent, words).encode("iso-8859-1"))
    return root


def test_parse_meeting_segments_words_and_latin1(tmp_path):
    ami = _write_ami(tmp_path / "ami")
    _write_ami(tmp_path / "ami", meeting="ES0001a")
    assert b"caf\xe9" in (ami / "words" / "TS9999z.A.words.xml").read_bytes()
    assert conv.list_meetings(ami) == ["ES0001a", "TS9999z"]

    m = conv.parse_meeting(ami, "TS9999z")
    assert sorted(m) == ["A", "B"]
    assert [(g.start, g.end) for g in m["A"]] == [(0.9, 2.0), (2.9, 3.8), (5.9, 7.0)]
    assert [len(g.words) for g in m["A"]] == [2, 1, 2]
    assert [(g.start, g.end) for g in m["B"]] == [(1.95, 2.5), (4.0, 5.0), (7.0, 8.0)]
    assert [len(g.words) for g in m["B"]] == [1, 2, 1]
    assert m["A"][0].words == (Word(1.0, 1.4, "Hello"), Word(1.4, 1.9, "café"))
    assert [w.text for w in m["A"][1].words] == ["yes"]
    assert [w.text for w in m["B"][1].words] == ["so", "über"]
    assert [w.text for w in m["B"][2].words] == ["sure"]


def test_load_meetings_caches_under_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGHTVOICE_DATA", str(tmp_path / "data"))
    ami = _write_ami(tmp_path / "ami")
    first = conv.load_meetings(ami)
    assert conv.timeline_cache_path() == tmp_path / "data" / "cache" / "ami_timelines.json"
    assert conv.timeline_cache_path().exists()
    assert first == {"TS9999z": conv.parse_meeting(ami, "TS9999z")}
    (ami / "segments" / "TS9999z.B.segments.xml").unlink()
    assert conv.load_meetings(ami) == first
    rebuilt = conv.load_meetings(ami, rebuild=True)
    assert sorted(rebuilt["TS9999z"]) == ["A"]


def _seg(start, end):
    return Segment(start, end, (Word(start, end, "w"),))


TARGET = [_seg(1.0, 5.0), _seg(9.0, 12.0), _seg(13.0, 15.0)]
PARTNER = [_seg(4.0, 6.0), _seg(10.0, 10.5), _seg(16.0, 18.0)]


def test_activity_frames():
    act = conv.activity(TARGET, 0.0, 20.0)
    assert act.dtype == bool and len(act) == 1000
    assert act.sum() == 450
    assert not act[49] and act[50] and act[249] and not act[250]
    part = conv.activity(TARGET, 0.5, 2.0)
    assert len(part) == 100 and part.sum() == 75 and not part[:25].any()


def test_features_on_a_hand_built_pair():
    f = conv.features(TARGET, PARTNER, 0.0, 20.0)
    assert f["target_share"] == pytest.approx(0.45)
    assert f["partner_share"] == pytest.approx(0.225)
    # overlap: 4.0 to 5.0 and 10.0 to 10.5 out of 9.0 s of target speech
    assert f["overlap_share"] == pytest.approx(1.5 / 9.0, abs=1e-3)
    # target silent from 5.0 to 9.0, then speaks again
    assert abs(f["reentry_s"] - 4.0) <= 0.05
    assert f["target_turns"] == 3 and f["partner_turns"] == 3
    assert f["backchannels"] == 1
    # a silence running in from before the window start is not a re-entry
    later = conv.features(TARGET, PARTNER, 6.0, 10.0)
    assert abs(later["reentry_s"] - 1.0) <= 0.05
    # the partner segment ending exactly at 6.0 and the one starting at 16.0 lie outside
    assert later["target_turns"] == 2 and later["partner_turns"] == 1


def _meetings():
    a = [_seg(0.5, 3.0), _seg(5.0, 8.0), _seg(10.5, 13.0), _seg(15.0, 18.0),
         _seg(20.5, 23.0), _seg(25.0, 28.0)]
    b = [_seg(3.2, 4.8), _seg(8.2, 9.8), _seg(13.2, 14.8), _seg(18.2, 19.8)]
    c = [_seg(29.0, 30.0)]
    return {"M1": {"A": a, "B": b, "C": c}}


def test_candidate_windows_keeps_and_drops():
    meetings = _meetings()
    wins = conv.candidate_windows(meetings, 10.0, 10.0)
    keys = {(w.meeting, w.target, w.partner, w.start) for w in wins}
    assert ("M1", "A", "B", 0.0) in keys
    assert ("M1", "B", "A", 0.0) in keys
    assert ("M1", "A", "B", 20.0) not in keys       # partner silent in [20, 30)
    assert all(w.partner != "C" for w in wins)       # C barely speaks
    assert all(w.start + w.length <= 30.0 for w in wins)
    for w in wins:
        agents = meetings[w.meeting]
        assert w.features == conv.features(agents[w.target], agents[w.partner], w.start, w.length)
    assert conv.candidate_windows(meetings, 10.0, 10.0) == wins


def test_slots_for_clips_merges_and_drops():
    segs = {
        "A": [_seg(8.0, 12.0), _seg(12.1, 13.0), _seg(15.0, 15.4), _seg(19.0, 25.0)],
        "B": [_seg(13.5, 14.5), _seg(16.0, 16.3), _seg(16.4, 16.8), _seg(30.0, 31.0)],
    }
    window = Window("M", "A", "B", 10.0, 10.0, {})
    assert conv.slots_for(window, segs) == [
        Slot("target", 0.0, 3.0),     # clipped at the window start and merged across 0.1 s
        Slot("partner", 3.5, 4.5),
        Slot("partner", 6.0, 6.8),    # two short pieces merged into one long enough
        Slot("target", 9.0, 10.0),    # clipped at the window end
    ]


def _u(uid, speaker, seconds, chapter="100", text=None):
    return Utterance(id=uid, speaker=speaker, chapter=chapter, split="dev-clean",
                     path=f"dev-clean/{speaker}/{chapter}/{uid}.wav", text=text or f"text {uid}",
                     seconds=seconds, sample_rate=24000)


def _pool(speaker, durations, chapter="100"):
    return [_u(f"{speaker}_{chapter}_{i:06d}_000000", speaker, d, chapter)
            for i, d in enumerate(durations)]


def test_fill_matches_durations_and_is_deterministic():
    slots = [Slot("target", 0.0, 4.0), Slot("partner", 4.5, 7.5), Slot("target", 8.0, 10.0)]
    target = _pool("10", [4.1, 3.9, 3.8, 2.0, 2.1, 9.0, 6.0])
    partner = _pool("20", [3.0, 2.9, 3.2, 1.0])
    seen = set()
    for index in range(20):
        placed = conv.fill(slots, target, partner, stream(SEED, index, "utterances"))
        assert len(placed) == 3
        for slot, p in zip(slots, placed):
            length = slot.end - slot.start
            assert p.role == slot.role and p.start == slot.start and not p.trimmed
            assert abs(p.seconds - length) <= 0.15 * length
        assert len({p.utterance_id for p in placed}) == len(placed)
        again = conv.fill(slots, target, partner, stream(SEED, index, "utterances"))
        assert again == placed
        seen.add(placed[0].utterance_id)
    assert len(seen) > 1   # the draw among matching utterances is random


def test_fill_long_slot_short_slot_and_exhaustion():
    rng = stream(SEED, 0, "utterances")
    target = _pool("10", [5.0, 2.4, 1.0])
    placed = conv.fill([Slot("target", 0.0, 8.0)], target, [], rng)
    assert [(p.seconds, p.start) for p in placed] == [(5.0, 0.0), (2.4, 5.3)]
    assert placed[1].start == pytest.approx(placed[0].start + placed[0].seconds + 0.3)

    placed = conv.fill([Slot("target", 2.0, 3.0)], _pool("10", [3.0, 2.0]), [], rng)
    assert len(placed) == 1
    assert placed[0].trimmed and placed[0].seconds == pytest.approx(1.0)
    assert placed[0].utterance_id == "10_100_000001_000000"

    with pytest.raises(NotEnoughSpeech):
        conv.fill([Slot("target", 0.0, 2.0), Slot("target", 3.0, 5.0)],
                  _pool("10", [2.0]), [], rng)
    with pytest.raises(NotEnoughSpeech):
        conv.fill([Slot("partner", 0.0, 2.0)], _pool("10", [2.0]), [], rng)


def test_fill_does_not_run_into_the_same_role():
    rng = stream(SEED, 1, "utterances")
    slots = [Slot("target", 0.0, 4.0), Slot("target", 4.3, 6.0)]
    placed = conv.fill(slots, _pool("10", [4.5, 3.0, 1.7]), [], rng)
    first = [p for p in placed if p.start < 4.3]
    # 4.5 s would overlap the next target slot, so the longest shorter one goes first
    # and the 0.7 s left after the gap gets a trimmed utterance
    assert [(p.start, p.seconds, p.trimmed) for p in first] == [
        (0.0, 3.0, False), (3.3, pytest.approx(0.7), True)]
    assert all(p.seconds != 4.5 for p in first)
    placed = conv.fill([Slot("target", 0.0, 4.0)], _pool("10", [4.5, 3.0]), [], rng, end_s=4.2)
    assert placed[0].seconds == 3.0              # 4.5 would run past the window


def test_fill_random_pool_never_reuses():
    rng = stream(SEED, 2, "utterances")
    durations = np.round(rng.uniform(1.0, 12.0, size=80), 2)
    target = _pool("10", durations[:40])
    partner = _pool("20", durations[40:])
    slots = conv.slots_for(Window("M1", "A", "B", 0.0, 30.0, {}), _meetings()["M1"])
    placed = conv.fill(slots, target, partner, rng, end_s=30.0)
    ids = [p.utterance_id for p in placed]
    assert len(ids) == len(set(ids))
    for slot in slots:
        inside = [p for p in placed if p.role == slot.role and slot.start <= p.start < slot.end]
        assert inside and inside[0].start == slot.start
    assert all(p.start + p.seconds <= 30.0 + 1e-9 for p in placed)


def _catalogue():
    return (_pool("10", [2.4, 2.6, 3.1, 1.8, 3.0, 1.6, 2.7, 3.3], chapter="100")
            + _pool("10", [2.5, 3.0, 1.7, 2.6, 3.2, 1.9], chapter="101")
            + _pool("20", [1.6, 1.7, 1.8, 2.0, 1.5, 1.9, 1.6], chapter="200"))


def test_plan_conversation_keeps_the_sample_chapter_out():
    window = conv.candidate_windows(_meetings(), 10.0, 10.0)[0]
    assert (window.target, window.partner, window.start) == ("A", "B", 0.0)
    segs = _meetings()["M1"]
    cat = _catalogue()
    by_id = {u.id: u for u in cat}
    plan = conv.plan_conversation(window, segs, cat, "10", "20", stream(SEED, 3, "utterances"),
                                  sample_chapter="100")
    assert plan.sample_chapter == "100" and plan.placements
    for p in plan.placements:
        u = by_id[p.utterance_id]
        assert u.speaker == ("10" if p.role == "target" else "20")
        assert not (p.role == "target" and u.chapter == "100")
        assert p.text == u.text
    with pytest.raises(ValueError):
        conv.plan_conversation(window, segs, cat, "10", "10", stream(SEED, 3, "utterances"))


def test_voice_sample_candidates():
    cat = _catalogue()
    got = conv.voice_sample_candidates(cat, "10", "101", 2.0, 3.0)
    assert [u.seconds for u in got] == [2.5, 3.0, 2.6]
    assert all(u.chapter == "101" and u.speaker == "10" for u in got)


def _tone(seconds, rate=24000, freq=440.0, amp=0.5):
    t = np.arange(round(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_render_stems_places_trims_and_clips(tmp_path):
    corpus = tmp_path / "LibriTTS_R"
    cat = [_u("10_100_000000_000000", "10", 0.5), _u("10_100_000001_000000", "10", 2.0),
           _u("20_200_000000_000000", "20", 1.0, chapter="200")]
    for u in cat:
        (corpus / u.path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(corpus / u.path, _tone(u.seconds), 24000, subtype="FLOAT")
    window = Window("M1", "A", "B", 0.0, 3.0, {})
    placements = [
        Placement("target", cat[0].id, 1.0, 0.5, cat[0].text),
        Placement("target", cat[1].id, 2.0, 0.6, cat[1].text, trimmed=True),
        Placement("partner", cat[2].id, 2.5, 1.0, cat[2].text),   # runs past the window end
    ]
    plan = Conversation(window, "10", "20", placements, None)
    stems = conv.render_stems(plan, {u.id: u for u in cat}, corpus)
    sr = 16000
    assert set(stems) == {"target", "partner"}
    for x in stems.values():
        assert x.dtype == np.float32 and len(x) == 48000

    target = stems["target"].astype(np.float64)
    margin = int(0.003 * sr)

    def energy(x, a, b):
        return float(np.sum(x[max(round(a * sr), 0):round(b * sr)] ** 2))

    inside = energy(target, 1.0, 1.5) + energy(target, 2.0, 2.6)
    outside = (energy(target, 0.0, 1.0 - margin / sr) + energy(target, 1.5 + margin / sr, 2.0)
               + energy(target, 2.6, 3.0))
    assert inside > 1000 * outside
    loud = np.flatnonzero(np.abs(target) > 0.05)
    assert abs(loud[0] - sr) <= margin
    assert abs(loud[-1] - int(2.6 * sr)) <= int(0.01 * sr)   # trimmed and faded at 2.6 s
    assert not target[round(2.6 * sr):].any()
    tail = target[round(2.6 * sr) - 8:round(2.6 * sr)]
    assert np.abs(tail).max() < 0.01                          # the fade reaches zero

    partner = stems["partner"].astype(np.float64)
    assert energy(partner, 0.0, 2.5 - margin / sr) < 1e-6 * energy(partner, 2.5, 3.0)
    assert np.abs(partner[-8:]).max() < 0.01                  # cut at the window end, faded
    assert np.abs(partner[int(2.9 * sr):int(2.95 * sr)]).max() > 0.4


def test_to_dict_from_dict_round_trip():
    window = conv.candidate_windows(_meetings(), 10.0, 10.0)[0]
    plan = conv.plan_conversation(window, _meetings()["M1"], _catalogue(), "10", "20",
                                  stream(SEED, 4, "utterances"), sample_chapter="101")
    blob = json.loads(json.dumps(conv.to_dict(plan)))
    assert conv.from_dict(blob) == plan


@pytest.mark.skipif(not (REAL_AMI / "segments").is_dir(),
                    reason="AMI annotations are not unpacked under data/sources")
def test_real_ami_corpus():
    es = conv.parse_meeting(REAL_AMI, "ES2002a")
    assert sorted(es) == ["A", "B", "C", "D"]
    assert all(segs and all(g.words for g in segs) for segs in es.values())

    t0 = time.perf_counter()
    meetings = {m: conv.parse_meeting(REAL_AMI, m) for m in conv.list_meetings(REAL_AMI)}
    parse_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    wins = conv.candidate_windows(meetings, 30.0, 15.0)
    window_s = time.perf_counter() - t0
    assert len(wins) >= 500
    assert window_s < 60.0

    q = [0, 25, 50, 75, 100]
    overlap = np.percentile([w.features["overlap_share"] for w in wins], q)
    reentry = np.percentile([w.features["reentry_s"] for w in wins], q)
    print(f"\nAMI meetings parsed: {len(meetings)} in {parse_s:.1f} s")
    print(f"candidate windows (30 s, hop 15 s): {len(wins)} in {window_s:.2f} s")
    print("overlap_share min/q1/median/q3/max:", " ".join(f"{v:.3f}" for v in overlap))
    print("reentry_s     min/q1/median/q3/max:", " ".join(f"{v:.2f}" for v in reentry))
