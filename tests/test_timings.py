import json

import numpy as np
import pytest
import soundfile as sf

from rightvoice import cli, libritts, score, timings
from rightvoice.libritts import Utterance
from rightvoice.report import load_word_timings

SR = 24000
A = "1001_2001_000001_000000"   # placed
B = "1002_2002_000002_000000"   # a bystander piece
C = "1001_2003_000003_000000"   # a voice sample piece
TEXTS = {A: "Hello world.", B: "Hello there world.", C: "Hello big wide world."}
SECONDS = {A: 0.5, B: 0.5, C: 0.25}   # C is shorter, so the fake recogniser hears nothing in it
WORDS = [{"word": " Hello", "start": 0.05, "end": 0.2},
         {"word": " world.", "start": 0.25, "end": 0.45},
         {"word": " lost", "start": None, "end": 1.0}]


class FakeASR:
    name = "fake-asr"
    version = "1"

    def __init__(self, words=WORDS, fail_after=None):
        self.words = words
        self.fail_after = fail_after
        self.calls = []

    def transcribe(self, audio, sr):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("recogniser stopped")
        self.calls.append((len(audio), sr))
        return [] if len(audio) < 6000 else list(self.words)


def _catalogue(root):
    rows = []
    for uid, text in TEXTS.items():
        speaker, chapter = uid.split("_")[:2]
        rel = f"test-clean/{speaker}/{chapter}/{uid}.wav"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(round(SECONDS[uid] * SR)) / SR
        sf.write(root / rel, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)
        rows.append(Utterance(id=uid, speaker=speaker, chapter=chapter, split="test-clean",
                              path=rel, text=text, seconds=SECONDS[uid], sample_rate=SR))
    return rows


def _manifest(build):
    rows = [
        {"id": "rv1-00000", "length_s": 2.0,
         "placements": [{"role": "target", "utterance_id": A, "start": 0.0, "seconds": 0.5,
                         "text": TEXTS[A]}],
         "sample": {"pieces": [{"utterance_id": C, "seconds": 0.25, "text": TEXTS[C]}]},
         "bystanders": []},
        {"id": "rv1-00001", "length_s": 2.0,
         "placements": [{"role": "target", "utterance_id": A, "start": 1.0, "seconds": 0.5,
                         "text": TEXTS[A]}],
         "sample": {"pieces": [{"utterance_id": C, "seconds": 0.25, "text": TEXTS[C]}]},
         "bystanders": [{"pieces": [{"utterance_id": B, "seconds": 0.5, "text": TEXTS[B]}],
                         "start_s": 0.0, "seconds": 0.5}]},
    ]
    build.mkdir(parents=True, exist_ok=True)
    (build / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


@pytest.fixture
def setup(tmp_path):
    corpus = tmp_path / "corpus"
    cat = _catalogue(corpus)
    build = tmp_path / "build"
    rows = _manifest(build)
    return build, corpus, cat, rows


def test_used_ids_come_from_placements_sample_and_bystander_pieces(setup):
    build, _, _, _ = setup
    assert timings.used_utterance_ids(build) == {A, B, C}
    # Listed ids and ids nested under "conversation" count too.
    row = {"id": "x", "conversation": {"placements": [{"utterance_id": "p1"}]},
           "sample": {"utterance_ids": ["s1"]}, "bystanders": [{"utterance_id": "b1"}]}
    (build / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    assert timings.used_utterance_ids(build) == {"p1", "s1", "b1"}


def test_word_timings_writes_the_file_and_report(setup):
    build, corpus, cat, rows = setup
    asr = FakeASR()
    out = timings.word_timings(build, asr, catalogue=cat, corpus_root=corpus, progress=False)
    data = json.loads((build / "word_timings.json").read_text())
    assert data == out == load_word_timings(build)
    assert set(data) == {A, B, C}
    assert data[A] == [[0.05, 0.2, "Hello"], [0.25, 0.45, "world."]]
    assert data[C] == []
    # 24 kHz utterances reach the recogniser at 16 kHz.
    assert sorted(asr.calls) == [(4000, 16000), (8000, 16000), (8000, 16000)]

    rep = json.loads((build / "word_timings_report.json").read_text())
    assert rep["utterances_used"] == rep["utterances_done"] == 3
    assert rep["utterances_left"] == 0
    assert rep["words"] == 4
    assert rep["utterances_no_words"] == 1
    assert rep["not_in_catalogue"] == []
    assert rep["per_utterance"][B] == {"recognised": 2, "text": 3}
    assert rep["per_utterance"][C] == {"recognised": 0, "text": 4}
    assert rep["within_one_word"] == pytest.approx(2 / 3)
    assert rep["asr"] == {"name": "fake-asr", "version": "1"}

    # The scorer reads the file as timings, so the placed target words are not estimated.
    ref = score.reference_from_manifest(rows[0], data)
    assert not ref.timings_estimated
    assert [w.word for w in ref.target_words] == ["hello", "world"]


def test_word_timings_resumes_and_honours_limit(setup):
    build, corpus, cat, _ = setup
    first = FakeASR()
    timings.word_timings(build, first, catalogue=cat, corpus_root=corpus, limit=1,
                         progress=False)
    data = json.loads((build / "word_timings.json").read_text())
    assert list(data) == [A] and len(first.calls) == 1
    rep = json.loads((build / "word_timings_report.json").read_text())
    assert rep["utterances_done"] == 1 and rep["utterances_left"] == 2

    other = [{"word": "other", "start": 0.0, "end": 0.2}]
    second = FakeASR(words=other)
    out = timings.word_timings(build, second, catalogue=cat, corpus_root=corpus, progress=False)
    assert len(second.calls) == 2   # only B and C
    assert out[A] == [[0.05, 0.2, "Hello"], [0.25, 0.45, "world."]]
    assert out[B] == [[0.0, 0.2, "other"]]

    third = FakeASR(words=other)
    timings.word_timings(build, third, catalogue=cat, corpus_root=corpus, progress=False)
    assert third.calls == []


def test_word_timings_keeps_progress_when_the_recogniser_fails(setup, monkeypatch):
    build, corpus, cat, _ = setup
    monkeypatch.setattr(timings, "SAVE_EVERY", 1)
    with pytest.raises(RuntimeError, match="recogniser stopped"):
        timings.word_timings(build, FakeASR(fail_after=2), catalogue=cat, corpus_root=corpus,
                             progress=False)
    assert sorted(json.loads((build / "word_timings.json").read_text())) == [A, C]


def test_unknown_ids_are_reported_and_skipped(setup):
    build, corpus, cat, _ = setup
    out = timings.word_timings(build, FakeASR(), catalogue=[u for u in cat if u.id != B],
                               corpus_root=corpus, progress=False)
    assert set(out) == {A, C}
    rep = json.loads((build / "word_timings_report.json").read_text())
    assert rep["not_in_catalogue"] == [B] and rep["utterances_left"] == 1


def test_cli_timings_builds_the_recogniser_like_score(setup, monkeypatch, capsys):
    build, corpus, cat, _ = setup
    made = {}

    def fake_whisper(**kwargs):
        made.update(kwargs)
        return FakeASR()

    monkeypatch.setattr(score, "WhisperASR", fake_whisper)
    monkeypatch.setattr(libritts, "load", lambda root=None, **kw: cat)
    monkeypatch.setattr(libritts, "corpus_root", lambda root=None: corpus)
    code = cli.main(["timings", "--build", str(build), "--asr-model", "tiny", "--device", "cpu",
                     "--limit", "2"])
    assert code == 0
    assert made == {"device": "cpu", "cache_dir": None, "model": "tiny"}
    assert len(json.loads((build / "word_timings.json").read_text())) == 2
    assert "timed 2 of 3 utterances" in capsys.readouterr().out
