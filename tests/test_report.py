import json
import math
import re

import numpy as np
import pytest

from rightvoice import cli, report, score, subsets

SEED = 20261022
DATE = "2026-10-22"
N = 40
ABSENT = {3, 9, 15, 21, 27, 33}
LONG = {38, 39}
FIRST = "the quick brown fox jumps"
SECOND = "over the lazy dog today"
OVERLAP = ("<5", "5-25", "25-50", ">=50")
REENTRY = ("none", "<2", "2-5", "5-10", ">=10")


def item_id(i):
    return f"rv1-long-{i - 38:03d}" if i in LONG else f"rv1-{i:05d}"


def length_s(i):
    return 300.0 if i in LONG else 30.0


def n_ref(i):
    return 0 if i in ABSENT else (10 if i % 2 == 0 else 5)


def active_s(i):
    return 0.0 if i in ABSENT else (6.0 if i % 2 == 0 else 3.0)


def capture(i):
    return "near_clean" if i < 6 else "room" if i < 22 else "call"


def bystanders(i):
    return 2 if i < 5 else 1 if i < 20 else 0


def manifest_row(i):
    placements = [
        {"role": "target", "utterance_id": f"t{i}a", "start": 1.0, "seconds": 3.0, "text": FIRST},
        {"role": "partner", "utterance_id": f"p{i}", "start": 5.0, "seconds": 3.0,
         "text": "hello there my good friend"},
    ]
    if i % 2 == 0:
        placements.append({"role": "target", "utterance_id": f"t{i}b", "start": 10.0,
                           "seconds": 3.0, "text": SECOND})
    return {
        "id": item_id(i), "index": i, "seed": SEED, "long": i in LONG,
        "window": {"length": length_s(i)}, "placements": placements, "bystanders": [],
        "target_absent": i in ABSENT,
        "buckets": {
            "overlap": OVERLAP[i % 4], "reentry": REENTRY[i % 5],
            "tir_db": "<=0" if i % 2 else ">0", "bystanders": bystanders(i),
            "same_sex": None if i < 2 else bool(i % 2), "capture": capture(i),
            "sample_s": (3.2, 5.0, 10.0)[i % 3], "sample_device": ("same", "other")[i % 2],
            "target_absent": i in ABSENT, "long": i in LONG,
        },
    }


# Per item numbers of each fake system: errors, wrong-voice words, words over target silence,
# activity hits, false-alarm frames, voice match, quality, SI-SDR improvement, empty output.
def alpha(i, frames, absent):
    return {"errors": 0 if absent else 1, "wv": 1 if i % 3 == 0 else 0, "words": i % 2,
            "tp": frames - frames // 10, "fp": 5, "vm": None if absent else 0.8,
            "q": 3.0 + 0.1 * (i % 5), "sdr_i": 10.0}


def beta(i, frames, absent):
    return {"errors": 1 if absent else 2, "wv": 2 if i % 4 == 0 else 0, "words": 2,
            "tp": frames - 50 if frames else 0, "fp": 20, "vm": None if absent else 0.6,
            "q": None, "sdr_i": 5.0}


def gamma(i, frames, absent):
    return {"errors": 0 if absent else i % 4, "wv": 0, "words": 1 if absent else 0,
            "tp": frames // 2, "fp": 30, "vm": 0.7 if not absent and i % 2 == 0 else None,
            "q": None, "sdr_i": None}


def control_clean(i, frames, absent):
    return {"errors": 0 if absent or i % 5 else 1, "wv": 0, "words": 0, "tp": frames, "fp": 0,
            "vm": None if absent else 0.95, "q": 4.0, "sdr_i": 30.0}


def control_mixture(i, frames, absent):
    return {"errors": 5 if absent else 3, "wv": 3, "words": 10, "tp": frames, "fp": 400,
            "vm": 0.5, "q": 2.5, "sdr_i": 0.0}


def bad_mixture(i, frames, absent):
    return {**control_mixture(i, frames, absent), "wv": 0}


def control_silence(i, frames, absent):
    return {"errors": n_ref(i), "wv": 0, "words": 0, "tp": 0, "fp": 0, "vm": None, "q": None,
            "sdr_i": None, "empty": True}


SYSTEMS = {"alpha": alpha, "beta": beta, "gamma": gamma, "control_clean": control_clean,
           "control_mixture": control_mixture, "control_silence": control_silence}


def numbers(fn, i):
    frames = round(active_s(i) * 50)
    return {"empty": False, **fn(i, frames, i in ABSENT)}, frames


def score_line(i, system, fn):
    v, frames = numbers(fn, i)
    absent, ref = i in ABSENT, n_ref(i)
    errors = v["errors"]
    sub = 0 if absent else min(errors, ref)
    ins = errors - sub
    silent = length_s(i) - active_s(i)
    return {
        "id": item_id(i), "system": system, "length_s": length_s(i), "long": i in LONG,
        "target_absent": absent, "timings_estimated": True, "empty_output": v["empty"],
        "target_words_lost": {"sub": sub, "ins": ins, "del": 0, "equal": ref - sub,
                              "n_ref": ref, "wer": errors / ref if ref else None},
        "wrong_voice_words": {"count": v["wv"], "candidates": v["wv"], "per_target_word": None,
                              "per_minute": None, "by_voice": {}},
        "talk_over_silence": {"words": v["words"], "per_minute": v["words"] / (silent / 60),
                              "speech_share": 0.0, "silent_s": silent, "speech_s": 0.0},
        "activity_f1": {"precision": None, "recall": None, "f1": None, "tp": v["tp"],
                        "fp": v["fp"], "fn": frames - v["tp"]},
        "voice_match": v["vm"], "si_sdr": None, "si_sdr_improvement": v["sdr_i"],
        "quality": v["q"], "adapters": {},
    }


def write_build(root, systems=None, *, skip=None, selection=True):
    """A synthetic build: manifest, build info, selection and one scores file per system."""
    systems = SYSTEMS if systems is None else systems
    skip = skip or {}
    root.mkdir(parents=True, exist_ok=True)
    rows = [manifest_row(i) for i in range(N)]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "build_info.json").write_text(json.dumps({"seed": SEED}))
    if selection:
        pool = {"near_clean": 150, "room": 300, "call": 200, "phone_g711": 50}
        (root / "selection.json").write_text(json.dumps({
            "seed": SEED, "pool": {"capture": pool, "target_absent": 56},
            "selected": [{"index": i, "quotas": ["fill"], "id": item_id(i),
                          "buckets": manifest_row(i)["buckets"]} for i in range(38)]}))
    scores = root / "scores"
    scores.mkdir(exist_ok=True)
    for name, fn in systems.items():
        lines = [score_line(i, name, fn) for i in range(N) if i not in skip.get(name, ())]
        (scores / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return root


def hand(fn, items=range(N), weights=None):
    """The headline numbers of one fake system, summed by hand."""
    w = weights or {i: 1.0 for i in items}
    s = {k: 0.0 for k in ("err", "ref", "wvp", "wv", "min", "words", "silent", "aw", "am",
                          "tp", "fp", "fn", "vm", "vmn")}
    for i in items:
        v, frames = numbers(fn, i)
        absent = i in ABSENT
        silent = (length_s(i) - active_s(i)) / 60
        if not absent:
            s["err"] += w[i] * v["errors"]
            s["ref"] += w[i] * n_ref(i)
            s["wvp"] += w[i] * v["wv"]
        else:
            s["aw"] += w[i] * v["words"]
            s["am"] += w[i] * silent
        s["wv"] += w[i] * v["wv"]
        s["min"] += w[i] * length_s(i) / 60
        s["words"] += w[i] * v["words"]
        s["silent"] += w[i] * silent
        if frames:
            s["tp"] += w[i] * v["tp"]
            s["fp"] += w[i] * v["fp"]
            s["fn"] += w[i] * (frames - v["tp"])
        if v["vm"] is not None:
            s["vm"] += w[i] * v["vm"]
            s["vmn"] += w[i]

    def div(a, b):
        return a / b if b else math.nan

    return {"target_wer": div(s["err"], s["ref"]),
            "wrong_voice_per_100_words": 100 * div(s["wvp"], s["ref"]),
            "wrong_voice_per_minute": div(s["wv"], s["min"]),
            "silence_words_per_minute": div(s["words"], s["silent"]),
            "absent_words_per_minute": div(s["aw"], s["am"]),
            "activity_f1": div(2 * s["tp"], 2 * s["tp"] + s["fp"] + s["fn"]),
            "voice_match": div(s["vm"], s["vmn"])}


@pytest.fixture
def build(tmp_path):
    return write_build(tmp_path / "build")


def point(rep, system, metric):
    return rep["systems"][system]["metrics"][metric]["point"]


def test_headline_numbers_are_ratios_of_sums(build):
    rep = report.build_report(build, n_boot=200, date=DATE)
    assert rep["order"] == ["control_clean", "control_mixture", "control_silence",
                            "alpha", "beta", "gamma"]
    checks = {"alpha": "target_wer", "beta": "wrong_voice_per_minute", "gamma": "activity_f1",
              "control_clean": "silence_words_per_minute",
              "control_mixture": "wrong_voice_per_100_words",
              "control_silence": "absent_words_per_minute"}
    for system, metric in checks.items():
        assert point(rep, system, metric) == pytest.approx(hand(SYSTEMS[system])[metric],
                                                           rel=1e-12)
    # 34 errors over 270 target words, not the mean of per-item rates
    assert point(rep, "alpha", "target_wer") == pytest.approx(34 / 270)
    assert point(rep, "beta", "absent_words_per_minute") == pytest.approx(
        hand(beta)["absent_words_per_minute"])
    assert point(rep, "gamma", "voice_match") == pytest.approx(0.7)
    assert point(rep, "alpha", "quality") == pytest.approx(
        np.mean([3.0 + 0.1 * (i % 5) for i in range(N)]))
    assert math.isnan(point(rep, "beta", "quality"))
    assert point(rep, "alpha", "si_sdr_i") == pytest.approx(10.0)
    assert rep["systems"]["control_silence"]["empty_outputs"] == {"count": 40, "share": 1.0}
    assert rep["systems"]["alpha"]["empty_outputs"]["count"] == 0
    assert rep["build"] == {"dir": str(build), "seed": SEED, "items": 40, "clips": 38,
                            "long": 2, "target_absent": 6, "minutes": 29.0,
                            "word_timings": False}


def test_bootstrap_is_reproducible_and_paired(build):
    tables = {}
    for name in ("alpha", "beta"):
        lines = [score_line(i, name, SYSTEMS[name]) for i in range(N)]
        tables[name] = report.item_table(lines)
    tables["alpha_again"] = report.item_table([score_line(i, "x", alpha) for i in range(N)])
    fn = report.METRIC_FNS["target_wer"]
    one = report.paired_bootstrap(tables, fn, n=1000, seed=SEED)
    two = report.paired_bootstrap(tables, fn, n=1000, seed=SEED)
    assert one == two
    other = report.paired_bootstrap(tables, fn, n=1000, seed=SEED + 1)
    assert other["systems"]["beta"] != one["systems"]["beta"]
    same = one["differences"]["alpha"]["alpha_again"]
    assert same == {"point": 0.0, "lo": 0.0, "hi": 0.0}
    a = one["systems"]["alpha"]
    assert a["lo"] <= a["point"] <= a["hi"] and a["lo"] < a["hi"]
    d = one["differences"]["alpha"]["beta"]
    assert d["point"] == pytest.approx(a["point"] - one["systems"]["beta"]["point"])
    assert one["differences"]["beta"]["alpha"]["lo"] == pytest.approx(-d["hi"])
    # a report run twice gives the same intervals
    r1 = report.build_report(build, n_boot=300, seed=7, date=DATE)
    r2 = report.build_report(build, n_boot=300, seed=7, date=DATE)
    assert score.json_safe(r1) == score.json_safe(r2)


def wer_lines(errors, ref):
    """Score lines of a system with ``errors[k]`` edits against ``ref`` words on item k."""
    out = []
    for k, e in enumerate(errors):
        out.append({"id": f"i{k}", "length_s": 30.0, "target_absent": False,
                    "target_words_lost": {"sub": e, "del": 0, "ins": 0, "equal": ref - e,
                                          "n_ref": ref},
                    "wrong_voice_words": {"count": 0},
                    "talk_over_silence": {"words": 0, "silent_s": 10.0},
                    "activity_f1": {"tp": 1, "fp": 0, "fn": 0}})
    return out


def verdict(errors_a, errors_b, ref):
    res = report.paired_bootstrap({"a": wer_lines(errors_a, ref), "b": wer_lines(errors_b, ref)},
                                  report.METRIC_FNS["target_wer"], n=1000, seed=SEED)
    a, b = res["systems"]["a"]["point"], res["systems"]["b"]["point"]
    d = res["differences"]["a"]["b"]
    return a, b, d, report.clearly_better(a, b, "target_wer", (d["lo"], d["hi"]))


def test_clearly_better_rule():
    a, b, d, yes = verdict([1] * 40, [2] * 40, 10)
    assert (a, b) == (pytest.approx(0.10), pytest.approx(0.20))
    assert d["hi"] < 0 and yes
    assert not report.clearly_better(b, a, "target_wer", (-d["hi"], -d["lo"]))

    # 2% relative: the interval excludes zero but the difference is too small
    a, b, d, yes = verdict([9] * 8 + [10] * 32, [10] * 40, 50)
    assert a / b == pytest.approx(0.98) and d["hi"] < 0
    assert not yes

    # 25% relative, but one item carries it all, so the interval spans zero
    a, b, d, yes = verdict([60] + [0] * 39, [2] * 40, 10)
    assert (b - a) / b == pytest.approx(0.25)
    assert d["lo"] < 0 < d["hi"]
    assert not yes

    # direction follows the metric
    assert report.clearly_better(0.9, 0.8, "activity_f1", (0.05, 0.15))
    assert not report.clearly_better(0.8, 0.9, "activity_f1", (-0.15, -0.05))
    assert not report.clearly_better(0.9, 0.8, "activity_f1", (-0.01, 0.15))
    assert not report.clearly_better(math.nan, 0.8, "voice_match", (0.05, 0.15))


def test_missing_item_counts_as_empty_output(tmp_path):
    full = report.build_report(write_build(tmp_path / "full"), n_boot=100, date=DATE)
    holed = report.build_report(write_build(tmp_path / "holed", skip={"gamma": {0, 2}}),
                                n_boot=100, date=DATE)
    g_full, g_holed = full["systems"]["gamma"], holed["systems"]["gamma"]
    # items 0 and 2 had 0 and 2 errors over 10 words; empty output deletes all 20 words
    err = sum(i % 4 for i in range(N) if i not in ABSENT)
    assert point(full, "gamma", "target_wer") == pytest.approx(err / 270)
    assert point(holed, "gamma", "target_wer") == pytest.approx((err - 2 + 20) / 270)
    assert point(holed, "gamma", "target_wer") > point(full, "gamma", "target_wer")
    assert g_holed["missing_lines"] == 2 and g_holed["missing_ids"] == ["rv1-00000", "rv1-00002"]
    assert g_holed["empty_outputs"]["count"] == g_full["empty_outputs"]["count"] + 2
    assert "gamma: 2 of 40 items have no score line and count as empty output." in holed["notes"]
    # the missing items' frames are all misses, and nothing else changes for other systems
    assert point(holed, "gamma", "activity_f1") < point(full, "gamma", "activity_f1")
    assert holed["systems"]["alpha"]["metrics"] == full["systems"]["alpha"]["metrics"]
    # the line built for a missing item is the scorer's empty output on the manifest reference
    empty = score.empty_output_scores(score.reference_from_manifest(manifest_row(0)))
    assert empty["target_words_lost"]["del"] == 10
    assert empty["talk_over_silence"]["silent_s"] == pytest.approx(24.0)


def test_bucket_tables_count_items_and_skip_small_values(build):
    rep = report.build_report(build, n_boot=100, date=DATE)
    by = {e["value"]: e for e in rep["buckets"]["bystanders"]}
    assert [e["value"] for e in rep["buckets"]["bystanders"]] == ["0", "1", "2"]
    assert {v: e["items"] for v, e in by.items()} == {"0": 20, "1": 15, "2": 5}
    assert by["2"]["skipped"] and "systems" not in by["2"]
    assert not by["1"]["skipped"]
    items = [i for i in range(N) if bystanders(i) == 1]
    assert by["1"]["systems"]["alpha"]["target_wer"] == pytest.approx(
        hand(alpha, items)["target_wer"])
    assert by["1"]["systems"]["control_silence"]["empty_outputs"] == 15
    cap = {e["value"]: (e["items"], e["skipped"]) for e in rep["buckets"]["capture"]}
    assert cap == {"near_clean": (6, True), "room": (16, False), "call": (18, False)}
    assert [e["value"] for e in rep["buckets"]["capture"]] == ["near_clean", "room", "call"]
    assert [e["value"] for e in rep["buckets"]["overlap"]] == list(OVERLAP)
    assert [(e["value"], e["items"]) for e in rep["buckets"]["same_sex"]] == [
        ("False", 19), ("True", 19), ("unknown", 2)]
    assert [(e["value"], e["items"]) for e in rep["buckets"]["target_absent"]] == [
        ("False", 34), ("True", 6)]
    absent = {e["value"]: e for e in rep["buckets"]["target_absent"]}["True"]
    assert absent["skipped"]
    assert set(rep["buckets"]) == set(report.BUCKET_KEYS)
    md = report.render_markdown(rep)
    assert "### bystanders" in md and "not scored: 2 (5 items)." in md
    assert "near_clean (6 items)" in md


def test_controls_checks(tmp_path, build):
    good = report.build_report(build, n_boot=100, date=DATE)
    assert report.controls_pass(good) == []
    assert good["controls"]["warnings"] == []
    bad = report.build_report(
        write_build(tmp_path / "bad", {**SYSTEMS, "control_mixture": bad_mixture}),
        n_boot=100, date=DATE)
    warnings = report.controls_pass(bad)
    assert len(warnings) == 1 and warnings[0].startswith("control_mixture:")
    assert "under 5.0%" in warnings[0]
    # the JSON form, with None for undefined numbers, is read the same way
    safe = score.json_safe(bad)
    assert report.controls_pass(safe) == warnings
    safe["systems"]["control_silence"]["metrics"]["absent_words_per_minute"]["point"] = 0.5
    safe["systems"]["control_clean"]["metrics"]["target_wer"]["point"] = None
    found = report.controls_pass(safe)
    assert len(found) == 3
    assert any(w.startswith("control_silence:") and "above 0.00" in w for w in found)
    assert any(w.startswith("control_clean:") and "undefined" in w for w in found)
    # unscored controls give no warning
    only = report.build_report(write_build(tmp_path / "only", {"alpha": alpha}), n_boot=50,
                               date=DATE)
    assert report.controls_pass(only) == []
    assert only["controls"]["missing"] == list(report.CONTROLS)


def test_reweighting_to_the_pool(tmp_path, build):
    rep = report.build_report(build, n_boot=100, date=DATE)
    rw = rep["reweighting"]
    assert rw["available"] and rw["pool_items"] == 700
    shares = {"near_clean": 6 / 40, "room": 16 / 40, "call": 18 / 40}
    pool = {"near_clean": 150 / 700, "room": 300 / 700, "call": 200 / 700}
    for v, share in shares.items():
        assert rw["values"][v]["weight"] == pytest.approx(pool[v] / share)
    assert math.isnan(rw["values"]["phone_g711"]["weight"])
    assert rw["values"]["phone_g711"]["selected"] == 0
    assert any("phone_g711" in n for n in rep["notes"])
    w = {i: pool[capture(i)] / shares[capture(i)] for i in range(N)}
    expected = hand(alpha, weights=w)
    for metric in ("target_wer", "wrong_voice_per_minute", "activity_f1"):
        assert rep["systems"]["alpha"]["reweighted"][metric] == pytest.approx(expected[metric])
    assert rep["systems"]["alpha"]["reweighted"]["target_wer"] != pytest.approx(
        point(rep, "alpha", "target_wer"))

    plain = report.build_report(write_build(tmp_path / "nosel", selection=False), n_boot=100,
                                date=DATE)
    assert not plain["reweighting"]["available"]
    assert plain["systems"]["alpha"]["reweighted"]["target_wer"] == pytest.approx(
        point(plain, "alpha", "target_wer"))
    assert "Reweighting unavailable" in report.render_markdown(plain)


def test_write_report_writes_both_files(build, tmp_path):
    rep = report.write_report(build, n_boot=200, date=DATE)
    md = (build / "report.md").read_text()
    loaded = json.loads((build / "report.json").read_text())
    assert loaded == rep
    assert rep["files"] == {"markdown": str(build / "report.md"),
                            "json": str(build / "report.json")}
    assert md.startswith("# RightVoice report\n")
    assert md.index("## Positive controls") < md.index(f"## Results, {DATE}")
    for name in report.CONTROLS:
        assert f"- {name} " in md
    assert "near the recogniser's floor" in md and "high in wrong-voice words" in md
    assert "zero words per minute on target-absent items" in md
    assert "Every scored control is in its expected range." in md
    header = ("| System | Target words lost (%) | Wrong-voice words (% of target words) | "
              "Wrong-voice words per minute | Words per minute, target silent | "
              "Words per minute, target absent | Activity F1 | Voice match | Quality | "
              "Empty outputs |")
    assert md.splitlines().count(header) == 2
    a = rep["systems"]["alpha"]["metrics"]["target_wer"]
    cell = f"{100 * a['point']:.1f} ({100 * a['lo']:.1f} to {100 * a['hi']:.1f})"
    assert f"| alpha | {cell} |" in md
    assert ("alpha is clearly better than beta on target words lost: 12.6% against 25.2%."
            in md)
    assert "clearly better than control" not in md and "control_clean is clearly" not in md
    assert "SI-SDR" not in md.split("## Notes")[0]
    assert "## Reweighted to the natural pool" in md and "## By bucket" in md
    footer = md.rstrip().splitlines()[-1]
    assert footer == (f"Build seed {SEED}. 40 items (38 clips and 2 long; 6 target-absent). "
                      f"Report dated {DATE}, 200 bootstrap resamples with seed {SEED}.")
    assert chr(0x2014) not in md and re.search(r"</?[A-Za-z]", md) is None
    assert rep["systems"]["beta"]["metrics"]["quality"]["point"] is None
    assert rep["systems"]["alpha"]["metrics"]["si_sdr_i"]["point"] == pytest.approx(10.0)
    assert {v["metric"] for v in rep["clearly_better"]} <= set(report.HEADLINE)

    out = report.write_report(build, tmp_path / "elsewhere" / "r.md", tmp_path / "r.json",
                              n_boot=50, date=DATE)
    assert (tmp_path / "elsewhere" / "r.md").is_file() and (tmp_path / "r.json").is_file()
    assert out["bootstrap"]["resamples"] == 50


def test_cli_report_runs_on_a_build(build, capsys):
    assert cli.main(["report", "--build", str(build), "--boot", "60", "--seed", "3"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed == [str(build / "report.md"), str(build / "report.json")]
    blob = json.loads((build / "report.json").read_text())
    assert blob["bootstrap"] == {"resamples": 60, "seed": 3, "percentiles": [2.5, 97.5],
                                 "paired": True}
    assert "## Positive controls" in (build / "report.md").read_text()


def test_cli_score_wires_the_adapters(tmp_path, monkeypatch, capsys):
    build = tmp_path / "b"
    build.mkdir()
    timings = {"u1": [[0.0, 0.4, "hello"]]}
    (build / "word_timings.json").write_text(json.dumps(timings))
    (build / "manifest.jsonl").write_text('{"id": "i1"}\n{"id": "i2"}\n')
    (tmp_path / "out").mkdir()
    for i in ("i1", "i2"):
        (tmp_path / "out" / f"{i}.wav").write_bytes(b"")
    made = {}

    class Fake:
        def __init__(self, **kw):
            made[type(self).__name__] = kw

    class ASR(Fake):
        pass

    class VAD(Fake):
        pass

    class Embed(Fake):
        pass

    def fake_score_system(build_dir, system, **kw):
        made["score_system"] = (build_dir, system, kw)
        return [{"empty_output": False}, {"empty_output": True}]

    monkeypatch.setattr(score, "WhisperASR", ASR)
    monkeypatch.setattr(score, "SileroVAD", VAD)
    monkeypatch.setattr(score, "EcapaEmbedder", Embed)
    monkeypatch.setattr(score, "score_system", fake_score_system)
    assert cli.main(["score", "--build", str(build), "--system", "mine", "--outputs",
                     str(tmp_path / "out"), "--asr-model", "ckpt-a", "--device", "cpu",
                     "--no-quality"]) == 0
    assert made["ASR"] == {"device": "cpu", "cache_dir": None, "model": "ckpt-a"}
    assert made["VAD"] == {} and made["Embed"] == {"device": "cpu"}
    build_dir, system, kw = made["score_system"]
    assert (build_dir, system) == (build, "mine")
    assert kw["word_timings"] == timings and kw["outputs"] == tmp_path / "out"
    assert kw["mos"] is None
    assert "scored 2 items for mine (1 empty output)" in capsys.readouterr().out

    assert cli.main(["score", "--build", str(build), "--system", "../x"]) == 2
    assert cli.main(["score", "--build", str(build), "--system", "plain"]) == 2
    assert cli.main(["score", "--build", str(build), "--system", "plain",
                     "--allow-partial"]) == 0
    assert "model" not in made["ASR"] and made["ASR"]["device"] is None
    assert made["score_system"][2]["word_timings"] == timings


def _fake_scoring(monkeypatch):
    made = {}

    class Fake:
        def __init__(self, **kw):
            pass

    def fake_score_system(build_dir, system, **kw):
        made["call"] = (system, kw)
        return []

    for name in ("WhisperASR", "SileroVAD", "EcapaEmbedder"):
        monkeypatch.setattr(score, name, Fake)
    monkeypatch.setattr(score, "score_system", fake_score_system)
    return made


def test_cli_score_refuses_a_run_that_did_not_finish(tmp_path, monkeypatch, capsys):
    made = _fake_scoring(monkeypatch)
    build = tmp_path / "b"
    build.mkdir()
    rows = [{"id": f"i{k}"} for k in range(10)]
    (build / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = build / "outputs" / "mine"
    out.mkdir(parents=True)
    for k in range(8):
        (out / f"i{k}.wav").write_bytes(b"")
    args = ["score", "--build", str(build), "--system", "mine", "--no-quality"]
    assert cli.main(args) == 2 and "call" not in made
    assert "only 8 of 10 items" in capsys.readouterr().err
    (out / "runs.jsonl").write_text('{"id": "i8", "status": "abstained"}\n')
    assert cli.main(args) == 0 and made["call"][0] == "mine"
    assert "1 items have no output and no logged abstention" in capsys.readouterr().err
    assert cli.main(["score", "--build", str(build), "--system", "control_silence",
                     "--no-quality"]) == 0


def test_cli_score_quick_scores_the_subset_into_its_own_folder(tmp_path, monkeypatch, capsys):
    made = _fake_scoring(monkeypatch)
    build = write_build(tmp_path / "b")
    rows = [manifest_row(i) for i in range(N)]
    quick = subsets.quick_ids(rows)
    out = tmp_path / "out"
    out.mkdir()
    for i in quick:
        (out / f"{i}.wav").write_bytes(b"")
    assert cli.main(["score", "--build", str(build), "--system", "mine", "--outputs", str(out),
                     "--no-quality", "--quick"]) == 0
    _, kw = made["call"]
    assert kw["items"] == quick and kw["out_path"] == build / "scores" / "quick" / "mine.jsonl"
    assert cli.main(["items", "--build", str(build), "--quick"]) == 0
    assert capsys.readouterr().out.split()[-len(quick):] == quick


def test_quick_report_compares_a_quick_run_with_full_runs(build):
    rows = [manifest_row(i) for i in range(N)]
    quick = subsets.quick_ids(rows)
    folder = build / "scores" / "quick"
    folder.mkdir()
    for name, fn in (("newcomer", gamma), ("alpha", beta)):
        lines = [score_line(i, name, fn) for i in range(N) if item_id(i) in quick]
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    rep = report.build_report(build, n_boot=50, seed=1, date=DATE, quick=True)
    assert rep["subset"] == {"name": "quick", "items": len(quick), "of": N}
    assert rep["build"]["items"] == len(quick) and rep["title"].endswith("quick subset")
    assert "newcomer" in rep["order"] and set(SYSTEMS) <= set(rep["order"])
    assert all(rep["systems"][s]["missing_lines"] == 0 for s in rep["order"])
    assert not any("not in the manifest" in note for note in rep["notes"])
    assert rep["notes"][0].startswith(f"Quick subset: {len(quick)} of {N} items")
    # Full score files are read on the subset, and a quick file wins over a full one.
    for metric in ("activity_f1", "voice_match"):
        assert point(rep, "newcomer", metric) == pytest.approx(point(rep, "gamma", metric))
        assert point(rep, "alpha", metric) == pytest.approx(point(rep, "beta", metric))
    full = report.build_report(build, n_boot=50, seed=1, date=DATE)
    assert "newcomer" not in full["order"] and full["subset"] is None

    assert cli.main(["report", "--build", str(build), "--boot", "20", "--quick"]) == 0
    assert (build / "report-quick.md").is_file() and (build / "report-quick.json").is_file()
    assert not (build / "report.md").exists()
    assert "Quick subset" in (build / "report-quick.md").read_text()
