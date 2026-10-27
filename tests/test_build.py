"""The item builder on a tiny synthetic world.

The world is written under a temporary data root in the real layouts (LibriTTS-R tree with
speakers.tsv, AMI segments and words XML, FSD50K metadata, ground truth and audio, a BUT
ReverbDB tree) and ``RIGHTVOICE_DATA`` points the loaders at it. The simulated room bank holds
three rooms.
"""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
import soundfile as sf

from rightvoice import build, capture, listen
from rightvoice.build import CAPTURE_SHARES, QUOTA_DIMS, PlanFailed
from rightvoice.rng import stream

SEED = 20261022
SR_IN = 24000
SR = 16000
MEETING = "TS9999z"
MEETING_S = 400.0
# Conversation readers (two chapters each) and bystander readers (one chapter), all on the
# fixed reader list, with a sex and a base pitch so their stems differ.
CONVERSATION = {"61": ("F", 215.0), "84": ("F", 180.0), "116": ("M", 115.0), "260": ("M", 140.0)}
BYSTANDERS = {"367": ("M", 100.0), "533": ("F", 240.0)}
UTTERANCES_PER_CHAPTER = 50
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
HEAD = ('<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>\n'
        '<nite:root nite:id="{root_id}" xmlns:nite="http://nite.sourceforge.net/">\n')
TAIL = "</nite:root>\n"
STEMS_30S = ("mixture", "target", "partner", "others")


# The synthetic world


def _voice(rng, seconds, f0):
    """A harmonic tone with a syllable-rate envelope and short silent edges."""
    n = round(seconds * SR_IN)
    t = np.arange(n) / SR_IN
    f = f0 * (1.0 + 0.04 * np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * t))
    phase = 2 * np.pi * np.cumsum(f) / SR_IN
    x = sum(a * np.sin(k * phase) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(3.0, 5.0) * t + rng.uniform(0, 6.28))
    edge = min(n // 4, round(0.06 * SR_IN))
    ramp = np.ones(n)
    ramp[:edge] = 0.0
    ramp[n - edge:] = 0.0
    y = 0.2 * x * env * ramp + 0.002 * rng.standard_normal(n) * ramp
    return y.astype(np.float32)


def _write_libritts(root):
    corpus = root / "sources" / "libritts_r" / "LibriTTS_R"
    rng = np.random.default_rng(1)
    readers = [(spk, sex, f0, ("100", "101"), UTTERANCES_PER_CHAPTER, (1.0, 6.0), "dev-clean")
               for spk, (sex, f0) in CONVERSATION.items()]
    readers += [(spk, sex, f0, ("200",), 12, (2.0, 6.0), "test-clean")
                for spk, (sex, f0) in BYSTANDERS.items()]
    lines = ["READER\tGENDER\tSUBSET\tNAME"]
    for spk, sex, f0, chapters, per, (lo, hi), split in readers:
        lines.append(f"{spk}\t{sex}\t{split}\tReader {spk}")
        for chapter in chapters:
            folder = corpus / split / spk / chapter
            folder.mkdir(parents=True)
            for i in range(per):
                stem = f"{spk}_{chapter}_{i:06d}_000000"
                pitch = f0 * rng.uniform(0.9, 1.1)
                sf.write(folder / f"{stem}.wav", _voice(rng, rng.uniform(lo, hi), pitch), SR_IN,
                         subtype="PCM_16")
                (folder / f"{stem}.normalized.txt").write_text(f"reader {spk} line {i}\n")
    (corpus / "speakers.tsv").write_text("\n".join(lines) + "\n")


def _timeline(seed=2):
    """Two agents taking turns over 400 s, with overlaps, back-channels and long silences."""
    rng = np.random.default_rng(seed)
    turns = {"A": [], "B": []}
    t, who = 0.5, "A"
    while t < MEETING_S - 8.0:
        other = "B" if who == "A" else "A"
        run = 1 if rng.random() > 0.12 else int(rng.integers(2, 4))
        for _ in range(run):
            d = float(rng.uniform(1.2, 5.5))
            if t + d > MEETING_S - 0.5:
                break
            turns[who].append((t, t + d))
            if d > 3.0 and rng.random() < 0.25:     # a back-channel under this turn
                s = t + d / 2
                turns[other].append((s, s + float(rng.uniform(0.3, 0.8))))
            t += d + float(rng.uniform(0.2, 1.0))
        t = turns[who][-1][1] + float(rng.uniform(-0.8, 3.5))
        who = other
    return {a: sorted(segs) for a, segs in turns.items()}


def _write_ami(root):
    ami = root / "sources" / "ami_annotations"
    (ami / "segments").mkdir(parents=True)
    (ami / "words").mkdir(parents=True)
    for agent, segs in _timeline().items():
        seg_lines = [HEAD.format(root_id=f"{MEETING}.{agent}.segs")]
        word_lines = [HEAD.format(root_id=f"{MEETING}.{agent}.words")]
        w = 0
        for k, (s, e) in enumerate(segs):
            mid = (s + e) / 2
            first = w
            for a, b, text in ((s, mid, "hello"), (mid, e, "there")):
                word_lines.append(f'   <w nite:id="{MEETING}.{agent}.words{w}" '
                                  f'starttime="{a:.3f}" endtime="{b:.3f}">{text}</w>\n')
                w += 1
            ref = f"id({MEETING}.{agent}.words{first})..id({MEETING}.{agent}.words{w - 1})"
            seg_lines.append(f'   <segment nite:id="{MEETING}.sync.{agent}{k}" channel="0" '
                             f'transcriber_start="{s:.3f}" transcriber_end="{e:.3f}">\n'
                             f'      <nite:child href="{MEETING}.{agent}.words.xml#{ref}"/>\n'
                             "   </segment>\n")
        seg_lines.append(TAIL)
        word_lines.append(TAIL)
        (ami / "segments" / f"{MEETING}.{agent}.segments.xml").write_bytes(
            "".join(seg_lines).encode("iso-8859-1"))
        (ami / "words" / f"{MEETING}.{agent}.words.xml").write_bytes(
            "".join(word_lines).encode("iso-8859-1"))


def _write_fsd50k(root):
    fsd = root / "sources" / "fsd50k"
    for sub in ("FSD50K.metadata", "FSD50K.ground_truth", "FSD50K.eval_audio"):
        (fsd / sub).mkdir(parents=True)
    rng = np.random.default_rng(3)
    clips = [("101", "Rain,Water"), ("102", "Engine,Vehicle"), ("103", "Wind"),
             ("104", "Typing,Computer_keyboard"), ("201", "Guitar,Musical_instrument,Music"),
             ("202", "Piano,Musical_instrument,Music")]
    info = {cid: {"title": f"clip {cid}", "license": CC0, "uploader": f"user{cid}"}
            for cid, _ in clips}
    (fsd / "FSD50K.metadata" / "eval_clips_info_FSD50K.json").write_text(json.dumps(info))
    with (fsd / "FSD50K.ground_truth" / "eval.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fname", "labels", "mids"])
        for cid, labels in clips:
            writer.writerow([cid, labels, "/m/0"])
    for cid, _ in clips:
        x = 0.1 * rng.standard_normal(round(rng.uniform(3.0, 8.0) * 44100))
        sf.write(fsd / "FSD50K.eval_audio" / f"{cid}.wav", x.astype(np.float32), 44100,
                 subtype="PCM_16")


def _write_but(root):
    but = root / "sources" / "but_reverbdb"
    rng = np.random.default_rng(4)
    for k, distance in enumerate((1.0, 1.8, 2.6), start=1):
        folder = but / "Room_A" / "MicID01" / "SpkID01_20170901_S" / f"{k:02d}"
        (folder / "RIR").mkdir(parents=True)
        (folder / "mic_meta.txt").write_text(
            "$EnvName\tRoom_A\n$EnvType\tOffice\n"
            f"$EnvMicID\t{k}\n$EnvMic{k}RelDistance\t{distance}\n"
            f"$EnvMic{k}Placing\ton table\n$EnvMic{k}RelDistanceRIRMeasured\t{distance}\n"
            f"$EnvMic{k}RelRT60\t1.2\n")
        n = round(0.6 * SR)
        rir = np.zeros(n)
        rir[50] = 0.8
        tail = np.arange(n - 51) / SR
        rir[51:] = 0.1 * rng.uniform(-1, 1, tail.size) * np.exp(-tail / 0.15)
        sf.write(folder / "RIR" / "IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav", rir, SR,
                 subtype="FLOAT")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rightvoice") / "data"
    _write_libritts(root)
    _write_ami(root)
    _write_fsd50k(root)
    _write_but(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RIGHTVOICE_DATA", str(root))
        yield root


@pytest.fixture(scope="module")
def world(data_root):
    return build.load_world(SEED, n_rooms=3, long_lengths=(300.0,))


@pytest.fixture(scope="module")
def plans(world):
    out = {}
    for i in range(200):
        try:
            out[i] = build.plan_item(SEED, i, world)
        except PlanFailed:
            continue
    return out


def _first(plans, test):
    for i, plan in sorted(plans.items()):
        if test(plan):
            return i
    pytest.fail("no planned item matches")


def _read(path):
    x, sr = sf.read(path, dtype="float64")
    info = sf.info(path)
    assert sr == SR and x.ndim == 1 and info.subtype == "PCM_16"
    return x


# Tests


def test_the_world_loads(world):
    assert world.conversation_pool == sorted(CONVERSATION)
    assert world.bystander_pool == sorted(BYSTANDERS)
    assert len(world.clip_windows) > 10 and len(world.long_windows[300.0]) > 0
    assert [c.id for c in world.noise_clips] == ["101", "102", "103", "104"]
    assert [c.id for c in world.music_clips] == ["201", "202"]
    assert len(world.measured) == 3 and world.measured_rooms == ["Room_A"]


def _strip(plan):
    return {k: v for k, v in plan.items() if k not in ("id", "index", "stream_index")}


def test_plans_are_valid_repeatable_and_distinct(world):
    seen = set()
    for i in range(20):
        plan = build.plan_item(SEED, i, world)
        assert plan == build.plan_item(SEED, i, world)
        assert json.loads(json.dumps(plan)) == plan
        assert plan["id"] == f"rv1-{i:05d}" and not plan["long"]
        seen.add(json.dumps(_strip(plan), sort_keys=True))

        readers = plan["readers"]
        target, partner = readers["target"], readers["partner"]
        assert target != partner and {target, partner} <= set(world.conversation_pool)
        assert readers["target_sex"] == CONVERSATION[target][0]
        assert plan["window"]["length"] == build.CLIP_S
        assert plan["placements"]
        chapter = plan["sample"]["chapter"]
        for p in plan["placements"]:
            u = world.by_id[p["utterance_id"]]
            assert u.speaker == (target if p["role"] == "target" else partner)
            assert not (p["role"] == "target" and u.chapter == chapter)
            assert 0.0 <= p["start"] and p["start"] + p["seconds"] <= 30.0 + 1e-6

        sample = plan["sample"]
        assert sample["length_s"] in build.SAMPLE_BUCKETS_S
        assert sample["device"] in ("same", "other")
        assert 1 <= len(sample["pieces"]) <= 2
        total = sum(p["seconds"] for p in sample["pieces"]) + build.GAP_S * (len(sample["pieces"]) - 1)
        assert total == pytest.approx(sample["length_s"], abs=1e-3)
        placed = {p["utterance_id"] for p in plan["placements"]}
        for p in sample["pieces"]:
            u = world.by_id[p["utterance_id"]]
            assert (u.speaker, u.chapter) == (target, chapter) and u.id not in placed
            assert p["seconds"] <= u.seconds + 1e-6

        assert -6.0 <= plan["levels"]["partner_gain_db"] <= 6.0
        assert -30.0 <= plan["levels"]["mixture_dbfs"] <= -15.0

        bystanders = plan["bystanders"]
        assert len(bystanders) == plan["buckets"]["bystanders"] <= 2
        assert len({b["reader"] for b in bystanders}) == len(bystanders)
        for b in bystanders:
            assert b["reader"] in world.bystander_pool and b["reader"] not in (target, partner)
            assert 2.0 <= b["seconds"] <= 15.0 and -20.0 <= b["gain_db"] <= -8.0
            assert 0 <= b["start_sample"] and b["start_sample"] + b["samples"] <= 30 * SR

        cap = plan["capture"]
        assert cap["branch"] in CAPTURE_SHARES and cap["drawn_branch"] in CAPTURE_SHARES
        if not world.amr_wb_available:
            assert cap["branch"] != "phone_amr_wb"
        assert (cap["room"] is None) == (build.ROOM_KIND[cap["branch"]] is None)
        assert len(cap["background"]) == (2 if cap["branch"] == "call" else 1)
        lo, hi = capture.BRANCH_SNR_DB[build.SNR_BRANCH[cap["branch"]]]
        for bg in cap["background"]:
            assert lo <= bg["snr_db"] <= hi and bg["clip_id"] in world.clips_by_id

        buckets = plan["buckets"]
        assert buckets["overlap"] == build.overlap_bucket(plan["window"]["features"]["overlap_share"])
        assert buckets["reentry"] == build.reentry_bucket(plan["window"]["features"]["reentry_s"])
        assert buckets["same_sex"] == (readers["target_sex"] == readers["partner_sex"])
        assert buckets["capture"] == cap["branch"] and buckets["target_absent"] == plan["target_absent"]
    assert len(seen) == 20


def test_bucket_edges():
    assert [build.overlap_bucket(x) for x in (0.0, 0.049, 0.05, 0.2499, 0.25, 0.5, 0.9)] == [
        "<5", "<5", "5-25", "5-25", "25-50", ">=50", ">=50"]
    assert [build.reentry_bucket(x) for x in (0.0, 0.02, 1.99, 2.0, 5.0, 9.99, 10.0)] == [
        "none", "<2", "<2", "2-5", "5-10", "5-10", ">=10"]
    assert build.tir_bucket(0.0) == "<=0" and build.tir_bucket(0.01) == ">0"


def test_capture_draw_follows_the_quota_table():
    n = 2000
    draws = Counter(build.draw_capture(stream(SEED, i, "rooms")) for i in range(n))
    assert set(draws) <= set(CAPTURE_SHARES)
    assert sum(CAPTURE_SHARES.values()) == pytest.approx(1.0)
    for value, share in CAPTURE_SHARES.items():
        assert abs(draws[value] / n - share) <= 0.03, (value, draws[value] / n, share)


def _check_item(out, row, world):
    folder = out / "items" / row["id"]
    audio = {name: _read(folder / f"{name}.wav") for name in build.STEMS}
    n = round(row["window"]["length"] * SR)
    for name in STEMS_30S:
        assert audio[name].size == n, name
    for name, x in audio.items():
        assert np.max(np.abs(x)) < 1.0, name
    total = audio["target"] + audio["partner"] + audio["others"]
    assert np.max(np.abs(audio["mixture"] - total)) < 1e-3
    assert row["sample"]["length_s"] in build.SAMPLE_BUCKETS_S
    assert audio["sample"].size == round(row["sample"]["length_s"] * SR)
    assert row["sample"]["dbfs_before_chain"] == pytest.approx(-25.0, abs=0.01)
    if not row["sample"]["chain"]:
        assert capture.active_rms_dbfs(audio["sample"]) == pytest.approx(-25.0, abs=0.1)
    assert row["files"] == {name: f"items/{row['id']}/{name}.wav" for name in build.STEMS}
    assert row["buckets"]["tir_db"] in ("<=0", ">0")
    if row["target_absent"]:
        assert not np.any(audio["target"])
    else:
        assert capture.active_rms_dbfs(audio["target"]) > -60.0
    assert capture.active_rms_dbfs(audio["partner"]) > -60.0
    assert len(row["bystanders"]) == row["buckets"]["bystanders"]
    json.dumps(row, allow_nan=False)
    return audio


def test_render_items(world, plans, tmp_path):
    picks = [
        _first(plans, lambda p: p["capture"]["branch"] == "call"),
        _first(plans, lambda p: p["capture"]["room"] is not None
               and p["capture"]["room"]["source"] == "simulated" and p["bystanders"]),
        _first(plans, lambda p: p["capture"]["branch"] in ("near_clean", "room_reverberant")),
    ]
    for index in picks:
        row = build.render_item(SEED, index, world, tmp_path)
        assert row["id"] == f"rv1-{index:05d}" and row["capture"]["branch"] == plans[index]["capture"]["branch"]
        _check_item(tmp_path, row, world)
        if row["capture"]["branch"] == "call":
            assert isinstance(row["capture"]["background"], list)
            assert {b["channel"] for b in row["capture"]["background"]} == {"target", "partner"}
            assert {s["channel"] for s in row["chain"]} <= {"target", "partner"}
        else:
            assert isinstance(row["capture"]["background"], dict)
        if row["capture"]["room"] is not None:
            room = row["capture"]["room"]
            assert len(room["responses"]) == 2 + len(row["bystanders"])
            assert room["rt60_s"] is not None

    # rendering is a pure function of (seed, index)
    again = build.render_item(SEED, picks[1], world, tmp_path / "again")
    first = tmp_path / "items" / again["id"] / "mixture.wav"
    second = tmp_path / "again" / "items" / again["id"] / "mixture.wav"
    assert first.read_bytes() == second.read_bytes()


def test_target_absent_item_has_a_silent_target(world, plans, tmp_path):
    index = _first(plans, lambda p: p["target_absent"])
    row = build.render_item(SEED, index, world, tmp_path)
    assert row["target_absent"] and row["buckets"]["target_absent"]
    audio = _check_item(tmp_path, row, world)
    assert not np.any(audio["target"])
    assert np.any(audio["sample"]) and np.any(audio["mixture"])


def test_codec_stages_run_only_when_available(world, plans, tmp_path, monkeypatch):
    index = _first(plans, lambda p: p["capture"]["branch"] in
                   ("laptop_opus", "phone_amr_wb", "phone_amr_nb", "phone_g711"))
    row = build.render_item(SEED, index, world, tmp_path / "real")
    codecs = [s for s in row["chain"] if s["stage"] == "codec"]
    assert len(codecs) == 1
    if capture.codec_available(codecs[0]["codec"]):
        assert "skipped" not in codecs[0] and codecs[0]["encoder"]
    else:
        assert codecs[0]["skipped"] == "codec unavailable"
    _check_item(tmp_path / "real", row, world)

    # without any codec the same item still builds, and the codec stage is recorded as skipped
    monkeypatch.setattr(capture, "codec_available", lambda name: False)
    row = build.render_item(SEED, index, world, tmp_path / "none")
    codecs = [s for s in row["chain"] + row["sample"]["chain"] if s["stage"] == "codec"]
    assert codecs and all(s["skipped"] == "codec unavailable" for s in codecs)
    _check_item(tmp_path / "none", row, world)


def test_selection_meets_quotas_or_the_pool(world, plans):
    pool = {i: plans[i]["buckets"] for i in range(60) if i in plans}
    assert len(pool) >= 55
    result = build.select(pool, SEED, count=40, min_per_value=2, min_absent=2)
    taken = [s["index"] for s in result["selected"]]
    assert len(taken) == 40 == len(set(taken))
    assert sorted(result["order"]) == sorted(pool)
    chosen = [pool[i] for i in taken]
    for dim in QUOTA_DIMS:
        have = Counter(str(b[dim]) for b in pool.values())
        got = Counter(str(b[dim]) for b in chosen)
        for value, n in have.items():
            assert got[value] >= min(2, n), (dim, value)
    absent_pool = sum(b["target_absent"] for b in pool.values())
    assert sum(b["target_absent"] for b in chosen) >= min(2, absent_pool)
    # quota picks come first, in selection order, then the fill
    kinds = [s["quotas"] == ["fill"] for s in result["selected"]]
    assert kinds == sorted(kinds)
    quota = [i for i, s in zip(taken, result["selected"]) if s["quotas"] != ["fill"]]
    assert quota == [i for i in result["order"] if i in set(quota)]
    assert build.select(pool, SEED, count=40, min_per_value=2, min_absent=2) == result


def test_build_end_to_end_and_listening_page(data_root, tmp_path):
    out = tmp_path / "build"
    info = build.build(out, seed=SEED, count=6, long_count=1, candidates=24, n_rooms=3,
                       long_lengths=(300.0,), progress=False)
    assert info["counts"]["rendered"] == 7 and info["counts"]["render_failed"] == 0
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 7
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    long_rows = [r for r in rows if r["long"]]
    assert len(long_rows) == 1 and long_rows[0]["id"].startswith("rv1-long-")
    assert long_rows[0]["window"]["length"] == 300.0 and long_rows[0]["buckets"]["long"]
    for row in rows:
        _check_item(out, row, None)
    for name in ("build_info.json", "attribution.csv", "selection.json"):
        assert (out / name).is_file()

    built = json.loads((out / "build_info.json").read_text())
    assert built["seed"] == SEED and built["amr_wb_available"] == capture.codec_available("amr_wb")
    assert "ffmpeg" in built and built["versions"]["soxr"] and built["timing"]["total_s"] > 0
    selection = json.loads((out / "selection.json").read_text())
    assert len(selection["selected"]) == 6 and len(selection["long"]) == 1
    assert {s["id"] for s in selection["selected"]} == {r["id"] for r in rows if not r["long"]}
    assert all(s["quotas"] for s in selection["selected"])
    used = set()
    for r in rows:
        bg = r["capture"]["background"]
        used |= {b["clip_id"] for b in (bg if isinstance(bg, list) else [bg])}
    with (out / "attribution.csv").open(newline="") as fh:
        assert {row["clip_id"] for row in csv.DictReader(fh)} == used

    for page in (listen.write_page(out, count=20),
                 listen.write_page(out, tmp_path / "elsewhere" / "page.html", count=3, seed=1)):
        text = page.read_text()
        srcs = re.findall(r'<audio controls preload="none" src="([^"]+)"', text)
        items = re.findall(r'<section class="item" id="([^"]+)"', text)
        assert len(srcs) == 5 * len(items) and len(items) == (7 if page.name == "listen.html" else 3)
        for src in srcs:
            assert (page.parent / unquote(src)).is_file(), src
        assert "<script" not in text and "http" not in text
    assert (out / "listen.html").is_file()


def test_listening_page_spreads_over_capture_values():
    rows = [{"id": f"rv1-{i:05d}", "buckets": {"capture": c}, "files": {}}
            for i, c in enumerate(["room"] * 10 + ["call"] * 3 + ["near_clean"])]
    picked = listen.choose(rows, 5, seed=0)
    assert Counter(r["buckets"]["capture"] for r in picked) == {"room": 2, "call": 2, "near_clean": 1}
    assert listen.choose(rows, 5, seed=0) == picked
    assert len(listen.choose(rows, 50, seed=0)) == len(rows)


def test_long_items_draw_from_their_own_streams(world):
    plan = build.plan_item(SEED, 0, world, long=True)
    assert plan["id"] == "rv1-long-000" and plan["long"]
    assert plan["stream_index"] == build.LONG_STREAM_BASE
    assert plan["window"]["length"] == 300.0
    assert plan == build.plan_item(SEED, 0, world, long=True)
    assert math.isclose(sum(p["seconds"] for p in plan["sample"]["pieces"])
                        + build.GAP_S * (len(plan["sample"]["pieces"]) - 1),
                        plan["sample"]["length_s"], abs_tol=1e-3)


def test_cli_wires_build_and_listen(data_root, tmp_path, monkeypatch):
    from rightvoice import cli

    calls = {}

    def fake_build(out, **kw):
        calls["build"] = (out, kw)
        return {"counts": {"planned": 1, "candidates": 1, "plan_failed": 0, "selected": 1,
                           "long": 0, "rendered": 1, "render_failed": 0},
                "timing": {"total_s": 0.1}, "render_failures": []}

    monkeypatch.setattr(build, "build", fake_build)
    assert cli.main(["build", "--seed", "7", "--out", str(tmp_path / "b"), "--count", "3",
                     "--long-count", "0", "--candidates", "9", "--min-per-value", "1",
                     "--limit", "2", "--only-plan"]) == 0
    out, kw = calls["build"]
    assert Path(out) == tmp_path / "b"
    assert kw["seed"] == 7 and kw["count"] == 3 and kw["long_count"] == 0
    assert kw["candidates"] == 9 and kw["min_per_value"] == 1 and kw["limit"] == 2
    assert kw["only_plan"] is True and kw["n_rooms"] == 600
