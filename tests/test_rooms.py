import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rightvoice import rooms
from rightvoice.capture import SR
from rightvoice.rng import stream

ROOT = Path(__file__).resolve().parents[1]
REAL_BUT = ROOT / "data" / "sources" / "but_reverbdb"
WAV_NAME = "IR_sweep_15s_45Hzto22kHz_FS16kHz.{}.wav"


def _meta_text(mic_id, room="Room_A", room_type="Office", rt60="1.2", rt30="1.1",
               distance_rir="2.5", distance="3.0", placing="on table", delay="0.007"):
    n = mic_id
    lines = [
        "$EnvID\t1",
        f"$EnvName\t{room}",
        f"$EnvType\t{room_type}",
        "$EnvDepth\tNot Defined",
        "",
        f"$EnvMicID\t{n}",
        f"$EnvMic{n}RelDistance\t{distance}",
        f"$EnvMic{n}Placing\t{placing}",
        f"$EnvMic{n}RelDistanceRIRMeasured\t{distance_rir}",
        f"$EnvMic{n}RelDelayRIRMeasured\t{delay}",
        f"$EnvMic{n}RelRT60\t{rt60}",
        f"$EnvMic{n}RelRT30\t{rt30}",
        # another mic's keys must not leak in
        f"$EnvMic{n + 1}RelRT60\t9.9",
    ]
    return "\n".join(lines) + "\n"


def _impulse(sr, delay, n=None, seed=0):
    """A delayed unit impulse followed by a decaying noise tail that stays under 50% of it."""
    n = n or sr
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[delay] = 0.8
    t = np.arange(n - delay - 1) / sr
    x[delay + 1:] = 0.1 * rng.uniform(-1, 1, t.size) * np.exp(-t / 0.15)
    return x


def test_parse_mic_meta(tmp_path):
    p = tmp_path / "mic_meta.txt"
    p.write_text(_meta_text(7, rt30="Not Defined", distance_rir="#N/A", placing="Not Defined",
                            delay="#N/A"))
    meta = rooms.parse_mic_meta(p)
    assert meta == {"room": "Room_A", "room_type": "Office", "mic_id": 7, "rt60_s": 1.2,
                    "rt30_s": None, "distance_m": 3.0, "placing": None, "delay_s": None}

    p.write_text(_meta_text(12))
    meta = rooms.parse_mic_meta(p)
    assert meta["mic_id"] == 12 and meta["rt60_s"] == 1.2 and meta["rt30_s"] == 1.1
    assert meta["distance_m"] == 2.5          # the response-measured distance wins
    assert meta["placing"] == "on table" and meta["delay_s"] == 0.007

    p.write_text("$EnvName\tNot Defined\n$EnvMicID\t#N/A\n")
    meta = rooms.parse_mic_meta(p)
    assert meta["room"] is None and meta["mic_id"] is None and meta["rt60_s"] is None


def _but_tree(root):
    """Room_A: position 01 with v00 and v01, position 02 with v00. Room_B: position 01, stereo
    at 32 kHz. Every impulse sits 50 samples in at 16 kHz."""
    spec = [
        ("Room_A", "SpkID01_20170901_S", "01", 1, ["v00", "v01"], "1.2", "2.5", 16000, 1),
        ("Room_A", "SpkID01_20170901_S", "02", 2, ["v00"], "1.4", "4.0", 16000, 1),
        ("Room_B", "SpkID02_20171001_S", "01", 1, ["v00"], "1.8", "1.0", 32000, 2),
    ]
    for room, spk, pos, mic_id, versions, rt60, dist, sr, channels in spec:
        d = root / room / "MicID01" / spk / pos
        (d / "RIR").mkdir(parents=True)
        (d / "mic_meta.txt").write_text(_meta_text(mic_id, room=room, rt60=rt60,
                                                   distance_rir=dist))
        for k, v in enumerate(versions):
            x = _impulse(sr, 50 * sr // SR, seed=k)
            data = np.stack([x] * channels, axis=1) if channels > 1 else x
            sf.write(d / "RIR" / WAV_NAME.format(v), data, sr, subtype="PCM_16")
    (root / "Room_A" / "MicID01" / "SpkID01_20170901_S" / "01" / "silence").mkdir()
    (root / "read_me.txt").write_text("not a response\n")
    return root


def test_catalogue_measured_on_a_synthetic_tree(tmp_path):
    root = _but_tree(tmp_path / "but")
    cat = rooms.catalogue_measured(root)
    assert len(cat) == 4
    assert [r.path for r in cat] == sorted(r.path for r in cat)
    assert [r.version for r in cat] == ["v00", "v01", "v00", "v00"]
    assert len({r.id for r in cat}) == 4
    assert {r.room for r in cat} == {"Room_A", "Room_B"}
    assert all(r.source == "measured" and r.room_type == "Office" for r in cat)
    assert [r.rt60_s for r in cat] == [1.2, 1.2, 1.4, 1.8]
    assert [r.distance_m for r in cat] == [2.5, 2.5, 4.0, 1.0]
    assert [r.sample_rate for r in cat] == [16000, 16000, 16000, 32000]
    assert cat[0].path == "Room_A/MicID01/SpkID01_20170901_S/01/RIR/" + WAV_NAME.format("v00")

    assert [r.rt60_s for r in rooms.measured_within(cat, rt60=(1.3, 2.0))] == [1.4, 1.8]
    assert [r.distance_m for r in rooms.measured_within(cat, distance=(2.0, 3.0))] == [2.5, 2.5]
    assert rooms.measured_within(cat, rt60=(1.0, 1.5), distance=(3.0, 5.0)) == [cat[2]]
    blank = rooms.Response(id="x", source="measured", room="R", room_type="", rt60_s=None,
                           distance_m=1.0, placing="", path="x.wav", version="v00",
                           sample_rate=SR)
    assert rooms.measured_within([blank], distance=(0.0, 2.0)) == [blank]
    assert rooms.measured_within([blank], rt60=(0.0, 9.0)) == []

    for resp in cat:
        rir, direct = rooms.load_response(resp, root)
        assert rir.dtype == np.float32 and rir.ndim == 1
        assert rir.size == SR                    # 1 s files, shorter than the 2.5 s cut
        assert np.sum(rir.astype(np.float64) ** 2) == pytest.approx(1.0, abs=1e-5)
        assert direct == 50
    rir, direct = rooms.load_response(cat[0], root, max_s=0.1)
    assert rir.size == 50 + int(0.1 * SR) and direct == 50


def test_load_measured_catalogue_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGHTVOICE_DATA", str(tmp_path / "data"))
    root = _but_tree(tmp_path / "but")
    first = rooms.load_measured_catalogue(root)
    cache = tmp_path / "data" / "cache" / "rooms_measured.json"
    assert cache.exists() and json.loads(cache.read_text())["count"] == 4
    monkeypatch.setattr(rooms, "catalogue_measured", lambda *a, **k: pytest.fail("rescanned"))
    assert rooms.load_measured_catalogue(root) == first


def _inside(point, dims, margin):
    return all(margin - 1e-9 <= p <= d - margin + 1e-9 for p, d in zip(point, dims))


@pytest.fixture(scope="module")
def three_rooms():
    out = {}
    for i, target in enumerate((0.25, 0.35, 0.5)):
        out[target] = rooms.simulate_shoebox(stream(3, i, "rooms"), rt60_s=(target, target))
    return out


def test_shoebox_rt60_matches_target(three_rooms):
    for target, (_, params) in three_rooms.items():
        assert params["rt60_target_s"] == target
        for measured in params["rt60_measured_s"]:
            assert abs(measured - target) <= 0.25 * target, (target, params["rt60_measured_s"])


def test_shoebox_geometry_and_direct_path(three_rooms):
    import pyroomacoustics as pra

    offset = pra.constants.get("frac_delay_length") // 2
    for rirs, params in three_rooms.values():
        dims = params["dims"]
        assert 3.0 <= dims[0] <= 8.0 and 2.5 <= dims[1] <= 6.0 and 2.4 <= dims[2] <= 3.5
        assert _inside(params["mic"], dims, 0.5)
        assert len(rirs) == len(params["sources"]) == len(params["distances_m"]) == 3
        for rir, src, dist, direct in zip(rirs, params["sources"], params["distances_m"],
                                          params["direct_index"]):
            assert _inside(src, dims, 0.3)
            assert 0.3 <= dist <= 2.0
            assert np.linalg.norm(np.subtract(src, params["mic"])) == pytest.approx(dist)
            assert abs((direct - offset) / SR * 343.0 - dist) <= 0.05
            assert rir.dtype == np.float32 and rir.size <= direct + int(2.5 * SR)
            assert np.sum(rir.astype(np.float64) ** 2) == pytest.approx(1.0, abs=1e-5)
        json.dumps(params)


def test_shoebox_is_deterministic():
    a, pa = rooms.simulate_shoebox(stream(1, 7, "rooms"))
    b, pb = rooms.simulate_shoebox(stream(1, 7, "rooms"))
    assert pa == pb
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c, pc = rooms.simulate_shoebox(stream(2, 7, "rooms"))
    assert pc["dims"] != pa["dims"]
    assert not np.array_equal(a[0][:min(a[0].size, c[0].size)], c[0][:min(a[0].size, c[0].size)])


def test_build_bank_round_trip_and_resume(tmp_path, monkeypatch):
    out = tmp_path / "bank"
    calls = []
    real = rooms.simulate_shoebox

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(rooms, "simulate_shoebox", counted)
    built = rooms.build_bank(out, seed=5, n_rooms=4, shard_size=3)
    assert len(calls) == 4
    assert sorted(p.name for p in (out / "shards").iterdir()) == ["rirs_00000.npz",
                                                                   "rirs_00001.npz"]
    assert len(built) == 12 and len({r.id for r in built}) == 12
    assert all(r.source == "simulated" and r.placing == "simulated" and r.version == ""
               for r in built)
    assert built[0].path == "shards/rirs_00000.npz#room_0/0"
    assert built[-1].path == "shards/rirs_00001.npz#room_3/2"
    assert rooms.load_bank(out) == built

    lines = (out / "rooms.jsonl").read_text().splitlines()
    assert [json.loads(line)["index"] for line in lines] == [0, 1, 2, 3]
    bank = json.loads((out / "bank.json").read_text())
    assert bank["seed"] == 5 and bank["n_rooms"] == 4 and bank["n_responses"] == 12
    assert bank["rt60_s"] == [0.2, 0.5] and bank["distance_m"] == [0.3, 2.0]
    assert bank["pyroomacoustics"]

    with np.load(out / "shards" / "rirs_00001.npz") as z:
        assert sorted(z.files) == ["room_3"]
        stored = z["room_3"]
    assert stored.shape[0] == 3 and stored.dtype == np.float32
    resp = built[10]                          # room 3, source 1
    rir, direct = rooms.load_response(resp, out)
    row = np.trim_zeros(stored[1], "b")
    np.testing.assert_allclose(rir, row, rtol=1e-5, atol=1e-7)
    record = json.loads(lines[3])
    assert direct == record["direct_index"][1]
    assert resp.rt60_s == record["rt60_measured_s"][1]
    assert resp.distance_m == record["distances_m"][1]

    # the bank matches a direct simulation of the same stream
    direct_rirs, _ = real(stream(5, 3, "rooms"))
    assert np.array_equal(row, direct_rirs[1])

    mtimes = {p: p.stat().st_mtime_ns for p in (out / "shards").iterdir()}
    again = rooms.build_bank(out, seed=5, n_rooms=4, shard_size=3)
    assert len(calls) == 4 and again == built
    assert {p: p.stat().st_mtime_ns for p in (out / "shards").iterdir()} == mtimes

    grown = rooms.build_bank(out, seed=5, n_rooms=5, shard_size=3)
    assert len(calls) == 5 and len(grown) == 15 and grown[:12] == built
    with np.load(out / "shards" / "rirs_00001.npz") as z:
        assert sorted(z.files) == ["room_3", "room_4"]

    with pytest.raises(ValueError, match="other settings"):
        rooms.build_bank(out, seed=6, n_rooms=4, shard_size=3)


def test_summary_counts_and_ranges(tmp_path):
    cat = rooms.catalogue_measured(_but_tree(tmp_path / "but"))
    s = rooms.summary(cat)
    assert s["total"] == 4 and s["by_source"] == {"measured": 4} and s["rooms"] == 2
    assert s["by_room"] == {"Room_A": 3, "Room_B": 1}
    assert s["rt60_s"] == {"count": 4, "min": 1.2, "median": pytest.approx(1.3), "max": 1.8}
    assert s["distance_m"] == {"count": 4, "min": 1.0, "median": 2.5, "max": 4.0}


@pytest.mark.skipif(not REAL_BUT.is_dir(), reason="BUT ReverbDB is not unpacked under data/sources")
def test_real_but_reverbdb():
    cat = rooms.catalogue_measured(REAL_BUT)
    assert len(cat) == 2325
    assert len({r.room for r in cat}) == 9
    rt60 = [r.rt60_s for r in cat if r.rt60_s is not None]
    assert min(rt60) >= 0.9
    rir, direct = rooms.load_response(cat[0], REAL_BUT)
    assert rir.dtype == np.float32 and direct < 0.1 * SR
    s = rooms.summary(cat)
    print("\nBUT ReverbDB summary:")
    print(json.dumps(s, indent=2))
    by_room: dict[str, list[float]] = {}
    for r in cat:
        if r.rt60_s is not None:
            by_room.setdefault(r.room, []).append(r.rt60_s)
    for room, values in sorted(by_room.items()):
        print(f"{room}: median RT60 {np.median(values):.2f} s over {len(values)} responses")
    print(f"first response {cat[0].path}: direct index {direct}, length {rir.size}")
