"""Room responses for the capture simulation: measured ones from BUT ReverbDB and simulated shoeboxes.

Measured responses come from the unpacked BUT ReverbDB tree,
``<Room>/<MicID>/<SpkID_date_S>/<position>/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.vNN.wav``, with
``mic_meta.txt`` beside each ``RIR/`` folder. Those rooms are all strongly reverberant, so
shoebox rooms simulated with pyroomacoustics fill the drier end. A bank
of simulated rooms is written as ``.npz`` shards plus ``rooms.jsonl``; every room is a pure
function of ``(seed, index)`` through ``rng.stream(seed, index, "rooms")``.

Every response, measured or simulated, is described by a ``Response`` and loaded with
``load_response`` as float32 at 16 kHz with unit energy, together with its direct-path index.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import rng as rng_mod
from .capture import SR, direct_index_of, normalise_rir
from .sources import data_root

MISSING_VALUES = frozenset({"", "not defined", "#n/a", "n/a", "nan"})

# Shoebox ranges, in metres and seconds.
ROOM_LENGTH_M = (3.0, 8.0)
ROOM_WIDTH_M = (2.5, 6.0)
ROOM_HEIGHT_M = (2.4, 3.5)
MIC_WALL_MARGIN_M = 0.5
SOURCE_WALL_MARGIN_M = 0.3
RIR_MAX_S = 2.5

# Achieved RT60 is checked against the target on the first source's response; outside this
# relative tolerance the absorption is rescaled and the room simulated again, at most this
# many more times.
RT60_TOLERANCE = 0.10
RT60_RETRIES = 3
ABSORPTION_RANGE = (0.01, 0.99)

# Directions tried for one drawn source distance before that distance is drawn again.
DIRECTION_TRIES = 1000

_VERSION_RE = re.compile(r"\.(v\d+)\.wav$", re.IGNORECASE)


@dataclass(frozen=True)
class Response:
    id: str
    source: str            # "measured" or "simulated"
    room: str
    room_type: str
    rt60_s: float | None
    distance_m: float | None
    placing: str
    path: str              # relative to the BUT root, or a shard reference into a bank
    version: str           # "v00" and so on for measured, "" for simulated
    sample_rate: int


# Measured responses


def resolve_but_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else data_root() / "sources" / "but_reverbdb"


def _value(raw: str | None) -> str | None:
    if raw is None:
        return None
    v = raw.strip()
    return None if v.lower() in MISSING_VALUES else v


def _float(raw: str | None) -> float | None:
    v = _value(raw)
    if v is None:
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _read_meta_lines(path: Path) -> dict[str, str]:
    """``$Key<TAB>value`` lines as a dict; lines without a leading ``$`` are ignored."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip("\r\n")
        if not line.startswith("$"):
            continue
        if "\t" in line:
            key, _, value = line.partition("\t")
        else:
            parts = line.split(None, 1)
            key, value = parts[0], parts[1] if len(parts) > 1 else ""
        out[key[1:].strip()] = value.strip()
    return out


def parse_mic_meta(path: Path | str) -> dict:
    """The room and microphone facts of one ``mic_meta.txt``; missing values come back as None.

    Keys: ``room``, ``room_type``, ``mic_id``, ``rt60_s``, ``rt30_s``, ``distance_m`` (the
    distance measured from the response, else the tape-measured one), ``placing`` and
    ``delay_s``. "Not Defined", "#N/A" and empty values count as missing.
    """
    meta = _read_meta_lines(Path(path))
    mic_id_f = _float(meta.get("EnvMicID"))
    mic_id = int(mic_id_f) if mic_id_f is not None else None

    def mic(key: str) -> str | None:
        return None if mic_id is None else meta.get(f"EnvMic{mic_id}{key}")

    distance = _float(mic("RelDistanceRIRMeasured"))
    if distance is None:
        distance = _float(mic("RelDistance"))
    return {
        "room": _value(meta.get("EnvName")),
        "room_type": _value(meta.get("EnvType")),
        "mic_id": mic_id,
        "rt60_s": _float(mic("RelRT60")),
        "rt30_s": _float(mic("RelRT30")),
        "distance_m": distance,
        "placing": _value(mic("Placing")),
        "delay_s": _float(mic("RelDelayRIRMeasured")),
    }


def catalogue_measured(but_root: Path | str | None = None) -> list[Response]:
    """One ``Response`` per RIR wav under the BUT ReverbDB root, sorted by path."""
    root = resolve_but_root(but_root)
    if not root.is_dir():
        raise FileNotFoundError(f"BUT ReverbDB not found at {root}")
    metas: dict[Path, dict] = {}
    rows: list[Response] = []
    for wav in root.glob("*/*/*/*/RIR/*.wav"):
        if wav.name.startswith("."):
            continue
        position = wav.parent.parent
        if position not in metas:
            meta_path = position / "mic_meta.txt"
            metas[position] = parse_mic_meta(meta_path) if meta_path.is_file() else {}
        meta = metas[position]
        rel = wav.relative_to(root).as_posix()
        match = _VERSION_RE.search(wav.name)
        version = match.group(1).lower() if match else ""
        rows.append(Response(
            id=f"{position.relative_to(root).as_posix()}/{version or wav.stem}",
            source="measured",
            room=meta.get("room") or rel.split("/", 1)[0],
            room_type=meta.get("room_type") or "",
            rt60_s=meta.get("rt60_s"),
            distance_m=meta.get("distance_m"),
            placing=meta.get("placing") or "",
            path=rel,
            version=version,
            sample_rate=int(sf.info(str(wav)).samplerate),
        ))
    rows.sort(key=lambda r: r.path)
    return rows


def measured_cache_path() -> Path:
    return data_root() / "cache" / "rooms_measured.json"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_measured_catalogue(but_root: Path | str | None = None,
                            rebuild: bool = False) -> list[Response]:
    """The measured catalogue, from its JSON cache when present and built from the same root."""
    root = resolve_but_root(but_root)
    key = str(root.resolve())
    path = measured_cache_path()
    if path.exists() and not rebuild:
        blob = json.loads(path.read_text())
        if blob.get("but_root") == key:
            return [Response(**row) for row in blob["responses"]]
    rows = catalogue_measured(root)
    blob = {"but_root": key, "count": len(rows), "responses": [asdict(r) for r in rows]}
    _write_text_atomic(path, json.dumps(blob, indent=0))
    return rows


def _within(value: float | None, bounds: tuple[float, float] | None) -> bool:
    if bounds is None:
        return True
    return value is not None and bounds[0] <= value <= bounds[1]


def measured_within(catalogue: list[Response], rt60: tuple[float, float] | None = None,
                    distance: tuple[float, float] | None = None) -> list[Response]:
    """Measured responses whose RT60 and distance lie inside the given closed ranges.

    A range left as None is not checked. Entries missing a checked value are dropped.
    """
    return [r for r in catalogue
            if r.source == "measured" and _within(r.rt60_s, rt60)
            and _within(r.distance_m, distance)]


# Loading


def parse_shard_ref(ref: str) -> tuple[str, str, int]:
    """``shards/rirs_00000.npz#room_12/1`` to ``("shards/rirs_00000.npz", "room_12", 1)``."""
    shard, sep, rest = ref.partition("#")
    key, sep2, row = rest.rpartition("/")
    if not sep or not sep2 or not key or not row.isdigit():
        raise ValueError(f"not a shard reference: {ref!r}")
    return shard, key, int(row)


def _read_shard_row(bank_dir: Path, ref: str) -> np.ndarray:
    shard, key, row = parse_shard_ref(ref)
    with np.load(bank_dir / shard) as z:
        arr = z[key]
    if not 0 <= row < arr.shape[0]:
        raise ValueError(f"{ref!r}: row {row} outside {arr.shape[0]} sources")
    return np.trim_zeros(np.asarray(arr[row], dtype=np.float32), "b")


def _cut_and_normalise(rir: np.ndarray, max_s: float | None, sr: int) -> tuple[np.ndarray, int]:
    direct = direct_index_of(rir)
    if max_s is not None:
        rir = rir[:direct + round(max_s * sr)]
    rir = normalise_rir(rir)
    return rir, direct_index_of(rir)


def load_response(resp: Response, root: Path | str | None = None,
                  max_s: float | None = RIR_MAX_S) -> tuple[np.ndarray, int]:
    """The response as float32 mono at 16 kHz with unit energy, and its direct-path index.

    ``root`` is the BUT ReverbDB root for measured responses (default under ``data_root()``)
    and the bank directory for simulated ones (required). Channels are averaged, other
    sample rates are converted with soxr, and the response is cut ``max_s`` seconds after
    the direct path before it is normalised.
    """
    if resp.source == "measured":
        path = resolve_but_root(root) / resp.path
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        rir = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    elif resp.source == "simulated":
        if root is None:
            raise ValueError("simulated responses need the bank directory as root")
        rir = _read_shard_row(Path(root), resp.path)
        sr = resp.sample_rate
    else:
        raise ValueError(f"unknown response source {resp.source!r}")
    if sr != SR:
        rir = soxr.resample(rir, sr, SR, quality="VHQ").astype(np.float32)
    return _cut_and_normalise(rir, max_s, SR)


# Shoebox simulation


def _pra():
    import pyroomacoustics as pra

    constants = getattr(pra, "constants", None)
    if constants is not None and hasattr(constants, "set"):
        constants.set("num_threads", 1)
    return pra


def _pra_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pyroomacoustics")
    except PackageNotFoundError:
        return str(getattr(_pra(), "__version__", ""))


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    z = float(rng.uniform(-1.0, 1.0))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([r * math.cos(phi), r * math.sin(phi), z])


def _place_source(rng: np.random.Generator, dims: np.ndarray, mic: np.ndarray,
                  distance_m: tuple[float, float]) -> tuple[np.ndarray, float]:
    """A point at a drawn distance from ``mic`` in a random direction, inside the margin.

    The direction is drawn again until the point is at least 0.3 m inside every wall. If no
    direction fits after ``DIRECTION_TRIES`` tries the distance is drawn again, so a distance
    that cannot fit in a small room does not loop forever.
    """
    lo, hi = SOURCE_WALL_MARGIN_M, dims - SOURCE_WALL_MARGIN_M
    while True:
        d = float(rng.uniform(*distance_m))
        for _ in range(DIRECTION_TRIES):
            p = mic + d * _unit_vector(rng)
            if np.all(p >= lo) and np.all(p <= hi):
                return p, d


def _shoebox_rirs(pra, dims: np.ndarray, absorption: float, max_order: int, mic: np.ndarray,
                  sources: list[np.ndarray], sr: int) -> list[np.ndarray]:
    room = pra.ShoeBox(dims.tolist(), fs=sr, materials=pra.Material(absorption),
                       max_order=max_order)
    for p in sources:
        room.add_source(p.tolist())
    room.add_microphone(mic.tolist())
    room.compute_rir()
    return [np.asarray(room.rir[0][s], dtype=np.float64) for s in range(len(sources))]


def simulate_shoebox(rng: np.random.Generator, rt60_s: tuple[float, float] = (0.2, 0.5),
                     distance_m: tuple[float, float] = (0.3, 2.0), n_sources: int = 3,
                     sr: int = SR) -> tuple[list[np.ndarray], dict]:
    """Simulate one shoebox room with one microphone and ``n_sources`` sources.

    Draws, in order: length, width, height, target RT60, microphone position (at least 0.5 m
    from every surface), then each source's distance and direction. Absorption and the
    reflection order come from Sabine's formula; the achieved RT60 of the first source's
    response (T30 extrapolated to 60 dB) is brought within 10% of the target by rescaling
    the absorption, at most three times. Returns the responses (float32, cut 2.5 s after the
    direct path, unit energy) and the drawn and achieved parameters as JSON-friendly values.
    """
    if n_sources < 1:
        raise ValueError("n_sources must be at least 1")
    pra = _pra()
    dims = np.array([rng.uniform(*ROOM_LENGTH_M), rng.uniform(*ROOM_WIDTH_M),
                     rng.uniform(*ROOM_HEIGHT_M)])
    target = float(rng.uniform(*rt60_s))
    mic = np.array([rng.uniform(MIC_WALL_MARGIN_M, d - MIC_WALL_MARGIN_M) for d in dims])
    placed = [_place_source(rng, dims, mic, distance_m) for _ in range(n_sources)]
    sources = [p for p, _ in placed]
    distances = [d for _, d in placed]

    e_absorption, max_order = pra.inverse_sabine(target, dims.tolist())
    absorption = float(e_absorption)
    max_order = int(max_order)
    measure = pra.experimental.rt60.measure_rt60
    attempts = 0
    while True:
        attempts += 1
        raw = _shoebox_rirs(pra, dims, absorption, max_order, mic, sources, sr)
        first = float(measure(raw[0], fs=sr, decay_db=30))
        if abs(first - target) <= RT60_TOLERANCE * target or attempts > RT60_RETRIES:
            break
        absorption = float(np.clip(absorption * first / target, *ABSORPTION_RANGE))

    measured = [first] + [float(measure(r, fs=sr, decay_db=30)) for r in raw[1:]]
    rirs, direct = [], []
    for r in raw:
        cut, idx = _cut_and_normalise(r, RIR_MAX_S, sr)
        rirs.append(cut)
        direct.append(idx)
    params = {
        "dims": [float(v) for v in dims],
        "rt60_target_s": target,
        "rt60_measured_s": measured,
        "absorption": absorption,
        "absorption_sabine": float(e_absorption),
        "max_order": max_order,
        "attempts": attempts,
        "mic": [float(v) for v in mic],
        "sources": [[float(v) for v in p] for p in sources],
        "distances_m": [float(d) for d in distances],
        "direct_index": direct,
        "sample_rate": int(sr),
    }
    return rirs, params


# Bank of simulated rooms

ROOMS_FILE = "rooms.jsonl"
BANK_FILE = "bank.json"


def shard_name(shard: int) -> str:
    return f"shards/rirs_{shard:05d}.npz"


def room_key(index: int) -> str:
    return f"room_{index}"


def _responses_of(record: dict, sample_rate: int) -> list[Response]:
    i = int(record["index"])
    return [
        Response(
            id=f"shoebox_{i:05d}_{s}",
            source="simulated",
            room=f"shoebox_{i:05d}",
            room_type="shoebox",
            rt60_s=float(record["rt60_measured_s"][s]),
            distance_m=float(record["distances_m"][s]),
            placing="simulated",
            path=f"{record['ref']}/{s}",
            version="",
            sample_rate=int(record.get("sample_rate", sample_rate)),
        )
        for s in range(len(record["distances_m"]))
    ]


def _read_records(out: Path) -> dict[int, dict]:
    path = out / ROOMS_FILE
    if not path.exists():
        return {}
    records: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            records[int(rec["index"])] = rec
    return records


def _write_records(out: Path, records: dict[int, dict]) -> None:
    text = "".join(json.dumps(records[i]) + "\n" for i in sorted(records))
    _write_text_atomic(out / ROOMS_FILE, text)


def _write_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _pad(rirs: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(rirs), max(r.size for r in rirs)), dtype=np.float32)
    for s, r in enumerate(rirs):
        out[s, :r.size] = r
    return out


def _write_bank_json(out: Path, config: dict, records: dict[int, dict]) -> None:
    blob = {**config, "n_rooms": len(records),
            "n_responses": sum(len(r["distances_m"]) for r in records.values()),
            "n_shards": len({r["shard"] for r in records.values()}),
            "pyroomacoustics": _pra_version()}
    _write_text_atomic(out / BANK_FILE, json.dumps(blob, indent=2) + "\n")


def build_bank(out_dir: Path | str, seed: int, n_rooms: int,
               rt60_s: tuple[float, float] = (0.2, 0.5),
               distance_m: tuple[float, float] = (0.3, 2.0), n_sources: int = 3,
               shard_size: int = 100) -> list[Response]:
    """Simulate rooms ``0 .. n_rooms - 1`` into ``out_dir`` and return their responses.

    Room ``i`` is drawn from ``rng.stream(seed, i, "rooms")`` and stored as ``room_<i>``
    (shape ``[n_sources, max_len]``, zero padded) in shard ``i // shard_size``. A room already
    listed in ``rooms.jsonl`` and present in its shard is not simulated again. ``bank.json``
    is written first, so a resume with other settings is refused. Each shard is rewritten
    whole, then ``rooms.jsonl``, both through a temporary file and a rename.
    """
    out = Path(out_dir)
    config = {"seed": int(seed), "rt60_s": [float(v) for v in rt60_s],
              "distance_m": [float(v) for v in distance_m], "n_sources": int(n_sources),
              "shard_size": int(shard_size), "sample_rate": SR}
    bank_path = out / BANK_FILE
    if bank_path.exists():
        old = json.loads(bank_path.read_text())
        changed = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
        if changed:
            raise ValueError(f"{out} holds a bank built with other settings: {changed}")
    records = _read_records(out)
    _write_bank_json(out, config, records)
    for shard in range(math.ceil(n_rooms / shard_size)):
        indices = range(shard * shard_size, min(n_rooms, (shard + 1) * shard_size))
        name = shard_name(shard)
        path = out / name
        arrays: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files}
        todo = [i for i in indices if not (i in records and room_key(i) in arrays)]
        if not todo:
            continue
        for i in todo:
            rirs, params = simulate_shoebox(rng_mod.stream(seed, i, "rooms"), rt60_s, distance_m,
                                            n_sources, SR)
            arrays[room_key(i)] = _pad(rirs)
            records[i] = {"index": i, "ref": f"{name}#{room_key(i)}", "shard": name,
                          "key": room_key(i), **params}
        _write_shard(path, arrays)
        _write_records(out, records)
    _write_bank_json(out, config, records)
    return [resp for i in sorted(records) if i < n_rooms for resp in _responses_of(records[i], SR)]


def load_bank(out_dir: Path | str) -> list[Response]:
    """Every response in a bank, in room then source order."""
    out = Path(out_dir)
    bank_path = out / BANK_FILE
    sample_rate = SR
    if bank_path.exists():
        sample_rate = json.loads(bank_path.read_text()).get("sample_rate", SR)
    records = _read_records(out)
    return [resp for i in sorted(records) for resp in _responses_of(records[i], sample_rate)]


# Summary


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)}


def summary(catalogue: list[Response]) -> dict:
    """Counts per source and per room, and RT60 and distance min, median and max."""
    return {
        "total": len(catalogue),
        "by_source": dict(sorted(Counter(r.source for r in catalogue).items())),
        "rooms": len({(r.source, r.room) for r in catalogue}),
        "by_room": dict(sorted(Counter(r.room for r in catalogue).items())),
        "rt60_s": _stats([r.rt60_s for r in catalogue if r.rt60_s is not None]),
        "distance_m": _stats([r.distance_m for r in catalogue if r.distance_m is not None]),
    }
