"""Simulated capture for phone, call and laptop audio.

Pure functions on float32 mono arrays at 16 kHz. Every random draw comes from a
``numpy.random.Generator`` passed in by the caller, normally ``rightvoice.rng.stream(seed,
index, stage)``, so each item stays a pure function of its seed and index. Stages that draw
values return them in a plain dict of JSON-friendly values so a manifest can record them.
"""
from __future__ import annotations

import functools
import math
import os
import shutil
import subprocess

import numpy as np
import soxr
from scipy import signal

SR = 16000

# Levels, in dBFS (full scale = 1.0) or dB.
SOURCE_LEVEL_DBFS = -25.0
MIXTURE_LEVEL_DBFS = (-30.0, -15.0)
PARTNER_GAIN_DB = (-6.0, 6.0)
BYSTANDER_GAIN_DB = (-20.0, -8.0)

# Noise and music SNR per branch, in dB of active-speech RMS over noise RMS.
BRANCH_SNR_DB = {"room": (15, 35), "call": (15, 35), "near_clean": (30, 45)}

# Device chain: probability of each stage, listed in the order the stages are applied.
DEVICE_STAGE_PROBS = {"agc": 0.2, "clip": 0.02, "codec": 0.4, "packet_loss": 0.05}
MAX_DEVICE_STAGES = 2
DEVICE_CODEC = "opus"
PHONE_TIERS = ("amr_wb", "amr_nb", "g711")

AGC_MAX_DB = 3.0
AGC_PERIOD_S = (1.0, 4.0)
CLIP_FRACTION = (0.001, 0.01)
PACKET_GAP_MS = (20, 100)
PACKET_MAX_FRACTION = 0.02
PACKET_FADE_MS = 2

# Codec delay is found by cross-correlation over one second, within +/- 100 ms.
ALIGN_WINDOW_S = 1.0
ALIGN_MAX_LAG_S = 0.1

# Pinned ffmpeg settings per codec. ``rate`` is the rate fed to the encoder and ``out_rate``
# the rate the pinned decoder returns; both are converted to and from 16 kHz with soxr.
_CODECS: dict[str, dict] = {
    "opus": {"encoder": "libopus", "decoder": "opus", "muxer": "ogg", "rate": 16000,
             "out_rate": 48000, "kbps": None,
             "flags": ("-application", "voip", "-vbr", "on", "-frame_duration", "20",
                       "-compression_level", "10")},
    "amr_nb": {"encoder": "libopencore_amrnb", "decoder": "amrnb", "muxer": "amr", "rate": 8000,
               "out_rate": 8000, "kbps": 12.2, "flags": ("-dtx", "0")},
    "amr_wb": {"encoder": "libvo_amrwbenc", "decoder": "amrwb", "muxer": "amr", "rate": 16000,
               "out_rate": 16000, "kbps": 23.85, "flags": ("-dtx", "0")},
    "g711": {"encoder": "pcm_mulaw", "decoder": "pcm_mulaw", "muxer": "mulaw", "rate": 8000,
             "out_rate": 8000, "kbps": 64.0, "flags": ()},
}
OPUS_KBPS = (16, 32)
TELEPHONE_BAND_HZ = (300.0, 3400.0)


class CodecUnavailable(RuntimeError):
    """The ffmpeg on this machine has no encoder for the requested codec."""


def _mono(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError(f"expected a mono 1-D array, got shape {a.shape}")
    return a


# Levels


def rms_db(x) -> float:
    """RMS of the whole signal in dBFS; -inf for silence or empty input."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return -math.inf
    ms = float(np.mean(a * a))
    return 10.0 * math.log10(ms) if ms > 0.0 else -math.inf


def active_rms_dbfs(x, frame_ms: float = 20, threshold_db: float = -50) -> float:
    """RMS in dBFS over the frames whose own RMS is above ``threshold_db``.

    Frames do not overlap; a short last frame counts as its own frame. Returns -inf when
    no frame is active.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return -math.inf
    frame = max(1, round(SR * frame_ms / 1000))
    starts = np.arange(0, a.size, frame)
    energy = np.add.reduceat(a * a, starts)
    lengths = np.diff(np.append(starts, a.size))
    active = energy / lengths > 10.0 ** (threshold_db / 10.0)
    if not np.any(active):
        return -math.inf
    return 10.0 * math.log10(energy[active].sum() / lengths[active].sum())


def level_to(x, target_dbfs: float) -> np.ndarray:
    """Scale so the active-speech RMS equals ``target_dbfs``. Silence comes back unchanged.

    A gain can move quiet frames across the activity threshold, so the gain is refined up to
    three more times.
    """
    y = _mono(x).astype(np.float64)
    for _ in range(4):
        level = active_rms_dbfs(y)
        if not math.isfinite(level) or abs(level - target_dbfs) < 1e-3:
            break
        y = y * 10.0 ** ((target_dbfs - level) / 20.0)
    return y.astype(np.float32)


def mixture_level(rng: np.random.Generator) -> float:
    """Active-speech level of the final mixture, uniform in -30 to -15 dBFS."""
    return float(rng.uniform(*MIXTURE_LEVEL_DBFS))


def partner_gain_db(rng: np.random.Generator) -> float:
    """Gain of the conversation partner relative to the target, uniform in -6 to +6 dB."""
    return float(rng.uniform(*PARTNER_GAIN_DB))


def peak_gain(x, peak: float = 0.99) -> float:
    """Gain that brings the absolute peak down to ``peak``; 1.0 if it is already at or below."""
    a = np.asarray(x)
    top = float(np.max(np.abs(a))) if a.size else 0.0
    return peak / top if top > peak else 1.0


def peak_guard(x, peak: float = 0.99) -> np.ndarray:
    """Scale down only if the absolute peak exceeds ``peak``. Never clips."""
    a = _mono(x)
    gain = peak_gain(a, peak)
    return a.copy() if gain == 1.0 else (a.astype(np.float64) * gain).astype(np.float32)


# Bystander playback chain: a voice heard through a TV or phone speaker.


def peaking_biquad(f0: float, gain_db: float, q: float, sr: int = SR):
    """Peaking EQ biquad from the Audio EQ Cookbook. Returns normalised ``(b, a)``."""
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b = np.array([1.0 + alpha * amp, -2.0 * cw, 1.0 - alpha * amp])
    a = np.array([1.0 + alpha / amp, -2.0 * cw, 1.0 - alpha / amp])
    return b / a[0], a / a[0]


def draw_playback_params(rng: np.random.Generator) -> dict:
    """Draw the playback chain parameters: two resonances, then the compressor."""
    resonances = [{"freq_hz": float(rng.uniform(400.0, 4000.0)),
                   "gain_db": float(rng.uniform(3.0, 9.0)),
                   "q": float(rng.uniform(1.5, 4.0))} for _ in range(2)]
    return {"highpass_hz": 150.0, "lowpass_hz": 6000.0, "resonances": resonances,
            "threshold_db_rel_peak": float(rng.uniform(-20.0, -8.0)),
            "ratio": float(rng.uniform(2.0, 6.0)), "attack_ms": 5.0, "release_ms": 50.0}


def playback_sos(params: dict) -> np.ndarray:
    """Second-order sections of the linear part of the playback chain."""
    hp = signal.butter(2, params["highpass_hz"], btype="highpass", fs=SR, output="sos")
    lp = signal.butter(2, params["lowpass_hz"], btype="lowpass", fs=SR, output="sos")
    peaks = [np.concatenate(peaking_biquad(r["freq_hz"], r["gain_db"], r["q"]))
             for r in params["resonances"]]
    return np.vstack([hp, lp, *peaks])


def _envelope(mag: np.ndarray, attack: float, release: float) -> np.ndarray:
    """One-pole follower: the attack coefficient while rising, the release one while falling."""
    out = []
    env = 0.0
    for m in mag.tolist():
        c = attack if m > env else release
        env = c * env + (1.0 - c) * m
        out.append(env)
    return np.asarray(out, dtype=np.float64)


def compress(x, threshold_dbfs: float, ratio: float, attack_ms: float = 5.0,
             release_ms: float = 50.0) -> np.ndarray:
    """Feed-forward compressor driven by a one-pole envelope follower on ``|x|``."""
    a = np.asarray(x, dtype=np.float64)
    attack = math.exp(-1.0 / (attack_ms * 1e-3 * SR))
    release = math.exp(-1.0 / (release_ms * 1e-3 * SR))
    env_db = 20.0 * np.log10(np.maximum(_envelope(np.abs(a), attack, release), 1e-12))
    gain_db = -np.maximum(env_db - threshold_dbfs, 0.0) * (1.0 - 1.0 / ratio)
    return (a * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def playback_chain(x, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """High-pass 150 Hz, low-pass 6 kHz, two resonances, then a compressor.

    The compressor threshold is set relative to the peak after the filters.
    """
    params = draw_playback_params(rng)
    y = signal.sosfilt(playback_sos(params), _mono(x).astype(np.float64))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.0:
        threshold = 20.0 * math.log10(peak) + params["threshold_db_rel_peak"]
        y = compress(y, threshold, params["ratio"], params["attack_ms"], params["release_ms"])
        params["threshold_dbfs"] = threshold
    else:
        params["threshold_dbfs"] = None
    return np.asarray(y, dtype=np.float32), params


def bystander_gain_db(rng: np.random.Generator) -> float:
    """Bystander level relative to the conversation's active-speech RMS: -20 to -8 dB."""
    return float(rng.uniform(*BYSTANDER_GAIN_DB))


def place_bystander(canvas_len: int, clip, rng: np.random.Generator) -> int:
    """Start sample that keeps ``clip`` (an array or a length) fully inside the canvas."""
    n = int(clip) if isinstance(clip, (int, np.integer)) else len(clip)
    if n > canvas_len:
        raise ValueError(f"clip of {n} samples does not fit a canvas of {canvas_len}")
    return int(rng.integers(0, canvas_len - n + 1))


# Rooms


def direct_index_of(rir) -> int:
    """Index of the first sample whose magnitude reaches 50% of the peak magnitude."""
    mag = np.abs(np.asarray(rir, dtype=np.float64))
    if mag.size == 0 or mag.max() <= 0.0:
        raise ValueError("room response is empty or silent")
    return int(np.argmax(mag >= 0.5 * mag.max()))


def normalise_rir(rir) -> np.ndarray:
    """Scale a room response to unit energy."""
    r = np.asarray(rir, dtype=np.float64)
    energy = float(np.sum(r * r))
    if energy <= 0.0:
        raise ValueError("room response is silent")
    return (r / math.sqrt(energy)).astype(np.float32)


def reverberate(x, rir, direct_index: int) -> np.ndarray:
    """Convolve with ``rir`` and drop the first ``direct_index`` samples.

    The direct path lands at time zero and the output keeps the input length.
    """
    a = _mono(x)
    r = np.asarray(rir, dtype=np.float64)
    if not 0 <= direct_index < r.size:
        raise ValueError(f"direct_index {direct_index} outside a response of {r.size}")
    if a.size == 0:
        return a.copy()
    wet = signal.fftconvolve(a.astype(np.float64), r, mode="full")
    return wet[direct_index:direct_index + a.size].astype(np.float32)


# Noise and music at SNR


def _fit_length(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Crop ``noise`` to ``n`` samples at a random offset, tiling it first if it is shorter."""
    m = noise.size
    if m == 0:
        raise ValueError("noise is empty")
    if m < n:
        noise = np.tile(noise, n // m + 2)
        offset = int(rng.integers(0, m))
    else:
        offset = int(rng.integers(0, m - n + 1))
    return noise[offset:offset + n]


def add_at_snr(speech, noise, snr_db: float, rng: np.random.Generator):
    """Add ``noise`` so active-speech RMS over noise RMS equals ``snr_db``.

    Returns ``(mixture, scaled_noise)``, both the length of ``speech``.
    """
    s = _mono(speech)
    seg = _fit_length(_mono(noise), s.size, rng).astype(np.float64)
    speech_db = active_rms_dbfs(s)
    noise_db = rms_db(seg)
    if not math.isfinite(speech_db):
        raise ValueError("speech has no active frames")
    if not math.isfinite(noise_db):
        raise ValueError("noise is silent")
    scaled = (seg * 10.0 ** ((speech_db - snr_db - noise_db) / 20.0)).astype(np.float32)
    mixture = (s.astype(np.float64) + scaled).astype(np.float32)
    return mixture, scaled


def draw_snr_db(branch: str, rng: np.random.Generator) -> float:
    """SNR for a branch, uniform over ``BRANCH_SNR_DB[branch]``."""
    if branch not in BRANCH_SNR_DB:
        raise ValueError(f"unknown branch {branch!r}; one of {tuple(BRANCH_SNR_DB)}")
    return float(rng.uniform(*BRANCH_SNR_DB[branch]))


# Device chain stages. Each takes rng and returns (audio, params).


def agc_drift(x, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Gain wandering within +/- 3 dB: a bounded random walk with cosine steps.

    Knots sit one period apart (period 1 to 4 s); each knot moves from the last by a normal
    step with a 1.5 dB spread, reflected back inside the bound.
    """
    a = _mono(x)
    period_s = float(rng.uniform(*AGC_PERIOD_S))
    step = max(1, round(period_s * SR))
    phase = float(rng.uniform(0.0, 1.0))
    n_knots = a.size // step + 3
    moves = rng.normal(0.0, AGC_MAX_DB / 2.0, size=n_knots)
    knots = np.empty(n_knots)
    level = float(rng.uniform(-AGC_MAX_DB, AGC_MAX_DB))
    for i in range(n_knots):
        knots[i] = level
        level += float(moves[i])
        if level > AGC_MAX_DB:
            level = 2.0 * AGC_MAX_DB - level
        elif level < -AGC_MAX_DB:
            level = -2.0 * AGC_MAX_DB - level
        level = min(max(level, -AGC_MAX_DB), AGC_MAX_DB)
    t = np.arange(a.size) / step + phase
    i = np.floor(t).astype(np.int64)
    w = 0.5 - 0.5 * np.cos(math.pi * (t - i))
    gain_db = knots[i] * (1.0 - w) + knots[i + 1] * w
    y = (a.astype(np.float64) * 10.0 ** (gain_db / 20.0)).astype(np.float32)
    return y, {
        "stage": "agc",
        "period_s": period_s,
        "step_sd_db": AGC_MAX_DB / 2.0,
        "start_db": float(knots[0]),
        "min_db": float(gain_db.min()) if a.size else 0.0,
        "max_db": float(gain_db.max()) if a.size else 0.0,
    }


def clip_fraction(x, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Hard-clip so that 0.1% to 1% of samples are clipped.

    The threshold is the quantile of ``|x|`` that leaves the drawn number of samples above
    it. Ties at the threshold can make the clipped share slightly smaller; the measured
    share is returned.
    """
    a = _mono(x)
    n = a.size
    target = float(rng.uniform(*CLIP_FRACTION))
    k = min(max(round(target * n), math.ceil(CLIP_FRACTION[0] * n)),
            math.floor(CLIP_FRACTION[1] * n))
    params = {"stage": "clip", "target_fraction": target, "threshold": None,
              "clipped_fraction": 0.0}
    if k < 1:
        return a.copy(), params
    mag = np.abs(a)
    threshold = np.float32(np.partition(mag, n - k - 1)[n - k - 1])
    if threshold <= 0.0:
        return a.copy(), params
    y = np.clip(a, -threshold, threshold)
    params.update(threshold=float(threshold), clipped_fraction=float(np.mean(mag > threshold)))
    return y, params


def packet_loss(x, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Zero gaps of 20 to 100 ms that cover at most 2% of the length.

    A coverage budget of 0.5% to 2% is drawn and filled with gaps. Each gap has a 2 ms
    linear fade out before it and a 2 ms fade in after it; other samples are untouched.
    """
    a = _mono(x)
    n = a.size
    fade = int(SR * PACKET_FADE_MS / 1000)
    gap_min, gap_max = (int(SR * ms / 1000) for ms in PACKET_GAP_MS)
    budget = int(rng.uniform(0.25, 1.0) * PACKET_MAX_FRACTION * n)
    lengths: list[int] = []
    total = 0
    while min(gap_max, budget - total) >= gap_min:
        g = int(rng.integers(gap_min, min(gap_max, budget - total) + 1))
        lengths.append(g)
        total += g
    free = n - total - 2 * fade * len(lengths)
    params = {"stage": "packet_loss", "gaps": [], "lost_fraction": 0.0,
              "fade_ms": PACKET_FADE_MS}
    if not lengths or free < 0:
        return a.copy(), params
    cuts = np.sort(rng.integers(0, free + 1, size=len(lengths)))
    env = np.ones(n)
    ramp = np.arange(1, fade + 1) / (fade + 1)
    used = 0
    for cut, g in zip(cuts.tolist(), lengths):
        start = cut + used + fade
        env[start - fade:start] = ramp[::-1]
        env[start:start + g] = 0.0
        env[start + g:start + g + fade] = ramp
        params["gaps"].append([start, g])
        used += g + 2 * fade
    params["lost_fraction"] = total / n
    return (a.astype(np.float64) * env).astype(np.float32), params


# Codecs through ffmpeg


def _ffmpeg_path() -> str | None:
    return os.environ.get("RIGHTVOICE_FFMPEG") or shutil.which("ffmpeg")


@functools.cache
def _probe(path: str | None) -> tuple[str | None, tuple[str, ...]]:
    if path is None:
        return None, ()
    try:
        run = functools.partial(subprocess.run, capture_output=True, text=True, check=True)
        version = run([path, "-hide_banner", "-version"]).stdout.splitlines()[0].strip()
        listing = run([path, "-hide_banner", "-encoders"]).stdout
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None, ()
    names: set[str] = set()
    past_legend = False
    for line in listing.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("------"):
            past_legend = True
        elif past_legend and len(parts) >= 2:
            names.add(parts[1])
    wanted = [spec["encoder"] for spec in _CODECS.values()]
    return version, tuple(e for e in wanted if e in names)


def ffmpeg_info() -> dict:
    """The ffmpeg path, its version line, and which of the four codec encoders it has."""
    path = _ffmpeg_path()
    version, encoders = _probe(path)
    return {"path": path, "version": version, "encoders": list(encoders)}


def _codec_spec(name: str) -> dict:
    if name not in _CODECS:
        raise ValueError(f"unknown codec {name!r}; one of {tuple(_CODECS)}")
    return _CODECS[name]


def codec_available(name: str) -> bool:
    """True if ffmpeg has the encoder for ``name`` (opus, amr_nb, amr_wb or g711)."""
    return _codec_spec(name)["encoder"] in ffmpeg_info()["encoders"]


def _ffmpeg(args: list[str], data: bytes) -> bytes:
    path = _ffmpeg_path()
    if path is None:
        raise CodecUnavailable("ffmpeg not found")
    cmd = [path, "-hide_banner", "-loglevel", "error", "-nostats", *args]
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with {proc.returncode}: {err}")
    return proc.stdout


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or x.size == 0:
        return x
    return soxr.resample(x, sr_in, sr_out, quality="VHQ")


def telephone_band(x) -> np.ndarray:
    """Zero-phase 300 to 3400 Hz band-limit (4th-order Butterworth, run forward and back)."""
    a = _mono(x).astype(np.float64)
    sos = signal.butter(4, TELEPHONE_BAND_HZ, btype="bandpass", fs=SR, output="sos")
    padlen = min(3 * (2 * len(sos) + 1), max(a.size - 1, 0))
    return signal.sosfiltfilt(sos, a, padlen=padlen).astype(np.float32)


def estimate_delay(out, ref, max_lag: int = int(ALIGN_MAX_LAG_S * SR),
                   window: int = int(ALIGN_WINDOW_S * SR)) -> int:
    """Samples by which ``out`` trails ``ref``, from cross-correlation over one second.

    Uses the first second of ``ref``; if that is below -60 dBFS, the loudest second instead.
    Periodic input gives near-equal peaks one period apart, so among local peaks within 2%
    of the best the smallest shift wins. Returns 0 when ``ref`` is silent.
    """
    ref = np.asarray(ref, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    window = min(window, ref.size)
    if window == 0 or out.size == 0:
        return 0
    start = 0
    if rms_db(ref[:window]) < -60.0:
        cum = np.concatenate(([0.0], np.cumsum(ref * ref)))
        start = int(np.argmax(cum[window:] - cum[:-window]))
    seg_ref = ref[start:start + window]
    if not np.any(seg_ref):
        return 0
    lo, hi = max(0, start - max_lag), min(out.size, start + window + max_lag)
    seg_out = out[lo:hi]
    corr = signal.correlate(seg_out, seg_ref, mode="full", method="fft")
    lags = signal.correlation_lags(seg_out.size, seg_ref.size, mode="full") + lo - start
    keep = np.abs(lags) <= max_lag
    corr, lags = corr[keep], lags[keep]
    best = int(np.argmax(corr))
    if corr.size < 3:
        return int(lags[best])
    peaks = np.r_[False, (corr[1:-1] >= corr[:-2]) & (corr[1:-1] >= corr[2:]), False]
    peaks[best] = True
    near = lags[peaks & (corr >= corr[best] - 0.02 * abs(corr[best]))]
    return int(near[np.argmin(np.abs(near))])


def _shift_to_length(y: np.ndarray, delay: int, n: int) -> np.ndarray:
    if delay > 0:
        y = y[delay:]
    elif delay < 0:
        y = np.concatenate([np.zeros(-delay, dtype=y.dtype), y])
    if y.size >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - y.size, dtype=y.dtype)])


def codec(x, name: str, rng: np.random.Generator,
          bitrate_kbps: float | None = None) -> tuple[np.ndarray, dict]:
    """Encode and decode through ffmpeg with pinned flags; output has the input length.

    opus: libopus, application voip, 16 to 32 kbps drawn unless given. amr_nb:
    libopencore_amrnb at 8 kHz, 12.2 kbps. amr_wb: libvo_amrwbenc at 16 kHz, 23.85 kbps.
    g711: band-limit to 300 to 3400 Hz, then pcm_mulaw at 8 kHz. Rate changes use soxr.
    Codec delay is removed by cross-correlation against the input (see ``estimate_delay``).
    Input beyond full scale is clipped by the encoders that take integer samples.
    """
    a = _mono(x)
    spec = _codec_spec(name)
    if not codec_available(name):
        raise CodecUnavailable(f"ffmpeg has no {spec['encoder']} encoder for {name!r}")
    if spec["kbps"] is not None:
        if bitrate_kbps is not None and float(bitrate_kbps) != spec["kbps"]:
            raise ValueError(f"{name} runs at a fixed {spec['kbps']} kbps")
        kbps = spec["kbps"]
    elif bitrate_kbps is not None:
        kbps = float(bitrate_kbps)
    else:
        kbps = float(rng.integers(OPUS_KBPS[0], OPUS_KBPS[1] + 1))
    params = {"stage": "codec", "codec": name, "encoder": spec["encoder"],
              "bitrate_kbps": kbps, "codec_rate": spec["rate"], "delay_samples": 0}
    if a.size == 0:
        return a.copy(), params
    rate = str(spec["rate"])
    bitrate = [] if name == "g711" else ["-b:a", f"{kbps:g}k"]
    encode = ["-f", "f32le", "-ar", rate, "-ac", "1", "-i", "pipe:0", "-c:a", spec["encoder"],
              *bitrate, *spec["flags"], "-fflags", "+bitexact", "-f", spec["muxer"], "pipe:1"]
    raw_in = ["-ar", rate, "-ac", "1"] if spec["muxer"] == "mulaw" else []
    decode = ["-c:a", spec["decoder"], *raw_in, "-f", spec["muxer"], "-i", "pipe:0",
              "-f", "f32le", "-ac", "1", "pipe:1"]
    src = telephone_band(a) if name == "g711" else a
    src = _resample(src, SR, spec["rate"])
    coded = _ffmpeg(encode, src.astype("<f4").tobytes())
    decoded = np.frombuffer(_ffmpeg(decode, coded), dtype="<f4")
    decoded = _resample(decoded, spec["out_rate"], SR)
    delay = estimate_delay(decoded, a)
    params["delay_samples"] = delay
    return _shift_to_length(decoded, delay, a.size).astype(np.float32), params


def phone_tier(x, tier: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Run one explicit phone tier: amr_wb, amr_nb or g711."""
    if tier not in PHONE_TIERS:
        raise ValueError(f"unknown phone tier {tier!r}; one of {PHONE_TIERS}")
    return codec(x, tier, rng)


# Device chain


def draw_device_stages(rng: np.random.Generator,
                       probs: dict[str, float] | None = None) -> tuple[list[str], list[str]]:
    """Draw which device stages run. Returns ``(kept, drawn)`` in application order.

    One uniform draw per stage, in a fixed order. When more than two are drawn the codec
    stays and the others are dropped at random down to two.
    """
    probs = DEVICE_STAGE_PROBS if probs is None else probs
    drawn = [name for name, p in probs.items() if rng.random() < p]
    if len(drawn) <= MAX_DEVICE_STAGES:
        return list(drawn), drawn
    fixed = [s for s in drawn if s == "codec"]
    others = [s for s in drawn if s != "codec"]
    picks = rng.choice(len(others), size=MAX_DEVICE_STAGES - len(fixed), replace=False)
    chosen = set(fixed) | {others[i] for i in picks.tolist()}
    return [s for s in drawn if s in chosen], drawn


def device_chain(x, rng: np.random.Generator, branch: str) -> tuple[np.ndarray, list[dict]]:
    """Apply at most two device stages: AGC, clipping, Opus, packet loss.

    ``branch`` is checked against ``BRANCH_SNR_DB``; the stage odds are the same for every
    branch. Returns the audio and the params of each stage that ran, in order.
    """
    if branch not in BRANCH_SNR_DB:
        raise ValueError(f"unknown branch {branch!r}; one of {tuple(BRANCH_SNR_DB)}")
    run = {"agc": agc_drift, "clip": clip_fraction, "packet_loss": packet_loss,
           "codec": lambda a, g: codec(a, DEVICE_CODEC, g)}
    kept, _ = draw_device_stages(rng)
    y, stages = _mono(x).copy(), []
    for stage in kept:
        y, params = run[stage](y, rng)
        stages.append(params)
    return y, stages


# Mixing helpers


def mix(stems: dict[str, np.ndarray]) -> np.ndarray:
    """Sum equal-length mono stems."""
    if not stems:
        raise ValueError("no stems to mix")
    lengths = {name: np.asarray(s).shape for name, s in stems.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"stems differ in length: {lengths}")
    total = np.zeros(next(iter(lengths.values())), dtype=np.float64)
    for s in stems.values():
        total += _mono(s)
    return total.astype(np.float32)
