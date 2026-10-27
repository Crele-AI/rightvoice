"""Known-answer tests for the capture simulation. All randomness comes from rng.stream."""
import math

import numpy as np
import pytest
from scipy import signal

from rightvoice import capture as cap
from rightvoice.rng import stream

SR = cap.SR
SEED = 20261022


def tone(seconds=1.0, freq=1000.0, amp=0.5):
    t = np.arange(round(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speechlike(seconds, index=0):
    """Gaussian noise under a 50 ms step envelope, levelled to -25 dBFS."""
    g = stream(SEED, index, "utterances")
    n = int(seconds * SR)
    env = np.repeat(g.uniform(0.0, 1.0, n // 800 + 1), 800)[:n]
    return cap.level_to(g.standard_normal(n) * env, -25.0)


def corr(a, b, skip=800):
    a = np.asarray(a[skip:-skip], dtype=np.float64)
    b = np.asarray(b[skip:-skip], dtype=np.float64)
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))


# Levels


def test_tone_levelled_to_minus_25_reads_minus_25():
    y = cap.level_to(tone(), -25.0)
    assert y.dtype == np.float32
    assert cap.active_rms_dbfs(y) == pytest.approx(-25.0, abs=0.1)
    assert cap.rms_db(y) == pytest.approx(-25.0, abs=0.1)
    quiet = cap.level_to(tone(amp=0.01), cap.SOURCE_LEVEL_DBFS)
    assert cap.active_rms_dbfs(quiet) == pytest.approx(-25.0, abs=0.1)


def test_silence_reads_minus_inf():
    z = np.zeros(SR, dtype=np.float32)
    assert cap.active_rms_dbfs(z) == -math.inf
    assert cap.active_rms_dbfs(np.zeros(0)) == -math.inf
    assert cap.rms_db(z) == -math.inf
    assert np.array_equal(cap.level_to(z, -25.0), z)


def test_level_uses_the_active_half_only():
    x = np.concatenate([np.zeros(SR // 2, np.float32), tone(0.5)])
    y = cap.level_to(x, -25.0)
    assert cap.active_rms_dbfs(y) == pytest.approx(-25.0, abs=0.1)
    assert cap.rms_db(y[SR // 2:]) == pytest.approx(-25.0, abs=0.1)
    assert not np.any(y[: SR // 2])
    # The whole-signal RMS is 3 dB lower because half of it is silent.
    assert cap.rms_db(y) == pytest.approx(-25.0 - 10 * math.log10(2), abs=0.1)


def test_rms_db_of_a_full_scale_sine():
    assert cap.rms_db(tone(1.0, 1000.0, 1.0)) == pytest.approx(-3.01, abs=0.01)


def test_level_draws_stay_in_range_and_repeat():
    g = stream(SEED, 0, "levels")
    levels = [cap.mixture_level(g) for _ in range(500)]
    partner = [cap.partner_gain_db(g) for _ in range(500)]
    assert -30.0 <= min(levels) and max(levels) <= -15.0
    assert -6.0 <= min(partner) and max(partner) <= 6.0
    bystander = [cap.bystander_gain_db(stream(SEED, i, "bystanders")) for i in range(200)]
    assert -20.0 <= min(bystander) and max(bystander) <= -8.0
    for branch, (lo, hi) in cap.BRANCH_SNR_DB.items():
        draws = [cap.draw_snr_db(branch, stream(SEED, i, "noise")) for i in range(200)]
        assert lo <= min(draws) and max(draws) <= hi
    assert cap.BRANCH_SNR_DB == {"room": (15, 35), "call": (15, 35), "near_clean": (30, 45)}
    assert cap.mixture_level(stream(SEED, 3, "levels")) == cap.mixture_level(
        stream(SEED, 3, "levels"))


def test_peak_guard():
    quiet = tone(amp=0.5)
    assert np.array_equal(cap.peak_guard(quiet), quiet)
    loud = tone(amp=1.7)
    out = cap.peak_guard(loud)
    assert out.dtype == np.float32 and out.shape == loud.shape
    assert np.max(np.abs(out)) == pytest.approx(0.99, abs=1e-6)
    assert np.allclose(out / 0.99, loud / np.max(np.abs(loud)), atol=1e-6)


# Bystander playback chain


def test_peaking_biquad_matches_the_cookbook_gain():
    b, a = cap.peaking_biquad(1000.0, 6.0, 2.0)
    _, h = signal.freqz(b, a, worN=[0.0, 1000.0, SR / 2], fs=SR)
    assert 20 * np.log10(np.abs(h)) == pytest.approx([0.0, 6.0, 0.0], abs=1e-6)


def test_playback_chain_shows_two_resonances_and_band_limits():
    impulse = np.zeros(8192, np.float32)
    impulse[0] = 1.0
    y, params = cap.playback_chain(impulse, stream(SEED, 0, "bystanders"))
    assert y.dtype == np.float32 and y.shape == impulse.shape
    power = np.abs(np.fft.rfft(y.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(y.size, 1 / SR)
    level = 10 * np.log10(power + 1e-30)

    def db_at(f):
        return level[np.argmin(np.abs(freqs - f))]

    res = [r["freq_hz"] for r in params["resonances"]]
    assert len(res) == 2
    for i, f in enumerate(res):
        # 1.5 octaves up or down inside the 150 Hz to 6 kHz passband, away from the other peak.
        options = [c for c in (f * 2**1.5, f / 2**1.5) if 150 < c < 6000]
        ref = max(options, key=lambda c: abs(math.log2(c / res[1 - i])))
        assert db_at(f) - db_at(ref) >= 2.0, (f, ref)
    passband = power[(freqs >= 300) & (freqs <= 5000)].mean()
    low = power[freqs < 100].mean()
    # At 16 kHz the spectrum ends at 8 kHz, so the top band checked is 7 to 8 kHz.
    high = power[freqs >= 7000].mean()
    assert 10 * np.log10(low / passband) <= -12.0
    assert 10 * np.log10(high / passband) <= -12.0


def test_playback_params_in_range_and_repeatable():
    x = speechlike(1.0)
    for i in range(20):
        y1, p1 = cap.playback_chain(x, stream(SEED, i, "bystanders"))
        y2, p2 = cap.playback_chain(x, stream(SEED, i, "bystanders"))
        assert np.array_equal(y1, y2) and p1 == p2
        for r in p1["resonances"]:
            assert 400 <= r["freq_hz"] <= 4000
            assert 3 <= r["gain_db"] <= 9
            assert 1.5 <= r["q"] <= 4
        assert -20 <= p1["threshold_db_rel_peak"] <= -8
        assert 2 <= p1["ratio"] <= 6
        assert (p1["attack_ms"], p1["release_ms"]) == (5.0, 50.0)


def test_compressor_reduces_only_above_threshold():
    x = tone(1.0, 1000.0, 0.5)
    assert np.array_equal(cap.compress(x, 0.0, 4.0), x)
    # The envelope of a 0.5 sine sits between -10 and -6 dBFS, so 30 to 34 dB over a
    # -40 dBFS threshold; at ratio 4 that is a 22.5 to 25.5 dB cut.
    y = cap.compress(x, -40.0, 4.0)
    cut = cap.rms_db(x[SR // 2:]) - cap.rms_db(y[SR // 2:])
    assert 22.0 <= cut <= 26.0


def test_place_bystander_keeps_the_clip_inside():
    clip = np.zeros(3 * SR, np.float32)
    canvas = 30 * SR
    starts = [cap.place_bystander(canvas, clip, stream(SEED, i, "bystanders")) for i in range(200)]
    assert min(starts) >= 0 and max(starts) + clip.size <= canvas
    assert cap.place_bystander(clip.size, clip, stream(SEED, 0, "bystanders")) == 0
    with pytest.raises(ValueError):
        cap.place_bystander(SR, clip, stream(SEED, 0, "bystanders"))


# Rooms


def test_reverberate_with_a_unit_impulse_returns_the_input():
    x = speechlike(1.0)
    rir = np.zeros(400, np.float32)
    rir[0] = 1.0
    assert cap.direct_index_of(rir) == 0
    y = cap.reverberate(x, rir, 0)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert np.allclose(y, x, atol=1e-6)


def test_reverberate_with_a_delayed_impulse_and_its_index_returns_the_input():
    x = speechlike(1.0)
    rir = np.zeros(400, np.float32)
    rir[57] = 1.0
    assert cap.direct_index_of(rir) == 57
    assert np.allclose(cap.reverberate(x, rir, 57), x, atol=1e-6)


def test_direct_index_is_the_first_sample_at_half_the_peak():
    rir = np.zeros(200)
    rir[[40, 55, 57]] = [0.3, -0.6, 1.0]
    assert cap.direct_index_of(rir) == 55
    unit = cap.normalise_rir(rir)
    assert float(np.sum(unit.astype(np.float64) ** 2)) == pytest.approx(1.0, abs=1e-6)


# Noise and music at SNR


def test_add_at_snr_white_noise_measures_20_db():
    speech = cap.level_to(tone(2.0), -25.0)
    white = stream(SEED, 1, "noise").standard_normal(3 * SR).astype(np.float32)
    mixture, scaled = cap.add_at_snr(speech, white, 20.0, stream(SEED, 2, "noise"))
    assert mixture.shape == scaled.shape == speech.shape
    assert cap.active_rms_dbfs(speech) - cap.rms_db(scaled) == pytest.approx(20.0, abs=0.5)
    assert np.allclose(mixture, speech + scaled, atol=1e-6)


def test_add_at_snr_tiles_short_noise():
    speech = speechlike(2.0)
    q = SR // 4
    short = stream(SEED, 3, "noise").standard_normal(q).astype(np.float32)
    _, scaled = cap.add_at_snr(speech, short, 15.0, stream(SEED, 4, "noise"))
    assert scaled.shape == speech.shape
    assert cap.active_rms_dbfs(speech) - cap.rms_db(scaled) == pytest.approx(15.0, abs=0.5)
    assert np.array_equal(scaled[:q], scaled[q:2 * q])
    with pytest.raises(ValueError):
        cap.add_at_snr(np.zeros(SR, np.float32), short, 15.0, stream(SEED, 4, "noise"))


# Device chain stages


def test_clip_fraction_clips_between_0_1_and_1_percent():
    x = speechlike(5.0)
    for i in range(20):
        y, p = cap.clip_fraction(x, stream(SEED, i, "chain"))
        changed = float(np.mean(y != x))
        assert 0.001 <= changed <= 0.01
        assert changed == pytest.approx(p["clipped_fraction"])
        assert np.max(np.abs(y)) == pytest.approx(p["threshold"])
        assert 0.001 <= p["target_fraction"] <= 0.01


def test_packet_loss_zeros_at_most_2_percent_and_keeps_the_rest():
    x = speechlike(10.0)
    fade = SR * 2 // 1000
    for i in range(10):
        y, p = cap.packet_loss(x, stream(SEED, i, "chain"))
        lost = float(np.mean((y == 0) & (x != 0)))
        assert 0 < lost <= 0.02
        assert lost == pytest.approx(p["lost_fraction"])
        touched = np.zeros(x.size, bool)
        for start, length in p["gaps"]:
            assert SR * 20 // 1000 <= length <= SR * 100 // 1000
            assert not np.any(y[start:start + length])
            for edge in (slice(start - fade, start), slice(start + length, start + length + fade)):
                ratio = y[edge] / x[edge]
                assert np.all((ratio > 0) & (ratio < 1))
            touched[start - fade:start + length + fade] = True
        assert np.array_equal(y[~touched], x[~touched])
    short = speechlike(0.5)
    y, p = cap.packet_loss(short, stream(SEED, 0, "chain"))
    assert p["gaps"] == [] and np.array_equal(y, short)


def test_agc_drift_gain_stays_within_3_2_db_and_moves_smoothly():
    x = tone(20.0, 440.0, 0.3)
    mask = np.abs(x) > 1e-3
    for i in range(10):
        y, p = cap.agc_drift(x, stream(SEED, i, "chain"))
        assert y.shape == x.shape and 1.0 <= p["period_s"] <= 4.0
        gain_db = 20 * np.log10(np.abs(y[mask]) / np.abs(x[mask]))
        assert np.max(np.abs(gain_db)) <= 3.2
        assert np.max(gain_db) - np.min(gain_db) > 0.5
        assert np.max(np.abs(np.diff(gain_db))) < 0.01


# Codecs


@pytest.mark.parametrize("name", ["opus", "amr_nb", "g711"])
def test_codec_keeps_length_and_a_tone(name):
    if not cap.codec_available(name):
        pytest.skip(f"ffmpeg has no encoder for {name}")
    x = tone(2.0, 1000.0, 0.3)
    y, p = cap.codec(x, name, stream(SEED, 0, "chain"))
    assert y.dtype == np.float32 and y.shape == x.shape
    assert p["codec"] == name
    assert corr(x, y) > 0.8
    odd = tone(12345 / SR, 1000.0, 0.3)
    assert cap.codec(odd, name, stream(SEED, 1, "chain"))[0].shape == odd.shape


@pytest.mark.parametrize("name", ["opus", "amr_nb", "g711"])
def test_codec_output_is_aligned(name):
    if not cap.codec_available(name):
        pytest.skip(f"ffmpeg has no encoder for {name}")
    x = speechlike(2.0)
    y, _ = cap.codec(x, name, stream(SEED, 0, "chain"))
    assert abs(cap.estimate_delay(y, x)) <= 1


def test_opus_bitrate_is_drawn_in_16_to_32_kbps():
    if not cap.codec_available("opus"):
        pytest.skip("ffmpeg has no libopus encoder")
    x = tone(0.2)
    rates = {cap.codec(x, "opus", stream(SEED, i, "chain"))[1]["bitrate_kbps"] for i in range(8)}
    assert all(16 <= r <= 32 for r in rates) and len(rates) > 1
    assert cap.codec(x, "opus", stream(SEED, 0, "chain"), bitrate_kbps=24)[1]["bitrate_kbps"] == 24


def test_amr_wb_runs_or_raises_codec_unavailable():
    x = tone(1.0, 1000.0, 0.3)
    if not cap.codec_available("amr_wb"):
        with pytest.raises(cap.CodecUnavailable):
            cap.codec(x, "amr_wb", stream(SEED, 0, "chain"))
        with pytest.raises(cap.CodecUnavailable):
            cap.phone_tier(x, "amr_wb", stream(SEED, 0, "chain"))
        pytest.skip("ffmpeg has no libvo_amrwbenc encoder")
    y, _ = cap.codec(x, "amr_wb", stream(SEED, 0, "chain"))
    assert y.shape == x.shape and corr(x, y) > 0.8


def test_codec_names_and_bitrates_are_checked():
    x = tone(0.2)
    with pytest.raises(ValueError):
        cap.codec(x, "mp3", stream(SEED, 0, "chain"))
    with pytest.raises(ValueError):
        cap.codec_available("mp3")
    with pytest.raises(ValueError):
        cap.phone_tier(x, "opus", stream(SEED, 0, "chain"))
    if cap.codec_available("amr_nb"):
        with pytest.raises(ValueError):
            cap.codec(x, "amr_nb", stream(SEED, 0, "chain"), bitrate_kbps=7.4)


def test_ffmpeg_info_lists_only_known_encoders():
    info = cap.ffmpeg_info()
    assert set(info["encoders"]) <= {"libopus", "libopencore_amrnb", "libvo_amrwbenc", "pcm_mulaw"}
    if info["path"] is None:
        pytest.skip("ffmpeg not found")
    assert info["version"].startswith("ffmpeg version")


def test_estimate_delay_recovers_known_shifts():
    x = speechlike(2.0)
    late = np.concatenate([np.zeros(37, np.float32), x])[:x.size]
    assert cap.estimate_delay(late, x) == 37
    assert cap.estimate_delay(x[20:], x) == -20
    lead = np.concatenate([np.zeros(2 * SR, np.float32), x])
    shifted = np.concatenate([np.zeros(55, np.float32), lead])[:lead.size]
    assert cap.estimate_delay(shifted, lead) == 55


# Device chain


def test_device_chain_never_runs_more_than_two_stages():
    if not cap.codec_available(cap.DEVICE_CODEC):
        pytest.skip("ffmpeg has no libopus encoder")
    x = speechlike(0.5)
    for i in range(200):
        kept, drawn = cap.draw_device_stages(stream(SEED, i, "chain"))
        y, stages = cap.device_chain(x, stream(SEED, i, "chain"), "room")
        names = [s["stage"] for s in stages]
        assert names == kept
        assert len(names) <= 2
        if "codec" in drawn:
            assert "codec" in names
        assert y.dtype == np.float32 and y.shape == x.shape
    with pytest.raises(ValueError):
        cap.device_chain(x, stream(SEED, 0, "chain"), "nope")


def test_device_stage_drop_keeps_the_codec():
    always = {name: 1.0 for name in cap.DEVICE_STAGE_PROBS}
    seen = set()
    for i in range(50):
        kept, drawn = cap.draw_device_stages(stream(SEED, i, "chain"), always)
        assert drawn == list(cap.DEVICE_STAGE_PROBS)
        assert len(kept) == 2 and "codec" in kept
        seen.add(tuple(kept))
    assert len(seen) == 3
    kept, drawn = cap.draw_device_stages(stream(SEED, 0, "chain"), dict(always, codec=0.0))
    assert len(drawn) == 3 and len(kept) == 2 and "codec" not in kept


def test_device_stage_odds():
    counts = dict.fromkeys(cap.DEVICE_STAGE_PROBS, 0)
    n = 5000
    for i in range(n):
        for name in cap.draw_device_stages(stream(SEED, i, "chain"))[1]:
            counts[name] += 1
    for name, p in cap.DEVICE_STAGE_PROBS.items():
        assert abs(counts[name] / n - p) < 4 * math.sqrt(p * (1 - p) / n), name


# Mixing


def test_mix_sums_equal_length_stems():
    a, b = tone(1.0, 300.0, 0.2), tone(1.0, 700.0, 0.1)
    out = cap.mix({"target": a, "partner": b})
    assert out.dtype == np.float32
    assert np.allclose(out, a + b, atol=1e-7)
    with pytest.raises(ValueError):
        cap.mix({"a": a, "b": a[:-1]})
    with pytest.raises(ValueError):
        cap.mix({})
