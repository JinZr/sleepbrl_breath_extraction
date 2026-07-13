from __future__ import annotations

import numpy as np
from scipy import signal

from radar_to_breath.config import ProcessingConfig
from radar_to_breath.processing import (
    fuse_pair_candidates,
    process_iq_pair,
    repair_fused_waveform,
)
from radar_to_breath.validation import estimate_minute_respiratory_rate


def _synthetic_pair(
    fs: float,
    duration_sec: float,
    frequency_hz: float,
    seed: int,
    noise: float,
    gain_i: float = 1.0,
    gain_q: float = 1.0,
):
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_sec * fs))) / fs
    displacement_phase = 1.2 * np.sin(2 * np.pi * frequency_hz * t)
    slow_drift = 0.15 * np.sin(2 * np.pi * 0.005 * t)
    carrier_phase = 0.8 + displacement_phase
    i = 2.0 + 0.4 * slow_drift + gain_i * np.cos(carrier_phase)
    q = -1.5 - 0.3 * slow_drift + gain_q * np.sin(carrier_phase)
    i += noise * rng.standard_normal(t.size)
    q += noise * rng.standard_normal(t.size)
    # A short gross-motion disturbance must be detected without changing the known rate.
    motion = (t >= 180) & (t < 184)
    i[motion] += 4.0 * rng.standard_normal(motion.sum())
    q[motion] += 4.0 * rng.standard_normal(motion.sum())
    return i.astype(float), q.astype(float)


def test_recovers_known_synthetic_breathing_frequency() -> None:
    fs = 50.0
    target_frequency = 0.25  # 15 breaths/min
    duration = 600.0
    config = ProcessingConfig(baseline_window_sec=180.0)
    results = []
    for pair in range(8):
        i, q = _synthetic_pair(
            fs,
            duration,
            target_frequency,
            seed=pair,
            noise=0.015 + 0.01 * pair,
            gain_i=1.0,
            gain_q=0.9 + 0.02 * pair,
        )
        if pair == 7:
            i[:] = 0.0  # one deliberately broken quadrature channel
        results.append(
            process_iq_pair(
                i,
                q,
                fs,
                pair,
                physical_min_i=-20,
                physical_max_i=20,
                lsb_i=1e-6,
                physical_min_q=-20,
                physical_max_q=20,
                lsb_q=1e-6,
                config=config,
            )
        )
    fusion = fuse_pair_candidates(results, config)
    assert fusion.global_quality_scores[7] == 0.0
    assert fusion.global_fusion_weights[7] == 0.0
    sos = signal.bessel(
        3, [0.1, 0.35], btype="bandpass", fs=4.0, output="sos", norm="phase"
    )
    repaired, artifact_mask, _ = repair_fused_waveform(
        fusion.waveform, fusion.artifact_mask, 4.0, config
    )
    x = signal.sosfiltfilt(sos, repaired)
    x, artifact_mask, _ = repair_fused_waveform(
        x, artifact_mask, 4.0, config, expand_initial=False
    )
    x = signal.sosfiltfilt(sos, x)
    x = ((x - np.mean(x)) / np.std(x)).astype(np.float32)
    rates = estimate_minute_respiratory_rate(x, artifact_mask, 4.0)
    valid = rates.rate_bpm[rates.valid]
    assert valid.size >= 6
    assert abs(float(np.median(valid)) - 15.0) < 0.6


def test_post_fusion_transient_is_marked_and_repaired() -> None:
    fs = 4.0
    config = ProcessingConfig()
    t = np.arange(int(300 * fs)) / fs
    clean = np.sin(2 * np.pi * 0.2 * t)
    corrupted = clean.copy()
    corrupted[int(120 * fs) : int(121 * fs)] += 25.0
    repaired, mask, detected = repair_fused_waveform(
        corrupted, np.zeros(corrupted.size, dtype=bool), fs, config
    )
    assert detected[int(120 * fs)]
    assert mask[int(120 * fs)]
    assert np.max(np.abs(repaired[mask])) <= 1.1
    assert np.max(np.abs(corrupted[mask])) > 20.0
