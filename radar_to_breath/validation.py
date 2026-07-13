from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy import signal


@dataclass(frozen=True)
class RateResult:
    time_sec: NDArray[np.float32]
    rate_bpm: NDArray[np.float32]
    valid: NDArray[np.bool_]
    confidence: NDArray[np.float32]
    artifact_fraction: NDArray[np.float32]


def respiratory_bandpower_ratio(
    x: NDArray[np.float64] | NDArray[np.float32], fs: float
) -> float:
    freqs, psd = signal.periodogram(np.asarray(x, dtype=np.float64), fs=fs)
    numerator = np.sum(psd[(freqs >= 0.10) & (freqs <= 0.35)])
    denominator = np.sum(psd[(freqs >= 0.05) & (freqs <= min(1.0, fs / 2 - 1e-6))])
    return float(numerator / max(float(denominator), np.finfo(float).eps))


def estimate_minute_respiratory_rate(
    breath: NDArray[np.float32] | NDArray[np.float64],
    artifact_mask: NDArray[np.bool_],
    fs: float,
    window_sec: float = 60.0,
) -> RateResult:
    x = np.asarray(breath, dtype=np.float64)
    if x.shape != artifact_mask.shape:
        raise ValueError("breath and artifact_mask shapes differ")
    win = max(8, int(round(window_sec * fs)))
    starts = np.arange(0, x.size, win, dtype=np.int64)
    rates: list[float] = []
    valid_list: list[bool] = []
    confidences: list[float] = []
    artifact_fractions: list[float] = []
    times: list[float] = []
    for start in starts:
        stop = min(x.size, int(start + win))
        segment = x[int(start) : stop]
        art = artifact_mask[int(start) : stop]
        times.append((float(start) + float(stop)) / (2.0 * fs))
        artifact_fraction = float(np.mean(art))
        artifact_fractions.append(artifact_fraction)
        if segment.size < max(8, int(round(30 * fs))):
            rates.append(0.0)
            confidences.append(0.0)
            valid_list.append(False)
            continue
        freqs, psd = signal.periodogram(
            segment, fs=fs, window="hann", detrend="constant", scaling="density"
        )
        search = (freqs >= 0.05) & (freqs <= 0.50)
        if not search.any() or np.sum(psd[search]) <= 0:
            rates.append(0.0)
            confidences.append(0.0)
            valid_list.append(False)
            continue
        local_psd = psd[search]
        local_freqs = freqs[search]
        peak_index = int(np.argmax(local_psd))
        peak_frequency = float(local_freqs[peak_index])
        rate = peak_frequency * 60.0
        # Integrate a neighborhood around the peak.  A single-bin ratio is
        # strongly dependent on FFT zero padding and underestimated confidence
        # for slowly varying breathing.  +/-0.033 Hz spans roughly two native
        # 60-second Fourier bins on each side.
        peak_neighborhood = search & (np.abs(freqs - peak_frequency) <= 0.033)
        confidence = float(
            np.sum(psd[peak_neighborhood])
            / max(np.sum(psd[search]), np.finfo(float).eps)
        )
        plausible = 6.0 <= rate <= 21.0
        valid = plausible and artifact_fraction <= 0.40 and confidence >= 0.25
        rates.append(rate)
        confidences.append(confidence)
        valid_list.append(valid)
    return RateResult(
        time_sec=np.asarray(times, dtype=np.float32),
        rate_bpm=np.asarray(rates, dtype=np.float32),
        valid=np.asarray(valid_list, dtype=bool),
        confidence=np.asarray(confidences, dtype=np.float32),
        artifact_fraction=np.asarray(artifact_fractions, dtype=np.float32),
    )


def validate_breath_contract(
    breath: NDArray[np.float32],
    fs: float,
    source_duration_sec: float,
) -> dict[str, float | int | str]:
    if breath.dtype != np.float32:
        raise TypeError(f"breath dtype is {breath.dtype}, expected float32")
    if breath.ndim != 1:
        raise ValueError("breath must be one-dimensional")
    if not np.isfinite(breath).all():
        raise FloatingPointError("breath contains NaN/Inf")
    if breath.size < 2:
        raise ValueError("breath is empty")
    output_duration = breath.size / fs
    duration_error_samples = abs(output_duration - source_duration_sec) * fs
    if duration_error_samples > 1.000001:
        raise ValueError(
            f"duration error is {duration_error_samples:.3f} output samples"
        )
    mean = float(np.mean(breath, dtype=np.float64))
    std = float(np.std(breath, dtype=np.float64))
    if abs(mean) > 5e-5:
        raise ValueError(f"breath mean is not approximately zero: {mean}")
    if not math.isclose(std, 1.0, rel_tol=5e-5, abs_tol=5e-5):
        raise ValueError(f"breath standard deviation is not one: {std}")
    return {
        "dtype": str(breath.dtype),
        "n_samples": int(breath.size),
        "output_duration_sec": float(output_duration),
        "duration_error_samples": float(duration_error_samples),
        "mean": mean,
        "std": std,
        "bandpower_ratio_0p1_0p35": respiratory_bandpower_ratio(breath, fs),
    }
