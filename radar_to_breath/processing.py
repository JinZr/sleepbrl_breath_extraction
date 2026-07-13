from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

from .config import ProcessingConfig


_EPS = np.finfo(np.float64).eps


@dataclass
class PairResult:
    pair_index: int
    breath: NDArray[np.float64]
    broad: NDArray[np.float64]
    phase: NDArray[np.float64]
    radius: NDArray[np.float64]
    artifact_mask: NDArray[np.bool_]
    jump_mask: NDArray[np.bool_]
    motion_mask: NDArray[np.bool_]
    saturation_mask: NDArray[np.bool_]
    iq_balance: float
    artifact_fraction: float
    jump_fraction: float
    motion_fraction: float
    saturation_fraction: float
    center_i_range: float
    center_q_range: float


@dataclass
class FusionResult:
    waveform: NDArray[np.float64]
    artifact_mask: NDArray[np.bool_]
    fusion_weights: NDArray[np.float32]
    fusion_signs: NDArray[np.int8]
    window_scores: NDArray[np.float32]
    window_start_sec: NDArray[np.float32]
    global_quality_scores: NDArray[np.float32]
    global_fusion_weights: NDArray[np.float32]
    selected_pairs: NDArray[np.int16]


def _robust_scale(x: NDArray[np.float64]) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= _EPS:
        scale = float(np.std(x))
    return max(scale, _EPS)


def _safe_sosfiltfilt(
    sos: NDArray[np.float64], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    if x.size < 16:
        raise ValueError("signal is too short for zero-phase filtering")
    default_pad = 3 * (2 * len(sos) + 1)
    padlen = min(default_pad, x.size - 2)
    return np.asarray(signal.sosfiltfilt(sos, x, padlen=padlen), dtype=np.float64)


def _fit_iq_circle_center(
    i_block: NDArray[np.float64], q_block: NDArray[np.float64]
) -> tuple[float, float] | None:
    """Estimate an axis-scaled algebraic circle center for one long I/Q block.

    Scaling each axis before the fit makes the estimate tolerant of moderate I/Q
    gain imbalance. Degenerate near-line trajectories are rejected because their
    arctangent phase is intrinsically unstable.
    """

    step = max(1, i_block.size // 5000)
    x = np.asarray(i_block[::step], dtype=np.float64)
    y = np.asarray(q_block[::step], dtype=np.float64)
    if x.size < 32:
        return None
    mx = float(np.median(x))
    my = float(np.median(y))
    sx = _robust_scale(x - mx)
    sy = _robust_scale(y - my)
    keep = (np.abs(x - mx) <= 8.0 * sx) & (np.abs(y - my) <= 8.0 * sy)
    x = x[keep]
    y = y[keep]
    if x.size < 32:
        return None
    mx = float(np.median(x))
    my = float(np.median(y))
    sx = _robust_scale(x - mx)
    sy = _robust_scale(y - my)
    xn = (x - mx) / sx
    yn = (y - my) / sy
    design = np.column_stack((2.0 * xn, 2.0 * yn, np.ones(xn.size)))
    target = xn * xn + yn * yn
    try:
        solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        condition = float(np.linalg.cond(design))
    except np.linalg.LinAlgError:
        return None
    cx = float(solution[0])
    cy = float(solution[1])
    radii = np.hypot(xn - cx, yn - cy)
    radius_med = float(np.median(radii))
    radius_cv = _robust_scale(radii) / max(radius_med, _EPS)
    covariance = np.cov(np.column_stack((xn, yn)), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    rank_ratio = float(eigenvalues[0] / max(eigenvalues[-1], _EPS))
    if (
        not np.isfinite([cx, cy, radius_med, radius_cv, condition, rank_ratio]).all()
        or radius_med <= _EPS
        or radius_cv > 0.80
        or condition > 150.0
        or rank_ratio < 0.005
        or math.hypot(cx, cy) > 20.0
    ):
        return None
    return mx + cx * sx, my + cy * sy


def _iq_center_tracks(
    i_signal: NDArray[np.float64],
    q_signal: NDArray[np.float64],
    fs: float,
    window_sec: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    n = i_signal.size
    block = max(1, int(round(window_sec * fs)))
    starts = np.arange(0, n, block, dtype=np.int64)
    sample_centers: list[float] = []
    center_i: list[float] = []
    center_q: list[float] = []
    valid: list[bool] = []
    for start in starts:
        stop = min(n, int(start + block))
        sample_centers.append((float(start) + float(stop - 1)) / 2.0)
        fitted = _fit_iq_circle_center(i_signal[int(start) : stop], q_signal[int(start) : stop])
        if fitted is None:
            center_i.append(float(np.median(i_signal[int(start) : stop])))
            center_q.append(float(np.median(q_signal[int(start) : stop])))
            valid.append(False)
        else:
            center_i.append(float(fitted[0]))
            center_q.append(float(fitted[1]))
            valid.append(True)

    ci = np.asarray(center_i, dtype=np.float64)
    cq = np.asarray(center_q, dtype=np.float64)
    centers = np.asarray(sample_centers, dtype=np.float64)
    valid_arr = np.asarray(valid, dtype=bool)
    # Interpolate across rejected blocks using trustworthy neighboring fits.
    if valid_arr.any():
        good = np.flatnonzero(valid_arr)
        ci = np.interp(np.arange(ci.size), good, ci[good])
        cq = np.interp(np.arange(cq.size), good, cq[good])
    if ci.size >= 3:
        ci = ndimage.median_filter(ci, size=3, mode="nearest")
        cq = ndimage.median_filter(cq, size=3, mode="nearest")
    if ci.size == 1:
        track_i = np.full(n, ci[0], dtype=np.float64)
        track_q = np.full(n, cq[0], dtype=np.float64)
    else:
        indices = np.arange(n, dtype=np.float64)
        track_i = np.interp(indices, centers, ci, left=ci[0], right=ci[-1])
        track_q = np.interp(indices, centers, cq, left=cq[0], right=cq[-1])
    return track_i, track_q, float(np.ptp(ci)), float(np.ptp(cq))


def _expand_mask(mask: NDArray[np.bool_], samples: int) -> NDArray[np.bool_]:
    if samples <= 0 or not mask.any():
        return mask.copy()
    size = 2 * samples + 1
    return ndimage.maximum_filter1d(mask.astype(np.uint8), size=size, mode="nearest").astype(bool)


def _consecutive_equal_mask(x: NDArray[np.float64], fs: float) -> NDArray[np.bool_]:
    equal = np.zeros(x.size, dtype=bool)
    equal[1:] = np.diff(x) == 0
    run = max(2, int(round(0.5 * fs)))
    local = ndimage.uniform_filter1d(equal.astype(float), size=run, mode="nearest")
    return local > 0.95


def _stitch_and_interpolate_phase(
    phase: NDArray[np.float64], artifact_mask: NDArray[np.bool_], fs: float
) -> NDArray[np.float64]:
    valid = np.isfinite(phase) & ~artifact_mask
    if valid.sum() < max(4, int(round(2 * fs))):
        finite = np.isfinite(phase)
        if finite.sum() < 2:
            return np.zeros_like(phase)
        idx = np.flatnonzero(finite)
        return np.interp(np.arange(phase.size), idx, phase[idx]).astype(np.float64)

    work = phase.copy()
    padded = np.pad(valid.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    edge = max(1, int(round(2.0 * fs)))
    previous_tail: float | None = None
    for start, stop in zip(starts, stops, strict=True):
        if previous_tail is not None:
            head_stop = min(stop, start + edge)
            head = float(np.median(work[start:head_stop]))
            work[start:stop] -= head - previous_tail
        tail_start = max(start, stop - edge)
        previous_tail = float(np.median(work[tail_start:stop]))

    idx = np.flatnonzero(valid)
    repaired = np.interp(np.arange(phase.size), idx, work[idx])
    return np.asarray(repaired, dtype=np.float64)


def _resample_ratio(source_fs: float, target_fs: float) -> tuple[int, int]:
    ratio = Fraction(target_fs / source_fs).limit_denominator(10_000)
    return ratio.numerator, ratio.denominator


def _resample_exact(
    x: NDArray[np.float64], source_fs: float, target_fs: float, target_length: int
) -> NDArray[np.float64]:
    up, down = _resample_ratio(source_fs, target_fs)
    y = np.asarray(signal.resample_poly(x, up, down), dtype=np.float64)
    if y.size > target_length:
        return y[:target_length]
    if y.size < target_length:
        if y.size == 0:
            return np.zeros(target_length, dtype=np.float64)
        return np.pad(y, (0, target_length - y.size), mode="edge")
    return y


def _downsample_mask(
    mask: NDArray[np.bool_], source_fs: float, target_fs: float, target_length: int
) -> NDArray[np.bool_]:
    dilation = max(1, int(math.ceil(source_fs / target_fs)))
    conservative = ndimage.maximum_filter1d(
        mask.astype(np.uint8), size=dilation, mode="nearest"
    ).astype(bool)
    times = np.arange(target_length, dtype=np.float64) / target_fs
    indices = np.rint(times * source_fs).astype(np.int64)
    indices = np.clip(indices, 0, mask.size - 1)
    return conservative[indices]


def _select_direct_iq_component(
    i_norm: NDArray[np.float64],
    q_norm: NDArray[np.float64],
    fs: float,
    config: ProcessingConfig,
) -> NDArray[np.float64]:
    components = np.stack((i_norm, q_norm))
    resp_sos = signal.bessel(
        3,
        [config.respiration_low_hz, config.respiration_high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
        norm="phase",
    )
    broad_sos = signal.butter(
        4,
        [config.broad_low_hz, config.broad_high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    respiratory = np.stack([_safe_sosfiltfilt(resp_sos, x) for x in components])
    broad = np.stack([_safe_sosfiltfilt(broad_sos, x) for x in components])

    n = i_norm.size
    win = max(8, int(round(config.fusion_window_sec * fs)))
    hop = max(1, int(round(config.fusion_hop_sec * fs)))
    starts = list(range(0, max(1, n - 1), hop))
    if not starts or starts[-1] + win < n:
        starts.append(max(0, n - win))
    starts = sorted(set(starts))

    accumulator = np.zeros(n, dtype=np.float64)
    respiratory_accumulator = np.zeros(n, dtype=np.float64)
    taper_accumulator = np.zeros(n, dtype=np.float64)
    for start in starts:
        stop = min(n, start + win)
        length = stop - start
        existing = taper_accumulator[start:stop] > 0
        respiratory_window = respiratory[:, start:stop]
        scores, _, _ = _window_feature_scores(
            respiratory_window,
            broad[:, start:stop],
            np.ones_like(respiratory_window),
            np.zeros_like(respiratory_window, dtype=bool),
            np.ones(2, dtype=np.float64),
            fs,
        )
        signs = np.ones(2, dtype=np.int8)
        if existing.sum() >= max(8, int(round(10 * fs))):
            previous = (
                respiratory_accumulator[start:stop][existing]
                / taper_accumulator[start:stop][existing]
            )
            for component in range(2):
                current = respiratory_window[component, existing]
                if np.std(previous) > _EPS and np.std(current) > _EPS:
                    overlap_corr = float(np.corrcoef(previous, current)[0, 1])
                    if np.isfinite(overlap_corr):
                        scores[component] *= 0.8 + 0.2 * abs(overlap_corr)
                        if overlap_corr < 0:
                            signs[component] = -1
        selected = int(np.argmax(scores))
        sign_value = int(signs[selected])
        taper = np.maximum(signal.windows.tukey(length, alpha=0.5), 0.05)
        accumulator[start:stop] += taper * sign_value * components[selected, start:stop]
        respiratory_accumulator[start:stop] += taper * sign_value * respiratory[selected, start:stop]
        taper_accumulator[start:stop] += taper

    if np.any(taper_accumulator <= 0):
        raise RuntimeError("direct I/Q overlap-add left uncovered samples")
    return accumulator / taper_accumulator


def process_iq_pair(
    i_signal: NDArray[np.float64],
    q_signal: NDArray[np.float64],
    fs: float,
    pair_index: int,
    physical_min_i: float,
    physical_max_i: float,
    lsb_i: float,
    physical_min_q: float,
    physical_max_q: float,
    lsb_q: float,
    config: ProcessingConfig,
) -> PairResult:
    """Convert one I/Q pair into a direct respiratory-band candidate."""

    if i_signal.shape != q_signal.shape or i_signal.ndim != 1:
        raise ValueError("I and Q must be same-length one-dimensional arrays")
    if i_signal.size < int(round(30 * fs)):
        raise ValueError("record is shorter than 30 seconds")

    nonfinite = ~np.isfinite(i_signal) | ~np.isfinite(q_signal)
    if nonfinite.any():
        finite = ~nonfinite
        if finite.sum() < 2:
            raise ValueError(f"I/Q pair {pair_index + 1} contains no usable values")
        idx = np.flatnonzero(finite)
        i_signal = np.interp(np.arange(i_signal.size), idx, i_signal[idx])
        q_signal = np.interp(np.arange(q_signal.size), idx, q_signal[idx])

    iq_lowpass = signal.butter(
        4, config.iq_lowpass_hz, btype="lowpass", fs=fs, output="sos"
    )
    i_smooth = _safe_sosfiltfilt(iq_lowpass, i_signal)
    q_smooth = _safe_sosfiltfilt(iq_lowpass, q_signal)

    baseline_i, baseline_q, center_i_range, center_q_range = _iq_center_tracks(
        i_smooth, q_smooth, fs, config.baseline_window_sec
    )
    i_centered = i_smooth - baseline_i
    q_centered = q_smooth - baseline_q
    scale_i = _robust_scale(i_centered)
    scale_q = _robust_scale(q_centered)
    balance = float(min(scale_i, scale_q) / max(scale_i, scale_q))

    i_norm = i_centered / scale_i
    q_norm = q_centered / scale_q
    radius = np.hypot(i_norm, q_norm)
    phase = _select_direct_iq_component(i_norm, q_norm, fs, config)

    dphase = np.diff(phase, prepend=phase[0])
    dphase_med = float(np.median(dphase))
    dphase_mad = _robust_scale(dphase)
    jump_threshold = max(
        config.jump_abs_threshold_rad,
        abs(dphase_med) + config.jump_mad_multiplier * dphase_mad,
    )
    jump = np.abs(dphase - dphase_med) > jump_threshold

    di = np.diff(i_norm, prepend=i_norm[0])
    dq = np.diff(q_norm, prepend=q_norm[0])
    motion_energy = np.hypot(di, dq)
    smooth_samples = max(1, int(round(fs)))
    motion_smooth = np.sqrt(
        ndimage.uniform_filter1d(
            motion_energy * motion_energy, size=smooth_samples, mode="nearest"
        )
    )
    motion_med = float(np.median(motion_smooth))
    motion_mad = _robust_scale(motion_smooth)
    motion = motion_smooth > (
        motion_med + config.motion_mad_multiplier * motion_mad
    )

    saturation = (
        (i_signal <= physical_min_i + 1.5 * lsb_i)
        | (i_signal >= physical_max_i - 1.5 * lsb_i)
        | (q_signal <= physical_min_q + 1.5 * lsb_q)
        | (q_signal >= physical_max_q - 1.5 * lsb_q)
        | _consecutive_equal_mask(i_signal, fs)
        | _consecutive_equal_mask(q_signal, fs)
    )

    radius_med = max(float(np.median(radius)), _EPS)
    low_radius = radius < config.low_radius_fraction * radius_med
    expand = int(round(config.artifact_expand_sec * fs))
    jump_expanded = _expand_mask(jump, expand)
    motion_expanded = _expand_mask(motion, expand)
    saturation_expanded = _expand_mask(saturation, max(1, int(round(0.5 * fs))))
    low_radius_expanded = _expand_mask(low_radius, max(1, int(round(0.2 * fs))))
    artifact = (
        nonfinite
        | jump_expanded
        | motion_expanded
        | saturation_expanded
        | low_radius_expanded
    )

    repaired_phase = _stitch_and_interpolate_phase(phase, artifact, fs)
    resp_sos = signal.bessel(
        3,
        [config.respiration_low_hz, config.respiration_high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
        norm="phase",
    )
    broad_sos = signal.butter(
        4,
        [config.broad_low_hz, config.broad_high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    breath_50 = _safe_sosfiltfilt(resp_sos, repaired_phase)
    broad_50 = _safe_sosfiltfilt(broad_sos, repaired_phase)

    target_length = int(round(i_signal.size / fs * config.target_fs))
    breath = _resample_exact(breath_50, fs, config.target_fs, target_length)
    broad = _resample_exact(broad_50, fs, config.target_fs, target_length)
    phase_out = _resample_exact(repaired_phase, fs, config.target_fs, target_length)
    radius_out = _resample_exact(radius, fs, config.target_fs, target_length)
    artifact_out = _downsample_mask(artifact, fs, config.target_fs, target_length)
    jump_out = _downsample_mask(jump_expanded, fs, config.target_fs, target_length)
    motion_out = _downsample_mask(motion_expanded, fs, config.target_fs, target_length)
    saturation_out = _downsample_mask(
        saturation_expanded, fs, config.target_fs, target_length
    )

    for name, array in (
        ("breath", breath),
        ("broad", broad),
        ("phase", phase_out),
        ("radius", radius_out),
    ):
        if not np.isfinite(array).all():
            raise FloatingPointError(f"pair {pair_index + 1} {name} contains NaN/Inf")

    return PairResult(
        pair_index=pair_index,
        breath=breath,
        broad=broad,
        phase=phase_out,
        radius=radius_out,
        artifact_mask=artifact_out,
        jump_mask=jump_out,
        motion_mask=motion_out,
        saturation_mask=saturation_out,
        iq_balance=balance,
        artifact_fraction=float(np.mean(artifact_out)),
        jump_fraction=float(np.mean(jump_out)),
        motion_fraction=float(np.mean(motion_out)),
        saturation_fraction=float(np.mean(saturation_out)),
        center_i_range=center_i_range,
        center_q_range=center_q_range,
    )



def repair_fused_waveform(
    waveform: NDArray[np.float64],
    initial_artifact_mask: NDArray[np.bool_],
    fs: float,
    config: ProcessingConfig,
    *,
    expand_initial: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Detect and interpolate gross post-fusion transients before final filtering.

    Each pair candidate has already been respiratory-band filtered, yet a phase
    stitch or a window-to-window carrier change can still create an isolated
    high-amplitude transient.  Feeding such a transient directly to the final
    zero-phase band-pass can ring for many seconds.  This function marks robust
    amplitude/envelope/derivative outliers, combines them with the fusion quality
    mask, expands their boundaries, and linearly interpolates through them.  The
    returned mask is retained in the NPZ so interpolation is never hidden from
    downstream users.
    """

    x = np.asarray(waveform, dtype=np.float64)
    initial = np.asarray(initial_artifact_mask, dtype=bool)
    if x.ndim != 1 or initial.shape != x.shape:
        raise ValueError("waveform and initial_artifact_mask must be same-length 1-D arrays")
    if fs <= 0:
        raise ValueError("fs must be positive")

    finite = np.isfinite(x)
    if finite.sum() < max(2, int(round(10.0 * fs))):
        raise FloatingPointError("fused waveform has fewer than ten seconds of finite data")
    work = x.copy()
    if not finite.all():
        idx = np.flatnonzero(finite)
        work = np.interp(np.arange(work.size), idx, work[idx]).astype(np.float64)

    center = float(np.median(work[finite]))
    global_scale = _robust_scale(work[finite] - center)
    absolute_outlier = (
        np.abs(work - center)
        > config.post_fusion_absolute_z_threshold * global_scale
    )

    amplitude_window = max(3, int(round(3.0 * fs)))
    envelope_power = ndimage.uniform_filter1d(
        (work - center) * (work - center),
        size=amplitude_window,
        mode="nearest",
    )
    envelope = np.sqrt(np.maximum(envelope_power, 0.0))
    envelope_med = float(np.median(envelope[finite]))
    envelope_scale = _robust_scale(envelope[finite] - envelope_med)
    envelope_outlier = envelope > (
        envelope_med
        + config.post_fusion_amplitude_mad_multiplier * envelope_scale
    )

    derivative = np.diff(work, prepend=work[0]) * fs
    derivative_window = max(3, int(round(1.0 * fs)))
    derivative_power = ndimage.uniform_filter1d(
        derivative * derivative,
        size=derivative_window,
        mode="nearest",
    )
    derivative_rms = np.sqrt(np.maximum(derivative_power, 0.0))
    derivative_med = float(np.median(derivative_rms[finite]))
    derivative_scale = _robust_scale(derivative_rms[finite] - derivative_med)
    derivative_outlier = derivative_rms > (
        derivative_med
        + config.post_fusion_derivative_mad_multiplier * derivative_scale
    )

    detected = absolute_outlier | envelope_outlier | derivative_outlier | ~finite
    expand = int(round(config.post_fusion_expand_sec * fs))
    detected = _expand_mask(detected, expand)
    initial_for_repair = _expand_mask(initial, expand) if expand_initial else initial
    combined = initial_for_repair | detected

    valid = ~combined & np.isfinite(work)
    if valid.sum() < max(2, int(round(10.0 * fs))):
        raise RuntimeError(
            "post-fusion artifact repair left fewer than ten seconds of valid data"
        )
    idx = np.flatnonzero(valid)
    repaired = np.interp(np.arange(work.size), idx, work[idx]).astype(np.float64)
    if not np.isfinite(repaired).all():
        raise FloatingPointError("post-fusion repaired waveform contains NaN/Inf")
    return repaired, combined, detected

def _window_feature_scores(
    candidates: NDArray[np.float64],
    broad: NDArray[np.float64],
    radius: NDArray[np.float64],
    artifacts: NDArray[np.bool_],
    balances: NDArray[np.float64],
    fs: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_pairs, n = candidates.shape
    base = np.zeros(n_pairs, dtype=np.float64)
    valid_fraction = np.zeros(n_pairs, dtype=np.float64)
    standardized = np.zeros_like(candidates)
    variability = np.zeros(n_pairs, dtype=np.float64)

    for pair in range(n_pairs):
        valid = ~artifacts[pair] & np.isfinite(candidates[pair])
        valid_fraction[pair] = float(np.mean(valid))
        if valid.sum() < max(8, int(round(10 * fs))):
            continue
        x = candidates[pair]
        xb = broad[pair]
        med = float(np.median(x[valid]))
        scale = _robust_scale(x[valid])
        standardized[pair] = (x - med) / scale
        variability[pair] = scale
        resp_energy = float(np.mean((x[valid] - med) ** 2))
        broad_center = xb[valid] - np.median(xb[valid])
        broad_energy = float(np.mean(broad_center * broad_center))
        band_ratio = resp_energy / max(broad_energy, _EPS)
        band_quality = float(np.clip((band_ratio - 0.08) / 0.72, 0.0, 1.0))
        continuity_quality = float(
            np.clip((valid_fraction[pair] - 0.25) / 0.75, 0.0, 1.0)
        )
        r = radius[pair, valid]
        radius_cv = _robust_scale(r) / max(abs(float(np.median(r))), _EPS)
        radius_quality = float(1.0 / (1.0 + radius_cv))
        balance_quality = float(np.sqrt(np.clip(balances[pair], 0.0, 1.0)))

        freqs, psd = signal.periodogram(x[valid], fs=fs, detrend="constant")
        in_band = (freqs >= 0.10) & (freqs <= 0.35)
        if in_band.any() and np.sum(psd[in_band]) > 0:
            concentration = float(np.max(psd[in_band]) / np.sum(psd[in_band]))
        else:
            concentration = 0.0
        spectral_quality = float(np.clip((concentration - 0.03) / 0.35, 0.0, 1.0))
        base[pair] = (
            0.32 * band_quality
            + 0.25 * continuity_quality
            + 0.15 * radius_quality
            + 0.13 * balance_quality
            + 0.15 * spectral_quality
        )

    positive = variability[variability > 0]
    if positive.size:
        reference_scale = float(np.median(positive))
        variability_quality = np.clip(variability / max(reference_scale, _EPS), 0.0, 1.0)
        base *= 0.8 + 0.2 * variability_quality

    correlations = np.eye(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            shared = ~artifacts[i] & ~artifacts[j]
            if shared.sum() < max(8, int(round(10 * fs))):
                corr = 0.0
            else:
                xi = standardized[i, shared]
                xj = standardized[j, shared]
                if np.std(xi) <= _EPS or np.std(xj) <= _EPS:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(xi, xj)[0, 1])
                    if not np.isfinite(corr):
                        corr = 0.0
            correlations[i, j] = corr
            correlations[j, i] = corr

    consensus = np.zeros(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        others = np.delete(np.abs(correlations[i]), i)
        consensus[i] = float(np.median(others)) if others.size else 0.0
    scores = 0.75 * base + 0.25 * consensus
    scores *= np.clip(valid_fraction / 0.5, 0.0, 1.0)
    return scores, correlations, standardized


def fuse_pair_candidates(
    pair_results: list[PairResult], config: ProcessingConfig
) -> FusionResult:
    if len(pair_results) != 8:
        raise ValueError(f"expected 8 pair results, received {len(pair_results)}")
    candidates = np.stack([item.breath for item in pair_results])
    broad = np.stack([item.broad for item in pair_results])
    radius = np.stack([item.radius for item in pair_results])
    artifacts = np.stack([item.artifact_mask for item in pair_results])
    balances = np.asarray([item.iq_balance for item in pair_results], dtype=np.float64)
    n_pairs, n = candidates.shape
    fs = config.target_fs
    standardized_candidates = np.zeros_like(candidates)
    for pair in range(n_pairs):
        valid = ~artifacts[pair] & np.isfinite(candidates[pair])
        if valid.sum() < max(8, int(round(10 * fs))):
            continue
        x = candidates[pair]
        med = float(np.median(x[valid]))
        standardized_candidates[pair] = (x - med) / _robust_scale(x[valid])
    win = max(8, int(round(config.fusion_window_sec * fs)))
    hop = max(1, int(round(config.fusion_hop_sec * fs)))
    starts = list(range(0, max(1, n - 1), hop))
    if not starts or starts[-1] + win < n:
        starts.append(max(0, n - win))
    starts = sorted(set(starts))

    accumulator = np.zeros(n, dtype=np.float64)
    taper_accumulator = np.zeros(n, dtype=np.float64)
    validity_accumulator = np.zeros(n, dtype=np.float64)
    all_weights = np.zeros((len(starts), n_pairs), dtype=np.float32)
    all_signs = np.ones((len(starts), n_pairs), dtype=np.int8)
    all_scores = np.zeros((len(starts), n_pairs), dtype=np.float32)

    for wi, start in enumerate(starts):
        stop = min(n, start + win)
        length = stop - start
        cand = candidates[:, start:stop]
        brd = broad[:, start:stop]
        rad = radius[:, start:stop]
        art = artifacts[:, start:stop]
        scores, corr, _ = _window_feature_scores(
            cand, brd, rad, art, balances, fs
        )
        standardized = standardized_candidates[:, start:stop]
        all_scores[wi] = scores.astype(np.float32)
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            chosen = np.asarray([best], dtype=int)
        else:
            threshold = max(
                config.minimum_pair_score, 0.55 * float(scores[best])
            )
            ranked = np.argsort(scores)[::-1]
            chosen = ranked[scores[ranked] >= threshold][: config.max_fusion_pairs]
            if chosen.size == 0:
                chosen = np.asarray([best], dtype=int)

        reference_values = scores[chosen] * (
            0.5 + 0.5 * np.median(np.abs(corr[np.ix_(chosen, chosen)]), axis=1)
        )
        reference = int(chosen[int(np.argmax(reference_values))])
        signs = np.ones(n_pairs, dtype=np.int8)
        for pair in chosen:
            signs[pair] = -1 if corr[reference, pair] < 0 else 1
        raw_weights = np.square(np.maximum(scores[chosen], 1e-6))
        weights = raw_weights / np.sum(raw_weights)
        all_weights[wi, chosen] = weights.astype(np.float32)
        all_signs[wi] = signs

        numerator = np.zeros(length, dtype=np.float64)
        denominator = np.zeros(length, dtype=np.float64)
        for weight, pair in zip(weights, chosen, strict=True):
            valid = ~art[pair]
            numerator[valid] += (
                float(weight) * int(signs[pair]) * standardized[pair, valid]
            )
            denominator[valid] += float(weight)
        fused = np.zeros(length, dtype=np.float64)
        usable = denominator > 0
        fused[usable] = numerator[usable] / denominator[usable]
        if not usable.all():
            fallback = int(signs[reference]) * standardized[reference]
            fused[~usable] = fallback[~usable]

        existing = taper_accumulator[start:stop] > 0
        if existing.sum() >= max(8, int(round(10 * fs))):
            previous = accumulator[start:stop][existing] / taper_accumulator[start:stop][existing]
            current = fused[existing]
            if np.std(previous) > _EPS and np.std(current) > _EPS:
                overlap_corr = float(np.corrcoef(previous, current)[0, 1])
                if np.isfinite(overlap_corr) and overlap_corr < 0:
                    fused *= -1
                    all_signs[wi, chosen] *= -1

        taper = signal.windows.tukey(length, alpha=0.5)
        taper = np.maximum(taper, 0.05)
        accumulator[start:stop] += taper * fused
        taper_accumulator[start:stop] += taper
        validity_accumulator[start:stop] += taper * denominator

    if np.any(taper_accumulator <= 0):
        raise RuntimeError("fusion overlap-add left uncovered samples")
    waveform = accumulator / taper_accumulator
    valid_strength = validity_accumulator / taper_accumulator
    artifact_mask = valid_strength < config.minimum_window_valid_fraction

    lengths = np.asarray(
        [min(n, start + win) - start for start in starts], dtype=np.float64
    )
    global_scores = np.average(all_scores, axis=0, weights=lengths).astype(np.float32)
    global_weights = np.average(all_weights, axis=0, weights=lengths).astype(np.float32)
    selected = (np.flatnonzero(global_weights > 0.01) + 1).astype(np.int16)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(global_scores)) + 1], dtype=np.int16)

    if not np.isfinite(waveform).all():
        raise FloatingPointError("fused waveform contains NaN/Inf")
    return FusionResult(
        waveform=waveform,
        artifact_mask=artifact_mask,
        fusion_weights=all_weights,
        fusion_signs=all_signs,
        window_scores=all_scores,
        window_start_sec=(np.asarray(starts, dtype=np.float64) / fs).astype(np.float32),
        global_quality_scores=global_scores,
        global_fusion_weights=global_weights,
        selected_pairs=selected,
    )
