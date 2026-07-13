from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .config import ProcessingConfig


def choose_diagnostic_start(
    artifact_mask: NDArray[np.bool_], fs: float, window_sec: float
) -> int:
    win = min(artifact_mask.size, max(1, int(round(window_sec * fs))))
    if win >= artifact_mask.size:
        return 0
    bad = artifact_mask.astype(np.int64)
    cumulative = np.concatenate(([0], np.cumsum(bad)))
    step = max(1, int(round(30 * fs)))
    starts = np.arange(0, artifact_mask.size - win + 1, step, dtype=np.int64)
    counts = cumulative[starts + win] - cumulative[starts]
    return int(starts[int(np.argmin(counts))])


def create_diagnostic_plot(
    output_path: str | Path,
    record_id: str,
    raw_i: NDArray[np.float64],
    raw_q: NDArray[np.float64],
    raw_fs: float,
    phase: NDArray[np.float64],
    candidates: NDArray[np.float64],
    final_breath: NDArray[np.float32],
    artifact_mask: NDArray[np.bool_],
    output_fs: float,
    start_sample_out: int,
    top_pair: int,
    quality_scores: NDArray[np.float32],
    config: ProcessingConfig,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_out = min(
        final_breath.size - start_sample_out,
        int(round(config.diagnostic_window_sec * output_fs)),
    )
    stop_out = start_sample_out + n_out
    t_out = np.arange(n_out, dtype=float) / output_fs
    raw_seconds = n_out / output_fs
    raw_n = min(raw_i.size, int(round(raw_seconds * raw_fs)))
    t_raw = np.arange(raw_n, dtype=float) / raw_fs

    fig, axes = plt.subplots(5, 1, figsize=(15, 15), constrained_layout=True)
    axes[0].plot(t_raw, raw_i[:raw_n], linewidth=0.7, label=f"S{2 * top_pair - 1} (I)")
    axes[0].plot(t_raw, raw_q[:raw_n], linewidth=0.7, label=f"S{2 * top_pair} (Q)")
    axes[0].set_title(f"{record_id}: raw I/Q, pair {top_pair}")
    axes[0].set_ylabel("EDF physical units (V)")
    axes[0].legend(loc="upper right")

    phase_seg = phase[start_sample_out:stop_out]
    phase_detrended = signal.detrend(phase_seg, type="linear")
    axes[1].plot(t_out, phase_detrended, linewidth=0.8)
    axes[1].set_title("Continuous unwrapped phase after artifact stitching (linear trend removed for display)")
    axes[1].set_ylabel("radians")

    for pair in range(candidates.shape[0]):
        segment = candidates[pair, start_sample_out:stop_out]
        scale = np.std(segment)
        shown = segment / scale if scale > 0 else segment
        axes[2].plot(t_out, shown + 3.0 * pair, linewidth=0.65, label=f"P{pair + 1}")
    axes[2].set_title("Eight respiratory-band phase candidates before fusion")
    axes[2].set_ylabel("normalized + offset")

    breath_seg = final_breath[start_sample_out:stop_out]
    artifact_seg = artifact_mask[start_sample_out:stop_out]
    axes[3].plot(t_out, breath_seg, linewidth=0.9, label="breath")
    if artifact_seg.any():
        ymin, ymax = axes[3].get_ylim()
        axes[3].fill_between(t_out, ymin, ymax, where=artifact_seg, alpha=0.2, label="artifact")
    axes[3].set_title(
        "Final 4 Hz zero-mean/unit-variance breath candidate; "
        f"pair quality={np.array2string(quality_scores, precision=2)}"
    )
    axes[3].set_ylabel("z-score")
    axes[3].legend(loc="upper right")

    freqs, psd = signal.welch(final_breath.astype(float), fs=output_fs, nperseg=min(1024, final_breath.size))
    axes[4].semilogy(freqs, np.maximum(psd, np.finfo(float).tiny), linewidth=1.0)
    axes[4].axvspan(config.respiration_low_hz, config.respiration_high_hz, alpha=0.2)
    axes[4].set_xlim(0, min(1.0, output_fs / 2))
    axes[4].set_title("Final breath power spectral density; shaded band is 0.1-0.35 Hz")
    axes[4].set_xlabel("frequency (Hz)")
    axes[4].set_ylabel("PSD")

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
