from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy import signal

from .annotations import SleepStageAnnotations, read_wfdb_sleep_stages, to_stage5_epochs
from .config import ProcessingConfig
from .diagnostics import choose_diagnostic_start, create_diagnostic_plot
from .edf_io import EdfMetadata, open_sleepbrl_edf, read_iq_pair
from .processing import (
    PairResult,
    fuse_pair_candidates,
    process_iq_pair,
    repair_fused_waveform,
)
from .validation import (
    estimate_minute_respiratory_rate,
    validate_breath_contract,
)


LOGGER = logging.getLogger(__name__)


def _annotation_summary(
    annotation_path: Path,
    annotations: SleepStageAnnotations,
    metadata: EdfMetadata,
) -> dict[str, object]:
    steps = np.unique(np.diff(annotations.samples)) if annotations.samples.size > 1 else np.asarray([])
    coverage_sec = 0.0
    if annotations.samples.size:
        coverage_sec = (float(annotations.samples[-1]) + 30.0 * metadata.source_fs) / metadata.source_fs
    return {
        "path": annotation_path.name,
        "count": int(annotations.samples.size),
        "labels": annotations.counts,
        "sample_steps": [int(v) for v in steps],
        "coverage_sec": coverage_sec,
    }


def _normalize_final(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float64)
    if not np.isfinite(x).all():
        finite = np.isfinite(x)
        if finite.sum() < 2:
            raise FloatingPointError("fused waveform contains fewer than two finite samples")
        idx = np.flatnonzero(finite)
        x = np.interp(np.arange(x.size), idx, x[idx])
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(std) or std <= np.finfo(float).eps:
        raise FloatingPointError("fused waveform has zero or invalid variance")
    y = ((x - mean) / std).astype(np.float32)
    if not np.isfinite(y).all():
        raise FloatingPointError("normalized breath contains NaN/Inf")
    return y


def process_edf(
    edf_path: str | Path,
    output_dir: str | Path,
    config: ProcessingConfig | None = None,
    overwrite: bool = False,
    diagnostics: bool = False,
    diagnostic_dir: str | Path | None = None,
) -> dict[str, object]:
    """Process one SleepBRL EDF and save one NPZ output."""

    cfg = config or ProcessingConfig()
    edf_path = Path(edf_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{edf_path.stem}_breath.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output_path}")

    raw, metadata = open_sleepbrl_edf(edf_path)
    try:
        annotation_path = Path(str(edf_path) + ".atr")
        annotations = read_wfdb_sleep_stages(annotation_path)
        stage5 = to_stage5_epochs(annotations, metadata.source_fs, metadata.n_samples)
        annotation_summary = _annotation_summary(annotation_path, annotations, metadata)
        LOGGER.info(
            "%s: %d samples, %.1f Hz, %.1f seconds",
            metadata.record_id,
            metadata.n_samples,
            metadata.source_fs,
            metadata.duration_sec,
        )
        pair_results: list[PairResult] = []
        for pair_index in range(8):
            LOGGER.info("%s: processing I/Q pair %d/8", metadata.record_id, pair_index + 1)
            i_signal, q_signal = read_iq_pair(raw, pair_index)
            result = process_iq_pair(
                i_signal=i_signal,
                q_signal=q_signal,
                fs=metadata.source_fs,
                pair_index=pair_index,
                physical_min_i=float(metadata.physical_min[2 * pair_index]),
                physical_max_i=float(metadata.physical_max[2 * pair_index]),
                lsb_i=float(metadata.lsb[2 * pair_index]),
                physical_min_q=float(metadata.physical_min[2 * pair_index + 1]),
                physical_max_q=float(metadata.physical_max[2 * pair_index + 1]),
                lsb_q=float(metadata.lsb[2 * pair_index + 1]),
                config=cfg,
            )
            pair_results.append(result)

        fusion = fuse_pair_candidates(pair_results, cfg)
        final_sos = signal.bessel(
            3,
            [cfg.respiration_low_hz, cfg.respiration_high_hz],
            btype="bandpass",
            fs=cfg.target_fs,
            output="sos",
            norm="phase",
        )
        repaired_fusion, artifact_mask, post_fusion_outlier_mask = repair_fused_waveform(
            fusion.waveform, fusion.artifact_mask, cfg.target_fs, cfg
        )
        final_filtered = signal.sosfiltfilt(final_sos, repaired_fusion)
        repaired_filtered, artifact_mask, post_filter_outlier_mask = repair_fused_waveform(
            final_filtered,
            artifact_mask,
            cfg.target_fs,
            cfg,
            expand_initial=False,
        )
        # Refilter after removing filter-visible transients. This second pass
        # suppresses long zero-phase ringing around abrupt phase/carrier changes.
        final_filtered = signal.sosfiltfilt(final_sos, repaired_filtered)
        breath = _normalize_final(final_filtered)
        artifact_mask = np.asarray(artifact_mask, dtype=bool)
        valid_mask = ~artifact_mask
        rate = estimate_minute_respiratory_rate(
            breath, artifact_mask, cfg.target_fs, cfg.minute_window_sec
        )
        contract = validate_breath_contract(
            breath, cfg.target_fs, metadata.duration_sec
        )
        expected_duration_sec = 30 * stage5.size
        if metadata.duration_sec != expected_duration_sec:
            raise ValueError(
                f"EDF duration {metadata.duration_sec} does not match "
                f"{stage5.size} stage epochs ({expected_duration_sec} seconds)"
            )
        if breath.size != 120 * stage5.size:
            raise ValueError(
                f"breath has {breath.size} samples but {stage5.size} stage epochs "
                f"require {120 * stage5.size} samples at 4 Hz"
            )

        pair_artifact_fraction = np.asarray(
            [item.artifact_fraction for item in pair_results], dtype=np.float32
        )
        pair_jump_fraction = np.asarray(
            [item.jump_fraction for item in pair_results], dtype=np.float32
        )
        pair_motion_fraction = np.asarray(
            [item.motion_fraction for item in pair_results], dtype=np.float32
        )
        pair_saturation_fraction = np.asarray(
            [item.saturation_fraction for item in pair_results], dtype=np.float32
        )
        pair_iq_balance = np.asarray(
            [item.iq_balance for item in pair_results], dtype=np.float32
        )
        top_pair = int(np.argmax(fusion.global_fusion_weights)) + 1
        if np.max(fusion.global_fusion_weights) <= 0:
            top_pair = int(np.argmax(fusion.global_quality_scores)) + 1

        payload = {
            "breath": breath,
            "stage5": stage5,
            "fs": np.asarray(cfg.target_fs, dtype=np.float32),
            "source_record": np.asarray(metadata.record_id),
            "source_edf": np.asarray(edf_path.name),
            "source_fs": np.asarray(metadata.source_fs, dtype=np.float32),
            "source_n_samples": np.asarray(metadata.n_samples, dtype=np.int64),
            "source_duration_sec": np.asarray(metadata.duration_sec, dtype=np.float64),
            "source_start_time": np.asarray(metadata.start_time),
            "iq_pairs": np.asarray([[2 * i + 1, 2 * i + 2] for i in range(8)], dtype=np.int16),
            "selected_iq_pairs": fusion.selected_pairs,
            "top_iq_pair": np.asarray(top_pair, dtype=np.int16),
            "fusion_weights": fusion.fusion_weights,
            "fusion_signs": fusion.fusion_signs,
            "fusion_window_start_sec": fusion.window_start_sec,
            "quality_scores": fusion.global_quality_scores,
            "quality_scores_window": fusion.window_scores,
            "global_fusion_weights": fusion.global_fusion_weights,
            "pair_artifact_fraction": pair_artifact_fraction,
            "pair_jump_fraction": pair_jump_fraction,
            "pair_motion_fraction": pair_motion_fraction,
            "pair_saturation_fraction": pair_saturation_fraction,
            "pair_iq_balance": pair_iq_balance,
            "valid_mask": valid_mask,
            "artifact_mask": artifact_mask,
            "fusion_artifact_mask": np.asarray(fusion.artifact_mask, dtype=bool),
            "post_fusion_outlier_mask": np.asarray(post_fusion_outlier_mask, dtype=bool),
            "post_filter_outlier_mask": np.asarray(post_filter_outlier_mask, dtype=bool),
            "respiratory_rate_time_sec": rate.time_sec,
            "respiratory_rate_bpm": rate.rate_bpm,
            "respiratory_rate_valid": rate.valid,
            "respiratory_rate_confidence": rate.confidence,
            "respiratory_rate_artifact_fraction": rate.artifact_fraction,
            "annotation_summary_json": np.asarray(json.dumps(annotation_summary, sort_keys=True)),
            "processing_config_json": np.asarray(cfg.to_json()),
            "method": np.asarray("windowed quality-weighted fusion of unwrapped I/Q phase"),
            "ground_truth_status": np.asarray(
                "candidate respiratory waveform; no synchronized respiration/airflow ground truth in public SleepBRL files"
            ),
        }

        with tempfile.NamedTemporaryFile(
            dir=output_root, prefix=f".{metadata.record_id}_", suffix=".npz", delete=False
        ) as handle:
            temp_path = Path(handle.name)
        try:
            np.savez_compressed(temp_path, **payload)
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        diagnostic_path = ""
        if diagnostics:
            diag_root = Path(diagnostic_dir).resolve() if diagnostic_dir else output_root / "diagnostics"
            diag_root.mkdir(parents=True, exist_ok=True)
            start_out = choose_diagnostic_start(
                artifact_mask, cfg.target_fs, cfg.diagnostic_window_sec
            )
            start_sec = start_out / cfg.target_fs
            raw_start = int(round(start_sec * metadata.source_fs))
            raw_stop = min(
                metadata.n_samples,
                raw_start + int(round(cfg.diagnostic_window_sec * metadata.source_fs)),
            )
            i_name = f"S{2 * top_pair - 1}"
            q_name = f"S{2 * top_pair}"
            raw_pair = raw.get_data(picks=[i_name, q_name], start=raw_start, stop=raw_stop)
            candidates = np.stack([item.breath for item in pair_results])
            phase = pair_results[top_pair - 1].phase
            diagnostic_file = diag_root / f"{metadata.record_id}_diagnostic.png"
            create_diagnostic_plot(
                output_path=diagnostic_file,
                record_id=metadata.record_id,
                raw_i=np.asarray(raw_pair[0], dtype=np.float64),
                raw_q=np.asarray(raw_pair[1], dtype=np.float64),
                raw_fs=metadata.source_fs,
                phase=phase,
                candidates=candidates,
                final_breath=breath,
                artifact_mask=artifact_mask,
                output_fs=cfg.target_fs,
                start_sample_out=start_out,
                top_pair=top_pair,
                quality_scores=fusion.global_quality_scores,
                config=cfg,
            )
            diagnostic_path = str(diagnostic_file)

        valid_rates = rate.rate_bpm[rate.valid]
        summary: dict[str, object] = {
            "record": metadata.record_id,
            "status": "ok",
            "source_edf": str(edf_path),
            "output_npz": str(output_path),
            "diagnostic_png": diagnostic_path,
            "source_fs": metadata.source_fs,
            "duration_sec": metadata.duration_sec,
            "stage_epochs": int(stage5.size),
            "duration_hours": metadata.duration_sec / 3600.0,
            "output_samples": int(breath.size),
            "top_iq_pair": top_pair,
            "selected_iq_pairs": [int(v) for v in fusion.selected_pairs],
            "top_quality_score": float(fusion.global_quality_scores[top_pair - 1]),
            "quality_scores": [float(v) for v in fusion.global_quality_scores],
            "global_fusion_weights": [float(v) for v in fusion.global_fusion_weights],
            "artifact_fraction": float(np.mean(artifact_mask)),
            "fusion_artifact_fraction": float(np.mean(fusion.artifact_mask)),
            "post_fusion_outlier_fraction": float(np.mean(post_fusion_outlier_mask)),
            "post_filter_outlier_fraction": float(np.mean(post_filter_outlier_mask)),
            "bandpower_ratio_0p1_0p35": float(contract["bandpower_ratio_0p1_0p35"]),
            "respiratory_rate_valid_minutes": int(valid_rates.size),
            "respiratory_rate_total_minutes": int(rate.rate_bpm.size),
            "respiratory_rate_median_bpm": float(np.median(valid_rates)) if valid_rates.size else 0.0,
            "respiratory_rate_q1_bpm": float(np.percentile(valid_rates, 25)) if valid_rates.size else 0.0,
            "respiratory_rate_q3_bpm": float(np.percentile(valid_rates, 75)) if valid_rates.size else 0.0,
            "annotation_summary": annotation_summary,
            "warnings": [],
        }
        if metadata.source_fs != 50.0:
            summary["warnings"].append(f"unexpected source sampling rate {metadata.source_fs}")
        if valid_rates.size < 0.5 * rate.rate_bpm.size:
            summary["warnings"].append("fewer than half of minute-level respiratory rates passed quality checks")
        if np.mean(artifact_mask) > 0.5:
            summary["warnings"].append("more than half of the fused output is marked as artifact/low quality")
        return summary
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()
