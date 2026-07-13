#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from radar_to_breath.annotations import read_wfdb_sleep_stages
from radar_to_breath.edf_io import discover_edf_files, open_sleepbrl_edf


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit locally available SleepBRL EDF and WFDB annotation files")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--signal-stats",
        action="store_true",
        help="read all channel samples and report scale/stuck/rail diagnostics",
    )
    args = parser.parse_args()
    records = []
    files = discover_edf_files(input_dir=args.input_dir)
    for path in files:
        raw, meta = open_sleepbrl_edf(path)
        try:
            annotation_path = Path(str(path) + ".atr")
            annotation = None
            if annotation_path.is_file():
                annotation = read_wfdb_sleep_stages(annotation_path)
            record = {
                "record": meta.record_id,
                "edf": str(path),
                "channels": list(meta.channel_names),
                "channel_count": len(meta.channel_names),
                "channel_sampling_rates": list(meta.channel_sampling_rates),
                "source_fs": meta.source_fs,
                "n_samples": meta.n_samples,
                "duration_sec": meta.duration_sec,
                "duration_hours": meta.duration_sec / 3600,
                "start_time": meta.start_time,
                "annotation_file": str(annotation_path) if annotation_path.is_file() else "",
                "annotation_count": int(annotation.samples.size) if annotation else 0,
                "annotation_labels": annotation.counts if annotation else {},
                "annotation_sample_steps": (
                    [int(v) for v in np.unique(np.diff(annotation.samples))]
                    if annotation and annotation.samples.size > 1
                    else []
                ),
            }
            if args.signal_stats:
                data = np.asarray(raw.get_data(), dtype=np.float64)
                standard_deviations = np.std(data, axis=1)
                median_std = max(float(np.median(standard_deviations)), np.finfo(float).eps)
                channel_stats = []
                suspicious_channels = []
                for channel_index, channel_name in enumerate(meta.channel_names):
                    values = data[channel_index]
                    equal_fraction = float(np.mean(np.diff(values) == 0)) if values.size > 1 else 0.0
                    rail_fraction = float(
                        np.mean(
                            (values <= meta.physical_min[channel_index] + 1.5 * meta.lsb[channel_index])
                            | (values >= meta.physical_max[channel_index] - 1.5 * meta.lsb[channel_index])
                        )
                    )
                    std = float(standard_deviations[channel_index])
                    scale_ratio = std / median_std
                    reasons = []
                    if not np.isfinite(values).all():
                        reasons.append("nonfinite")
                    if scale_ratio < 0.01:
                        reasons.append("std_below_1_percent_of_record_median")
                    if equal_fraction > 0.05:
                        reasons.append("more_than_5_percent_equal_successive_samples")
                    if rail_fraction > 0.001:
                        reasons.append("more_than_0.1_percent_at_physical_rail")
                    item = {
                        "channel": channel_name,
                        "mean": float(np.mean(values)),
                        "std": std,
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "std_to_record_median_ratio": scale_ratio,
                        "equal_successive_fraction": equal_fraction,
                        "physical_rail_fraction": rail_fraction,
                        "nonfinite_fraction": float(np.mean(~np.isfinite(values))),
                        "flags": reasons,
                    }
                    channel_stats.append(item)
                    if reasons:
                        suspicious_channels.append({"channel": channel_name, "flags": reasons})
                record["channel_signal_stats"] = channel_stats
                record["suspicious_channels"] = suspicious_channels
                del data
            records.append(record)
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()
    payload = {
        "discovered_record_count": len(records),
        "records": records,
        "ground_truth_assessment": {
            "sleep_stage_annotations_present": all(r["annotation_count"] > 0 for r in records),
            "synchronized_respiration_waveform_present": False,
            "synchronized_ecg_bcg_rri_jji_present": False,
            "selected_target": "breath",
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "record",
                "channel_count",
                "source_fs",
                "n_samples",
                "duration_sec",
                "duration_hours",
                "start_time",
                "annotation_count",
                "annotation_labels",
                "annotation_sample_steps",
                "suspicious_channels",
                "edf",
                "annotation_file",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in records:
                item = dict(row)
                item["annotation_labels"] = json.dumps(item["annotation_labels"], sort_keys=True)
                item["annotation_sample_steps"] = json.dumps(item["annotation_sample_steps"])
                item["suspicious_channels"] = json.dumps(
                    item.get("suspicious_channels", []), sort_keys=True
                )
                writer.writerow(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
