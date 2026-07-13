from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import traceback

from .config import ProcessingConfig
from .edf_io import discover_edf_files
from .pipeline import process_edf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m radar_to_breath",
        description="Convert SleepBRL S1-S16 radar I/Q EDF signals into a 4 Hz candidate breath waveform.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="directory searched recursively for sbj*.edf")
    source.add_argument("--edf", type=Path, help="single EDF file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="write one diagnostic PNG per record")
    parser.add_argument("--diagnostic-dir", type=Path)
    parser.add_argument("--config", type=Path, help="optional JSON file overriding ProcessingConfig defaults")
    parser.add_argument("--max-records", type=int, default=None, help="process only the first N discovered EDFs")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def _write_reports(output_dir: Path, summaries: list[dict], failures: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
    )
    columns = [
        "record",
        "status",
        "duration_sec",
        "duration_hours",
        "top_iq_pair",
        "top_quality_score",
        "artifact_fraction",
        "bandpower_ratio_0p1_0p35",
        "respiratory_rate_valid_minutes",
        "respiratory_rate_total_minutes",
        "respiratory_rate_median_bpm",
        "respiratory_rate_q1_bpm",
        "respiratory_rate_q3_bpm",
        "selected_iq_pairs",
        "quality_scores",
        "global_fusion_weights",
        "warnings",
        "output_npz",
        "diagnostic_png",
    ]
    with (output_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            serialized = dict(row)
            for key in ("selected_iq_pairs", "quality_scores", "global_fusion_weights", "warnings"):
                serialized[key] = json.dumps(serialized.get(key, []), ensure_ascii=False)
            writer.writerow(serialized)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "run.log", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("radar_to_breath")
    config = ProcessingConfig.from_json_file(args.config) if args.config else ProcessingConfig()
    files = discover_edf_files(input_dir=args.input_dir, single_edf=args.edf)
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("--max-records must be positive")
        files = files[: args.max_records]
    logger.info("discovered %d EDF file(s)", len(files))

    summaries: list[dict] = []
    failures: list[dict] = []
    for index, path in enumerate(files, start=1):
        logger.info("[%d/%d] starting %s", index, len(files), path.name)
        try:
            summary = process_edf(
                edf_path=path,
                output_dir=args.output_dir,
                config=config,
                overwrite=args.overwrite,
                diagnostics=args.diagnostics,
                diagnostic_dir=args.diagnostic_dir,
            )
            summaries.append(summary)
            logger.info(
                "[%d/%d] completed %s: top pair=%s, median rate=%.2f bpm",
                index,
                len(files),
                path.name,
                summary["top_iq_pair"],
                summary["respiratory_rate_median_bpm"],
            )
        except Exception as exc:  # failures are recorded and processing continues
            failure = {
                "record": path.stem,
                "path": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            logger.exception("[%d/%d] failed %s", index, len(files), path.name)
        finally:
            _write_reports(args.output_dir, summaries, failures)

    logger.info("finished: %d successful, %d failed", len(summaries), len(failures))
    return 1 if failures else 0
