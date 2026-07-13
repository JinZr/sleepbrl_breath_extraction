from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook

from radar_to_breath import cli


def _write_subjects(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Subject", "Sex", "Age"])
    sheet.append(["sbj01", "F", 55])
    sheet.append(["sbj02", "M", 24])
    workbook.save(path)


def test_index_is_written_only_after_every_record_succeeds(tmp_path: Path, monkeypatch) -> None:
    subjects_path = tmp_path / "subjects.xlsx"
    _write_subjects(subjects_path)
    edfs = [tmp_path / "sbj01.edf", tmp_path / "sbj02.edf"]
    monkeypatch.setattr(cli, "discover_edf_files", lambda **kwargs: edfs)

    def process_edf(edf_path: Path, output_dir: Path, **kwargs):
        subject = edf_path.stem
        return {
            "record": subject,
            "output_npz": str(output_dir / f"{subject}_breath.npz"),
            "stage_epochs": 10,
            "top_iq_pair": 1,
            "respiratory_rate_median_bpm": 12.0,
        }

    monkeypatch.setattr(cli, "process_edf", process_edf)
    output_dir = tmp_path / "success"
    result = cli.main(
        [
            "--input-dir",
            str(tmp_path),
            "--subjects-xlsx",
            str(subjects_path),
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert result == 0
    with (output_dir / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "path": str(output_dir / "sbj01_breath.npz"),
            "dataset": "sleepbrl",
            "source": "sleepbrl",
            "subject_id": "sbj01",
            "session_id": "sbj01",
            "duration": "300",
            "age": "55",
            "sex": "0",
            "breath_mask": "1",
            "stage_mask": "1",
            "split": "test",
        },
        {
            "path": str(output_dir / "sbj02_breath.npz"),
            "dataset": "sleepbrl",
            "source": "sleepbrl",
            "subject_id": "sbj02",
            "session_id": "sbj02",
            "duration": "300",
            "age": "24",
            "sex": "1",
            "breath_mask": "1",
            "stage_mask": "1",
            "split": "test",
        },
    ]

    def fail_second(edf_path: Path, **kwargs):
        if edf_path.stem == "sbj02":
            raise RuntimeError("synthetic failure")
        return process_edf(edf_path=edf_path, **kwargs)

    monkeypatch.setattr(cli, "process_edf", fail_second)
    failed_output_dir = tmp_path / "failed"
    result = cli.main(
        [
            "--input-dir",
            str(tmp_path),
            "--subjects-xlsx",
            str(subjects_path),
            "--split",
            "test",
            "--output-dir",
            str(failed_output_dir),
        ]
    )
    assert result == 1
    assert not (failed_output_dir / "index.csv").exists()
