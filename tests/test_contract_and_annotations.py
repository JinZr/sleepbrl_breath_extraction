from __future__ import annotations

from pathlib import Path

import numpy as np

from radar_to_breath.annotations import (
    SleepStageAnnotations,
    read_wfdb_sleep_stages,
    to_stage5_epochs,
)
from radar_to_breath.validation import validate_breath_contract


def test_breath_output_contract() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4 * 600).astype(np.float64)
    x = ((x - x.mean()) / x.std()).astype(np.float32)
    report = validate_breath_contract(x, 4.0, 600.0)
    assert report["n_samples"] == 2400
    assert report["dtype"] == "float32"


def test_minimal_wfdb_aux_parser(tmp_path: Path) -> None:
    # Annotation at sample 0, type 1, AUX "W", then EOF.
    payload = bytes([0x00, 0x04, 0x01, 0xFC, ord("W"), 0x00, 0x00, 0x00])
    path = tmp_path / "x.atr"
    path.write_bytes(payload)
    result = read_wfdb_sleep_stages(path)
    assert result.samples.tolist() == [0]
    assert result.labels == ("W",)


def test_stage5_mapping_requires_full_30_second_alignment() -> None:
    stages = SleepStageAnnotations(
        samples=np.asarray([0, 30, 60, 90, 120], dtype=np.int64),
        labels=("W", "1", "2", "3", "R"),
    )
    stage5 = to_stage5_epochs(stages, source_fs=1.0, source_n_samples=150)
    assert stage5.dtype == np.int64
    assert stage5.tolist() == [0, 1, 2, 3, 4]

    misaligned = SleepStageAnnotations(
        samples=np.asarray([0, 30, 60, 91, 120], dtype=np.int64),
        labels=stages.labels,
    )
    with np.testing.assert_raises_regex(ValueError, "start at sample 0"):
        to_stage5_epochs(misaligned, source_fs=1.0, source_n_samples=150)
