from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
from numpy.typing import NDArray


EXPECTED_CHANNELS = tuple(f"S{i}" for i in range(1, 17))
IQ_PAIRS = tuple((f"S{2 * i + 1}", f"S{2 * i + 2}") for i in range(8))


@dataclass(frozen=True)
class EdfMetadata:
    path: Path
    record_id: str
    channel_names: tuple[str, ...]
    channel_sampling_rates: tuple[float, ...]
    source_fs: float
    n_samples: int
    duration_sec: float
    start_time: str
    physical_min: NDArray[np.float64]
    physical_max: NDArray[np.float64]
    lsb: NDArray[np.float64]


def discover_edf_files(
    input_dir: str | Path | None = None, single_edf: str | Path | None = None
) -> list[Path]:
    if (input_dir is None) == (single_edf is None):
        raise ValueError("provide exactly one of input_dir or single_edf")
    if single_edf is not None:
        path = Path(single_edf).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".edf":
            raise ValueError(f"not an EDF file: {path}")
        return [path]
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(
        (p for p in root.rglob("*.edf") if "__MACOSX" not in p.parts),
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"no EDF files found under {root}")
    return files


def open_sleepbrl_edf(path: str | Path):
    edf_path = Path(path).resolve()
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    channel_names = tuple(raw.ch_names)
    if channel_names != EXPECTED_CHANNELS:
        missing = sorted(set(EXPECTED_CHANNELS) - set(channel_names))
        extra = sorted(set(channel_names) - set(EXPECTED_CHANNELS))
        raise ValueError(
            f"{edf_path.name}: expected exactly S1-S16 in order; "
            f"missing={missing}, extra={extra}, actual={channel_names}"
        )

    extras = raw._raw_extras[0]
    record_length = np.asarray(extras["record_length"], dtype=float).reshape(-1)[0]
    n_samps = np.asarray(extras["n_samps"], dtype=float)
    channel_fs = n_samps / record_length
    if channel_fs.shape != (16,) or not np.allclose(channel_fs, channel_fs[0]):
        raise ValueError(f"{edf_path.name}: channel sampling rates differ: {channel_fs}")
    source_fs = float(channel_fs[0])
    if not np.isclose(source_fs, float(raw.info["sfreq"])):
        raise ValueError(
            f"{edf_path.name}: header sampling rate mismatch: "
            f"{source_fs} versus MNE {raw.info['sfreq']}"
        )

    units = np.asarray(extras["units"], dtype=float)
    cal = np.asarray(extras["cal"], dtype=float) * units
    offsets = np.asarray(extras["offsets"], dtype=float) * units
    digital_max = np.asarray(extras["digital_max"], dtype=float)
    digital_min = np.full_like(digital_max, -32768.0)
    physical_max = digital_max * cal + offsets
    physical_min = digital_min * cal + offsets

    meas_date = raw.info.get("meas_date")
    if isinstance(meas_date, datetime):
        start_time = meas_date.isoformat()
    else:
        start_time = ""
    metadata = EdfMetadata(
        path=edf_path,
        record_id=edf_path.stem,
        channel_names=channel_names,
        channel_sampling_rates=tuple(float(v) for v in channel_fs),
        source_fs=source_fs,
        n_samples=int(raw.n_times),
        duration_sec=float(raw.n_times / source_fs),
        start_time=start_time,
        physical_min=physical_min.astype(np.float64),
        physical_max=physical_max.astype(np.float64),
        lsb=np.abs(cal).astype(np.float64),
    )
    return raw, metadata


def read_iq_pair(raw, pair_index: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not 0 <= pair_index < 8:
        raise IndexError(pair_index)
    i_name, q_name = IQ_PAIRS[pair_index]
    data = raw.get_data(picks=[i_name, q_name])
    if data.shape[0] != 2:
        raise RuntimeError(f"failed to read {i_name}/{q_name}")
    return (
        np.asarray(data[0], dtype=np.float64),
        np.asarray(data[1], dtype=np.float64),
    )
