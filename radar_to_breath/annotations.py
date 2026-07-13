from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SleepStageAnnotations:
    samples: NDArray[np.int64]
    labels: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.labels).items()))


def read_wfdb_sleep_stages(path: str | Path) -> SleepStageAnnotations:
    """Read the small WFDB annotation subset used by SleepBRL.

    SleepBRL stores one ordinary annotation plus an AUX text field (W, 1, 2,
    3, or R) every 30 seconds. This reader supports standard SKIP, SUB, CHAN,
    NUM, and AUX fields and intentionally has no dependency on wfdb-python.
    """

    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if raw.size % 2:
        raise ValueError(f"WFDB annotation file has odd byte count: {path}")
    pairs = raw.reshape(-1, 2)
    bpi = 0
    sample_total = 0
    samples: list[int] = []
    labels: list[str] = []

    while bpi < len(pairs) - 1:
        sample_diff = 0
        while (int(pairs[bpi, 1]) >> 2) == 59:  # SKIP
            if bpi + 2 >= len(pairs):
                raise ValueError(f"truncated SKIP field in {path}")
            skip = (
                (int(pairs[bpi + 1, 0]) << 16)
                + (int(pairs[bpi + 1, 1]) << 24)
                + int(pairs[bpi + 2, 0])
                + (int(pairs[bpi + 2, 1]) << 8)
            )
            if skip > 2_147_483_647:
                skip -= 4_294_967_296
            sample_diff += skip
            bpi += 3

        label_store = int(pairs[bpi, 1]) >> 2
        dt = int(pairs[bpi, 0]) + 256 * (int(pairs[bpi, 1]) & 3)
        if label_store == 0 and dt == 0:
            break
        sample_total += sample_diff + dt
        bpi += 1

        aux: str | None = None
        while bpi < len(pairs):
            extra_code = int(pairs[bpi, 1]) >> 2
            if extra_code <= 59:
                break
            if extra_code in (60, 61, 62):  # NUM, SUB, CHAN
                bpi += 1
                continue
            if extra_code == 63:  # AUX
                n_bytes = int(pairs[bpi, 0])
                n_pairs = (n_bytes + 1) // 2
                if bpi + n_pairs >= len(pairs):
                    raise ValueError(f"truncated AUX field in {path}")
                aux_bytes = pairs[bpi + 1 : bpi + 1 + n_pairs].reshape(-1)[:n_bytes]
                aux = bytes(int(v) for v in aux_bytes).decode("ascii", errors="strict")
                bpi += 1 + n_pairs
                continue
            raise ValueError(f"unsupported WFDB extra code {extra_code} in {path}")

        samples.append(sample_total)
        labels.append(aux if aux is not None else str(label_store))

    return SleepStageAnnotations(
        samples=np.asarray(samples, dtype=np.int64), labels=tuple(labels)
    )
