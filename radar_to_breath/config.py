from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessingConfig:
    """Numerical settings for the deterministic radar-to-breath pipeline."""

    target_fs: float = 4.0
    respiration_low_hz: float = 0.10
    respiration_high_hz: float = 0.35
    iq_lowpass_hz: float = 0.80
    broad_low_hz: float = 0.05
    broad_high_hz: float = 0.80
    baseline_window_sec: float = 300.0
    artifact_expand_sec: float = 1.0
    low_radius_fraction: float = 0.08
    jump_abs_threshold_rad: float = 0.80
    jump_mad_multiplier: float = 10.0
    motion_mad_multiplier: float = 10.0
    fusion_window_sec: float = 120.0
    fusion_hop_sec: float = 60.0
    max_fusion_pairs: int = 3
    minimum_pair_score: float = 0.20
    minimum_window_valid_fraction: float = 0.35
    post_fusion_absolute_z_threshold: float = 8.0
    post_fusion_amplitude_mad_multiplier: float = 10.0
    post_fusion_derivative_mad_multiplier: float = 10.0
    post_fusion_expand_sec: float = 3.0
    minute_window_sec: float = 60.0
    diagnostic_window_sec: float = 180.0

    def __post_init__(self) -> None:
        if self.target_fs <= 2 * self.respiration_high_hz:
            raise ValueError("target_fs must exceed twice the respiration high cutoff")
        if not 0 < self.respiration_low_hz < self.respiration_high_hz:
            raise ValueError("invalid respiration band")
        if not 0 < self.broad_low_hz < self.broad_high_hz:
            raise ValueError("invalid broad quality band")
        if self.broad_high_hz >= self.target_fs / 2:
            raise ValueError("broad_high_hz must be below target Nyquist")
        if self.baseline_window_sec <= 2 / self.respiration_low_hz:
            raise ValueError("baseline_window_sec is too short relative to breathing")
        if self.fusion_window_sec <= 0 or self.fusion_hop_sec <= 0:
            raise ValueError("fusion window and hop must be positive")
        if self.fusion_hop_sec > self.fusion_window_sec:
            raise ValueError("fusion_hop_sec cannot exceed fusion_window_sec")
        if not 1 <= self.max_fusion_pairs <= 8:
            raise ValueError("max_fusion_pairs must be in [1, 8]")
        if self.post_fusion_absolute_z_threshold <= 0:
            raise ValueError("post_fusion_absolute_z_threshold must be positive")
        if self.post_fusion_amplitude_mad_multiplier <= 0:
            raise ValueError("post_fusion_amplitude_mad_multiplier must be positive")
        if self.post_fusion_derivative_mad_multiplier <= 0:
            raise ValueError("post_fusion_derivative_mad_multiplier must be positive")
        if self.post_fusion_expand_sec < 0:
            raise ValueError("post_fusion_expand_sec cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ProcessingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"unknown configuration keys: {unknown}")
        return cls(**payload)
