# SleepBRL radar-to-breath

This project converts the 16 SleepBRL radar channels into a continuous 4 Hz `float32` candidate respiratory waveform. The final `breath` field follows the downstream numerical convention used by the supplied `breath.py`: respiratory-band filtering, resampling, zero mean, unit variance, finite values, and a one-dimensional array.

## Scientific scope

The public SleepBRL record contains eight I/Q pairs (`S1/S2`, ..., `S15/S16`) sampled at 50 Hz plus 30-second sleep-stage labels (`W`, `1`, `2`, `3`, `R`). It does not contain a synchronized respiratory inductance belt, airflow, ECG, BCG, RRI, or JJI waveform. Therefore, this code produces a **signal-processing-derived candidate respiratory waveform**. It has not been validated against synchronized respiration or airflow ground truth.

The code does not train a model and does not use sleep stages as respiratory labels. The supplied `heartbeat.py` requires beat positions plus RRI/JJI values, which are absent from SleepBRL, so heartbeat output is intentionally not generated.

## Processing chain

For each frequency pair, the code:

1. reads I and Q from EDF and validates all 16 channels and their per-channel sampling rates;
2. low-pass filters I/Q at 0.8 Hz to suppress out-of-band noise;
3. estimates slowly varying I/Q centers in 5-minute blocks with an axis-scaled algebraic circle fit, rejects degenerate near-line fits, falls back to block medians where needed, and interpolates accepted centers over time;
4. robustly scales I and Q, forms `z = I + jQ`, and computes `unwrap(angle(z))`;
5. marks low-radius intervals, phase jumps, derivative-energy motion, clipping/rail samples, and long stuck-value runs;
6. stitches phase across artifact intervals, applies a zero-phase third-order Bessel 0.1-0.35 Hz band-pass, and resamples to 4 Hz;
7. scores all eight candidates in overlapping 120-second windows using respiratory-band energy ratio, continuity, spectral concentration, I/Q balance, amplitude stability, and cross-frequency agreement;
8. sign-aligns and quality-weights at most three candidates per window, then overlap-adds the fused signal;
9. detects gross post-fusion amplitude/envelope/derivative transients, interpolates through every marked interval while retaining an explicit mask, and repeats this check after the first final filter to suppress zero-phase ringing;
10. applies the final 0.1-0.35 Hz zero-phase third-order Bessel filter and performs whole-record zero-mean/unit-variance normalization.

The 0.1-0.35 Hz band corresponds to approximately 6-21 breaths/min.

No absolute displacement conversion is performed. For a calibrated monostatic continuous-wave radar, a common small-motion relation is `x = lambda * phase / (4*pi)`. SleepBRL spans stepped carrier frequencies around 3.6-4.0 GHz, and the public files do not provide the exact per-pair carrier table, phase-center calibration, or I/Q imbalance calibration needed for defensible absolute displacement.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Command line

Process a directory:

```bash
python -m radar_to_breath \
  --input-dir /path/to/sleepbrl/1.0.0 \
  --output-dir /path/to/output \
  --diagnostics
```

Process one EDF:

```bash
python -m radar_to_breath \
  --edf /path/to/sbj01.edf \
  --output-dir /path/to/output \
  --overwrite
```

An optional configuration file can be passed with `--config default_config.json`. Progress is printed and written to `run.log`. The command also writes `run_summary.csv`, `run_summary.json`, and `failures.json`; exceptions are recorded with tracebacks and are not silently discarded.

## NPZ fields

The field directly usable as the downstream breath channel is:

- `breath`: 4 Hz, one-dimensional, zero-mean/unit-variance `float32`, finite.

Important accompanying fields include:

- `fs`: `4.0`;
- `source_record`, `source_edf`, `source_fs`, `source_n_samples`, `source_duration_sec`, `source_start_time`;
- `selected_iq_pairs`, `top_iq_pair`, `fusion_weights`, `fusion_signs`, `fusion_window_start_sec`;
- `quality_scores`, `quality_scores_window`, `global_fusion_weights`;
- `pair_artifact_fraction`, `pair_jump_fraction`, `pair_motion_fraction`, `pair_saturation_fraction`, `pair_iq_balance`;
- `valid_mask`, `artifact_mask`, plus `fusion_artifact_mask`, `post_fusion_outlier_mask`, and `post_filter_outlier_mask` for provenance of repaired intervals;
- minute-level `respiratory_rate_bpm`, `respiratory_rate_valid`, `respiratory_rate_confidence`, and `respiratory_rate_artifact_fraction`;
- `ground_truth_status`, which explicitly states the lack of synchronized respiratory validation.

## Audit and tests

Audit local files and decode the WFDB stage annotations:

```bash
python scripts/audit_sleepbrl.py \
  --input-dir /path/to/sleepbrl/1.0.0 \
  --signal-stats \
  --output-json audit.json \
  --output-csv audit.csv
```

Run tests:

```bash
pytest
```

The synthetic test generates quadrature I/Q signals with a known 0.25 Hz respiratory component, slow drift, noise, one broken channel, and a short gross-motion interval. It verifies that the broken pair receives zero quality/weight and that the recovered median minute-level rate remains close to 15 breaths/min. A separate test verifies post-fusion transient detection and interpolation.

## Limitations

Multipath, I/Q imbalance, a changing radar-subject geometry, large body movement, and phase-center changes can all distort arctangent demodulation. A narrow I/Q arc can also make circle-center estimation ill-conditioned. Windowed multi-frequency fusion reduces reliance on a single carrier but does not create physiological ground truth. Artifact-filled samples remain continuous for downstream tensor compatibility and are identified by `artifact_mask`; analyses should use the mask. The global sign of the normalized waveform is arbitrary because radar phase polarity depends on geometry and sign alignment. Processing reads one full I/Q pair at a time rather than loading all 16 channels simultaneously; full-record zero-phase filtering still requires substantial RAM for an overnight record.
