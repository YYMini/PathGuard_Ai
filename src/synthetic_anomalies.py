"""Reproducible synthetic movement anomalies for Stage 3 experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, INPUT_COLUMNS, generate_trajectory_features

RANDOM_SEED = 42
ANOMALY_TYPES = ("route_deviation", "abnormal_speed", "long_stop", "direction_change")
META_COLUMNS = ["sample_id", "source_trajectory_id", "is_synthetic", "anomaly_label", "anomaly_type", "anomaly_segment_id"]


def _segment(length: int, rng: np.random.Generator, min_len: int = 10, max_len: int = 30) -> tuple[int, int]:
    if length < min_len + 2:
        raise ValueError(f"합성 이상을 만들기에는 trajectory가 너무 짧습니다: {length}행")
    size = min(int(rng.integers(min_len, min(max_len, length - 2) + 1)), length - 2)
    start = int(rng.integers(1, length - size))
    return start, start + size


def _recalculate(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[INPUT_COLUMNS].copy()
    featured, removed = generate_trajectory_features(base)
    if removed:
        raise ValueError("합성 후 timestamp 순서가 올바르지 않습니다.")
    return featured


def _validate(frame: pd.DataFrame) -> None:
    numeric = frame[["latitude", "longitude"] + FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all(): raise ValueError("합성 특징에 NaN 또는 무한대가 있습니다.")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("합성 timestamp가 증가하지 않거나 중복됩니다.")
    if not frame["latitude"].between(-90, 90).all() or not frame["longitude"].between(-180, 180).all():
        raise ValueError("합성 좌표가 유효 범위를 벗어났습니다.")
    if (frame[["distance_m", "speed_mps"]] < 0).any().any(): raise ValueError("합성 거리/속도가 음수입니다.")
    if not frame["bearing_deg"].between(0, 360, inclusive="left").all(): raise ValueError("방위각 범위가 잘못되었습니다.")
    if not frame["direction_change_deg"].between(0, 180).all(): raise ValueError("방향 변화 범위가 잘못되었습니다.")


def generate_anomaly_sample(
    trajectory: pd.DataFrame, anomaly_type: str, sample_number: int = 1,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate one independently transformed trajectory and its manifest row."""
    if anomaly_type not in ANOMALY_TYPES: raise ValueError(f"지원하지 않는 이상 유형: {anomaly_type}")
    original = trajectory.copy(deep=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    start, end = _segment(len(original), rng)
    changed = original.copy(deep=True)
    lat = changed["latitude"].to_numpy(dtype=float)
    lon = changed["longitude"].to_numpy(dtype=float)
    if anomaly_type == "route_deviation":
        metres = float(rng.uniform(100, 300)); weights = np.sin(np.linspace(0, np.pi, end - start))
        heading = np.arctan2(lat[end - 1] - lat[start], (lon[end - 1] - lon[start]) * np.cos(np.radians(lat[start])))
        north, east = -np.cos(heading) * metres * weights, np.sin(heading) * metres * weights
        lat[start:end] += north / 111_320.0
        lon[start:end] += east / (111_320.0 * np.maximum(np.cos(np.radians(lat[start:end])), 1e-6))
    elif anomaly_type == "abnormal_speed":
        median = float(original.loc[start:end - 1, "speed_mps"].replace(0, np.nan).median())
        target = float(np.clip((median if np.isfinite(median) else 3.0) * rng.uniform(3, 5), 10, 45))
        timestamps = pd.to_datetime(changed["timestamp"]).tolist()
        for position in range(start, end):
            distance = max(float(original.loc[position, "distance_m"]), 0.001)
            interval = max(distance / target, 0.001)
            timestamps[position] = timestamps[position - 1] + pd.to_timedelta(interval, unit="s")
        shift = timestamps[end - 1] - pd.Timestamp(original.loc[end - 1, "timestamp"])
        for position in range(end, len(timestamps)):
            timestamps[position] = pd.Timestamp(original.loc[position, "timestamp"]) + shift
        changed["timestamp"] = timestamps
    elif anomaly_type == "long_stop":
        timestamps = pd.to_datetime(changed["timestamp"])
        # Select enough consecutive points for 120 seconds where possible.
        while end < len(changed) - 1 and (timestamps.iloc[end - 1] - timestamps.iloc[start]).total_seconds() < 120:
            end += 1
        jitter = rng.uniform(-1.0, 1.0, end - start)
        lat[start:end] = lat[start] + jitter / 111_320.0
        lon[start:end] = lon[start] + jitter / (111_320.0 * max(np.cos(np.radians(lat[start])), 1e-6))
    else:
        metres = float(rng.uniform(15, 40)); signs = np.where(np.arange(end - start) % 2 == 0, 1.0, -1.0)
        lat[start:end] += signs * metres / 111_320.0
    changed["latitude"], changed["longitude"] = lat, lon
    featured = _recalculate(changed)
    _validate(featured)
    trajectory_id = str(original["trajectory_id"].iloc[0])
    sample_id = f"{trajectory_id}__{anomaly_type}__{sample_number:02d}"
    featured["sample_id"] = sample_id
    featured["source_trajectory_id"] = trajectory_id
    featured["is_synthetic"] = True
    featured["anomaly_label"] = 0
    featured.loc[start:end - 1, "anomaly_label"] = 1
    featured["anomaly_type"] = anomaly_type
    featured["anomaly_segment_id"] = ""
    featured.loc[start:end - 1, "anomaly_segment_id"] = f"{sample_id}__segment"
    manifest = {
        "sample_id": sample_id, "source_trajectory_id": trajectory_id, "anomaly_type": anomaly_type,
        "anomaly_start_time": featured.loc[start, "timestamp"], "anomaly_end_time": featured.loc[end - 1, "timestamp"],
        "anomaly_point_count": end - start, "original_point_count": len(original),
        "generated_point_count": len(featured), "random_seed": seed,
    }
    return featured, manifest


def generate_synthetic_anomalies(
    eligible: pd.DataFrame, samples_per_type: int = 1, seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples, manifests = [], []
    for trajectory_index, (_, group) in enumerate(eligible.groupby("trajectory_id", sort=True)):
        for type_index, anomaly_type in enumerate(ANOMALY_TYPES):
            for number in range(1, samples_per_type + 1):
                sample_seed = seed + trajectory_index * 10_000 + type_index * 1_000 + number
                sample, manifest = generate_anomaly_sample(group, anomaly_type, number, sample_seed)
                samples.append(sample); manifests.append(manifest)
    columns = INPUT_COLUMNS + FEATURE_COLUMNS + META_COLUMNS
    return (pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=columns), pd.DataFrame(manifests))
