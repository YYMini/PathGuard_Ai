"""GPS measurement-quality flags and trajectory eligibility checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LONG_GAP_THRESHOLD_SEC = 300.0
UNREALISTIC_SPEED_THRESHOLD_MPS = 50.0
GPS_JUMP_DISTANCE_THRESHOLD_M = 1000.0
GPS_JUMP_MAX_TIME_SEC = 60.0
MIN_TRAJECTORY_POINTS = 100
MAX_LOW_QUALITY_RATIO = 0.05

REQUIRED_COLUMNS = [
    "user_id", "trajectory_id", "timestamp", "latitude", "longitude", "altitude",
    "time_diff_sec", "distance_m", "speed_mps", "acceleration_mps2", "bearing_deg",
    "direction_change_deg", "stop_duration_sec",
]
FEATURE_COLUMNS = REQUIRED_COLUMNS[6:]


def load_feature_data(path: str | Path) -> pd.DataFrame:
    """Read Stage 2 features while preserving identifier strings."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"입력 특징 CSV를 찾을 수 없습니다: {path}")
    frame = pd.read_csv(path, dtype={"user_id": "string", "trajectory_id": "string"})
    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"입력 특징 CSV에 필수 컬럼이 없습니다: {', '.join(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in REQUIRED_COLUMNS[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def add_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every input row with measurement-quality flags appended."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"품질 검사에 필요한 컬럼이 없습니다: {', '.join(missing)}")
    result = frame.copy(deep=True)
    first = result.groupby("trajectory_id", sort=False).cumcount().eq(0)
    result["is_invalid_time"] = result["time_diff_sec"].le(0) & ~first
    result["is_long_gap"] = result["time_diff_sec"].gt(LONG_GAP_THRESHOLD_SEC)
    result["is_unrealistic_speed"] = result["speed_mps"].gt(UNREALISTIC_SPEED_THRESHOLD_MPS)
    result["is_gps_jump"] = (
        result["distance_m"].gt(GPS_JUMP_DISTANCE_THRESHOLD_M)
        & result["time_diff_sec"].le(GPS_JUMP_MAX_TIME_SEC)
        & result["time_diff_sec"].gt(0)
    )
    flags = ["is_invalid_time", "is_long_gap", "is_unrealistic_speed", "is_gps_jump"]
    result["is_low_quality"] = result[flags].any(axis=1)
    names = ["invalid_time", "long_gap", "unrealistic_speed", "gps_jump"]
    result["quality_reason"] = [
        ";".join(name for name, flag in zip(names, values) if flag)
        for values in result[flags].itertuples(index=False, name=None)
    ]
    result["is_training_eligible"] = ~result["is_low_quality"] & ~first
    return result


def summarize_trajectories(
    frame: pd.DataFrame,
    min_points: int = MIN_TRAJECTORY_POINTS,
    max_low_quality_ratio: float = MAX_LOW_QUALITY_RATIO,
) -> pd.DataFrame:
    """Summarize and select trajectories suitable for synthetic experiments."""
    required = REQUIRED_COLUMNS + ["is_low_quality"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"trajectory 요약에 필요한 컬럼이 없습니다: {', '.join(missing)}")
    rows = []
    for trajectory_id, group in frame.groupby("trajectory_id", sort=True):
        timestamp = pd.to_datetime(group["timestamp"], errors="coerce")
        numeric = group[FEATURE_COLUMNS + ["latitude", "longitude"]].apply(
            pd.to_numeric, errors="coerce"
        )
        reasons = []
        if len(group) < min_points:
            reasons.append(f"point_count<{min_points}")
        low_count = int(group["is_low_quality"].sum())
        ratio = low_count / len(group) if len(group) else 1.0
        if ratio > max_low_quality_ratio:
            reasons.append(f"low_quality_ratio>{max_low_quality_ratio:.2f}")
        if timestamp.isna().any() or not timestamp.is_monotonic_increasing or timestamp.duplicated().any():
            reasons.append("invalid_timestamp_order")
        values = numeric.to_numpy(dtype=float)
        if np.isnan(values).any() or np.isinf(values).any():
            reasons.append("non_finite_required_feature")
        start = timestamp.min()
        end = timestamp.max()
        rows.append({
            "trajectory_id": str(trajectory_id), "point_count": len(group),
            "start_time": start, "end_time": end,
            "duration_sec": (end - start).total_seconds() if pd.notna(start) and pd.notna(end) else np.nan,
            "low_quality_count": low_count, "low_quality_ratio": ratio,
            "eligible_for_dataset": not reasons, "exclusion_reason": ";".join(reasons),
        })
    return pd.DataFrame(rows)


def quality_counts(frame: pd.DataFrame) -> dict[str, int]:
    flags = ["is_invalid_time", "is_long_gap", "is_unrealistic_speed", "is_gps_jump"]
    return {flag: int(frame[flag].sum()) for flag in flags}
