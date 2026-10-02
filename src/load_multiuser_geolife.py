"""Stage 5-A GeoLife loading, with global IDs and an input audit trail."""

from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.feature_engineering import generate_trajectory_features
from src.load_geolife import (
    DEFAULT_DATA_ROOT, find_trajectory_files, read_plt_file, validate_gps_data,
)

ID_COLUMNS = (
    "user_id", "trajectory_id", "original_trajectory_id", "source_trajectory_id",
    "sample_id", "anomaly_segment_id", "dataset_split", "retained_trajectory_id",
)


def validate_user_ids(user_ids: Sequence[str]) -> list[str]:
    """Require explicit zero-padded IDs rather than silently changing identity."""
    users = list(user_ids)
    if not users or any(not isinstance(u, str) or len(u) != 3 or not u.isascii()
                        or not u.isdigit() for u in users):
        raise ValueError("user_id must be a three-digit string, e.g. '000'.")
    if len(set(users)) != len(users):
        raise ValueError("Duplicate user_id in requested users.")
    return sorted(users)


def read_dataset_csv(path: Path) -> pd.DataFrame:
    """Preserve IDs, nanosecond timestamps, and empty reason strings on disk."""
    frame = pd.read_csv(path, dtype={c: "string" for c in ID_COLUMNS},
                        keep_default_na=False)
    for column in ("timestamp", "start_time", "end_time",
                   "anomaly_start_time", "anomaly_end_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise", format="mixed")
    for column in ("is_synthetic", "is_low_quality", "is_training_eligible",
                   "source_quality_valid", "synthetic_value_valid",
                   "is_invalid_time", "is_long_gap", "is_unrealistic_speed",
                   "is_gps_jump", "eligible_for_dataset", "quality_eligible_for_dataset", "is_exact_duplicate"):
        if column in frame:
            values = frame[column].astype("string").str.lower()
            if not values.isin(["true", "false"]).all():
                raise ValueError(f"Invalid boolean values: {column}")
            frame[column] = values.eq("true").astype(bool)
    return frame


def trajectory_fingerprint(frame: pd.DataFrame) -> str:
    """SHA256 of lat/lon at 7 decimals and elapsed nanoseconds, in time order.

    User ID, altitude, and absolute date are excluded to detect copies with a
    shifted start time. This is a duplicate-screening definition, not a claim
    that two rounded paths are behaviorally identical.
    """
    ordered = frame.sort_values("timestamp", kind="stable")
    timestamps = pd.to_datetime(ordered["timestamp"])
    coordinates = ordered[["latitude", "longitude"]].to_numpy(dtype=float)
    if ordered.empty or timestamps.isna().any() or not np.isfinite(coordinates).all():
        raise ValueError("Cannot fingerprint empty or invalid GPS data.")
    elapsed = (timestamps - timestamps.iloc[0]).to_numpy(dtype="timedelta64[ns]").astype(np.int64)
    content = "\n".join(
        f"{lat:.7f},{lon:.7f},{int(ns)}"
        for (lat, lon), ns in zip(coordinates, elapsed)
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_multiuser_features(
    data_root: Path = DEFAULT_DATA_ROOT,
    user_ids: Sequence[str] = tuple(f"{i:03}" for i in range(20)),
    files_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select filename-first files; reuse Stage 1/2 without combining routes."""
    data_root = Path(data_root)
    users = validate_user_ids(user_ids)
    if files_per_user < 1:
        raise ValueError("files_per_user must be positive.")
    frames, manifest = [], []
    for user in users:
        for file_path in find_trajectory_files(data_root, user, files_per_user):
            raw = read_plt_file(file_path, user)
            original_id = file_path.stem
            global_id = f"{user}__{original_id}"
            raw["trajectory_id"] = global_id
            invalid = (
                raw["timestamp"].isna()
                | ~raw["latitude"].between(-90, 90)
                | ~raw["longitude"].between(-180, 180)
            )
            valid_raw = raw.loc[~invalid]
            duplicate_count = int(valid_raw.duplicated(["trajectory_id", "timestamp"]).sum())
            reverse_count = int(valid_raw["timestamp"].diff().dt.total_seconds().lt(0).sum())
            audit = {
                "user_id": user, "original_trajectory_id": original_id,
                "trajectory_id": global_id, "source_trajectory_id": global_id,
                "input_path": str(file_path.relative_to(data_root)),
                "input_sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                "raw_row_count": len(raw), "invalid_coordinate_timestamp_rows": int(invalid.sum()),
                "duplicate_timestamp_rows": duplicate_count,
                "raw_timestamp_reverse_count": reverse_count,
            }
            if valid_raw.empty:
                audit.update(cleaned_row_count=0, feature_removed_rows=0,
                             content_fingerprint="", load_status="all_rows_invalid")
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    clean = validate_gps_data(raw)
                    featured, removed = generate_trajectory_features(clean)
                featured["original_trajectory_id"] = original_id
                featured["source_trajectory_id"] = global_id
                featured["source_point_index"] = np.arange(len(featured), dtype=np.int64)
                for column in ID_COLUMNS:
                    if column in featured:
                        featured[column] = featured[column].astype("string")
                frames.append(featured)
                audit.update(cleaned_row_count=len(featured), feature_removed_rows=removed,
                             content_fingerprint=trajectory_fingerprint(featured), load_status="loaded")
            manifest.append(audit)
    inputs = pd.DataFrame(manifest)
    if inputs["trajectory_id"].duplicated().any():
        raise ValueError("Global trajectory ID collision.")
    if not frames:
        raise ValueError("No valid GPS trajectories remain.")
    return pd.concat(frames, ignore_index=True), inputs
