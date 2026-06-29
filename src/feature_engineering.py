"""Create movement features from the processed GeoLife GPS CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gps_raw.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gps_features.csv"

EARTH_RADIUS_M = 6_371_000.0
BEARING_MIN_DISTANCE_M = 1.0
STOP_DISTANCE_THRESHOLD_M = 3.0
STOP_SPEED_THRESHOLD_MPS = 0.5
EXTREME_SPEED_THRESHOLD_MPS = 50.0

INPUT_COLUMNS = [
    "user_id",
    "trajectory_id",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
]
FEATURE_COLUMNS = [
    "time_diff_sec",
    "distance_m",
    "speed_mps",
    "acceleration_mps2",
    "bearing_deg",
    "direction_change_deg",
    "stop_duration_sec",
]
OUTPUT_COLUMNS = INPUT_COLUMNS + FEATURE_COLUMNS


def load_gps_csv(input_path: Path) -> pd.DataFrame:
    """Load the raw GPS CSV with identifiers preserved as strings."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"입력 CSV를 찾을 수 없습니다: {input_path}\n"
            "먼저 python -m src.load_geolife 를 실행하세요."
        )
    try:
        frame = pd.read_csv(
            input_path,
            dtype={"user_id": "string", "trajectory_id": "string"},
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"입력 CSV를 읽을 수 없습니다: {input_path} ({exc})") from exc

    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"입력 CSV에 필수 컬럼이 없습니다: {', '.join(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("latitude", "longitude", "altitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[INPUT_COLUMNS].sort_values(
        ["trajectory_id", "timestamp"], kind="stable", na_position="last"
    ).reset_index(drop=True)


def validate_input_data(frame: pd.DataFrame) -> None:
    """Validate columns and values required for safe feature calculation."""
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"입력 데이터에 필수 컬럼이 없습니다: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("입력 CSV에 GPS 데이터가 없습니다.")

    null_counts = frame[INPUT_COLUMNS].isna().sum()
    invalid_coordinates = (
        ~frame["latitude"].between(-90, 90, inclusive="both")
        | ~frame["longitude"].between(-180, 180, inclusive="both")
    )
    problems = [
        f"{column} 결측치 {int(count):,}개"
        for column, count in null_counts.items()
        if count
    ]
    if invalid_coordinates.any():
        problems.append(f"범위를 벗어난 위도/경도 {int(invalid_coordinates.sum()):,}개")
    if problems:
        raise ValueError("입력 데이터가 유효하지 않습니다: " + ", ".join(problems))


def haversine_distance(
    latitude1: float | np.ndarray,
    longitude1: float | np.ndarray,
    latitude2: float | np.ndarray,
    longitude2: float | np.ndarray,
) -> float | np.ndarray:
    """Return great-circle distance in metres using a 6,371,000 m Earth radius."""
    lat1 = np.radians(latitude1)
    lon1 = np.radians(longitude1)
    lat2 = np.radians(latitude2)
    lon2 = np.radians(longitude2)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(delta_lon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    distance = EARTH_RADIUS_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(distance) if np.ndim(distance) == 0 else distance


def calculate_bearing(
    latitude1: float | np.ndarray,
    longitude1: float | np.ndarray,
    latitude2: float | np.ndarray,
    longitude2: float | np.ndarray,
) -> float | np.ndarray:
    """Return initial bearing in degrees normalized to [0, 360)."""
    lat1 = np.radians(latitude1)
    lon1 = np.radians(longitude1)
    lat2 = np.radians(latitude2)
    lon2 = np.radians(longitude2)
    delta_lon = lon2 - lon1
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        delta_lon
    )
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return float(bearing) if np.ndim(bearing) == 0 else bearing


def calculate_direction_change(
    previous_bearing: float | np.ndarray,
    current_bearing: float | np.ndarray,
) -> float | np.ndarray:
    """Return the smallest angular difference in the range [0, 180]."""
    difference = np.abs(
        (np.asarray(current_bearing) - np.asarray(previous_bearing) + 180.0)
        % 360.0
        - 180.0
    )
    return float(difference) if np.ndim(difference) == 0 else difference


def calculate_stop_durations(group: pd.DataFrame) -> np.ndarray:
    """Accumulate elapsed seconds while consecutive points meet stop thresholds."""
    durations = np.zeros(len(group), dtype=float)
    elapsed = 0.0
    for position in range(1, len(group)):
        row = group.iloc[position]
        if (
            row["distance_m"] <= STOP_DISTANCE_THRESHOLD_M
            and row["speed_mps"] <= STOP_SPEED_THRESHOLD_MPS
        ):
            elapsed += float(row["time_diff_sec"])
            durations[position] = elapsed
        else:
            elapsed = 0.0
    return durations


def _remove_invalid_time_intervals(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    preliminary_diff = frame.groupby("trajectory_id", sort=False)["timestamp"].diff()
    invalid = preliminary_diff.dt.total_seconds().le(0)
    invalid_count = int(invalid.sum())
    if invalid_count:
        print(
            "경고: 유효하지 않은 시간 간격(time_diff_sec <= 0) "
            f"{invalid_count:,}개 행을 특징 계산에서 제외합니다."
        )
    return frame.loc[~invalid].reset_index(drop=True), invalid_count


def _create_group_features(group: pd.DataFrame) -> pd.DataFrame:
    result = group.copy().reset_index(drop=True)
    count = len(result)
    time_diff = result["timestamp"].diff().dt.total_seconds().fillna(0.0).to_numpy()

    distance = np.zeros(count, dtype=float)
    if count > 1:
        distance[1:] = haversine_distance(
            result["latitude"].to_numpy()[:-1],
            result["longitude"].to_numpy()[:-1],
            result["latitude"].to_numpy()[1:],
            result["longitude"].to_numpy()[1:],
        )

    speed = np.zeros(count, dtype=float)
    np.divide(distance, time_diff, out=speed, where=time_diff > 0)
    acceleration = np.zeros(count, dtype=float)
    if count > 1:
        np.divide(
            speed[1:] - speed[:-1],
            time_diff[1:],
            out=acceleration[1:],
            where=time_diff[1:] > 0,
        )

    bearing = np.zeros(count, dtype=float)
    for position in range(1, count):
        if distance[position] < BEARING_MIN_DISTANCE_M:
            bearing[position] = bearing[position - 1]
        else:
            bearing[position] = calculate_bearing(
                result.at[position - 1, "latitude"],
                result.at[position - 1, "longitude"],
                result.at[position, "latitude"],
                result.at[position, "longitude"],
            )

    direction_change = np.zeros(count, dtype=float)
    if count > 1:
        direction_change[1:] = calculate_direction_change(bearing[:-1], bearing[1:])

    result["time_diff_sec"] = time_diff
    result["distance_m"] = distance
    result["speed_mps"] = speed
    result["acceleration_mps2"] = acceleration
    result["bearing_deg"] = bearing
    result["direction_change_deg"] = direction_change
    result["stop_duration_sec"] = calculate_stop_durations(result)
    return result


def generate_trajectory_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Generate independent movement features for every trajectory."""
    validate_input_data(frame)
    sorted_frame = frame.sort_values(
        ["trajectory_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    valid_frame, invalid_count = _remove_invalid_time_intervals(sorted_frame)
    featured = pd.concat(
        [
            _create_group_features(group)
            for _, group in valid_frame.groupby("trajectory_id", sort=False)
        ],
        ignore_index=True,
    )
    featured["user_id"] = featured["user_id"].astype("string")
    featured["trajectory_id"] = featured["trajectory_id"].astype("string")
    return featured[OUTPUT_COLUMNS], invalid_count


def validate_result_data(frame: pd.DataFrame) -> dict[str, int]:
    """Print output quality checks and return their counts."""
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    required_nulls = int(frame[OUTPUT_COLUMNS].isna().sum().sum()) if not missing_columns else -1
    numeric = frame.select_dtypes(include=[np.number])
    values = numeric.to_numpy(dtype=float)
    checks = {
        "missing_columns": len(missing_columns),
        "required_nulls": required_nulls,
        "nan": int(np.isnan(values).sum()),
        "positive_infinity": int(np.isposinf(values).sum()),
        "negative_infinity": int(np.isneginf(values).sum()),
        "negative_distance": int((frame["distance_m"] < 0).sum()),
        "negative_speed": int((frame["speed_mps"] < 0).sum()),
        "invalid_bearing": int(
            ((frame["bearing_deg"] < 0) | (frame["bearing_deg"] >= 360)).sum()
        ),
        "invalid_direction_change": int(
            (
                (frame["direction_change_deg"] < 0)
                | (frame["direction_change_deg"] > 180)
            ).sum()
        ),
        "extreme_speed": int((frame["speed_mps"] > EXTREME_SPEED_THRESHOLD_MPS).sum()),
    }

    print("\n=== 데이터 품질 검증 ===")
    print(f"누락된 필수 컬럼: {', '.join(missing_columns) if missing_columns else '없음'}")
    labels = {
        "required_nulls": "필수 컬럼 결측치",
        "nan": "NaN",
        "positive_infinity": "양의 무한대",
        "negative_infinity": "음의 무한대",
        "negative_distance": "음수 거리",
        "negative_speed": "음수 속도",
        "invalid_bearing": "범위 밖 bearing",
        "invalid_direction_change": "범위 밖 direction_change",
        "extreme_speed": "속도 50m/s 초과(참고용)",
    }
    for key, label in labels.items():
        print(f"{label}: {checks[key]:,}개")

    trajectory_summary = frame.groupby("trajectory_id", sort=True).agg(
        gps_point_count=("timestamp", "size"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
    )
    print("trajectory별 GPS 포인트 수 및 시작·종료 시간:")
    print(trajectory_summary.to_string())
    return checks


def save_features_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    """Save generated features as a UTF-8 CSV in the fixed column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[OUTPUT_COLUMNS].to_csv(
        output_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )
    return output_path


def print_summary(
    frame: pd.DataFrame,
    input_path: Path,
    output_path: Path,
    input_rows: int,
    invalid_time_count: int,
) -> None:
    """Print paths, row counts, and min/mean/max feature statistics."""
    print("\n=== GPS 이동 특징 생성 완료 ===")
    print(f"입력 CSV 경로: {input_path}")
    print(f"출력 CSV 경로: {output_path}")
    print(f"전체 입력 행 수: {input_rows:,}개")
    print(f"전체 출력 행 수: {len(frame):,}개")
    print(f"전체 trajectory 수: {frame['trajectory_id'].nunique():,}개")
    print(f"유효하지 않은 시간 간격 행 수: {invalid_time_count:,}개")
    print(
        f"속도 50m/s 초과 행 수: "
        f"{int((frame['speed_mps'] > EXTREME_SPEED_THRESHOLD_MPS).sum()):,}개"
    )
    print("특징 통계:")
    statistics = frame[FEATURE_COLUMNS].agg(["min", "mean", "max"]).transpose()
    print(statistics.to_string(float_format=lambda value: f"{value:.6f}"))


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gps_raw.csv에서 trajectory별 GPS 이동 특징을 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Run GPS feature engineering from the command line."""
    options = parse_args(args)
    try:
        raw_frame = load_gps_csv(options.input)
        featured_frame, invalid_count = generate_trajectory_features(raw_frame)
        validate_result_data(featured_frame)
        saved_path = save_features_csv(featured_frame, options.output)
        print_summary(
            featured_frame,
            options.input,
            saved_path,
            len(raw_frame),
            invalid_count,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
