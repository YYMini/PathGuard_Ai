"""Load GeoLife PLT trajectories, validate them, and save a combined CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "geolife" / "Data"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gps_raw.csv"

PLT_COLUMNS = [
    "latitude",
    "longitude",
    "unused",
    "altitude",
    "date_days",
    "date",
    "time",
]
OUTPUT_COLUMNS = [
    "user_id",
    "trajectory_id",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
]


def read_plt_file(file_path: Path, user_id: str) -> pd.DataFrame:
    """Read one GeoLife PLT file and return the project output columns."""
    try:
        frame = pd.read_csv(
            file_path,
            skiprows=6,
            names=PLT_COLUMNS,
            usecols=range(len(PLT_COLUMNS)),
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"PLT 파일을 읽을 수 없습니다: {file_path} ({exc})") from exc

    frame["timestamp"] = pd.to_datetime(
        frame["date"].astype("string") + " " + frame["time"].astype("string"),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["altitude"] = pd.to_numeric(frame["altitude"], errors="coerce")
    frame["user_id"] = user_id
    frame["trajectory_id"] = file_path.stem
    return frame[OUTPUT_COLUMNS]


def find_trajectory_files(
    data_root: Path, user_id: str, limit: int | None = None
) -> list[Path]:
    """Find a user's PLT files in filename order, optionally limiting the count."""
    trajectory_dir = data_root / user_id / "Trajectory"
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(
            "GeoLife Trajectory 폴더를 찾을 수 없습니다: "
            f"{trajectory_dir}\n"
            "GeoLife 데이터를 data/raw/geolife/Data/ 아래에 넣었는지 확인하세요."
        )

    files = sorted(trajectory_dir.glob("*.plt"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(
            f"PLT 파일이 없습니다: {trajectory_dir}\n"
            "해당 폴더에 .plt 파일이 있는지 확인하세요."
        )
    return files[:limit] if limit is not None else files


def load_trajectory_files(files: list[Path], user_id: str) -> pd.DataFrame:
    """Read multiple PLT files and combine them into one DataFrame."""
    if not files:
        raise ValueError("불러올 PLT 파일 목록이 비어 있습니다.")
    return pd.concat(
        [read_plt_file(file_path, user_id) for file_path in files],
        ignore_index=True,
    )


def validate_gps_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Report invalid coordinates/timestamps and duplicates, then remove them."""
    missing_coordinates = frame["latitude"].isna() | frame["longitude"].isna()
    out_of_range = (
        ~frame["latitude"].between(-90, 90, inclusive="both")
        | ~frame["longitude"].between(-180, 180, inclusive="both")
    )
    invalid_timestamp = frame["timestamp"].isna()
    invalid_rows = missing_coordinates | out_of_range | invalid_timestamp

    print(f"유효하지 않은 위도/경도 행: {(missing_coordinates | out_of_range).sum():,}개")
    print(f"timestamp 변환 실패 행: {invalid_timestamp.sum():,}개")
    print(f"삭제할 유효하지 않은 행: {invalid_rows.sum():,}개")

    cleaned = frame.loc[~invalid_rows].copy()
    duplicates = cleaned.duplicated(
        subset=["trajectory_id", "timestamp"], keep="first"
    )
    print(f"중복 trajectory_id/timestamp 행: {duplicates.sum():,}개")
    if duplicates.any():
        print(f"삭제할 중복 행: {duplicates.sum():,}개")
        cleaned = cleaned.loc[~duplicates].copy()

    if cleaned.empty:
        raise ValueError("검증 후 남은 유효한 GPS 포인트가 없습니다.")
    return cleaned


def save_to_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    """Sort validated GPS data and save it as UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_frame = frame.sort_values(
        ["trajectory_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    sorted_frame.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return output_path


def print_summary(
    frame: pd.DataFrame, user_id: str, file_count: int, output_path: Path
) -> None:
    """Print a compact summary of the completed GeoLife import."""
    counts = frame.groupby("trajectory_id", sort=True).size()
    print("\n=== GeoLife 불러오기 완료 ===")
    print(f"불러온 사용자 ID: {user_id}")
    print(f"불러온 .plt 파일 수: {file_count:,}개")
    print(f"전체 GPS 포인트 수: {len(frame):,}개")
    print("경로별 GPS 포인트 수:")
    for trajectory_id, count in counts.items():
        print(f"  - {trajectory_id}: {count:,}개")
    print(f"전체 시작 시간: {frame['timestamp'].min()}")
    print(f"전체 종료 시간: {frame['timestamp'].max()}")
    print(f"저장된 CSV 경로: {output_path}")


def main() -> None:
    """Load the first five trajectories for GeoLife user 000."""
    user_id = "000"
    try:
        files = find_trajectory_files(DEFAULT_DATA_ROOT, user_id, limit=5)
        raw_frame = load_trajectory_files(files, user_id)
        valid_frame = validate_gps_data(raw_frame)
        saved_path = save_to_csv(valid_frame, DEFAULT_OUTPUT_PATH)
        print_summary(valid_frame, user_id, len(files), saved_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
