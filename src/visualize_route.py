"""Visualize a selected trajectory from the processed GeoLife CSV with Folium."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import folium
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "gps_raw.csv"
DEFAULT_MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
REQUIRED_COLUMNS = {"trajectory_id", "timestamp", "latitude", "longitude"}


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for route selection."""
    parser = argparse.ArgumentParser(
        description="GeoLife GPS 이동 경로를 OpenStreetMap 지도에 표시합니다."
    )
    parser.add_argument(
        "--trajectory-id",
        type=str,
        help="시각화할 trajectory_id. 생략하면 CSV의 첫 번째 경로를 사용합니다.",
    )
    return parser.parse_args(args)


def load_trajectory(
    csv_path: Path, trajectory_id: str | None = None
) -> tuple[str, pd.DataFrame]:
    """Read the CSV and return the requested trajectory in timestamp order."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"처리된 CSV를 찾을 수 없습니다: {csv_path}\n"
            "먼저 python -m src.load_geolife 를 실행하세요."
        )

    try:
        frame = pd.read_csv(
            csv_path,
            dtype={"trajectory_id": "string"},
            parse_dates=["timestamp"],
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"CSV를 읽을 수 없습니다: {csv_path} ({exc})") from exc

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("CSV에 시각화할 GPS 데이터가 없습니다.")

    available_ids = frame["trajectory_id"].dropna().drop_duplicates().tolist()
    if not available_ids:
        raise ValueError("CSV에 사용할 수 있는 trajectory_id가 없습니다.")

    selected_id = trajectory_id if trajectory_id is not None else available_ids[0]
    if selected_id not in available_ids:
        id_list = "\n".join(f"  - {item}" for item in available_ids)
        raise ValueError(
            f"입력한 경로 ID: {selected_id}\n"
            f"사용할 수 있는 전체 경로 ID 목록:\n{id_list}"
        )

    route = frame.loc[frame["trajectory_id"] == selected_id].copy()
    route["latitude"] = pd.to_numeric(route["latitude"], errors="coerce")
    route["longitude"] = pd.to_numeric(route["longitude"], errors="coerce")
    valid = (
        route["latitude"].between(-90, 90, inclusive="both")
        & route["longitude"].between(-180, 180, inclusive="both")
        & route["timestamp"].notna()
    )
    invalid_count = int((~valid).sum())
    if invalid_count:
        print(f"경고: 지도 생성에서 제외할 유효하지 않은 행: {invalid_count:,}개")
        route = route.loc[valid].copy()
    if route.empty:
        raise ValueError(f"경로 {selected_id}에 유효한 GPS 포인트가 없습니다.")

    return selected_id, route.sort_values("timestamp", kind="stable")


def create_route_map(
    trajectory_id: str, route: pd.DataFrame, output_path: Path
) -> Path:
    """Create and save an OpenStreetMap route with start/end markers."""
    coordinates = list(zip(route["latitude"], route["longitude"]))
    route_map = folium.Map(location=coordinates[0], tiles="OpenStreetMap", zoom_start=14)
    route_info = f"경로 ID: {trajectory_id}<br>GPS 포인트 수: {len(route):,}개"

    folium.PolyLine(
        coordinates,
        color="#2563eb",
        weight=5,
        opacity=0.85,
        tooltip=route_info,
        popup=folium.Popup(route_info, max_width=300),
    ).add_to(route_map)
    folium.Marker(
        coordinates[0],
        tooltip="출발 지점",
        popup=f"출발 지점<br>{route.iloc[0]['timestamp']}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(route_map)
    folium.Marker(
        coordinates[-1],
        tooltip="도착 지점",
        popup=f"도착 지점<br>{route.iloc[-1]['timestamp']}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(route_map)
    route_map.fit_bounds(coordinates, padding=(30, 30))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    route_map.save(output_path)
    return output_path


def main(args: Sequence[str] | None = None) -> None:
    """Create a route map selected by command-line arguments."""
    options = parse_args(args)
    try:
        trajectory_id, route = load_trajectory(
            DEFAULT_CSV_PATH, options.trajectory_id
        )
        output_path = DEFAULT_MAP_DIR / f"route_{trajectory_id}.html"
        saved_path = create_route_map(trajectory_id, route, output_path)
        print("=== 이동 경로 지도 생성 완료 ===")
        print(f"선택한 경로 ID: {trajectory_id}")
        print(f"GPS 포인트 수: {len(route):,}개")
        print(f"시작 시간: {route['timestamp'].min()}")
        print(f"종료 시간: {route['timestamp'].max()}")
        print(f"저장된 HTML 경로: {saved_path}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
