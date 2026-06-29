"""Visualize trajectory movement features as charts and an interactive map."""

from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gps_features.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_MAPS_DIR = PROJECT_ROOT / "outputs" / "maps"
DEFAULT_PORTFOLIO_DIR = PROJECT_ROOT / "docs" / "images" / "stage2"

SPEED_CHECK_THRESHOLD_MPS = 50.0
TIME_GAP_CHECK_THRESHOLD_SEC = 300.0
STOP_CHECK_THRESHOLD_SEC = 60.0
DIRECTION_TOP_POINT_COUNT = 5
FIGURE_DPI = 160

REQUIRED_COLUMNS = [
    "user_id",
    "trajectory_id",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "time_diff_sec",
    "distance_m",
    "speed_mps",
    "acceleration_mps2",
    "bearing_deg",
    "direction_change_deg",
    "stop_duration_sec",
]
NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "altitude",
    "time_diff_sec",
    "distance_m",
    "speed_mps",
    "acceleration_mps2",
    "bearing_deg",
    "direction_change_deg",
    "stop_duration_sec",
]
FIGURE_FILENAMES = [
    "speed_timeline.png",
    "acceleration_timeline.png",
    "direction_change_timeline.png",
    "stop_duration_timeline.png",
    "time_gap_timeline.png",
]
SUMMARY_COLUMNS = [
    "trajectory_id",
    "gps_point_count",
    "start_time",
    "end_time",
    "duration_sec",
    "total_distance_m",
    "average_speed_mps",
    "maximum_speed_mps",
    "minimum_acceleration_mps2",
    "maximum_acceleration_mps2",
    "average_direction_change_deg",
    "maximum_direction_change_deg",
    "maximum_stop_duration_sec",
    "maximum_time_gap_sec",
    "speed_over_50_count",
    "time_gap_over_300_count",
    "stop_over_60_count",
]


def validate_required_columns(frame: pd.DataFrame) -> None:
    """Raise a clear error when required feature columns are unavailable."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"특징 CSV에 필수 컬럼이 없습니다: {', '.join(missing)}")


def load_feature_csv(input_path: Path) -> pd.DataFrame:
    """Load and validate the generated GPS feature CSV."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"GPS 특징 CSV를 찾을 수 없습니다: {input_path}\n"
            "먼저 python -m src.feature_engineering 을 실행하세요."
        )
    try:
        frame = pd.read_csv(
            input_path,
            dtype={"user_id": "string", "trajectory_id": "string"},
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"GPS 특징 CSV를 읽을 수 없습니다: {input_path} ({exc})") from exc

    validate_required_columns(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    null_counts = frame[REQUIRED_COLUMNS].isna().sum()
    invalid = [
        f"{column} {int(count):,}개"
        for column, count in null_counts.items()
        if count
    ]
    numeric_values = frame[NUMERIC_COLUMNS].to_numpy(dtype=float)
    non_finite_count = int((~np.isfinite(numeric_values)).sum())
    if invalid or non_finite_count:
        details = []
        if invalid:
            details.append("결측치: " + ", ".join(invalid))
        if non_finite_count:
            details.append(f"유한하지 않은 숫자: {non_finite_count:,}개")
        raise ValueError("특징 CSV 값이 유효하지 않습니다: " + "; ".join(details))
    if frame.empty:
        raise ValueError("특징 CSV에 시각화할 GPS 데이터가 없습니다.")
    return frame


def select_trajectory(
    frame: pd.DataFrame, trajectory_id: str | None = None
) -> tuple[str, pd.DataFrame]:
    """Select one trajectory and return a timestamp-sorted independent copy."""
    validate_required_columns(frame)
    available_ids = frame["trajectory_id"].astype("string").dropna().drop_duplicates().tolist()
    if not available_ids:
        raise ValueError("특징 CSV에 사용할 수 있는 trajectory_id가 없습니다.")
    selected_id = trajectory_id if trajectory_id is not None else str(available_ids[0])
    if selected_id not in available_ids:
        id_list = "\n".join(f"  - {item}" for item in available_ids)
        raise ValueError(
            f"입력한 trajectory ID: {selected_id}\n"
            f"사용 가능한 trajectory ID 목록:\n{id_list}"
        )
    route = frame.loc[frame["trajectory_id"] == selected_id].copy()
    route = route.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if route.empty:
        raise ValueError(f"trajectory {selected_id}에 시각화할 포인트가 없습니다.")
    return selected_id, route


def create_output_directories(
    trajectory_id: str,
    figures_root: Path = DEFAULT_FIGURES_DIR,
    maps_root: Path = DEFAULT_MAPS_DIR,
) -> tuple[Path, Path]:
    """Create and return the trajectory figure directory and map directory."""
    figure_dir = figures_root / trajectory_id
    figure_dir.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)
    return figure_dir, maps_root


def _new_figure() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(12, 5.5))


def _finish_figure(fig: plt.Figure, axis: plt.Axes, output_path: Path) -> Path:
    axis.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_speed_figure(route: pd.DataFrame, trajectory_id: str, output_path: Path) -> Path:
    """Create the speed timeline and highlight data-quality check points."""
    fig, axis = _new_figure()
    axis.plot(route["timestamp"], route["speed_mps"], color="#2563eb", linewidth=1.2)
    flagged = route["speed_mps"] > SPEED_CHECK_THRESHOLD_MPS
    axis.scatter(
        route.loc[flagged, "timestamp"],
        route.loc[flagged, "speed_mps"],
        color="#dc2626",
        s=38,
        zorder=3,
        label="Data check: speed > 50 m/s",
    )
    axis.axhline(
        SPEED_CHECK_THRESHOLD_MPS,
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="50 m/s reference",
    )
    axis.set(title=f"Trajectory {trajectory_id} - Speed", xlabel="Timestamp", ylabel="Speed (m/s)")
    axis.legend(loc="best")
    return _finish_figure(fig, axis, output_path)


def create_acceleration_figure(
    route: pd.DataFrame, trajectory_id: str, output_path: Path
) -> Path:
    """Create acceleration timeline with its minimum and maximum emphasized."""
    fig, axis = _new_figure()
    axis.plot(
        route["timestamp"], route["acceleration_mps2"], color="#0f766e", linewidth=1.1
    )
    minimum_index = route["acceleration_mps2"].idxmin()
    maximum_index = route["acceleration_mps2"].idxmax()
    axis.scatter(
        [route.at[minimum_index, "timestamp"], route.at[maximum_index, "timestamp"]],
        [route.at[minimum_index, "acceleration_mps2"], route.at[maximum_index, "acceleration_mps2"]],
        color=["#7c3aed", "#dc2626"],
        s=42,
        zorder=3,
        label="Minimum / maximum",
    )
    axis.axhline(0, color="#374151", linestyle="--", linewidth=1)
    axis.set(
        title=f"Trajectory {trajectory_id} - Acceleration",
        xlabel="Timestamp",
        ylabel="Acceleration (m/s²)",
    )
    axis.legend(loc="best")
    return _finish_figure(fig, axis, output_path)


def create_direction_change_figure(
    route: pd.DataFrame, trajectory_id: str, output_path: Path
) -> Path:
    """Create direction-change timeline and highlight the largest values."""
    fig, axis = _new_figure()
    axis.plot(
        route["timestamp"], route["direction_change_deg"], color="#7c3aed", linewidth=1.1
    )
    top = route.nlargest(min(DIRECTION_TOP_POINT_COUNT, len(route)), "direction_change_deg")
    axis.scatter(
        top["timestamp"],
        top["direction_change_deg"],
        color="#f97316",
        s=36,
        zorder=3,
        label=f"Top {len(top)} values",
    )
    axis.set_ylim(0, 180)
    axis.set(
        title=f"Trajectory {trajectory_id} - Direction Change",
        xlabel="Timestamp",
        ylabel="Direction change (degrees)",
    )
    axis.legend(loc="best")
    return _finish_figure(fig, axis, output_path)


def create_stop_duration_figure(
    route: pd.DataFrame, trajectory_id: str, output_path: Path
) -> Path:
    """Create accumulated stop-duration timeline with 60-second check points."""
    fig, axis = _new_figure()
    axis.plot(
        route["timestamp"], route["stop_duration_sec"], color="#059669", linewidth=1.2
    )
    flagged = route["stop_duration_sec"] >= STOP_CHECK_THRESHOLD_SEC
    axis.scatter(
        route.loc[flagged, "timestamp"],
        route.loc[flagged, "stop_duration_sec"],
        color="#f59e0b",
        s=30,
        zorder=3,
        label="Data check: stop duration ≥ 60 s",
    )
    axis.set(
        title=f"Trajectory {trajectory_id} - Stop Duration",
        xlabel="Timestamp",
        ylabel="Stop duration (s)",
    )
    axis.legend(loc="best")
    return _finish_figure(fig, axis, output_path)


def create_time_gap_figure(
    route: pd.DataFrame, trajectory_id: str, output_path: Path
) -> Path:
    """Create GPS recording-gap timeline and annotate the maximum gap."""
    fig, axis = _new_figure()
    axis.plot(route["timestamp"], route["time_diff_sec"], color="#475569", linewidth=1.1)
    flagged = route["time_diff_sec"] > TIME_GAP_CHECK_THRESHOLD_SEC
    axis.scatter(
        route.loc[flagged, "timestamp"],
        route.loc[flagged, "time_diff_sec"],
        color="#7c3aed",
        s=36,
        zorder=3,
        label="Data check: gap > 300 s",
    )
    maximum_index = route["time_diff_sec"].idxmax()
    maximum_time = route.at[maximum_index, "timestamp"]
    maximum_gap = float(route.at[maximum_index, "time_diff_sec"])
    axis.annotate(
        f"Maximum: {maximum_gap:.0f} s",
        xy=(maximum_time, maximum_gap),
        xytext=(8, 12),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#111827"},
    )
    axis.set(
        title=f"Trajectory {trajectory_id} - GPS Recording Gap",
        xlabel="Timestamp",
        ylabel="Time gap (s)",
    )
    axis.legend(loc="best")
    return _finish_figure(fig, axis, output_path)


def create_feature_figures(
    route: pd.DataFrame, trajectory_id: str, output_dir: Path
) -> list[Path]:
    """Create all five feature charts without modifying the source frame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    creators = [
        create_speed_figure,
        create_acceleration_figure,
        create_direction_change_figure,
        create_stop_duration_figure,
        create_time_gap_figure,
    ]
    return [
        creator(route, trajectory_id, output_dir / filename)
        for creator, filename in zip(creators, FIGURE_FILENAMES)
    ]


def _marker_details(row: pd.Series) -> tuple[list[str], list[tuple[str, str]], str]:
    labels: list[str] = []
    fields: dict[str, str] = {"timestamp": str(row["timestamp"])}
    colors: list[str] = []
    if row["speed_mps"] > SPEED_CHECK_THRESHOLD_MPS:
        labels.append("고속도 확인")
        colors.append("red")
        fields.update(
            {
                "latitude": f"{row['latitude']:.6f}",
                "longitude": f"{row['longitude']:.6f}",
                "speed_mps": f"{row['speed_mps']:.3f}",
                "distance_m": f"{row['distance_m']:.3f}",
                "time_diff_sec": f"{row['time_diff_sec']:.3f}",
            }
        )
    if row["time_diff_sec"] > TIME_GAP_CHECK_THRESHOLD_SEC:
        labels.append("기록 공백 확인")
        colors.append("purple")
        fields.update(
            {
                "time_diff_sec": f"{row['time_diff_sec']:.3f}",
                "distance_m": f"{row['distance_m']:.3f}",
                "speed_mps": f"{row['speed_mps']:.3f}",
            }
        )
    if row["stop_duration_sec"] >= STOP_CHECK_THRESHOLD_SEC:
        labels.append("장시간 정지 확인")
        colors.append("orange")
        fields.update(
            {
                "stop_duration_sec": f"{row['stop_duration_sec']:.3f}",
                "speed_mps": f"{row['speed_mps']:.3f}",
                "direction_change_deg": f"{row['direction_change_deg']:.3f}",
            }
        )
    color = colors[0] if len(set(colors)) == 1 else "darkblue"
    return labels, list(fields.items()), color


def create_feature_map(
    route: pd.DataFrame, trajectory_id: str, output_path: Path
) -> Path:
    """Create a Folium route map with neutral data-quality check markers."""
    coordinates = list(zip(route["latitude"], route["longitude"]))
    route_map = folium.Map(location=coordinates[0], tiles="OpenStreetMap", zoom_start=14)
    folium.PolyLine(
        coordinates,
        color="#2563eb",
        weight=4,
        opacity=0.8,
        tooltip=f"trajectory {trajectory_id}",
    ).add_to(route_map)
    folium.Marker(
        coordinates[0], tooltip="출발 지점", icon=folium.Icon(color="green", icon="play")
    ).add_to(route_map)
    folium.Marker(
        coordinates[-1], tooltip="도착 지점", icon=folium.Icon(color="red", icon="stop")
    ).add_to(route_map)

    marker_cluster = MarkerCluster(name="데이터 품질 확인 지점").add_to(route_map)
    checks = (
        (route["speed_mps"] > SPEED_CHECK_THRESHOLD_MPS)
        | (route["time_diff_sec"] > TIME_GAP_CHECK_THRESHOLD_SEC)
        | (route["stop_duration_sec"] >= STOP_CHECK_THRESHOLD_SEC)
    )
    for _, row in route.loc[checks].iterrows():
        labels, fields, color = _marker_details(row)
        popup_lines = [f"<strong>{html.escape(' · '.join(labels))}</strong>"]
        popup_lines.extend(
            f"{html.escape(name)}: {html.escape(value)}" for name, value in fields
        )
        folium.Marker(
            [row["latitude"], row["longitude"]],
            tooltip=" · ".join(labels),
            popup=folium.Popup("<br>".join(popup_lines), max_width=360),
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(marker_cluster)

    start_time = route["timestamp"].min()
    end_time = route["timestamp"].max()
    info = f"""
    <div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;
                border:2px solid #64748b;padding:10px;font-size:13px;">
      <strong>Trajectory {html.escape(trajectory_id)}</strong><br>
      GPS 포인트: {len(route):,}개<br>
      시작: {start_time}<br>
      종료: {end_time}
    </div>
    """
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
                border:2px solid #64748b;padding:10px;font-size:13px;">
      <strong>데이터 품질 확인 범례</strong><br>
      <span style="color:#dc2626">●</span> 고속도 확인 (&gt; 50m/s)<br>
      <span style="color:#7c3aed">●</span> 기록 공백 확인 (&gt; 300초)<br>
      <span style="color:#f59e0b">●</span> 장시간 정지 확인 (≥ 60초)<br>
      <span style="color:#1e3a8a">●</span> 여러 조건 동시 충족
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(info + legend))
    folium.LayerControl().add_to(route_map)
    route_map.fit_bounds(coordinates, padding=(30, 30))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    route_map.save(output_path)
    return output_path


def calculate_trajectory_summary(
    route: pd.DataFrame, trajectory_id: str
) -> pd.DataFrame:
    """Calculate one-row portfolio statistics for a selected trajectory."""
    start_time = route["timestamp"].min()
    end_time = route["timestamp"].max()
    summary = {
        "trajectory_id": trajectory_id,
        "gps_point_count": len(route),
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": float((end_time - start_time).total_seconds()),
        "total_distance_m": float(route["distance_m"].sum()),
        "average_speed_mps": float(route["speed_mps"].mean()),
        "maximum_speed_mps": float(route["speed_mps"].max()),
        "minimum_acceleration_mps2": float(route["acceleration_mps2"].min()),
        "maximum_acceleration_mps2": float(route["acceleration_mps2"].max()),
        "average_direction_change_deg": float(route["direction_change_deg"].mean()),
        "maximum_direction_change_deg": float(route["direction_change_deg"].max()),
        "maximum_stop_duration_sec": float(route["stop_duration_sec"].max()),
        "maximum_time_gap_sec": float(route["time_diff_sec"].max()),
        "speed_over_50_count": int((route["speed_mps"] > SPEED_CHECK_THRESHOLD_MPS).sum()),
        "time_gap_over_300_count": int(
            (route["time_diff_sec"] > TIME_GAP_CHECK_THRESHOLD_SEC).sum()
        ),
        "stop_over_60_count": int((route["stop_duration_sec"] >= STOP_CHECK_THRESHOLD_SEC).sum()),
    }
    return pd.DataFrame([summary], columns=SUMMARY_COLUMNS)


def save_summary_csv(summary: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return output_path


def export_portfolio_results(
    generated_files: Sequence[Path], trajectory_id: str, portfolio_root: Path
) -> list[Path]:
    """Copy generated PNG and summary CSV files into the Git-trackable docs tree."""
    destination = portfolio_root / trajectory_id
    destination.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for source in generated_files:
        target = destination / source.name
        shutil.copy2(source, target)
        exported.append(target)
    return exported


def print_summary(
    summary: pd.DataFrame,
    png_paths: Sequence[Path],
    map_path: Path,
    portfolio_exported: bool,
) -> None:
    row = summary.iloc[0]
    print("\n=== GPS 특징 시각화 완료 ===")
    print(f"선택한 trajectory ID: {row['trajectory_id']}")
    print(f"GPS 포인트 수: {int(row['gps_point_count']):,}개")
    print(f"시작 시간: {row['start_time']}")
    print(f"종료 시간: {row['end_time']}")
    print(f"총 이동 거리: {row['total_distance_m']:.3f}m")
    print(f"평균 속도: {row['average_speed_mps']:.3f}m/s")
    print(f"최대 속도: {row['maximum_speed_mps']:.3f}m/s")
    print(f"최대 정지 시간: {row['maximum_stop_duration_sec']:.3f}초")
    print(f"최대 GPS 측정 간격: {row['maximum_time_gap_sec']:.3f}초")
    print("생성된 PNG 파일 목록:")
    for path in png_paths:
        print(f"  - {path}")
    print(f"생성된 HTML 지도 경로: {map_path}")
    print(f"포트폴리오 이미지 저장 여부: {'예' if portfolio_exported else '아니요'}")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="trajectory별 GPS 이동 특징 그래프와 품질 확인 지도를 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--trajectory-id", type=str)
    parser.add_argument("--export-portfolio", action="store_true")
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    options = parse_args(args)
    try:
        frame = load_feature_csv(options.input)
        trajectory_id, route = select_trajectory(frame, options.trajectory_id)
        figure_dir, maps_dir = create_output_directories(trajectory_id)
        png_paths = create_feature_figures(route, trajectory_id, figure_dir)
        summary = calculate_trajectory_summary(route, trajectory_id)
        summary_path = save_summary_csv(summary, figure_dir / "feature_summary.csv")
        map_path = create_feature_map(
            route, trajectory_id, maps_dir / f"feature_route_{trajectory_id}.html"
        )
        if options.export_portfolio:
            export_portfolio_results(
                [*png_paths, summary_path], trajectory_id, DEFAULT_PORTFOLIO_DIR
            )
        print_summary(summary, png_paths, map_path, options.export_portfolio)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
