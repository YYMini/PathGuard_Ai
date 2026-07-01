"""Visualize Stage 3 dataset quality, splits, and synthetic examples."""

from __future__ import annotations

import argparse
import shutil
from html import escape
from pathlib import Path
from typing import Sequence

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.feature_engineering import haversine_distance

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed"
DEFAULT_FIGURES = ROOT / "outputs" / "figures" / "stage3"
DEFAULT_MAP = ROOT / "outputs" / "maps" / "stage3_synthetic_examples.html"
DEFAULT_PORTFOLIO = ROOT / "docs" / "images" / "stage3"
PNG_NAMES = ["quality_flag_counts.png", "split_distribution.png", "anomaly_type_counts.png", "normal_vs_synthetic_features.png"]
ANOMALY_TYPES = ("route_deviation", "abnormal_speed", "long_stop", "direction_change")
LAYER_NAMES = {"normal": "정상 경로", "route_deviation": "경로 이탈", "abnormal_speed": "비정상 속도", "long_stop": "장시간 정지", "direction_change": "급격한 방향 변화"}
COLORS = {"route_deviation": "#D32F2F", "abnormal_speed": "#F57C00", "long_stop": "#7B1FA2", "direction_change": "#1976D2"}


def _read(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.is_file(): raise FileNotFoundError(f"시각화 입력 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path, dtype={"trajectory_id": "string", "source_trajectory_id": "string"})


def _anomaly_segments(sample: pd.DataFrame) -> list[pd.DataFrame]:
    """Return labeled segments, grouped by segment id or contiguous row runs."""
    labeled = sample[sample["anomaly_label"].eq(1)].copy()
    if labeled.empty:
        return []
    ids = labeled["anomaly_segment_id"].fillna("").astype(str)
    contiguous = labeled.index.to_series().diff().ne(1).cumsum().astype(str)
    keys = ids.where(ids.ne(""), "unlabeled") + "::" + contiguous
    return [group for _, group in labeled.groupby(keys, sort=False)]


def _segment_with_context(sample: pd.DataFrame, segment: pd.DataFrame) -> pd.DataFrame:
    start, end = sample.index.get_loc(segment.index[0]), sample.index.get_loc(segment.index[-1])
    return sample.iloc[max(0, start - 1):min(len(sample), end + 2)]


def _popup_rows(segment: pd.DataFrame, position: str) -> str:
    row = segment.iloc[0] if position == "start" else segment.iloc[-1]
    position_name = "이상 구간 시작" if position == "start" else "이상 구간 종료"
    return (f"<b>{position_name}</b><br>"
            f"이상 유형: {LAYER_NAMES[str(row['anomaly_type'])]}<br>샘플 ID: {escape(str(row['sample_id']))}<br>"
            f"시간: {escape(str(row['timestamp']))}<br>이상 구간 ID: {escape(str(row['anomaly_segment_id']))}<br>"
            f"변형 포인트 수: {len(segment)}")


def _add_end_markers(layer: folium.FeatureGroup, segment: pd.DataFrame) -> None:
    for position, row, color in (("start", segment.iloc[0], "green"), ("end", segment.iloc[-1], "red")):
        folium.Marker(
            [row["latitude"], row["longitude"]],
            tooltip="이상 구간 시작" if position == "start" else "이상 구간 종료",
            popup=folium.Popup(_popup_rows(segment, position), max_width=360),
            icon=folium.Icon(color=color, icon="flag"),
        ).add_to(layer)


def _add_anomaly_segment(layer: folium.FeatureGroup, sample: pd.DataFrame,
                         segment: pd.DataFrame, original: pd.DataFrame) -> list[list[float]]:
    """Render one labeled segment with anomaly-specific visual encoding."""
    anomaly_type = str(segment["anomaly_type"].iloc[0])
    context = _segment_with_context(sample, segment)
    coordinates = context[["latitude", "longitude"]].astype(float).values.tolist()
    mean_speed, max_speed = segment["speed_mps"].mean(), segment["speed_mps"].max()
    start_time, end_time = segment["timestamp"].iloc[0], segment["timestamp"].iloc[-1]
    common = (f"{LAYER_NAMES[anomaly_type]} 이상 구간 | 샘플 ID={segment['sample_id'].iloc[0]} | "
              f"시작 시간={start_time} | 종료 시간={end_time} | 포인트 수={len(segment)}")
    if anomaly_type == "route_deviation":
        matched = original[pd.to_datetime(original["timestamp"]).isin(pd.to_datetime(segment["timestamp"]))]
        deviation_text = ""
        if len(matched) == len(segment):
            distances = haversine_distance(segment["latitude"].to_numpy(), segment["longitude"].to_numpy(), matched["latitude"].to_numpy(), matched["longitude"].to_numpy())
            deviation_text = f" | 평균 이탈 거리={float(distances.mean()):.1f}m | 최대 이탈 거리={float(distances.max()):.1f}m"
            folium.PolyLine(matched[["latitude", "longitude"]].values.tolist(), color="#263238", weight=5, dash_array="8 6", opacity=.9, tooltip="동일 시간대 원본 경로").add_to(layer)
        folium.PolyLine(coordinates, color=COLORS[anomaly_type], weight=8, opacity=.9, tooltip=common + deviation_text).add_to(layer)
    elif anomaly_type == "abnormal_speed":
        points = context.reset_index(drop=True)
        for i in range(1, len(points)):
            speed = float(points.loc[i, "speed_mps"])
            color = "#9E9E9E" if speed < 10 else "#FBC02D" if speed < 20 else "#FB8C00" if speed < 30 else "#D50000"
            folium.PolyLine(points.loc[i-1:i, ["latitude", "longitude"]].values.tolist(), color=color, weight=8, opacity=.82,
                            tooltip=f"비정상 속도 구간 | 속도={speed:.2f}m/s | 평균 속도={mean_speed:.2f}m/s | 최대 속도={max_speed:.2f}m/s | 포인트 수={len(segment)}").add_to(layer)
    elif anomaly_type == "long_stop":
        folium.PolyLine(coordinates, color=COLORS[anomaly_type], weight=3, opacity=.55, tooltip="장시간 정지 전후 이동선").add_to(layer)
        duration = float(segment["stop_duration_sec"].max())
        center = [float(segment["latitude"].mean()), float(segment["longitude"].mean())]
        popup = (f"<b>장시간 정지 위치</b><br>누적 정지 시간: {duration:.1f}초<br>정지 시작 시간: {start_time}<br>"
                 f"정지 종료 시간: {end_time}<br>정지 포인트 수: {len(segment)}<br>평균 속도: {mean_speed:.3f}m/s")
        folium.CircleMarker(center, radius=min(22, 8 + duration / 30), color=COLORS[anomaly_type], fill=True, fill_opacity=.45,
                            tooltip="장시간 정지 위치", popup=folium.Popup(popup, max_width=360)).add_to(layer)
        for _, row in segment.iterrows():
            folium.CircleMarker([row["latitude"], row["longitude"]], radius=2, color=COLORS[anomaly_type], fill=True,
                                tooltip=f"정지 포인트 | {row['timestamp']}").add_to(layer)
    else:
        folium.PolyLine(coordinates, color=COLORS[anomaly_type], weight=8, opacity=.9, tooltip=common).add_to(layer)
        for _, row in segment.iterrows():
            change = float(row["direction_change_deg"])
            popup = (f"시간: {row['timestamp']}<br>방향 변화량: {change:.2f}°<br>속도: {float(row['speed_mps']):.2f}m/s<br>"
                     f"위도: {float(row['latitude']):.7f}<br>경도: {float(row['longitude']):.7f}")
            folium.CircleMarker([row["latitude"], row["longitude"]], radius=3 + min(change, 180) / 45,
                                color=COLORS[anomaly_type], fill=True, fill_opacity=.8,
                                tooltip="방향 변화 지점", popup=folium.Popup(popup, max_width=320)).add_to(layer)
    _add_end_markers(layer, segment)
    return segment[["latitude", "longitude"]].astype(float).values.tolist()


def _legend() -> str:
    return ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:10px;'
            'border:1px solid #777;font-size:12px"><b>합성 이동 경로 비교</b><br><br>'
            '<span style="color:#263238">┅┅┅</span> 정상 경로<br>'
            '<span style="color:#D32F2F">━━━━</span> 합성 이상 구간<br>'
            '🚩 이상 구간 시작·종료<br>● 정지 또는 방향 변화 지점<br><br>'
            '<b>본 지도는 합성 데이터 비교용이며<br>실제 위험 판정 결과가 아닙니다.</b></div>')


def _title(text: str) -> str:
    return (f'<div style="position:fixed;top:12px;left:50px;z-index:9998;background:rgba(255,255,255,.92);'
            f'padding:9px 14px;border:1px solid #555;font-size:16px;font-weight:bold">{text}</div>')


def create_visualizations(data_dir: Path = DEFAULT_DATA, figures_dir: Path = DEFAULT_FIGURES,
                          map_path: Path = DEFAULT_MAP, export_portfolio: bool = False,
                          portfolio_dir: Path = DEFAULT_PORTFOLIO) -> list[Path]:
    quality = _read(data_dir, "quality_checked.csv"); synthetic = _read(data_dir, "synthetic_anomalies.csv")
    manifest = _read(data_dir, "synthetic_anomaly_manifest.csv")
    splits = {name: _read(data_dir, f"{name}.csv") for name in ("train", "validation", "test")}
    trajectory_summary = _read(data_dir, "trajectory_quality_summary.csv")
    figures_dir.mkdir(parents=True, exist_ok=True); map_path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    flags = ["is_invalid_time", "is_long_gap", "is_unrealistic_speed", "is_gps_jump"]
    fig, ax = plt.subplots(figsize=(8, 4.5)); quality[flags].sum().plot.bar(ax=ax, color="#4472C4")
    ax.set(title="GPS quality flag counts", ylabel="Rows", xlabel="Quality flag"); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    path = figures_dir / PNG_NAMES[0]; fig.savefig(path, dpi=150); plt.close(fig); created.append(path)
    distribution = pd.DataFrame({name: {"normal": int(frame["anomaly_label"].eq(0).sum()), "anomaly": int(frame["anomaly_label"].eq(1).sum())} for name, frame in splits.items()}).T
    fig, ax = plt.subplots(figsize=(7, 4.5)); distribution.plot.bar(ax=ax, color=["#70AD47", "#C00000"])
    ax.set(title="Dataset split distribution", ylabel="Rows", xlabel="Split"); ax.tick_params(axis="x", rotation=0); fig.tight_layout()
    path = figures_dir / PNG_NAMES[1]; fig.savefig(path, dpi=150); plt.close(fig); created.append(path)
    fig, ax = plt.subplots(figsize=(7, 4.5)); manifest["anomaly_type"].value_counts().reindex(["route_deviation", "abnormal_speed", "long_stop", "direction_change"], fill_value=0).plot.bar(ax=ax, color="#ED7D31")
    ax.set(title="Synthetic samples by anomaly type", ylabel="Samples", xlabel="Synthetic anomaly type"); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    path = figures_dir / PNG_NAMES[2]; fig.savefig(path, dpi=150); plt.close(fig); created.append(path)
    features = ["speed_mps", "acceleration_mps2", "direction_change_deg", "stop_duration_sec"]
    normal = pd.concat(splits.values(), ignore_index=True); normal = normal[~normal["is_synthetic"].astype(bool)]
    anomaly = synthetic[synthetic["anomaly_label"].eq(1)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, feature in zip(axes.flat, features):
        ax.boxplot([normal[feature].clip(normal[feature].quantile(.01), normal[feature].quantile(.99)), anomaly[feature].clip(anomaly[feature].quantile(.01), anomaly[feature].quantile(.99))], tick_labels=["Normal", "Synthetic"], showfliers=False)
        ax.set_title(feature); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Normal vs synthetic anomaly features"); fig.tight_layout()
    path = figures_dir / PNG_NAMES[3]; fig.savefig(path, dpi=150); plt.close(fig); created.append(path)
    center = [float(quality["latitude"].median()), float(quality["longitude"].median())]
    fmap = folium.Map(location=center, zoom_start=12, control_scale=True)
    route_layers = {
        anomaly_type: folium.FeatureGroup(
            name=layer_name,
            overlay=True,
            control=True,
            show=anomaly_type == "normal",
        ).add_to(fmap)
        for anomaly_type, layer_name in LAYER_NAMES.items()
    }
    if not manifest.empty:
        source_id = str(manifest.iloc[0]["source_trajectory_id"])
        normal_route = quality[quality["trajectory_id"].astype(str).eq(source_id)].copy()
        folium.PolyLine(
            normal_route[["latitude", "longitude"]].values.tolist(),
            color="#263238", weight=5, opacity=.9, dash_array="8 6", tooltip=LAYER_NAMES["normal"],
        ).add_to(route_layers["normal"])
        detail_paths = []
        for anomaly_type in ANOMALY_TYPES:
            candidates = manifest[(manifest["source_trajectory_id"].astype(str) == source_id) & (manifest["anomaly_type"] == anomaly_type)]
            if candidates.empty: continue
            sample = synthetic[synthetic["sample_id"] == candidates.iloc[0]["sample_id"]].copy().reset_index(drop=True)
            segments = _anomaly_segments(sample)
            bounds = []
            for segment in segments:
                bounds.extend(_add_anomaly_segment(route_layers[anomaly_type], sample, segment, normal_route))

            detail_map = folium.Map(location=center, zoom_start=15, control_scale=True)
            original_layer = folium.FeatureGroup(name=LAYER_NAMES["normal"], show=True).add_to(detail_map)
            folium.PolyLine(normal_route[["latitude", "longitude"]].values.tolist(), color="#263238", weight=5, opacity=.9,
                            dash_array="8 6", tooltip=LAYER_NAMES["normal"]).add_to(original_layer)
            detail_layer = folium.FeatureGroup(name=LAYER_NAMES[anomaly_type], show=True).add_to(detail_map)
            detail_bounds = []
            for segment in segments:
                detail_bounds.extend(_add_anomaly_segment(detail_layer, sample, segment, normal_route))
            if detail_bounds:
                detail_map.fit_bounds(detail_bounds, padding=(30, 30))
            folium.LayerControl(collapsed=False).add_to(detail_map)
            detail_map.get_root().html.add_child(folium.Element(_legend()))
            detail_titles = {
                "route_deviation": "정상 경로와 경로 이탈 비교",
                "abnormal_speed": "정상 경로와 비정상 속도 구간 비교",
                "long_stop": "정상 경로와 장시간 정지 구간 비교",
                "direction_change": "정상 경로와 급격한 방향 변화 비교",
            }
            detail_map.get_root().html.add_child(folium.Element(_title(detail_titles[anomaly_type])))
            detail_path = map_path.parent / f"stage3_{anomaly_type}_detail.html"
            detail_map.save(detail_path); detail_paths.append(detail_path)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend()))
    fmap.get_root().html.add_child(folium.Element(_title("Stage 3 합성 이상 경로 비교")))
    fmap.save(map_path); created.append(map_path)
    created.extend(detail_paths if not manifest.empty else [])
    if export_portfolio:
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        for name in PNG_NAMES: shutil.copy2(figures_dir / name, portfolio_dir / name); created.append(portfolio_dir / name)
        summary_rows = [{"metric": "total_trajectories", "category": "all", "value": quality["trajectory_id"].nunique()}, {"metric": "eligible_trajectories", "category": "all", "value": int(trajectory_summary["eligible_for_dataset"].astype(str).str.lower().eq("true").sum())}]
        summary_rows += [{"metric": "quality_flag_count", "category": flag, "value": int(quality[flag].sum())} for flag in flags]
        summary_rows += [{"metric": "synthetic_sample_count", "category": key, "value": int(value)} for key, value in manifest["anomaly_type"].value_counts().items()]
        for split, frame in splits.items():
            summary_rows += [{"metric": "split_row_count", "category": split, "value": len(frame)}, {"metric": "split_normal_rows", "category": split, "value": int(frame["anomaly_label"].eq(0).sum())}, {"metric": "split_anomaly_rows", "category": split, "value": int(frame["anomaly_label"].eq(1).sum())}]
        summary_path = portfolio_dir / "dataset_summary.csv"; pd.DataFrame(summary_rows).to_csv(summary_path, index=False); created.append(summary_path)
    return created


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 데이터셋을 시각화합니다.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA); parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--map-output", type=Path, default=DEFAULT_MAP); parser.add_argument("--portfolio-dir", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--export-portfolio", action="store_true"); return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    options = parse_args(args)
    try:
        paths = create_visualizations(options.data_dir, options.figures_dir, options.map_output, options.export_portfolio, options.portfolio_dir)
        print("Stage 3 시각화 생성 완료:"); [print(f"- {path}") for path in paths]
    except (FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}"); raise SystemExit(1) from exc


if __name__ == "__main__": main()
