"""Build the Stage 3 quality-checked, synthetic, and split datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data_quality import (MAX_LOW_QUALITY_RATIO, MIN_TRAJECTORY_POINTS,
    REQUIRED_COLUMNS, add_quality_flags, load_feature_data, quality_counts, summarize_trajectories)
from src.synthetic_anomalies import META_COLUMNS, RANDOM_SEED, generate_synthetic_anomalies

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "gps_features.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
FEATURE_COLUMNS = REQUIRED_COLUMNS[6:]


def add_normal_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    result["sample_id"] = result["trajectory_id"].astype(str) + "__normal"
    result["source_trajectory_id"] = result["trajectory_id"].astype(str)
    result["is_synthetic"] = False; result["anomaly_label"] = 0
    result["anomaly_type"] = "normal"; result["anomaly_segment_id"] = ""
    return result


def assign_splits(ids: list[str], seed: int = RANDOM_SEED) -> dict[str, str]:
    """Deterministically assign source trajectories, retaining validation/test when possible."""
    ids = sorted(map(str, ids)); rng = np.random.default_rng(seed); rng.shuffle(ids)
    n = len(ids)
    if n < 3: raise ValueError("train/validation/test 분할에는 적격 trajectory가 최소 3개 필요합니다.")
    validation = max(1, round(n * .2)); test = max(1, round(n * .2)); train = n - validation - test
    if train < 1: train, validation, test = 1, 1, n - 2
    return {value: ("train" if i < train else "validation" if i < train + validation else "test") for i, value in enumerate(ids)}


def build_split_frames(normal: pd.DataFrame, synthetic: pd.DataFrame, assignments: dict[str, str]) -> dict[str, pd.DataFrame]:
    normal = add_normal_metadata(normal)
    frames = {}
    for split in ("train", "validation", "test"):
        ids = {key for key, value in assignments.items() if value == split}
        selected_normal = normal[normal["source_trajectory_id"].isin(ids)].copy()
        if split == "train":
            selected_normal = selected_normal[selected_normal["is_training_eligible"]].copy()
            combined = selected_normal
        else:
            selected_synthetic = synthetic[synthetic["source_trajectory_id"].isin(ids)].copy()
            combined = pd.concat([selected_normal, selected_synthetic], ignore_index=True, sort=False)
        combined["dataset_split"] = split
        frames[split] = combined
    return frames


def validate_splits(frames: dict[str, pd.DataFrame]) -> None:
    errors = []
    id_sets = {name: set(frame["source_trajectory_id"].astype(str)) for name, frame in frames.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if id_sets[left] & id_sets[right]: errors.append(f"{left}/{right} source trajectory 중복")
    train = frames["train"]
    if train["is_synthetic"].astype(bool).any(): errors.append("train에 합성 데이터 존재")
    if train["anomaly_label"].eq(1).any(): errors.append("train에 이상 라벨 존재")
    for split in ("validation", "test"):
        frame = frames[split]
        if frame.empty or not frame["is_synthetic"].astype(bool).any() or not (~frame["is_synthetic"].astype(bool)).any():
            errors.append(f"{split}에 정상/합성 데이터가 모두 존재하지 않음")
    seen = {}
    for split, frame in frames.items():
        for sample in frame["sample_id"].dropna().unique():
            if sample in seen and seen[sample] != split: errors.append(f"sample_id split 중복: {sample}")
            seen[sample] = split
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all(): errors.append(f"{split} 특징에 NaN/무한대 존재")
    if errors: raise ValueError("데이터 누수/품질 검증 실패: " + "; ".join(errors))


def create_split_manifest(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, frame in frames.items():
        for source, group in frame.groupby("source_trajectory_id", sort=True):
            synthetic = group[group["is_synthetic"].astype(bool)]
            rows.append({"source_trajectory_id": source, "dataset_split": split,
                "normal_row_count": int((~group["is_synthetic"].astype(bool)).sum()),
                "synthetic_sample_count": synthetic["sample_id"].nunique(),
                "synthetic_anomaly_row_count": int(synthetic["anomaly_label"].eq(1).sum())})
    return pd.DataFrame(rows)


def prepare_dataset(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
                    samples_per_type: int = 1, seed: int = RANDOM_SEED,
                    min_trajectory_points: int = MIN_TRAJECTORY_POINTS,
                    max_low_quality_ratio: float = MAX_LOW_QUALITY_RATIO) -> dict[str, object]:
    source = load_feature_data(input_path); checked = add_quality_flags(source)
    summary = summarize_trajectories(checked, min_trajectory_points, max_low_quality_ratio)
    eligible_ids = summary.loc[summary["eligible_for_dataset"], "trajectory_id"].astype(str).tolist()
    eligible = checked[checked["trajectory_id"].astype(str).isin(eligible_ids)].copy()
    synthetic, manifest = generate_synthetic_anomalies(eligible, samples_per_type, seed)
    assignments = assign_splits(eligible_ids, seed)
    frames = build_split_frames(eligible, synthetic, assignments); validate_splits(frames)
    split_manifest = create_split_manifest(frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"quality_checked": checked, "trajectory_quality_summary": summary,
             "synthetic_anomalies": synthetic, "synthetic_anomaly_manifest": manifest,
             "train": frames["train"], "validation": frames["validation"], "test": frames["test"],
             "split_manifest": split_manifest}
    paths = []
    for name, frame in files.items():
        path = output_dir / f"{name}.csv"; frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S"); paths.append(path)
        pd.read_csv(path, nrows=2)
    return {"source": source, "checked": checked, "summary": summary, "synthetic": synthetic,
            "manifest": manifest, "frames": frames, "split_manifest": split_manifest, "paths": paths}


def print_summary(result: dict[str, object], input_path: Path) -> None:
    source, checked, summary, manifest, frames = (result[k] for k in ("source", "checked", "summary", "manifest", "frames"))
    print("\n=== Stage 3 학습 데이터 구성 완료 ===")
    print(f"입력 파일: {input_path}\n전체 입력 행 수: {len(source):,}\n전체 trajectory 수: {source['trajectory_id'].nunique():,}")
    print(f"학습 적격 trajectory 수: {int(summary['eligible_for_dataset'].sum()):,}\n제외 trajectory 수: {int((~summary['eligible_for_dataset']).sum()):,}")
    print("품질 플래그별 개수:", quality_counts(checked)); print(f"생성된 합성 sample 수: {len(manifest):,}")
    print("이상 유형별 sample 수:", manifest["anomaly_type"].value_counts().to_dict())
    for split, frame in frames.items():
        print(f"{split}: trajectory {frame['source_trajectory_id'].nunique():,}, 행 {len(frame):,}, 정상 {int(frame['anomaly_label'].eq(0).sum()):,}, 이상 {int(frame['anomaly_label'].eq(1).sum()):,}")
    print("데이터 누수 검사 결과: 통과"); print("생성된 파일 목록:"); [print(f"- {p}") for p in result["paths"]]


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 학습 데이터셋을 구성합니다.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-type", type=int, default=1); parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--min-trajectory-points", type=int, default=MIN_TRAJECTORY_POINTS)
    parser.add_argument("--max-low-quality-ratio", type=float, default=MAX_LOW_QUALITY_RATIO)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    options = parse_args(args)
    try:
        result = prepare_dataset(options.input, options.output_dir, options.samples_per_type, options.seed,
                                 options.min_trajectory_points, options.max_low_quality_ratio)
        print_summary(result, options.input)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"오류: {exc}"); raise SystemExit(1) from exc


if __name__ == "__main__": main()
