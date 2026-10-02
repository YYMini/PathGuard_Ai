"""Stage 5-A: prepare user-disjoint data and audit lineage; never train a model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data_quality import (
    MAX_LOW_QUALITY_RATIO, MIN_TRAJECTORY_POINTS, add_quality_flags,
    quality_counts, summarize_trajectories,
)
from src.feature_engineering import FEATURE_COLUMNS
from src.load_geolife import DEFAULT_DATA_ROOT
from src.load_multiuser_geolife import (
    ID_COLUMNS, load_multiuser_features, read_dataset_csv, validate_user_ids,
)
from src.prepare_dataset import add_normal_metadata
from src.synthetic_anomalies import ANOMALY_TYPES, generate_anomaly_sample
from src.trajectory_deduplication import (
    DEDUP_COLUMNS, FingerprintLeakageError, deduplicate_trajectory_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "geolife_u000_019_first5_seed42_dedup"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed" / "stage5"
LINEAGE_KEY = ["source_trajectory_id", "source_point_index"]
MODEL_FEATURE_COLUMNS = [
    "time_diff_sec", "distance_m", "speed_mps", "acceleration_mps2",
    "direction_change_deg", "stop_duration_sec", "bearing_sin", "bearing_cos",
]
QUALITY_FIELDS = [
    "is_invalid_time", "is_long_gap", "is_unrealistic_speed", "is_gps_jump",
    "is_low_quality", "quality_reason", "is_training_eligible", "source_quality_valid",
]
SPLITS = ("train", "validation", "test")


def assign_user_splits(users: Sequence[str], seed: int = 42) -> dict[str, str]:
    ids = validate_user_ids(users)
    if len(ids) < 3:
        raise ValueError("At least three users are required for user-level splits.")
    np.random.default_rng(seed).shuffle(ids)
    validation = max(1, round(len(ids) * .2))
    test = max(1, round(len(ids) * .2))
    train = len(ids) - validation - test
    return {user: "train" if i < train else "validation" if i < train + validation else "test"
            for i, user in enumerate(ids)}


def _check_lineage_keys(frame: pd.DataFrame, unique: bool = False) -> None:
    if any(c not in frame for c in LINEAGE_KEY) or frame[LINEAGE_KEY].isna().any().any():
        raise ValueError("Missing lineage key.")
    index = pd.to_numeric(frame["source_point_index"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(index).all() or (index < 0).any() or (index != np.floor(index)).any():
        raise ValueError("Invalid source_point_index.")
    if frame["source_trajectory_id"].astype(str).eq("").any():
        raise ValueError("Empty source_trajectory_id.")
    if unique and frame.duplicated(LINEAGE_KEY).any():
        raise ValueError("Duplicate lineage key in original source.")


def validate_numeric_values(frame: pd.DataFrame) -> None:
    columns = FEATURE_COLUMNS + ["latitude", "longitude"]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("NaN/inf in movement features or coordinates.")
    if not frame["latitude"].between(-90, 90).all() or not frame["longitude"].between(-180, 180).all():
        raise ValueError("Coordinates out of range.")
    if frame[["distance_m", "speed_mps", "time_diff_sec", "stop_duration_sec"]].lt(0).any().any():
        raise ValueError("Negative distance, speed, time or stop duration.")
    if not frame["bearing_deg"].between(0, 360, inclusive="left").all():
        raise ValueError("Invalid bearing.")
    if not frame["direction_change_deg"].between(0, 180).all():
        raise ValueError("Invalid direction change.")


def attach_source_quality(sample: pd.DataFrame, originals: pd.DataFrame) -> pd.DataFrame:
    """Many samples may share a source key; each original key must be unique."""
    _check_lineage_keys(originals, unique=True)
    _check_lineage_keys(sample)
    if sample.duplicated(["sample_id"] + LINEAGE_KEY).any():
        raise ValueError("Duplicate lineage key within sample.")
    identity = ["user_id", "original_trajectory_id"]
    source = originals[LINEAGE_KEY + identity + QUALITY_FIELDS].rename(
        columns={c: f"_source_{c}" for c in identity + QUALITY_FIELDS})
    result = sample.drop(columns=QUALITY_FIELDS, errors="ignore").merge(
        source, on=LINEAGE_KEY, how="left", validate="many_to_one", indicator=True, sort=False)
    if result["_merge"].ne("both").any():
        raise ValueError("Lineage points to missing original source row.")
    for column in identity:
        if not result[column].astype(str).eq(result[f"_source_{column}"].astype(str)).all():
            raise ValueError("Lineage identity mismatch.")
    if not result["trajectory_id"].astype(str).eq(result["source_trajectory_id"].astype(str)).all():
        raise ValueError("Synthetic trajectory/source identity mismatch.")
    for column in QUALITY_FIELDS:
        result[column] = result.pop(f"_source_{column}")
    return result.drop(columns=["_merge"] + [f"_source_{c}" for c in identity])


def synthetic_seed_for(source: str, anomaly_type: str, seed: int) -> int:
    payload = json.dumps([seed, source, anomaly_type, 1], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def generate_evaluation_synthetic(
    originals: pd.DataFrame, assignments: dict[str, str], seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples, manifests = [], []
    for source, group in originals.groupby("source_trajectory_id", sort=True):
        user = str(group["user_id"].iloc[0])
        split = assignments[user]
        if split == "train":
            continue
        ordered = group.sort_values("source_point_index", kind="stable").reset_index(drop=True)
        if not ordered["timestamp"].is_monotonic_increasing:
            raise ValueError("Source point order does not match timestamp order.")
        for anomaly_type in ANOMALY_TYPES:
            sample_seed = synthetic_seed_for(str(source), anomaly_type, seed)
            sample, manifest = generate_anomaly_sample(ordered, anomaly_type, seed=sample_seed)
            if len(sample) != len(ordered):
                raise ValueError("Synthetic generation changed source point count.")
            sample["original_trajectory_id"] = ordered["original_trajectory_id"].iloc[0]
            sample["source_point_index"] = ordered["source_point_index"].to_numpy()
            sample["dataset_split"] = split
            validate_numeric_values(sample)
            sample["synthetic_value_valid"] = True
            sample = attach_source_quality(sample, originals)
            manifests.append({**manifest, "user_id": user,
                              "original_trajectory_id": sample["original_trajectory_id"].iloc[0],
                              "dataset_split": split,
                              "source_anomaly_quality_valid_rows": int(
                                  (sample["anomaly_label"].eq(1) & sample["source_quality_valid"]).sum())})
            samples.append(sample)
    if not samples:
        raise ValueError("No eligible Validation/Test trajectories for synthetic data.")
    return pd.concat(samples, ignore_index=True), pd.DataFrame(manifests)


def evaluation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Use each original normal once and only valid synthetic anomaly segments."""
    original = ~frame["is_synthetic"] & frame["anomaly_label"].eq(0)
    anomaly = frame["is_synthetic"] & frame["anomaly_label"].eq(1)
    return frame.loc[(original | anomaly) & frame["source_quality_valid"]
                     & frame["synthetic_value_valid"]].copy().reset_index(drop=True)


def duplicate_fingerprint_report(inputs: pd.DataFrame, assignments: dict[str, str]) -> dict:
    groups = []
    populated = inputs.loc[inputs["content_fingerprint"].ne("")]
    for fingerprint, group in populated.groupby("content_fingerprint", sort=True):
        if len(group) < 2:
            continue
        records = []
        for row in group.to_dict("records"):
            records.append({"user_id": row["user_id"], "trajectory_id": row["trajectory_id"],
                            "eligible_for_dataset": bool(row["eligible_for_dataset"]),
                            "dataset_split": assignments[str(row["user_id"])]
                            if row["eligible_for_dataset"] else "excluded"})
        splits = {r["dataset_split"] for r in records if r["dataset_split"] != "excluded"}
        groups.append({"fingerprint": fingerprint, "trajectories": records,
                       "cross_split": len(splits) > 1})
    return {"duplicate_fingerprint_count": len(groups),
            "duplicate_trajectory_count": sum(len(g["trajectories"]) for g in groups),
            "cross_split_duplicate_fingerprint_count": sum(g["cross_split"] for g in groups),
            "groups": groups,
            "normalization": "latitude/longitude 7 decimal places; elapsed nanoseconds; sorted timestamp",
            "policy": "Report all copies; do not delete. Cross-split copies require review before Stage 5-B."}


def validate_dataset(
    frames: dict[str, pd.DataFrame], originals: pd.DataFrame,
    assignments: dict[str, str],
) -> dict:
    """Fail closed on identity, split, lineage, labels, and numeric corruption."""
    _check_lineage_keys(originals, unique=True)
    identity = originals[["trajectory_id", "user_id", "original_trajectory_id"]].drop_duplicates()
    if identity["trajectory_id"].duplicated().any():
        raise ValueError("Global trajectory ID collision.")
    expected = originals["user_id"].astype(str) + "__" + originals["original_trajectory_id"].astype(str)
    if not originals["trajectory_id"].astype(str).eq(expected).all():
        raise ValueError("Global trajectory identity mismatch.")
    users, sources, samples = {}, {}, {}
    for split in SPLITS:
        frame = frames[split]
        if frame.empty:
            raise ValueError(f"Empty {split} split.")
        _check_lineage_keys(frame)
        validate_numeric_values(frame)
        if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).eq("").any():
            raise ValueError("Missing sample_id.")
        if frame.duplicated(["sample_id"] + LINEAGE_KEY).any():
            raise ValueError("Duplicate lineage key within sample.")
        if not frame["anomaly_label"].isin([0, 1]).all():
            raise ValueError("Invalid anomaly label.")
        for c in ("is_synthetic", "source_quality_valid", "synthetic_value_valid"):
            if frame[c].isna().any() or not frame[c].isin([True, False]).all():
                raise ValueError(f"Invalid boolean: {c}")
        if not frame["synthetic_value_valid"].all():
            raise ValueError("Synthetic value validity contradicts valid numeric data.")
        users[split] = set(frame["user_id"].astype(str))
        sources[split] = set(frame["source_trajectory_id"].astype(str))
        samples[split] = set(frame["sample_id"].astype(str))
    for left, right in itertools.combinations(SPLITS, 2):
        if users[left] & users[right]:
            raise ValueError(f"User overlap: {left}/{right}.")
        if sources[left] & sources[right]:
            raise ValueError(f"Source trajectory overlap: {left}/{right}.")
        if samples[left] & samples[right]:
            raise ValueError(f"Sample overlap: {left}/{right}.")
    for split in SPLITS:
        frame = frames[split]
        if not frame["user_id"].map(assignments).eq(split).all() or not frame["dataset_split"].eq(split).all():
            raise ValueError("Split metadata/assignment mismatch.")
        matched = attach_source_quality(frame, originals)
        for column in QUALITY_FIELDS:
            if not frame[column].reset_index(drop=True).equals(matched[column].reset_index(drop=True)):
                raise ValueError(f"Source quality mismatch: {column}")
        normal = frame.loc[~frame["is_synthetic"]]
        if normal["anomaly_label"].ne(0).any():
            raise ValueError("Original data contains anomaly labels.")
        for sample_id, group in frame.groupby("sample_id", sort=False):
            if group["source_trajectory_id"].nunique() != 1 or group["is_synthetic"].nunique() != 1:
                raise ValueError(f"Sample identity collision: {sample_id}")
        if split == "train":
            if frame["is_synthetic"].any() or frame["anomaly_label"].ne(0).any():
                raise ValueError("Train contains synthetic/anomaly rows.")
            if not frame["source_quality_valid"].all():
                raise ValueError("Train contains invalid source quality.")
        else:
            for source, group in frame.groupby("source_trajectory_id", sort=False):
                original_keys = set(originals.loc[originals["source_trajectory_id"].eq(source), "source_point_index"])
                source_normal = group.loc[~group["is_synthetic"]]
                if set(source_normal["source_point_index"]) != original_keys:
                    raise ValueError("Incomplete original source trajectory.")
                synthetic = group.loc[group["is_synthetic"]]
                if synthetic["sample_id"].nunique() != len(ANOMALY_TYPES):
                    raise ValueError("Expected exactly one sample per anomaly type.")
                if set(synthetic["anomaly_type"]) != set(ANOMALY_TYPES):
                    raise ValueError("Missing synthetic anomaly types.")
                for _, sample in synthetic.groupby("sample_id", sort=False):
                    if sample["anomaly_type"].nunique() != 1 or not sample["anomaly_label"].eq(1).any():
                        raise ValueError("Invalid synthetic sample labels/types.")
                    if set(sample["source_point_index"]) != original_keys:
                        raise ValueError("Incomplete synthetic lineage.")
                    timestamps = sample.sort_values("source_point_index")["timestamp"]
                    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
                        raise ValueError("Invalid synthetic timestamp order.")
            selected = evaluation_rows(frame)
            if selected.empty or set(selected["anomaly_label"]) != {0, 1}:
                raise ValueError("Evaluation requires normal and anomaly rows.")
    if set(MODEL_FEATURE_COLUMNS) & set(ID_COLUMNS + ("source_point_index", "anomaly_label")):
        raise ValueError("Model feature schema includes metadata.")
    return {"user_overlap": 0, "source_trajectory_overlap": 0, "sample_overlap": 0,
            "global_trajectory_id_collision": 0, "missing_lineage_count": 0,
            "duplicate_source_lineage_count": 0, "duplicate_sample_lineage_count": 0,
            "train_synthetic_rows": 0, "train_anomaly_rows": 0,
            "non_finite_feature_count": 0, "metadata_in_model_features": False,
            "lineage_key": LINEAGE_KEY, "structural_checks_passed": True}



def validate_saved_dataset(output_dir: Path) -> dict:
    """Read-only verification of an exported dataset, including byte checksums."""
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    for name, expected in summary["file_checksums"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe checksum file name.")
        actual = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    users = read_dataset_csv(output_dir / "user_split_manifest.csv")
    if users["user_id"].duplicated().any():
        raise ValueError("Duplicate user split manifest entry.")
    validate_user_ids(users["user_id"].tolist())
    assignments = dict(zip(users["user_id"], users["dataset_split"]))
    if assignments != assign_user_splits(users["user_id"].tolist(), summary["split_seed"]):
        raise ValueError("User split manifest does not match split seed.")
    frames = {s: read_dataset_csv(output_dir / f"{s}.csv") for s in SPLITS}
    originals = read_dataset_csv(output_dir / "source_normal.csv")
    report = validate_dataset(frames, originals, assignments)
    for split in ("validation", "test"):
        selected = read_dataset_csv(output_dir / f"{split}_evaluation.csv")
        pd.testing.assert_frame_equal(selected, evaluation_rows(frames[split]), check_dtype=False)
    inputs = read_dataset_csv(output_dir / "input_manifest.csv")
    if "deduplication" in summary:
        before_dedup = inputs.drop(columns=DEDUP_COLUMNS).copy()
        before_dedup["eligible_for_dataset"] = inputs["quality_eligible_for_dataset"]
        before_dedup["exclusion_reason"] = inputs["exclusion_reason"].map(
            lambda reason: ";".join(r for r in reason.split(";") if r != "exact_duplicate_trajectory"))
        annotated, deduplication = deduplicate_trajectory_manifest(before_dedup, assignments)
        pd.testing.assert_frame_equal(inputs, annotated, check_dtype=False)
        if deduplication != summary["deduplication"]:
            raise ValueError("Saved deduplication report does not match manifest.")
        trajectories = read_dataset_csv(output_dir / "trajectory_quality_summary.csv")
        columns = ["trajectory_id", "content_fingerprint", "eligible_for_dataset", "exclusion_reason"] + DEDUP_COLUMNS
        pd.testing.assert_frame_equal(inputs[columns], trajectories[columns], check_dtype=False)
        expected_sources = set(inputs.loc[inputs["eligible_for_dataset"], "trajectory_id"])
        if expected_sources != set(originals["source_trajectory_id"]):
            raise ValueError("Source normal includes excluded duplicates or misses retained trajectories.")
        report.update(deduplication)
    else:
        report.update(duplicate_fingerprint_report(inputs, assignments))
    report["csv_roundtrip_passed"] = True
    report["leading_zero_preserved"] = True
    report["checksums_verified"] = True
    report["ready_for_stage5_b"] = report["cross_split_duplicate_fingerprint_count"] == 0
    return report


def prepare_multiuser_dataset(
    data_root: Path = DEFAULT_DATA_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_id: str = DEFAULT_DATASET_ID,
    user_ids: Sequence[str] = tuple(f"{i:03}" for i in range(20)),
    files_per_user: int = 5, split_seed: int = 42, synthetic_seed: int = 42,
) -> dict:
    started = time.perf_counter()
    if not dataset_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in dataset_id):
        raise ValueError("dataset_id must be a simple directory name.")
    users = validate_user_ids(user_ids)
    output_dir = Path(output_root) / dataset_id
    if output_dir.exists():
        raise FileExistsError(f"Dataset directory already exists: {output_dir}. Use a new dataset_id.")
    featured, inputs = load_multiuser_features(Path(data_root), users, files_per_user)
    validate_numeric_values(featured)
    checked = add_quality_flags(featured)
    trajectories = summarize_trajectories(checked)
    trajectories = inputs[["user_id", "original_trajectory_id", "trajectory_id", "load_status"]].merge(
        trajectories, on="trajectory_id", how="left", validate="one_to_one")
    empty = trajectories["load_status"].eq("all_rows_invalid")
    trajectories.loc[empty, "eligible_for_dataset"] = False
    trajectories.loc[empty, "exclusion_reason"] = "all_rows_invalid"
    for c in ("point_count", "low_quality_count"):
        trajectories[c] = trajectories[c].fillna(0).astype(int)
    trajectories["eligible_for_dataset"] = trajectories["eligible_for_dataset"].astype(bool)
    inputs = inputs.merge(trajectories[["trajectory_id", "eligible_for_dataset", "exclusion_reason"]],
                          on="trajectory_id", validate="one_to_one")
    inputs["eligible_normal_row_count"] = inputs["trajectory_id"].map(
        checked.groupby("trajectory_id")["is_training_eligible"].sum()).fillna(0).astype(int)
    assignments = assign_user_splits(users, split_seed)
    try:
        inputs, deduplication = deduplicate_trajectory_manifest(inputs, assignments)
    except FingerprintLeakageError as exc:
        # Persist the error without producing split CSVs or synthetic samples.
        output_dir.mkdir(parents=True)
        inputs.to_csv(output_dir / "input_manifest.csv", index=False)
        (output_dir / "leakage_report.json").write_text(
            json.dumps(exc.report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    annotation = inputs[["trajectory_id", "content_fingerprint", "eligible_for_dataset",
                         "exclusion_reason"] + DEDUP_COLUMNS + ["eligible_normal_row_count"]]
    trajectories = trajectories.drop(columns=["eligible_for_dataset", "exclusion_reason"]).merge(
        annotation, on="trajectory_id", validate="one_to_one")
    quality_rows = []
    for user in users:
        paths = trajectories.loc[trajectories["user_id"].eq(user)]
        rows = checked.loc[checked["user_id"].eq(user)]
        eligible_ids = set(paths.loc[paths["eligible_for_dataset"], "trajectory_id"])
        eligible_rows = rows.loc[rows["trajectory_id"].isin(eligible_ids)]
        reasons = paths.loc[~paths["eligible_for_dataset"], "exclusion_reason"].value_counts().to_dict()
        quality_rows.append({"user_id": user, "input_trajectory_count": len(paths),
                             "eligible_trajectory_count": len(eligible_ids),
                             "quality_eligible_trajectory_count_before_dedup": int(paths["quality_eligible_for_dataset"].sum()),
                             "excluded_duplicate_trajectory_count": int(paths["is_exact_duplicate"].sum()),
                             "excluded_duplicate_rows": int(paths["duplicate_excluded_rows"].sum()),
                             "excluded_duplicate_normal_rows": int(paths["duplicate_excluded_normal_rows"].sum()),
                             "excluded_trajectory_count": int((~paths["eligible_for_dataset"]).sum()),
                             "normal_quality_row_count": int((~rows["is_low_quality"]).sum()),
                             "low_quality_row_count": int(rows["is_low_quality"].sum()),
                             "eligible_normal_row_count": int(eligible_rows["is_training_eligible"].sum()),
                             "exclusion_reasons": json.dumps(reasons, ensure_ascii=False),
                             **{k.removeprefix("is_") + "_row_count": v for k, v in quality_counts(rows).items()}})
    user_quality = pd.DataFrame(quality_rows)
    if user_quality["quality_eligible_trajectory_count_before_dedup"].eq(0).any():
        raise ValueError("Requested user has no quality-eligible trajectory; revise cohort explicitly, not silently.")
    eligible = checked.loc[checked["trajectory_id"].isin(
        trajectories.loc[trajectories["eligible_for_dataset"], "trajectory_id"])].copy()
    originals = add_normal_metadata(eligible)
    originals["source_quality_valid"] = ~originals["is_low_quality"] & originals["is_training_eligible"]
    originals["synthetic_value_valid"] = True
    originals["dataset_split"] = originals["user_id"].map(assignments)
    synthetic, synthetic_manifest = generate_evaluation_synthetic(originals, assignments, synthetic_seed)
    frames = {}
    for split in SPLITS:
        normal = originals.loc[originals["dataset_split"].eq(split)]
        frames[split] = (normal.loc[normal["source_quality_valid"]].copy() if split == "train"
                         else pd.concat([normal, synthetic.loc[synthetic["dataset_split"].eq(split)]], ignore_index=True))
        frames[split] = frames[split].reset_index(drop=True)
    leakage = validate_dataset(frames, originals, assignments)
    leakage.update(deduplication)
    leakage["deduplication_applied_before_synthetic"] = True
    user_split = user_quality.copy()
    user_split["dataset_split"] = user_split["user_id"].map(assignments)
    eval_frames = {s: evaluation_rows(frames[s]) for s in ("validation", "test")}
    csvs = {"input_manifest": inputs, "user_quality_summary": user_quality,
            "trajectory_quality_summary": trajectories, "user_split_manifest": user_split,
            "synthetic_anomaly_manifest": synthetic_manifest, "quality_checked": checked,
            "source_normal": originals, **frames,
            **{f"{s}_evaluation": f for s, f in eval_frames.items()}}
    output_dir.mkdir(parents=True)
    for name, frame in csvs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        reread = read_dataset_csv(output_dir / f"{name}.csv")
        if len(reread) != len(frame):
            raise ValueError(f"CSV row count mismatch: {name}")
        for column in ID_COLUMNS + ("source_point_index", "anomaly_label"):
            if column in frame:
                if not frame[column].astype("string").reset_index(drop=True).equals(
                        reread[column].astype("string").reset_index(drop=True)):
                    raise ValueError(f"CSV identity/lineage mismatch: {name}/{column}")
        if "timestamp" in frame and not pd.to_datetime(frame["timestamp"]).reset_index(drop=True).equals(
                reread["timestamp"].reset_index(drop=True)):
            raise ValueError(f"CSV timestamp mismatch: {name}")
    roundtrip = {s: read_dataset_csv(output_dir / f"{s}.csv") for s in SPLITS}
    validate_dataset(roundtrip, read_dataset_csv(output_dir / "source_normal.csv"), assignments)
    for split in eval_frames:
        reread = read_dataset_csv(output_dir / f"{split}_evaluation.csv")
        expected = evaluation_rows(roundtrip[split])
        pd.testing.assert_frame_equal(reread, expected, check_dtype=False)
    leakage["csv_roundtrip_passed"] = True
    leakage["leading_zero_preserved"] = True
    split_summary = {}
    for split in SPLITS:
        frame = frames[split]
        selected = frame if split == "train" else eval_frames[split]
        normal = selected.loc[~selected["is_synthetic"]]
        anomalies = selected.loc[selected["is_synthetic"]]
        split_summary[split] = {
            "users": sorted(u for u, s in assignments.items() if s == split),
            "user_count": sum(s == split for s in assignments.values()),
            "trajectory_count": int(frame["source_trajectory_id"].nunique()),
            "stored_rows": len(frame), "normal_original_rows": len(normal),
            "synthetic_sample_count": int(frame.loc[frame["is_synthetic"], "sample_id"].nunique()),
            "synthetic_anomaly_rows": len(anomalies),
            "synthetic_anomaly_rows_before_quality_filter": int(
                (frame["is_synthetic"] & frame["anomaly_label"].eq(1)).sum()),
            "evaluation_rows": len(selected),
            "anomaly_rows_by_type": {t: int(anomalies["anomaly_type"].eq(t).sum()) for t in ANOMALY_TYPES},
        }
    payload = {
        "dataset_id": dataset_id, "stage": "5-A", "user_count": len(users),
        "input_trajectory_count": len(inputs),
        "quality_eligible_trajectory_count_before_dedup": int(trajectories["quality_eligible_for_dataset"].sum()),
        "eligible_trajectory_count": int(trajectories["eligible_for_dataset"].sum()),
        "deduplication": deduplication,
        "excluded_trajectory_count": int((~trajectories["eligible_for_dataset"]).sum()),
        "raw_row_count": int(inputs["raw_row_count"].sum()),
        "cleaned_row_count": len(checked),
        "eligible_normal_row_count": int(originals["source_quality_valid"].sum()),
        "quality": {"low_quality_rows": int(checked["is_low_quality"].sum()), **quality_counts(checked)},
        "splits": split_summary, "split_seed": split_seed, "synthetic_seed": synthetic_seed,
        "synthetic_seed_derivation": "SHA256 JSON [seed, global_source_id, anomaly_type, sample_number] first 8 bytes",
        "files_per_user": files_per_user, "selection_policy": "filename-first; no replacement of excluded paths",
        "quality_criteria": {"min_trajectory_points": MIN_TRAJECTORY_POINTS,
                             "max_low_quality_ratio": MAX_LOW_QUALITY_RATIO},
        "model_feature_columns_for_stage5_b": MODEL_FEATURE_COLUMNS,
        "leakage": {k: v for k, v in leakage.items() if k not in ("groups", "normalization", "policy")},
        "ready_for_stage5_b": leakage["ready_for_stage5_b"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "split_csv_policy": "train: valid originals; validation/test: full originals + full synthetic copies for audit",
        "evaluation_csv_policy": "valid original normal + source-valid synthetic anomaly rows only",
        "limitations": ["Original normal is a research assumption, not verified behavioral ground truth.",
                        "Synthetic labels do not establish real-world safety performance."],
    }
    payload["file_checksums"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(output_dir.glob("*.csv"))}
    payload["csv_bytes"] = sum(p.stat().st_size for p in output_dir.glob("*.csv"))
    payload["elapsed_seconds"] = time.perf_counter() - started
    for name, data in (("leakage_report", leakage), ("duplicate_report", deduplication),
                       ("dataset_summary", payload)):
        (output_dir / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": output_dir, "summary": payload, "leakage": leakage}


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 5-A multi-user dataset preparation (no model training).")
    parser.add_argument("--validate-existing", type=Path, help="Verify an exported dataset without changing files.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--users", nargs="+", default=[f"{i:03}" for i in range(20)])
    parser.add_argument("--files-per-user", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    options = parser.parse_args(args)
    if options.validate_existing is not None:
        report = validate_saved_dataset(options.validate_existing)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["ready_for_stage5_b"]:
            raise SystemExit(2)
        return
    result = prepare_multiuser_dataset(options.data_root, options.output_root, options.dataset_id,
                                       options.users, options.files_per_user,
                                       options.split_seed, options.synthetic_seed)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Saved to: {result['output_dir']}")
    if not result["leakage"]["ready_for_stage5_b"]:
        print("Cross-split duplicate fingerprints require review before Stage 5-B; nothing was deleted.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
