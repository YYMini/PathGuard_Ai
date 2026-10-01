"""Train and evaluate the Stage 4 PyTorch autoencoder anomaly detector."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import Autoencoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = ROOT / "models" / "stage4"
DEFAULT_METRICS_DIR = ROOT / "outputs" / "metrics"
DEFAULT_FIGURE_DIR = ROOT / "outputs" / "figures" / "stage4"
DEFAULT_DOCS_DIR = ROOT / "docs" / "images" / "stage4"

BASE_FEATURE_COLUMNS = [
    "time_diff_sec",
    "distance_m",
    "speed_mps",
    "acceleration_mps2",
    "direction_change_deg",
    "stop_duration_sec",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ["bearing_sin", "bearing_cos"]
RESULT_ID_COLUMNS = [
    "user_id",
    "trajectory_id",
    "source_trajectory_id",
    "sample_id",
    "timestamp",
    "anomaly_label",
    "anomaly_type",
]
QUALITY_COLUMNS = ["is_low_quality", "quality_reason", "is_training_eligible", "source_quality_valid"]
PREDICTION_FEATURE_COLUMNS = [
    "time_diff_sec",
    "distance_m",
    "speed_mps",
    "acceleration_mps2",
    "direction_change_deg",
    "stop_duration_sec",
]
MODEL_ARCHITECTURE = {
    "input_dim": 8,
    "encoder": [8, 16, 8, 4],
    "decoder": [4, 8, 16, 8],
    "activation": "ReLU",
    "loss": "MSELoss",
    "optimizer": "Adam",
}


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 15
    threshold_percentile: float = 95.0
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_csvs(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    frames = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 3 split file is missing: {path}")
        frames[split] = pd.read_csv(path)
    return frames


def add_bearing_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "bearing_deg" not in frame.columns:
        raise ValueError("Missing required column for bearing transform: bearing_deg")
    result = frame.copy()
    radians = np.radians(pd.to_numeric(result["bearing_deg"], errors="coerce"))
    result["bearing_sin"] = np.sin(radians)
    result["bearing_cos"] = np.cos(radians)
    return result


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Return True/False/<NA> without treating missing or unknown values as False."""
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result[series == True] = True
    result[series == False] = False
    strings = series.astype("string").str.strip().str.lower()
    result[strings.isin(["true", "1", "yes"])] = True
    result[strings.isin(["false", "0", "no"])] = False
    return result


def bool_is_true(series: pd.Series) -> pd.Series:
    return parse_bool_series(series).eq(True).fillna(False)


def bool_is_false(series: pd.Series) -> pd.Series:
    return parse_bool_series(series).eq(False).fillna(False)


def filter_train_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["anomaly_label", "is_synthetic", "is_training_eligible", "is_low_quality"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Train data is missing required columns: {', '.join(missing)}")
    mask = (
        frame["anomaly_label"].eq(0)
        & bool_is_false(frame["is_synthetic"])
        & bool_is_true(frame["is_training_eligible"])
        & bool_is_false(frame["is_low_quality"])
    )
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError("Train filter produced no rows.")
    if selected["anomaly_label"].max() != 0:
        raise ValueError("Train data contains anomaly_label other than 0 after filtering.")
    if bool_is_true(selected["is_synthetic"]).any():
        raise ValueError("Train data contains synthetic rows after filtering.")
    return selected


def _attach_source_quality(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["source_trajectory_id", "sample_id", "timestamp", "is_synthetic", "anomaly_label",
                "is_low_quality", "quality_reason", "is_training_eligible"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Evaluation data is missing required columns: {', '.join(missing)}")
    result = frame.copy()
    result["_row_order"] = np.arange(len(result))
    result["_is_synthetic_bool"] = bool_is_true(result["is_synthetic"])
    original = result.loc[~result["_is_synthetic_bool"] & result["anomaly_label"].eq(0)].copy()
    if original.empty:
        raise ValueError("Evaluation data has no original normal rows for source quality matching.")
    if original.duplicated(["source_trajectory_id", "timestamp"]).any():
        duplicates = original.loc[original.duplicated(["source_trajectory_id", "timestamp"], keep=False),
                                  ["source_trajectory_id", "timestamp"]].drop_duplicates()
        raise ValueError("Original normal rows are not unique by source_trajectory_id/timestamp: "
                         + duplicates.head().to_dict("records").__repr__())

    original = original.sort_values(["source_trajectory_id", "timestamp"], kind="stable")
    original["_source_position"] = original.groupby("source_trajectory_id", sort=False).cumcount()
    source_quality = original[[
        "source_trajectory_id",
        "_source_position",
        "is_low_quality",
        "quality_reason",
        "is_training_eligible",
    ]].rename(columns={
        "is_low_quality": "_source_is_low_quality",
        "quality_reason": "_source_quality_reason",
        "is_training_eligible": "_source_is_training_eligible",
    })
    source_counts = original.groupby("source_trajectory_id").size()
    synthetic = result.loc[result["_is_synthetic_bool"]].copy()
    if not synthetic.empty:
        if synthetic["sample_id"].isna().any():
            raise ValueError("Synthetic evaluation rows contain missing sample_id values.")
        synthetic = synthetic.sort_values(["source_trajectory_id", "sample_id", "timestamp"], kind="stable")
        synthetic["_source_position"] = synthetic.groupby(["source_trajectory_id", "sample_id"], sort=False).cumcount()
        sample_counts = synthetic.groupby(["source_trajectory_id", "sample_id"]).size()
        bad_counts = []
        for (source, sample), count in sample_counts.items():
            expected = source_counts.get(source)
            if expected is None or int(expected) != int(count):
                bad_counts.append({"source_trajectory_id": source, "sample_id": sample,
                                   "synthetic_rows": int(count),
                                   "source_rows": None if expected is None else int(expected)})
        if bad_counts:
            raise ValueError("Synthetic rows cannot be safely matched to original source rows by position: "
                             + repr(bad_counts[:5]))
        merged = synthetic.merge(source_quality, on=["source_trajectory_id", "_source_position"],
                                 how="left", validate="many_to_one")
        if merged[["_source_is_low_quality", "_source_is_training_eligible"]].isna().any().any():
            raise ValueError("Synthetic source quality merge produced missing quality values.")
        positions = merged["_row_order"].to_numpy(dtype=int)
        result.iloc[positions, result.columns.get_loc("is_low_quality")] = merged["_source_is_low_quality"].to_numpy()
        result.iloc[positions, result.columns.get_loc("quality_reason")] = merged["_source_quality_reason"].to_numpy()
        result.iloc[positions, result.columns.get_loc("is_training_eligible")] = merged["_source_is_training_eligible"].to_numpy()

    result["source_quality_valid"] = (
        bool_is_false(result["is_low_quality"]) & bool_is_true(result["is_training_eligible"])
    )
    return result.drop(columns=["_row_order", "_is_synthetic_bool"], errors="ignore")


def filter_evaluation_rows(frame: pd.DataFrame, split_name: str = "evaluation") -> pd.DataFrame:
    frame = _attach_source_quality(frame)
    required = ["anomaly_label", "is_synthetic", "is_low_quality", "is_training_eligible", "source_quality_valid"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Evaluation data is missing required columns: {', '.join(missing)}")
    synthetic_mask = bool_is_true(frame["is_synthetic"])
    source_quality_valid = bool_is_true(frame["source_quality_valid"])
    original_normal_candidate = ~synthetic_mask & frame["anomaly_label"].eq(0)
    synthetic_anomaly_candidate = synthetic_mask & frame["anomaly_label"].eq(1)
    original_normal = original_normal_candidate & source_quality_valid
    synthetic_anomaly = synthetic_anomaly_candidate & source_quality_valid
    selected = frame.loc[original_normal | synthetic_anomaly].copy()
    if selected.empty:
        raise ValueError("Evaluation filter produced no rows.")
    if (bool_is_true(selected["is_synthetic"]) & selected["anomaly_label"].eq(0)).any():
        raise ValueError("Evaluation rows include synthetic normal rows.")
    selected.attrs["filter_stats"] = {
        f"excluded_{split_name}_low_quality_normal_rows": int((original_normal_candidate & ~original_normal).sum()),
        f"excluded_{split_name}_low_quality_synthetic_anomaly_rows": int((synthetic_anomaly_candidate & ~synthetic_anomaly).sum()),
        f"final_{split_name}_normal_rows": int(selected["anomaly_label"].eq(0).sum()),
        f"final_{split_name}_anomaly_rows": int(selected["anomaly_label"].eq(1).sum()),
    }
    return selected


def validate_feature_values(frame: pd.DataFrame, split_name: str) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{split_name} is missing model feature columns: {', '.join(missing)}")
    numeric = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = []
        for column in FEATURE_COLUMNS:
            column_values = numeric[column].to_numpy(dtype=float)
            if not np.isfinite(column_values).all():
                bad.append(column)
        raise ValueError(f"{split_name} has NaN or inf in model features: {', '.join(bad)}")


def prepare_stage4_data(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prepared = {name: add_bearing_features(frame) for name, frame in frames.items()}
    result = {
        "train": filter_train_rows(prepared["train"]),
        "validation": filter_evaluation_rows(prepared["validation"], "validation"),
        "test": filter_evaluation_rows(prepared["test"], "test"),
    }
    for split, frame in result.items():
        validate_feature_values(frame, split)
    return result


def fit_train_scaler(train: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    values = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return scaler.fit(values), scaler.transform(values).astype(np.float32)


def transform_with_scaler(scaler: StandardScaler, frame: pd.DataFrame) -> np.ndarray:
    return scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)).astype(np.float32)


def make_loader(values: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.tensor(values, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: Autoencoder,
    train_values: np.ndarray,
    validation_normal_values: np.ndarray,
    options: TrainingOptions,
    device: torch.device,
    model_path: Path,
) -> tuple[pd.DataFrame, int]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    train_loader = make_loader(train_values, options.batch_size, True)
    validation_tensor = torch.tensor(validation_normal_values, dtype=torch.float32, device=device)
    best_loss = math.inf
    best_epoch = 0
    patience_count = 0
    history = []
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, options.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_rows += len(batch)
        train_loss = total_loss / max(total_rows, 1)

        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_tensor), validation_tensor).item())

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_normal_loss": validation_loss,
            "best_loss": best_loss,
            "patience": patience_count,
        })
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"validation_normal_loss={validation_loss:.6f} best_loss={best_loss:.6f} "
            f"patience={patience_count}/{options.patience}"
        )
        if patience_count >= options.patience:
            break
    return pd.DataFrame(history), best_epoch


def reconstruction_errors(model: Autoencoder, values: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    errors = []
    loader = make_loader(values, batch_size, False)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            errors.append(torch.mean((batch - reconstructed) ** 2, dim=1).cpu().numpy())
    return np.concatenate(errors) if errors else np.array([], dtype=float)


def calculate_threshold(errors: np.ndarray, labels: pd.Series, percentile: float) -> float:
    normal_errors = errors[labels.to_numpy(dtype=int) == 0]
    if len(normal_errors) == 0:
        raise ValueError("No validation normal rows are available for threshold calculation.")
    return float(np.percentile(normal_errors, percentile))


def prediction_frame(frame: pd.DataFrame, errors: np.ndarray, threshold: float) -> pd.DataFrame:
    requested = RESULT_ID_COLUMNS + QUALITY_COLUMNS + PREDICTION_FEATURE_COLUMNS
    columns = [column for column in requested if column in frame.columns]
    result = frame[columns].copy()
    result["reconstruction_error"] = errors
    result["predicted_anomaly"] = (errors > threshold).astype(int)
    return result


def calculate_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int | None]:
    y_true = frame["anomaly_label"].astype(int).to_numpy()
    scores = frame["reconstruction_error"].to_numpy(dtype=float)
    y_pred = frame["predicted_anomaly"].astype(int).to_numpy()
    labels_present = set(y_true.tolist())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(y_true, scores)) if len(labels_present) == 2 else None
    pr_auc = float(average_precision_score(y_true, scores)) if len(labels_present) == 2 else None
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": pr_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(threshold),
        "normal_error_mean": float(np.mean(normal_scores)) if len(normal_scores) else None,
        "anomaly_error_mean": float(np.mean(anomaly_scores)) if len(anomaly_scores) else None,
        "row_count": int(len(frame)),
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_training_loss(history: pd.DataFrame, best_epoch: int, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["validation_normal_loss"], label="validation normal")
    plt.axvline(best_epoch, color="black", linestyle="--", label=f"best epoch {best_epoch}")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def _plot_error_distribution(predictions: pd.DataFrame, threshold: float, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    transformed_threshold = np.log10(1.0 + threshold)
    for label, name in ((0, "normal"), (1, "synthetic anomaly")):
        values = predictions.loc[predictions["anomaly_label"].eq(label), "reconstruction_error"]
        plt.hist(np.log10(1.0 + values.to_numpy(dtype=float)), bins=40, alpha=0.6, label=name)
    plt.axvline(transformed_threshold, color="black", linestyle="--",
                label=f"threshold log10(1+x)={transformed_threshold:.4f}")
    plt.xlabel("log10(1 + reconstruction error)")
    plt.ylabel("Rows")
    plt.title("Test reconstruction error distribution (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def _plot_error_distribution_zoom(predictions: pd.DataFrame, threshold: float, path: Path) -> None:
    scores = predictions["reconstruction_error"].to_numpy(dtype=float)
    upper = float(np.percentile(scores, 99))
    outside_count = int((scores > upper).sum())
    visible = predictions[predictions["reconstruction_error"].le(upper)]
    plt.figure(figsize=(8, 5))
    for label, name in ((0, "normal"), (1, "synthetic anomaly")):
        values = visible.loc[visible["anomaly_label"].eq(label), "reconstruction_error"]
        plt.hist(values, bins=40, alpha=0.6, label=name)
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.6f}")
    plt.xlim(left=0, right=upper)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Rows")
    plt.title("Test reconstruction error distribution zoomed to 99th percentile")
    plt.text(0.98, 0.95, f"Outside 99th percentile: {outside_count}",
             transform=plt.gca().transAxes, ha="right", va="top",
             bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def _plot_confusion(predictions: pd.DataFrame, path: Path) -> None:
    matrix = confusion_matrix(predictions["anomaly_label"], predictions["predicted_anomaly"], labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["normal", "anomaly"]); plt.yticks([0, 1], ["normal", "anomaly"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for y in range(2):
        for x in range(2):
            plt.text(x, y, str(matrix[y, x]), ha="center", va="center", color="black")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_roc(predictions: pd.DataFrame, path: Path) -> None:
    y_true = predictions["anomaly_label"].astype(int)
    if y_true.nunique() < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, predictions["reconstruction_error"])
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def _plot_pr(predictions: pd.DataFrame, path: Path) -> None:
    y_true = predictions["anomaly_label"].astype(int)
    if y_true.nunique() < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, predictions["reconstruction_error"])
    plt.figure(figsize=(6, 5)); plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def _plot_by_type(predictions: pd.DataFrame, path: Path) -> None:
    values = predictions.copy()
    values["display_type"] = np.where(values["anomaly_label"].eq(0), "normal", values.get("anomaly_type", "anomaly"))
    order = ["normal", "route_deviation", "abnormal_speed", "long_stop", "direction_change"]
    grouped = [values.loc[values["display_type"].eq(name), "reconstruction_error"] for name in order]
    grouped = [series for series in grouped if len(series)]
    labels = [name for name in order if values["display_type"].eq(name).any()]
    plt.figure(figsize=(8, 5)); plt.boxplot(grouped, tick_labels=labels, showfliers=False)
    plt.ylabel("Reconstruction error"); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def create_figures(history: pd.DataFrame, validation_predictions: pd.DataFrame, test_predictions: pd.DataFrame,
                   best_epoch: int, threshold: float, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_dir / "training_loss.png",
        figure_dir / "reconstruction_error_distribution.png",
        figure_dir / "reconstruction_error_distribution_zoom.png",
        figure_dir / "confusion_matrix.png",
        figure_dir / "roc_curve.png",
        figure_dir / "precision_recall_curve.png",
        figure_dir / "anomaly_score_by_type.png",
    ]
    _plot_training_loss(history, best_epoch, paths[0])
    _plot_error_distribution(test_predictions, threshold, paths[1])
    _plot_error_distribution_zoom(test_predictions, threshold, paths[2])
    _plot_confusion(test_predictions, paths[3])
    _plot_roc(test_predictions, paths[4])
    _plot_pr(test_predictions, paths[5])
    _plot_by_type(test_predictions, paths[6])
    return [path for path in paths if path.exists()]


def copy_portfolio_outputs(figure_paths: list[Path], summary_path: Path, docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for path in figure_paths:
        shutil.copy2(path, docs_dir / path.name)
    if summary_path.exists():
        shutil.copy2(summary_path, docs_dir / summary_path.name)


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    options: TrainingOptions = TrainingOptions(),
) -> dict[str, object]:
    set_seed(options.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    frames = prepare_stage4_data(load_split_csvs(data_dir))
    filter_stats = {}
    filter_stats.update(frames["validation"].attrs.get("filter_stats", {}))
    filter_stats.update(frames["test"].attrs.get("filter_stats", {}))
    scaler, train_values = fit_train_scaler(frames["train"])
    validation_values = transform_with_scaler(scaler, frames["validation"])
    test_values = transform_with_scaler(scaler, frames["test"])
    validation_normal_values = validation_values[frames["validation"]["anomaly_label"].to_numpy(dtype=int) == 0]
    if len(validation_normal_values) == 0:
        raise ValueError("Validation split has no original normal rows for early stopping.")

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pathguard_autoencoder.pt"
    model = Autoencoder(len(FEATURE_COLUMNS)).to(device)
    history, best_epoch = train_model(model, train_values, validation_normal_values, options, device, model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    validation_errors = reconstruction_errors(model, validation_values, device)
    threshold = calculate_threshold(validation_errors, frames["validation"]["anomaly_label"], options.threshold_percentile)
    test_errors = reconstruction_errors(model, test_values, device)
    validation_predictions = prediction_frame(frames["validation"], validation_errors, threshold)
    test_predictions = prediction_frame(frames["test"], test_errors, threshold)
    validation_metrics = calculate_metrics(validation_predictions, threshold)
    test_metrics = calculate_metrics(test_predictions, threshold)

    scaler_path = model_dir / "scaler.joblib"
    feature_path = model_dir / "feature_columns.json"
    config_path = model_dir / "training_config.json"
    joblib.dump(scaler, scaler_path)
    save_json(feature_path, {"feature_columns": FEATURE_COLUMNS})
    history_path = metrics_dir / "stage4_training_history.csv"
    validation_metrics_path = metrics_dir / "stage4_validation_metrics.json"
    test_metrics_path = metrics_dir / "stage4_test_metrics.json"
    validation_predictions_path = metrics_dir / "stage4_validation_predictions.csv"
    test_predictions_path = metrics_dir / "stage4_test_predictions.csv"
    summary_path = metrics_dir / "stage4_summary.csv"
    history.to_csv(history_path, index=False)
    save_json(validation_metrics_path, validation_metrics)
    save_json(test_metrics_path, test_metrics)
    validation_predictions.to_csv(validation_predictions_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)
    summary = pd.DataFrame([{"split": "validation", **validation_metrics}, {"split": "test", **test_metrics}])
    summary.to_csv(summary_path, index=False)

    config = {
        "feature_columns": FEATURE_COLUMNS,
        "model_architecture": MODEL_ARCHITECTURE,
        **asdict(options),
        "actual_epochs": int(history["epoch"].max()),
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "device": str(device),
        "train_rows": int(len(frames["train"])),
        "validation_eval_rows": int(len(frames["validation"])),
        "test_eval_rows": int(len(frames["test"])),
        **filter_stats,
    }
    save_json(config_path, config)
    figure_paths = create_figures(history, validation_predictions, test_predictions, best_epoch, threshold, figure_dir)
    copy_portfolio_outputs(figure_paths, summary_path, docs_dir)

    print("\n=== Stage 4 training summary ===")
    print(f"Device: {device}")
    print(f"Train rows: {len(frames['train']):,}")
    print(f"Validation rows: {len(frames['validation']):,}")
    print(f"Test rows: {len(frames['test']):,}")
    for key, value in filter_stats.items():
        print(f"{key}: {value:,}")
    print(f"Best epoch: {best_epoch}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Validation F1: {validation_metrics['f1_score']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']}")
    print(f"Test PR-AUC: {test_metrics['average_precision']}")
    print(f"Model path: {model_path}")
    print(f"Figure directory: {figure_dir}")
    return {
        "frames": frames,
        "history": history,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "paths": {
            "model": model_path,
            "scaler": scaler_path,
            "training_config": config_path,
            "feature_columns": feature_path,
            "history": history_path,
            "validation_metrics": validation_metrics_path,
            "test_metrics": test_metrics_path,
            "validation_predictions": validation_predictions_path,
            "test_predictions": test_predictions_path,
            "summary": summary_path,
            "figures": figure_paths,
        },
    }


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 4 PathGuard autoencoder.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    options = TrainingOptions(
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        learning_rate=parsed.learning_rate,
        patience=parsed.patience,
        threshold_percentile=parsed.threshold_percentile,
        seed=parsed.seed,
    )
    try:
        run_pipeline(parsed.data_dir, parsed.model_dir, parsed.metrics_dir, parsed.figure_dir, parsed.docs_dir, options)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
