import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.autoencoder import Autoencoder
from src.prepare_dataset import assign_splits, build_split_frames
from src.synthetic_anomalies import generate_synthetic_anomalies
from src.train_autoencoder import (
    FEATURE_COLUMNS,
    TrainingOptions,
    add_bearing_features,
    calculate_threshold,
    filter_evaluation_rows,
    filter_train_rows,
    fit_train_scaler,
    prepare_stage4_data,
    prediction_frame,
    reconstruction_errors,
    run_pipeline,
    transform_with_scaler,
    validate_feature_values,
)
from tests.stage3_helpers import checked_many


class Stage4AutoencoderTests(unittest.TestCase):
    def setUp(self):
        normal = checked_many(count=5, points=120)
        synthetic, _ = generate_synthetic_anomalies(normal, seed=42)
        assignments = assign_splits([f"t{i}" for i in range(5)], 42)
        self.frames = build_split_frames(normal, synthetic, assignments)

    def small_eval_frame(self, low_quality=False, training_eligible=True, synthetic=True):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="10s")
        base = []
        for position, timestamp in enumerate(timestamps):
            base.append({
                "user_id": "000",
                "trajectory_id": "t1",
                "source_trajectory_id": "t1",
                "sample_id": "t1__normal",
                "timestamp": timestamp,
                "is_synthetic": False,
                "anomaly_label": 0,
                "anomaly_type": "normal",
                "is_low_quality": low_quality if position == 1 else False,
                "quality_reason": "long_gap" if low_quality and position == 1 else np.nan,
                "is_training_eligible": training_eligible,
                "time_diff_sec": float(position),
                "distance_m": 1.0,
                "speed_mps": 0.1,
                "acceleration_mps2": 0.0,
                "bearing_deg": 90.0,
                "direction_change_deg": 0.0,
                "stop_duration_sec": 0.0,
            })
        if synthetic:
            for position, timestamp in enumerate(timestamps):
                row = base[position].copy()
                row["trajectory_id"] = "t1"
                row["sample_id"] = "t1__route_deviation__01"
                row["is_synthetic"] = True
                row["anomaly_label"] = 1 if position in (1, 2) else 0
                row["anomaly_type"] = "route_deviation"
                row["is_low_quality"] = np.nan
                row["quality_reason"] = np.nan
                row["is_training_eligible"] = np.nan
                base.append(row)
        return pd.DataFrame(base)

    def test_model_input_output_shape_matches(self):
        model = Autoencoder(input_dim=8)
        values = torch.randn(1, 8)
        self.assertEqual(model(values).shape, values.shape)

    def test_model_handles_batch_input(self):
        model = Autoencoder(input_dim=8)
        values = torch.randn(7, 8)
        self.assertEqual(model(values).shape, values.shape)

    def test_bearing_deg_is_transformed_to_sin_cos(self):
        frame = pd.DataFrame({"bearing_deg": [0.0, 90.0, 180.0]})
        result = add_bearing_features(frame)
        self.assertTrue(np.allclose(result["bearing_sin"], [0.0, 1.0, 0.0], atol=1e-7))
        self.assertTrue(np.allclose(result["bearing_cos"], [1.0, 0.0, -1.0], atol=1e-7))

    def test_train_filter_excludes_synthetic_rows(self):
        frame = self.frames["validation"]
        result = filter_train_rows(
            frame.assign(is_training_eligible=True, is_low_quality=False)
        )
        self.assertFalse(result["is_synthetic"].astype(bool).any())

    def test_train_filter_excludes_anomaly_rows(self):
        frame = self.frames["validation"].assign(is_training_eligible=True, is_low_quality=False)
        result = filter_train_rows(frame)
        self.assertFalse(result["anomaly_label"].eq(1).any())

    def test_evaluation_filter_excludes_synthetic_normal_rows(self):
        result = filter_evaluation_rows(self.frames["validation"])
        self.assertFalse((result["is_synthetic"].astype(bool) & result["anomaly_label"].eq(0)).any())

    def test_evaluation_filter_keeps_original_normal_and_synthetic_anomaly(self):
        result = filter_evaluation_rows(self.frames["validation"])
        self.assertTrue((~result["is_synthetic"].astype(bool) & result["anomaly_label"].eq(0)).any())
        self.assertTrue((result["is_synthetic"].astype(bool) & result["anomaly_label"].eq(1)).any())

    def test_evaluation_filter_excludes_low_quality_original_normal_rows(self):
        result = filter_evaluation_rows(self.small_eval_frame(low_quality=True, synthetic=False), "validation")
        self.assertFalse(result["quality_reason"].eq("long_gap").any())
        self.assertEqual(result.attrs["filter_stats"]["excluded_validation_low_quality_normal_rows"], 1)

    def test_evaluation_filter_keeps_clean_original_normal_rows(self):
        result = filter_evaluation_rows(self.small_eval_frame(low_quality=False, synthetic=False), "validation")
        self.assertEqual(len(result), 3)

    def test_evaluation_filter_handles_string_bool_values(self):
        frame = self.small_eval_frame(low_quality=False, synthetic=False)
        frame["is_synthetic"] = "False"
        frame["is_low_quality"] = "False"
        frame["is_training_eligible"] = "True"
        result = filter_evaluation_rows(frame, "validation")
        self.assertEqual(len(result), 3)

    def test_evaluation_filter_does_not_treat_nan_quality_as_normal(self):
        frame = self.small_eval_frame(low_quality=False, synthetic=False)
        frame["is_low_quality"] = frame["is_low_quality"].astype("object")
        frame.loc[1, "is_low_quality"] = np.nan
        result = filter_evaluation_rows(frame, "validation")
        self.assertEqual(len(result), 2)

    def test_evaluation_filter_excludes_synthetic_anomaly_from_low_quality_source(self):
        result = filter_evaluation_rows(self.small_eval_frame(low_quality=True, synthetic=True), "validation")
        anomalies = result[result["anomaly_label"].eq(1)]
        self.assertEqual(len(anomalies), 1)
        self.assertFalse(anomalies["quality_reason"].eq("long_gap").any())
        self.assertEqual(result.attrs["filter_stats"]["excluded_validation_low_quality_synthetic_anomaly_rows"], 1)

    def test_scaler_is_fit_on_train_only(self):
        prepared = prepare_stage4_data(self.frames)
        scaler, train_values = fit_train_scaler(prepared["train"])
        _ = transform_with_scaler(scaler, prepared["validation"])
        self.assertEqual(train_values.shape, (len(prepared["train"]), len(FEATURE_COLUMNS)))
        self.assertTrue(np.allclose(scaler.mean_, prepared["train"][FEATURE_COLUMNS].mean().to_numpy()))
        self.assertEqual(int(scaler.n_samples_seen_), len(prepared["train"]))

    def test_reconstruction_error_shape_matches_rows(self):
        model = Autoencoder(input_dim=8)
        values = np.random.default_rng(42).normal(size=(11, 8)).astype(np.float32)
        errors = reconstruction_errors(model, values, torch.device("cpu"), batch_size=4)
        self.assertEqual(errors.shape, (11,))

    def test_threshold_uses_validation_normal_only(self):
        errors = np.array([1.0, 2.0, 100.0, 200.0])
        labels = pd.Series([0, 0, 1, 1])
        self.assertEqual(calculate_threshold(errors, labels, 50), 1.5)

    def test_test_data_is_not_needed_for_threshold(self):
        errors = np.array([1.0, 2.0, 100.0])
        labels = pd.Series([0, 0, 1])
        first = calculate_threshold(errors, labels, 95)
        second = calculate_threshold(errors, labels, 95)
        self.assertEqual(first, second)

    def test_prediction_frame_keeps_quality_and_feature_columns(self):
        frame = filter_evaluation_rows(self.small_eval_frame(low_quality=False, synthetic=True), "validation")
        errors = np.arange(len(frame), dtype=float)
        predictions = prediction_frame(frame, errors, threshold=1.0)
        for column in (
            "is_low_quality",
            "quality_reason",
            "is_training_eligible",
            "source_quality_valid",
            "time_diff_sec",
            "distance_m",
            "speed_mps",
            "acceleration_mps2",
            "direction_change_deg",
            "stop_duration_sec",
        ):
            self.assertIn(column, predictions.columns)

    def test_nan_and_inf_validation(self):
        frame = add_bearing_features(self.frames["train"].copy())
        frame.loc[frame.index[0], FEATURE_COLUMNS[0]] = np.inf
        with self.assertRaises(ValueError):
            validate_feature_values(frame, "train")

    def test_pipeline_saves_model_scaler_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            for name, frame in self.frames.items():
                frame.to_csv(data_dir / f"{name}.csv", index=False)
            result = run_pipeline(
                data_dir=data_dir,
                model_dir=root / "models",
                metrics_dir=root / "metrics",
                figure_dir=root / "figures",
                docs_dir=root / "docs",
                options=TrainingOptions(epochs=3, batch_size=32, patience=2, seed=42),
            )
            paths = result["paths"]
            self.assertTrue(paths["model"].is_file())
            self.assertTrue(paths["scaler"].is_file())
            self.assertTrue(paths["validation_predictions"].is_file())
            self.assertTrue(paths["test_predictions"].is_file())
            self.assertTrue(paths["summary"].is_file())
            self.assertTrue(paths["figures"])
            figure_names = {path.name for path in paths["figures"]}
            self.assertIn("reconstruction_error_distribution.png", figure_names)
            self.assertIn("reconstruction_error_distribution_zoom.png", figure_names)
            saved = pd.read_csv(paths["test_predictions"])
            self.assertIn("source_quality_valid", saved.columns)
            self.assertIn("quality_reason", saved.columns)

    def test_same_seed_reproducible_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            for name, frame in self.frames.items():
                frame.to_csv(data_dir / f"{name}.csv", index=False)
            first = run_pipeline(data_dir, root / "m1", root / "o1", root / "f1", root / "d1",
                                 TrainingOptions(epochs=2, batch_size=32, patience=2, seed=7))
            second = run_pipeline(data_dir, root / "m2", root / "o2", root / "f2", root / "d2",
                                  TrainingOptions(epochs=2, batch_size=32, patience=2, seed=7))
            self.assertAlmostEqual(first["threshold"], second["threshold"], places=7)


if __name__ == "__main__":
    unittest.main()
