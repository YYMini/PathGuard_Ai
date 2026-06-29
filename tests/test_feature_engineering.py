"""Unit tests for GPS movement feature engineering."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import (
    calculate_bearing,
    calculate_direction_change,
    generate_trajectory_features,
    haversine_distance,
)


def make_frame(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "trajectory_id",
            "timestamp",
            "latitude",
            "longitude",
            "altitude",
        ],
    ).assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))


class DistanceAndDirectionTests(unittest.TestCase):
    def test_same_coordinate_distance_is_zero(self) -> None:
        self.assertAlmostEqual(haversine_distance(37.5, 127.0, 37.5, 127.0), 0.0)

    def test_known_haversine_distance(self) -> None:
        # One degree of longitude at the equator is approximately 111.195 km.
        distance = haversine_distance(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(distance, 111_194.93, delta=1.0)

    def test_cardinal_bearings(self) -> None:
        self.assertAlmostEqual(calculate_bearing(0, 0, 1, 0), 0.0, delta=0.01)
        self.assertAlmostEqual(calculate_bearing(0, 0, 0, 1), 90.0, delta=0.01)
        self.assertAlmostEqual(calculate_bearing(0, 0, -1, 0), 180.0, delta=0.01)
        self.assertAlmostEqual(calculate_bearing(0, 0, 0, -1), 270.0, delta=0.01)

    def test_direction_change_wraps_at_north(self) -> None:
        self.assertAlmostEqual(calculate_direction_change(350.0, 10.0), 20.0)


class FeatureGenerationTests(unittest.TestCase):
    def test_stop_duration_accumulates(self) -> None:
        frame = make_frame(
            [
                ("000", "a", "2024-01-01 00:00:00", 37.0, 127.0, 0),
                ("000", "a", "2024-01-01 00:00:10", 37.0, 127.0, 0),
                ("000", "a", "2024-01-01 00:00:25", 37.0, 127.0, 0),
            ]
        )
        result, _ = generate_trajectory_features(frame)
        self.assertEqual(result["stop_duration_sec"].tolist(), [0.0, 10.0, 25.0])

    def test_stop_duration_resets_after_movement(self) -> None:
        frame = make_frame(
            [
                ("000", "a", "2024-01-01 00:00:00", 37.0, 127.0, 0),
                ("000", "a", "2024-01-01 00:00:10", 37.0, 127.0, 0),
                ("000", "a", "2024-01-01 00:00:20", 37.001, 127.0, 0),
            ]
        )
        result, _ = generate_trajectory_features(frame)
        self.assertEqual(result["stop_duration_sec"].tolist(), [0.0, 10.0, 0.0])

    def test_trajectory_boundary_does_not_connect_routes(self) -> None:
        frame = make_frame(
            [
                ("000", "a", "2024-01-01 00:00:00", 0.0, 0.0, 0),
                ("000", "a", "2024-01-01 00:00:10", 0.0, 0.001, 0),
                ("000", "b", "2024-01-02 00:00:00", 50.0, 100.0, 0),
                ("000", "b", "2024-01-02 00:00:10", 50.001, 100.0, 0),
            ]
        )
        result, _ = generate_trajectory_features(frame)
        first_rows = result.groupby("trajectory_id", sort=False).head(1)
        for column in (
            "time_diff_sec",
            "distance_m",
            "speed_mps",
            "acceleration_mps2",
            "bearing_deg",
            "direction_change_deg",
            "stop_duration_sec",
        ):
            self.assertTrue((first_rows[column] == 0).all(), column)

    def test_result_has_no_nan_or_infinity(self) -> None:
        frame = make_frame(
            [
                ("000", "a", "2024-01-01 00:00:00", 37.0, 127.0, 10),
                ("000", "a", "2024-01-01 00:00:10", 37.0, 127.0, 10),
                ("000", "a", "2024-01-01 00:00:20", 37.001, 127.001, 10),
            ]
        )
        result, _ = generate_trajectory_features(frame)
        numeric = result.select_dtypes(include=[np.number]).to_numpy()
        self.assertFalse(np.isnan(numeric).any())
        self.assertTrue(np.isfinite(numeric).all())

    def test_non_positive_time_interval_is_reported_and_removed(self) -> None:
        frame = make_frame(
            [
                ("000", "a", "2024-01-01 00:00:00", 37.0, 127.0, 0),
                ("000", "a", "2024-01-01 00:00:00", 37.1, 127.1, 0),
                ("000", "a", "2024-01-01 00:00:10", 37.2, 127.2, 0),
            ]
        )
        result, invalid_count = generate_trajectory_features(frame)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(len(result), 2)
        self.assertTrue((result["time_diff_sec"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
