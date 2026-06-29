"""Unit tests for stage-2 feature visualization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.visualize_features import (
    FIGURE_FILENAMES,
    calculate_trajectory_summary,
    create_feature_figures,
    create_feature_map,
    export_portfolio_results,
    select_trajectory,
    validate_required_columns,
)


def make_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": pd.Series(["000", "000", "000", "000"], dtype="string"),
            "trajectory_id": pd.Series(["a", "a", "a", "b"], dtype="string"),
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:01:20",
                    "2024-01-02 00:00:00",
                ]
            ),
            "latitude": [37.0, 37.0001, 37.0002, 38.0],
            "longitude": [127.0, 127.0001, 127.0002, 128.0],
            "altitude": [10.0, 11.0, 12.0, 20.0],
            "time_diff_sec": [0.0, 10.0, 70.0, 0.0],
            "distance_m": [0.0, 100.0, 200.0, 0.0],
            "speed_mps": [0.0, 10.0, 60.0, 0.0],
            "acceleration_mps2": [0.0, 1.0, -2.0, 0.0],
            "bearing_deg": [0.0, 45.0, 90.0, 0.0],
            "direction_change_deg": [0.0, 45.0, 120.0, 0.0],
            "stop_duration_sec": [0.0, 0.0, 70.0, 0.0],
        }
    )


class SelectionAndSummaryTests(unittest.TestCase):
    def test_existing_trajectory_is_selected(self) -> None:
        selected_id, route = select_trajectory(make_feature_frame(), "a")
        self.assertEqual(selected_id, "a")
        self.assertEqual(len(route), 3)
        self.assertTrue((route["trajectory_id"] == "a").all())

    def test_unknown_trajectory_error_lists_available_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "입력한 trajectory ID: missing") as context:
            select_trajectory(make_feature_frame(), "missing")
        self.assertIn("사용 가능한 trajectory ID 목록", str(context.exception))
        self.assertIn("  - a", str(context.exception))

    def test_missing_required_column_raises_error(self) -> None:
        frame = make_feature_frame().drop(columns="speed_mps")
        with self.assertRaisesRegex(ValueError, "speed_mps"):
            validate_required_columns(frame)

    def test_trajectory_summary_is_exact(self) -> None:
        _, route = select_trajectory(make_feature_frame(), "a")
        summary = calculate_trajectory_summary(route, "a").iloc[0]
        self.assertEqual(summary["gps_point_count"], 3)
        self.assertEqual(summary["duration_sec"], 80.0)
        self.assertEqual(summary["total_distance_m"], 300.0)
        self.assertAlmostEqual(summary["average_speed_mps"], 70.0 / 3.0)
        self.assertEqual(summary["maximum_speed_mps"], 60.0)
        self.assertEqual(summary["minimum_acceleration_mps2"], -2.0)
        self.assertEqual(summary["maximum_acceleration_mps2"], 1.0)
        self.assertEqual(summary["speed_over_50_count"], 1)
        self.assertEqual(summary["stop_over_60_count"], 1)


class OutputTests(unittest.TestCase):
    def test_png_files_are_created_in_temporary_directory(self) -> None:
        _, route = select_trajectory(make_feature_frame(), "a")
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = create_feature_figures(route, "a", Path(temporary_directory))
            self.assertEqual([path.name for path in paths], FIGURE_FILENAMES)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))

    def test_feature_map_html_is_created(self) -> None:
        _, route = select_trajectory(make_feature_frame(), "a")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "feature_route_a.html"
            result = create_feature_map(route, "a", output_path)
            self.assertTrue(result.is_file())
            content = result.read_text(encoding="utf-8")
            self.assertIn("Trajectory a", content)
            self.assertIn("고속도 확인", content)

    def test_portfolio_export_copies_png_files(self) -> None:
        _, route = select_trajectory(make_feature_frame(), "a")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated = create_feature_figures(route, "a", root / "generated")
            exported = export_portfolio_results(generated, "a", root / "portfolio")
            self.assertEqual(len(exported), 5)
            self.assertTrue(all(path.is_file() for path in exported))

    def test_visualization_does_not_modify_source_dataframe(self) -> None:
        _, route = select_trajectory(make_feature_frame(), "a")
        original = route.copy(deep=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            create_feature_figures(route, "a", root / "figures")
            create_feature_map(route, "a", root / "map.html")
            calculate_trajectory_summary(route, "a")
        assert_frame_equal(route, original)


if __name__ == "__main__":
    unittest.main()
