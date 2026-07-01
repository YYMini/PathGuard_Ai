import unittest
import numpy as np
from pandas.testing import assert_frame_equal
from src.synthetic_anomalies import generate_anomaly_sample
from tests.stage3_helpers import trajectory

class SyntheticAnomalyTests(unittest.TestCase):
    def setUp(self): self.frame = trajectory(points=150)
    def test_reproducible_and_input_unchanged(self):
        before = self.frame.copy(); a, _ = generate_anomaly_sample(self.frame, "route_deviation", seed=7); b, _ = generate_anomaly_sample(self.frame, "route_deviation", seed=7)
        assert_frame_equal(a, b); assert_frame_equal(self.frame, before)
    def test_route_deviation_changes_coordinates(self):
        result, _ = generate_anomaly_sample(self.frame, "route_deviation", seed=7); mask = result.anomaly_label.eq(1)
        self.assertGreater(np.abs(result.loc[mask, "latitude"].to_numpy() - self.frame.loc[mask, "latitude"].to_numpy()).max(), .0001)
    def test_abnormal_speed_is_below_quality_threshold(self):
        result, _ = generate_anomaly_sample(self.frame, "abnormal_speed", seed=7); self.assertLess(result.loc[result.anomaly_label.eq(1), "speed_mps"].max(), 50)
    def test_long_stop_increases_stop_duration(self):
        result, _ = generate_anomaly_sample(self.frame, "long_stop", seed=7); self.assertGreater(result.stop_duration_sec.max(), self.frame.stop_duration_sec.max())
    def test_direction_change_increases(self):
        result, _ = generate_anomaly_sample(self.frame, "direction_change", seed=7); self.assertGreater(result.direction_change_deg.mean(), self.frame.direction_change_deg.mean())
    def test_only_segment_is_labeled(self):
        result, manifest = generate_anomaly_sample(self.frame, "long_stop", seed=7); self.assertEqual(int(result.anomaly_label.sum()), manifest["anomaly_point_count"]); self.assertTrue(result.anomaly_label.eq(0).any())
    def test_features_are_finite(self):
        result, _ = generate_anomaly_sample(self.frame, "direction_change", seed=7); self.assertTrue(np.isfinite(result.select_dtypes(include=np.number)).all().all())

if __name__ == "__main__": unittest.main()
