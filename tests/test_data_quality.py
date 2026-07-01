import unittest
from pandas.testing import assert_frame_equal
from src.data_quality import add_quality_flags
from tests.stage3_helpers import trajectory

class DataQualityTests(unittest.TestCase):
    def setUp(self): self.frame = trajectory(points=12)
    def test_long_gap(self):
        self.frame.loc[3, "time_diff_sec"] = 301; self.assertTrue(add_quality_flags(self.frame).loc[3, "is_long_gap"])
    def test_unrealistic_speed(self):
        self.frame.loc[3, "speed_mps"] = 51; self.assertTrue(add_quality_flags(self.frame).loc[3, "is_unrealistic_speed"])
    def test_gps_jump(self):
        self.frame.loc[3, ["distance_m", "time_diff_sec"]] = [1001, 60]; self.assertTrue(add_quality_flags(self.frame).loc[3, "is_gps_jump"])
    def test_reasons_are_combined(self):
        self.frame.loc[3, ["distance_m", "time_diff_sec", "speed_mps"]] = [1001, 301, 51]
        self.assertEqual(add_quality_flags(self.frame).loc[3, "quality_reason"], "long_gap;unrealistic_speed")
    def test_rows_and_input_unchanged(self):
        before = self.frame.copy(); result = add_quality_flags(self.frame); self.assertEqual(len(result), len(before)); assert_frame_equal(self.frame, before)
    def test_first_row_is_not_invalid_but_not_trainable(self):
        result = add_quality_flags(self.frame); self.assertFalse(result.loc[0, "is_invalid_time"]); self.assertFalse(result.loc[0, "is_training_eligible"])

if __name__ == "__main__": unittest.main()
