import tempfile, unittest
from pathlib import Path
import pandas as pd
from src.prepare_dataset import assign_splits, build_split_frames, prepare_dataset, validate_splits
from src.synthetic_anomalies import generate_synthetic_anomalies
from tests.stage3_helpers import checked_many

class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.normal = checked_many(); self.synthetic, _ = generate_synthetic_anomalies(self.normal, seed=42); self.mapping = assign_splits([f"t{i}" for i in range(5)], 42); self.frames = build_split_frames(self.normal, self.synthetic, self.mapping)
    def test_no_source_overlap(self): validate_splits(self.frames)
    def test_train_is_normal_only(self): self.assertFalse(self.frames["train"].is_synthetic.any()); self.assertFalse(self.frames["train"].anomaly_label.any())
    def test_validation_and_test_have_both(self):
        for name in ("validation", "test"): self.assertEqual(set(self.frames[name].is_synthetic), {False, True})
    def test_sample_ids_do_not_cross_splits(self): validate_splits(self.frames)
    def test_split_is_reproducible(self): self.assertEqual(self.mapping, assign_splits([f"t{i}" for i in range(5)], 42))
    def test_outputs_are_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); source = path / "input.csv"; self.normal.drop(columns=[c for c in self.normal if c.startswith("is_") or c == "quality_reason"]).to_csv(source, index=False)
            result = prepare_dataset(source, path / "out", seed=42)
            for output in result["paths"]: self.assertIsInstance(pd.read_csv(output), pd.DataFrame)

if __name__ == "__main__": unittest.main()
