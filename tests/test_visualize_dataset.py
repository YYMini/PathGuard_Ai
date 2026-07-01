import tempfile, unittest
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
from src.prepare_dataset import prepare_dataset
from src.visualize_dataset import PNG_NAMES, create_visualizations
from tests.stage3_helpers import checked_many

class VisualizeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); source = self.root / "input.csv"
        checked_many().iloc[:, :13].to_csv(source, index=False); prepare_dataset(source, self.root / "data")
    def tearDown(self): self.tmp.cleanup()
    def test_png_and_html_are_created(self):
        paths = create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        self.assertTrue(all((self.root / "figures" / name).is_file() for name in PNG_NAMES)); self.assertTrue((self.root / "map.html").is_file())
    def test_portfolio_is_created(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html", True, self.root / "portfolio")
        self.assertTrue((self.root / "portfolio" / "dataset_summary.csv").is_file())
    def test_folium_map_contains_layer_control(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        html = (self.root / "map.html").read_text(encoding="utf-8")
        self.assertIn("L.control.layers", html)
        for layer_name in ("정상 경로", "경로 이탈", "비정상 속도", "장시간 정지", "급격한 방향 변화"):
            self.assertIn(layer_name, html)
        self.assertIn("합성 이동 경로 비교", html)
        self.assertIn("실제 위험 판정 결과가 아닙니다", html)
    def test_anomaly_segment_coordinates_are_in_html(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        html = (self.root / "map.html").read_text(encoding="utf-8")
        synthetic = pd.read_csv(self.root / "data" / "synthetic_anomalies.csv")
        source = synthetic.iloc[0]["source_trajectory_id"]
        for anomaly_type in ("route_deviation", "abnormal_speed", "long_stop", "direction_change"):
            row = synthetic[(synthetic.source_trajectory_id == source) & (synthetic.anomaly_type == anomaly_type) & synthetic.anomaly_label.eq(1)].iloc[0]
            self.assertIn(str(round(float(row.latitude), 6)), html)
    def test_start_and_end_markers_are_created(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        html = (self.root / "map.html").read_text(encoding="utf-8")
        self.assertIn("이상 구간 시작", html); self.assertIn("이상 구간 종료", html)
        self.assertIn("이상 유형", html); self.assertIn("샘플 ID", html); self.assertIn("변형 포인트 수", html)
    def test_long_stop_circle_marker_is_created(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        html = (self.root / "map.html").read_text(encoding="utf-8")
        self.assertIn("장시간 정지 위치", html); self.assertIn("누적 정지 시간", html); self.assertIn("L.circleMarker", html)
    def test_direction_change_point_markers_are_created(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        html = (self.root / "map.html").read_text(encoding="utf-8")
        self.assertIn("방향 변화 지점", html); self.assertIn("방향 변화량", html)
    def test_abnormal_speed_statistics_are_in_html(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        html = (self.root / "map.html").read_text(encoding="utf-8")
        self.assertIn("비정상 속도 구간", html); self.assertIn("평균 속도=", html); self.assertIn("최대 속도=", html)
    def test_normal_and_anomaly_styles_are_distinct(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        html = (self.root / "map.html").read_text(encoding="utf-8")
        self.assertIn('"weight": 5', html); self.assertIn('"opacity": 0.9', html)
        self.assertIn('"weight": 8', html); self.assertIn('"opacity": 0.82', html)
        self.assertIn('"dashArray": "8 6"', html)
    def test_four_detail_maps_are_created(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html")
        for anomaly_type in ("route_deviation", "abnormal_speed", "long_stop", "direction_change"):
            detail = self.root / f"stage3_{anomaly_type}_detail.html"
            self.assertTrue(detail.is_file()); self.assertIn("fitBounds", detail.read_text(encoding="utf-8"))
    def test_portfolio_contains_all_png_files(self):
        create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html", True, self.root / "portfolio")
        self.assertTrue(all((self.root / "portfolio" / name).is_file() for name in PNG_NAMES))
    def test_source_data_is_unchanged(self):
        before = pd.read_csv(self.root / "data" / "quality_checked.csv"); create_visualizations(self.root / "data", self.root / "figures", self.root / "map.html"); after = pd.read_csv(self.root / "data" / "quality_checked.csv"); assert_frame_equal(before, after)

if __name__ == "__main__": unittest.main()
