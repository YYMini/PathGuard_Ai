"""Demo contract tests; artificial analytic example only, never observed GPS or the final cohort."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import run_pathguard_demo as demo
from src import stage7_protocol as p

class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example=p.ROOT/'examples/demo_trajectory.csv'
        cls.raw=pd.read_csv(cls.example,dtype={'user_id':str,'trajectory_id':str})
    def test_valid_csv(self):
        self.assertEqual(len(demo.validate_input(self.raw)),200)
    def test_missing_columns(self):
        with self.assertRaises(ValueError):demo.validate_input(self.raw.drop(columns=['latitude']))
    def test_nan_rejected(self):
        f=self.raw.copy();f.loc[0,'longitude']=np.nan
        with self.assertRaises(ValueError):demo.validate_input(f)
    def test_infinite_rejected(self):
        f=self.raw.copy();f.loc[0,'latitude']=np.inf
        with self.assertRaises(ValueError):demo.validate_input(f)
    def test_coordinate_range(self):
        f=self.raw.copy();f.loc[0,'latitude']=100
        with self.assertRaises(ValueError):demo.validate_input(f)
    def test_timestamp_reverse(self):
        with self.assertRaises(ValueError):demo.validate_input(self.raw.iloc[::-1])
    def test_duplicate_timestamp(self):
        f=self.raw.copy();f.loc[1,'timestamp']=f.loc[0,'timestamp']
        with self.assertRaises(ValueError):demo.validate_input(f)
    def test_invalid_timestamp(self):
        f=self.raw.copy();f.loc[0,'timestamp']='not a time'
        with self.assertRaises(ValueError):demo.validate_input(f)
    def test_empty_csv(self):
        with self.assertRaises(ValueError):demo.validate_input(self.raw.iloc[:0])
    def test_default_optional_fields(self):
        f=demo.validate_input(self.raw[['timestamp','latitude','longitude']])
        self.assertTrue(f.user_id.eq('demo').all());self.assertTrue(f.altitude.eq(0).all())
    def test_all_frozen_detector_loads(self):
        for detector in ['rule','if','ae']:
            m=demo.load_demo_detector(detector,42)
            self.assertEqual(m['record']['feature_list'],p.training.FEATURE_COLUMNS)
            self.assertEqual(m['scaler'].n_features_in_,8) if detector!='rule' else self.assertIsNone(m['scaler'])
    def test_unknown_seed_rejected(self):
        with self.assertRaises(ValueError):demo.load_demo_detector('ae',1234)
    def test_unknown_detector_rejected(self):
        with self.assertRaises(ValueError):demo.load_demo_detector('new-model',42)
    def test_feature_shape_predictions_and_alerts(self):
        for detector in ['rule','if','ae']:
            out,_=demo.predict_csv(self.example,detector,42)
            self.assertEqual(len(out),200);self.assertIn('anomaly_score',out)
            valid=out.loc[out.quality_valid_for_inference]
            self.assertTrue(np.isfinite(valid.anomaly_score).all())
            self.assertTrue(valid.point_prediction.isin([0,1]).all())
            self.assertTrue(out.notification.isin([True,False]).all())
    def test_unknown_quality_not_silently_normal(self):
        out,_=demo.predict_csv(self.example,'ae',42)
        self.assertEqual(out.prediction_status.iloc[0],'quality_unknown')
        self.assertTrue(pd.isna(out.point_prediction.iloc[0]))
        self.assertFalse(out.notification.iloc[0])
    def test_deterministic_predictions(self):
        a,_=demo.predict_csv(self.example,'ae',42);b,_=demo.predict_csv(self.example,'ae',42)
        pd.testing.assert_frame_equal(a,b)
    def test_no_threshold_recomputation(self):
        with patch('numpy.percentile',side_effect=AssertionError('No Final/demo threshold percentile')):
            out,_=demo.predict_csv(self.example,'if',42)
        self.assertEqual(len(out),200)
    def test_model_hash_rejection(self):
        with patch.object(demo,'load_frozen_detector',side_effect=ValueError('Frozen artifact changed')):
            with self.assertRaises(ValueError):demo.load_demo_detector('ae',42)
    def test_output_csv_and_map(self):
        with tempfile.TemporaryDirectory() as t:
            demo.run(self.example,t,'ae',42,True)
            self.assertTrue((Path(t)/'demo_predictions.csv').exists())
            html=(Path(t)/'demo_route.html').read_text(encoding='utf-8')
            self.assertIn('circleMarker',html);self.assertIn('quality unknown',html)
            self.assertEqual(p.read(Path(t)/'demo_manifest.json')['alert_policy'],'G0_C60')
    def test_demo_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'demo_predictions.csv').write_text('old')
            with self.assertRaises(FileExistsError):demo.run(self.example,t,'ae',42)

if __name__=='__main__':unittest.main()
