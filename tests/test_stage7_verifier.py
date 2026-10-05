import unittest
import numpy as np
import pandas as pd
from src.verify_final_validation import point, replay, run_length, fingerprint
from src.alert_policy import AlertPolicy,apply_policy
from src.load_multiuser_geolife import trajectory_fingerprint

class IndependentVerifierTests(unittest.TestCase):
    def stream(self,decisions,indices=None,seconds=10):
        n=len(decisions)
        return pd.DataFrame(dict(user_id='020',source_trajectory_id='fixture',sample_id='fixture',
            source_point_index=list(range(n)) if indices is None else indices,
            timestamp=pd.date_range('2008',periods=n,freq=f'{seconds}s'),predicted_anomaly=decisions))
    def test_scalar_point_metrics(self):
        f=pd.DataFrame(dict(anomaly_label=[0,0,1,1],predicted_anomaly=[0,1,0,1],anomaly_score=[0.,2.,1.,3.]))
        r=point(f)
        self.assertEqual(r['accuracy'],.5);self.assertEqual(r['f1_score'],.5)
        self.assertEqual(r['false_positive_rate'],.5);self.assertEqual(r['roc_auc'],.75)
    def test_empty_user_supported(self):
        f=pd.DataFrame(columns=['anomaly_label','predicted_anomaly','anomaly_score'])
        r=point(f);self.assertIsNone(r['recall']);self.assertIsNone(r['roc_auc'])
        self.assertEqual(r['normal_row_count'],0)
    def test_scalar_replay_matches_raw(self):
        f=self.stream([1,1,0,1,0,1])
        a=replay(f,0);b=apply_policy(f,AlertPolicy('G0',0))
        self.assertEqual(a.independent_emit.tolist(),b.notification_emitted.tolist())
    def test_scalar_replay_matches_cooldown(self):
        f=self.stream([1,0,1,0,1,0,1,0,1])
        a=replay(f,60);b=apply_policy(f,AlertPolicy('G0',60))
        self.assertEqual(a.independent_emit.tolist(),b.notification_emitted.tolist())
    def test_scalar_replay_hole_clock_retained(self):
        f=self.stream([1,1,1],[0,1,3])
        a=replay(f,60)
        self.assertEqual(a.independent_candidate.tolist(),[True,False,True])
        self.assertEqual(a.independent_emit.tolist(),[True,False,False])
    def test_scalar_long_gap_reset(self):
        f=self.stream([1,1],seconds=301)
        self.assertEqual(replay(f,60).independent_emit.tolist(),[True,True])
        self.assertEqual(run_length(f),(1,2))
    def test_scalar_cooldown_exact60(self):
        f=self.stream([1,0,1],seconds=30)
        self.assertEqual(replay(f,60).independent_emit.tolist(),[True,False,True])
    def test_independent_fingerprint(self):
        f=self.stream([0,0,0]);f['latitude']=[39.9,39.9001,39.9002];f['longitude']=116.3
        self.assertEqual(fingerprint(f),trajectory_fingerprint(f))

if __name__=='__main__':unittest.main()
