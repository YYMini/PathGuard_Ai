"""Causal context replay, observable-only scoring and frozen artifact integration."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import audit_context_reference as audit
from src import audit_route_reference_coverage as spatial
from src import audit_route_representation as old
from src import train_multiuser_autoencoder as single
from src import compare_stage5_baselines as baseline
from src import evaluate_multi_seed_stability as multi
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import prepare_multiuser_dataset
from tests.test_multiuser_dataset import write_plt


def stream(source='001__a',user='001',start='2024-01-01',count=120):
    return pd.DataFrame({'user_id':user,'trajectory_id':source,'source_trajectory_id':source,
        'source_point_index':np.arange(count),'timestamp':pd.date_range(start,periods=count,freq='10s'),
        'latitude':39.+np.arange(count)*.00001,'longitude':116.+np.arange(count)*.00002})


class ContextReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        ids=[f'{i:03}' for i in range(5)];raw=cls.root/'raw'
        for user in ids:
            write_plt(raw/user/'Trajectory/20240101000000.plt',user,gap=True)
            p=raw/user/'Trajectory/20240102000000.plt';write_plt(p,user)
            p.write_text(p.read_text().replace('2024-01-01','2024-01-02'))
        result=prepare_multiuser_dataset(raw,cls.root/'data/processed/stage5','context_fixture',ids,2)
        cls.data=result['output_dir'];summary,users=single.read_stage5_metadata(cls.data)
        train=single.load_stage5_split(cls.data,'train',summary,users)
        test=single.load_stage5_split(cls.data,'test',summary,users)
        ref=spatial.SpatialReference.build(train,users,read_dataset_csv(cls.data/'trajectory_quality_summary.csv'))
        normal=ref.query(test.loc[~test.is_synthetic]);route=test.loc[test.is_synthetic&test.anomaly_type.eq('route_deviation')]
        pairs=spatial.source_matched(route,normal,ref)
        stage62=cls.root/'outputs/metrics/stage6/route_reference_coverage';stage62.mkdir(parents=True)
        normal.to_csv(stage62/'normal_reference_distances.csv',index=False)
        ref.query(route).to_csv(stage62/'route_deviation_reference_distances.csv',index=False)
        pairs.to_csv(stage62/'source_matched_distance_delta.csv',index=False)
        single.save_json(stage62/'route_reference_audit.json',{'full_test_separation':spatial.separation(normal[spatial.DISTANCE],pairs.synthetic_reference_distance_m)})
        roots=[cls.data,cls.root/'models/stage5/context_fixture',cls.root/'outputs/metrics/stage5/context_fixture',
            cls.root/'outputs/figures/stage5/context_fixture',cls.root/'outputs/metrics/stage6/context_fixture/route_audit',
            cls.root/'outputs/figures/stage6/context_fixture/route_audit',stage62,cls.root/'outputs/figures/stage6/route_reference_coverage']
        for p in roots:p.mkdir(parents=True,exist_ok=True)
        cls.before={str(p.relative_to(cls.root)):multi.file_snapshot(p) for p in roots}
        cls.snapshot=cls.root/'protected.json';single.save_json(cls.snapshot,cls.before)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(single,'fit_and_select',side_effect=AssertionError('no train')), \
             patch.object(single,'reconstruction_errors',side_effect=AssertionError('no inference')), \
             patch.object(single,'calculate_threshold',side_effect=AssertionError('no threshold')), \
             patch.object(baseline,'fit_train_scaler',side_effect=AssertionError('no scaler')), \
             patch.object(baseline,'select_baseline',side_effect=AssertionError('no IF')):
            cls.result=audit.run_audit(cls.data,cls.root,cls.snapshot)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def replay(self,frame=None):
        f=stream() if frame is None else frame
        return audit.score_stream(f,f.iloc[:0],set(),'001')

    def test_causal_prefix_only(self):
        f=self.replay()
        np.testing.assert_array_equal(f.causal_prefix_context_points,np.arange(len(f)))
        valid=f.causal_prefix_context_points.gt(0)
        self.assertTrue(f.loc[valid,'causal_prefix_nearest_timestamp'].lt(f.loc[valid,'timestamp']).all())

    def test_future_coordinates_do_not_affect_past_scores(self):
        f=stream();a=self.replay(f);f.loc[60:,'latitude']+=1.;b=self.replay(f)
        np.testing.assert_allclose(a.causal_prefix_distance_m.iloc[:60],b.causal_prefix_distance_m.iloc[:60],equal_nan=True)

    def test_no_self_match(self):
        f=self.replay();self.assertTrue(f.causal_prefix_distance_m.iloc[1:].gt(0).all())
        self.assertTrue(f.causal_prefix_nearest_point_index.iloc[1:].lt(f.source_point_index.iloc[1:]).all())

    def test_prior_trajectory_only(self):
        current=stream(start='2024-01-02');past=stream(source='001__past')
        f=audit.score_stream(current,past,{'001__past'},'001')
        self.assertTrue(f.prior_personal_context_points.eq(len(past)).all())
        self.assertTrue(f.prior_personal_nearest_source_id.eq('001__past').all())

    def test_future_trajectory_fails(self):
        with self.assertRaises(ValueError):audit.score_stream(stream(),stream('001__future',start='2024-01-02'),{'001__future'},'001')

    def test_current_trajectory_in_history_fails(self):
        with self.assertRaises(ValueError):audit.score_stream(stream(),stream(),{'001__a'},'001')

    def test_prior_overlap_future_points_excluded(self):
        past=stream('001__past',start='2023-12-31 23:59:40');current=stream(count=10)
        f=audit.score_stream(current,past,{'001__past'},'001')
        self.assertEqual(f.prior_personal_context_points.iloc[0],2)
        self.assertTrue(f.prior_personal_nearest_timestamp.lt(f.timestamp).all())

    def test_first_trajectory_no_history(self):
        f=self.replay();self.assertTrue(f.prior_personal_status.eq('no_personal_history').all())
        self.assertTrue(f.prior_personal_distance_m.isna().all())

    def test_minimum_context_candidates_exact(self):
        f=self.replay();self.assertEqual(audit.MIN_CONTEXT,(10,25,50,100))
        for n in audit.MIN_CONTEXT:self.assertEqual(int(f.causal_prefix_context_points.ge(n).sum()),len(f)-n)

    def test_context_availability_and_insufficient_rates(self):
        a=self.result['tables']['context_availability'];s=self.result['tables']['support_analysis']
        self.assertTrue(a.normal_available_pct.between(0,100).all());self.assertTrue(s.normal_evaluable_pct.between(0,100).all())
        self.assertTrue(s.normal_supported_count.le(s.normal_total).all())

    def test_context_coverage_monotonic(self):
        c=self.result['tables']['coverage_by_reference']
        for _,g in c.groupby(['user_id','reference']):self.assertTrue(g.covered_count.diff().dropna().ge(0).all())

    def test_route_source_lineage_preserved(self):
        p=self.result['tables']['source_matched_context_delta'];r=self.result['tables']['route_deviation_context_distances']
        self.assertEqual(set(p[audit.SCORE_KEY].itertuples(index=False,name=None)),set(r[audit.SCORE_KEY].itertuples(index=False,name=None)))
        for ref in audit.REFERENCES:np.testing.assert_allclose(p[ref+'_delta_m'],p[ref+'_distance_m']-p['original_'+ref+'_distance_m'],equal_nan=True)

    def test_progression_bucket_and_first(self):
        r=self.result['tables']['route_deviation_context_distances']
        self.assertEqual(set(r.progression_bucket),{'early','middle','late'})
        self.assertTrue(r.loc[r.audit_is_segment_first,'segment_relative_position'].eq(0).all())

    def test_label_mutation_does_not_change_deployable_scores(self):
        f=stream();f['anomaly_label']=0;f['anomaly_start_time']=pd.Timestamp('2024-01-01')
        a=self.replay(f);f['anomaly_label']=1;f['source_quality_valid']=False;f['anomaly_start_time']=pd.Timestamp('1900-01-01');b=self.replay(f)
        pd.testing.assert_frame_equal(a,b)

    def test_source_matching_unavailable_to_scorer(self):
        f=stream();f['original_latitude']=99.;f['original_longitude']=999.
        pd.testing.assert_frame_equal(self.replay(f),self.replay(f[audit.OBS]))

    def test_roc_ap_direction(self):
        self.assertEqual(spatial.separation([1.,2.],[3.,4.])['roc_auc'],1.)
        self.assertEqual(spatial.separation([1.,2.],[3.,4.])['average_precision'],1.)

    def test_user_aggregation(self):
        t=self.result['tables']['per_user_context_metrics']
        self.assertTrue(set(t.user_id)-{'ALL'});self.assertEqual(t.loc[t.user_id.eq('ALL'),'reference'].nunique(),len(audit.REFERENCES))

    def test_zero_displacement_boundary(self):
        b=self.result['tables']['label_boundary_context_audit'];b=b.loc[b.zero_displacement]
        np.testing.assert_allclose(b.prior_personal_delta_m.dropna(),0.,atol=1e-6)
        # Prefix context may already differ despite a zero-offset current endpoint.
        self.assertTrue(len(b)>0)

    def test_nan_inf_blocked(self):
        for value in (np.nan,np.inf):
            f=stream();f.loc[10,'latitude']=value
            with self.assertRaises(ValueError):self.replay(f)

    def test_timestamp_order_fails(self):
        f=stream();f.loc[3,'timestamp']=f.loc[2,'timestamp']
        with self.assertRaises(ValueError):self.replay(f)
        with self.assertRaises(ValueError):self.replay(stream().iloc[::-1])

    def test_prior_catalog_order(self):
        a=stream('001__z',start='2024-01-01');b=stream('001__a',start='2024-01-02')
        c=audit.trajectory_catalog(pd.concat([b,a]),{'001'})
        self.assertEqual(c.source_trajectory_id.tolist(),['001__z','001__a'])
        points,_,count=audit.prior_pool(pd.concat([b,a]),c,'001','001__a');self.assertEqual(count,1);self.assertEqual(set(points.source_trajectory_id),{'001__z'})

    def test_cross_user_personal_history_fails(self):
        with self.assertRaises(ValueError):audit.score_stream(stream(),stream('002__a','002',start='2023-01-01'),{'002__a'},'001')

    def test_unauthorized_global_user_fails(self):
        with self.assertRaises(ValueError):audit.score_global(stream(user='999'),stream(),{'000'})

    def test_global_strict_past(self):
        train=stream('000__a','000');queries=stream()
        f=audit.score_global(train,queries,{'000'})
        self.assertTrue(f.global_train_causal_context_points.eq(np.arange(len(f))).all())
        self.assertTrue(np.isnan(f.global_train_causal_distance_m.iloc[0]))

    def test_exact_buffer_and_tree_against_bruteforce(self):
        points=stream(count=30);index=audit.PastIndex(points)
        with patch.object(audit,'BLOCK_SIZE',7):
            for i in range(1,len(points)):
                q=points.iloc[i];result=index.nearest(q.timestamp,q.latitude,q.longitude)
                expected=audit.haversine_distance(q.latitude,q.longitude,points.latitude.iloc[:i],points.longitude.iloc[:i]).min()
                self.assertAlmostEqual(result[0],expected,places=6)

    def test_query_time_backwards_fails(self):
        f=stream();index=audit.PastIndex(f);index.nearest(f.timestamp.iloc[5],39.,116.)
        with self.assertRaises(ValueError):index.nearest(f.timestamp.iloc[4],39.,116.)

    def test_observed_anomalous_prefix_is_retained(self):
        f=stream();f.loc[10:20,'latitude']+=.002;f['anomaly_label']=0;f.loc[10:20,'anomaly_label']=1
        a=self.replay(f);self.assertEqual(a.causal_prefix_context_points.iloc[21],21)

    def test_static_baseline_marked_noncausal(self):
        self.assertFalse(self.result['report']['static_baseline']['deployable_causal'])
        self.assertFalse(self.result['report']['source_matching_used_for_gate'])

    def test_protected_artifacts_unchanged(self):old.assert_protected(self.root,self.before)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):audit.run_audit(self.data,self.root,self.snapshot)

    def test_context_duration_uses_observed_span(self):
        f=self.replay()
        self.assertEqual(f.causal_prefix_duration_sec.iloc[0],0.)
        self.assertEqual(f.causal_prefix_duration_sec.iloc[1],0.)
        self.assertEqual(f.causal_prefix_duration_sec.iloc[10],90.)
        self.assertTrue(f.causal_prefix_last_context_timestamp.iloc[1:].lt(f.timestamp.iloc[1:]).all())

    def test_required_outputs(self):
        for name in ('normal_context_distances','route_deviation_context_distances','context_availability',
            'coverage_by_reference','per_user_context_metrics','source_matched_context_delta','anomaly_progression','support_analysis'):
            self.assertTrue((self.result['metrics_dir']/(name+'.csv')).exists())
        self.assertEqual(len(list(self.result['figure_dir'].glob('*.png'))),7)
        self.assertFalse(self.result['report']['test_parameter_tuning'])


if __name__=='__main__':unittest.main()
