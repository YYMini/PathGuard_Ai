"""Stage 6.4 causality, exact-grid Frechet, audit-only joins and frozen exports."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src import audit_trajectory_window_similarity as audit
from src import audit_context_reference as context
from src import audit_route_representation as previous
from src import evaluate_multi_seed_stability as multi
from src import train_multiuser_autoencoder as single
from src.load_multiuser_geolife import read_dataset_csv


def stream(source='001__current',user='001',start='2024-01-02',count=180):
    return pd.DataFrame({'user_id':user,'trajectory_id':source,'source_trajectory_id':source,
        'source_point_index':np.arange(count),'timestamp':pd.date_range(start,periods=count,freq='10s'),
        'latitude':39.+np.arange(count)*.00001,'longitude':116.+np.arange(count)*.00002})


def oracle(a,b):
    c=np.empty((len(a),len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            d=float(np.linalg.norm(a[i]-b[j]))
            if i==j==0:c[i,j]=d
            elif i==0:c[i,j]=max(d,c[i,j-1])
            elif j==0:c[i,j]=max(d,c[i-1,j])
            else:c[i,j]=max(d,min(c[i-1,j],c[i,j-1],c[i-1,j-1]))
    return c[-1,-1]


class WindowSimilarityTests(unittest.TestCase):
    def score(self,f=None,width=10,gap=0,prior=None,ends=None):
        f=stream() if f is None else f
        ends=np.arange(len(f)) if ends is None else ends
        return audit.score_stream(f,f.iloc[ends],[] if prior is None else prior,(39,116),width,gap)[0]

    def test_known_frechet(self):
        a=np.array([[0.,0.],[1.,0.],[2.,0.]])
        self.assertAlmostEqual(audit.frechet_distance(a,a+[0,3]),3)
        self.assertAlmostEqual(audit.frechet_distance(a,np.array([[0,0],[2,0]])),1)

    def test_identical_zero_and_symmetric(self):
        a=np.random.default_rng(8).normal(size=(25,3));b=a+.4
        self.assertEqual(audit.frechet_distance(a,a),0)
        self.assertEqual(audit.frechet_distance(a,b),audit.frechet_distance(b,a))

    def test_dp_matches_independent_oracle(self):
        rng=np.random.default_rng(30)
        for n,m in [(1,3),(10,10),(25,25),(50,50)]:
            a=rng.normal(size=(n,3));b=rng.normal(size=(m,3))
            self.assertAlmostEqual(audit.frechet_distance(a,b),oracle(a,b),places=12)

    def test_early_abandon_never_hides_better_distance(self):
        a=np.arange(30.).reshape(10,3);b=a+5
        exact=oracle(a,b)
        self.assertEqual(audit._frechet(a,b,exact),exact)
        self.assertTrue(np.isinf(audit._frechet(a,b,exact/2)))

    def test_projection_metres_and_absolute_translation(self):
        f=stream(count=10);g=f.copy();g.longitude+=.001
        x=audit.metric_coordinates(f,(39,116));y=audit.metric_coordinates(g,(39,116))
        self.assertTrue(80<audit.frechet_distance(x,y)<90)
        self.assertGreater(audit.frechet_distance(x,y),np.linalg.norm(x-x[0]-(y-y[0]),axis=1).max())

    def test_common_anchor_is_rigid(self):
        f=stream(count=10);g=f.copy();g.latitude+=.1
        a=audit.frechet_distance(audit.metric_coordinates(f,(39,116)),audit.metric_coordinates(g,(39,116)))
        b=audit.frechet_distance(audit.metric_coordinates(f,(0,0)),audit.metric_coordinates(g,(0,0)))
        self.assertAlmostEqual(a,b,places=7)

    def test_pruning_matches_all_grid_windows(self):
        f=stream(count=1600);rng=np.random.default_rng(12)
        f.latitude+=rng.normal(0,.0001,len(f));f.longitude+=rng.normal(0,.0001,len(f))
        for width in audit.WINDOWS:
            bank=audit.WindowBank([f],(39,116),width)
            x=audit.metric_coordinates(f,(39,116))
            for limit,end in [(8,220),(135,1500),(len(bank.meta),1599)]:
                q=x[end-width+1:end+1];result=bank.nearest(q,limit)
                values=np.array([oracle(q,b) for b in bank.windows[:limit]])
                self.assertAlmostEqual(result.score,float(values.min()),places=10)
                self.assertEqual(result.winner,int(values.argmin()))
                self.assertEqual(result.candidates,limit)

    def test_deterministic_tie_first_window(self):
        f=stream(count=1500);f.latitude=39.;f.longitude=116.
        bank=audit.WindowBank([f],(39,116),10)
        q=bank.windows[0]
        self.assertEqual(bank.nearest(q).winner,0)

    def test_causal_current_window(self):
        f=stream();r=self.score(f,ends=[9,50,179])
        self.assertEqual(r.current_window_start_index.tolist(),[0]*3+[41]*3+[170]*3)
        self.assertTrue(r.current_window_end_timestamp.le(f.timestamp.iloc[-1]).all())

    def test_future_coordinate_exclusion(self):
        f=stream();a=self.score(f,ends=[80]);f.loc[81:,'latitude']+=2
        b=self.score(f,ends=[80]);assert_frame_equal(a,b)

    def test_prefix_has_zero_overlap_and_gap(self):
        for gap in audit.GAPS:
            r=self.score(gap=gap);r=r.loc[r.selected_reference_type.eq('prefix')]
            self.assertTrue(r.reference_end_index.le(r.current_window_start_index-gap-1).all())
            self.assertTrue(r.reference_end_timestamp.lt(r.current_window_start_timestamp).all())
            self.assertTrue(r.overlap_count.eq(0).all())

    def test_self_reference_rejected(self):
        f=stream(count=10);meta=('001','001__current',0,9,f.timestamp.iloc[0].value,f.timestamp.iloc[-1].value)
        with self.assertRaises(ValueError):audit.assert_reference(meta,f,10,0,'prefix',f.timestamp.iloc[0])
        with self.assertRaises(ValueError):self.score(f,prior=[f])

    def test_equal_timestamp_reference_rejected(self):
        f=stream(count=10);meta=('001','001__past',0,9,f.timestamp.iloc[0].value-1,f.timestamp.iloc[0].value)
        with self.assertRaises(ValueError):audit.assert_reference(meta,f,10,0,'prior',f.timestamp.iloc[0])

    def test_completed_prior_stricter_than_earlier_start(self):
        current=stream();past=stream('001__past',start='2024-01-01');overlap=stream('001__overlap',start='2024-01-01 23:59:50')
        future=stream('001__future',start='2024-01-03');other=stream('008__past',user='008',start='2023-01-01')
        originals=pd.concat([past,overlap,current,future,other],ignore_index=True)
        catalog=context.trajectory_catalog(originals,{'001','008'})
        allowed=audit.completed_prior(originals,catalog,'001','001__current')
        self.assertEqual([g.source_trajectory_id.iloc[0] for g in allowed],['001__past'])

    def test_future_trajectory_rejected(self):
        with self.assertRaises(ValueError):self.score(prior=[stream('001__future',start='2024-01-03')])

    def test_overlapping_prior_trajectory_rejected(self):
        with self.assertRaises(ValueError):self.score(prior=[stream('001__overlap',start='2024-01-01 23:59:50')])

    def test_cross_user_prior_rejected(self):
        with self.assertRaises(ValueError):self.score(prior=[stream('008__past',user='008',start='2024-01-01')])

    def test_widths_and_gap_grid_no_tuning(self):
        self.assertEqual(audit.WINDOWS,(10,25,50));self.assertEqual(audit.GAPS,(0,10,25,50));self.assertEqual(audit.STRIDE,10)
        for width in audit.WINDOWS:
            r=self.score(width=width);p=r.loc[r.reference_type.eq('prefix')]
            self.assertEqual(int(p.status.eq('available').sum()),len(stream())-(2*width-1))

    def test_first_history_and_insufficient_history(self):
        r=self.score(ends=[0,9,19]);prior=r.loc[r.reference_type.eq('prior')];prefix=r.loc[r.reference_type.eq('prefix')]
        self.assertEqual(prior.status.tolist(),['insufficient_current_window','no_history','no_history'])
        self.assertEqual(prefix.status.tolist(),['insufficient_current_window','insufficient_history','available'])
        r=self.score(prior=[stream('001__short',start='2024-01-01',count=5)],ends=[9]);self.assertEqual(r.status.iloc[0],'insufficient_history')

    def test_prior_gap_invariance_and_combined_min(self):
        past=stream('001__past',start='2024-01-01');a=self.score(prior=[past],gap=0);b=self.score(prior=[past],gap=50)
        np.testing.assert_allclose(a.loc[a.reference_type.eq('prior'),audit.SCORE],b.loc[b.reference_type.eq('prior'),audit.SCORE],equal_nan=True)
        p=a.pivot(index='source_point_index',columns='reference_type',values=audit.SCORE)
        np.testing.assert_allclose(p.combined,np.fmin(p.prior,p.prefix),equal_nan=True)

    def test_auxiliary_distance_from_frechet_winner(self):
        f=stream();past=stream('001__past',start='2024-01-01');r=self.score(f,prior=[past],ends=[79]);bank=audit.WindowBank([past],(39,116),10)
        q=audit.metric_coordinates(f,(39,116))[70:80];winner=bank.nearest(q);row=r.iloc[0]
        self.assertAlmostEqual(row.aligned_mean_score_m,np.linalg.norm(q-bank.windows[winner.winner],axis=1).mean())

    def test_labels_pair_fields_features_do_not_affect_score(self):
        f=stream();a=self.score(f,ends=[80]);f['anomaly_label']=1;f['distance_m']=99999.;f['original_latitude']=0.;f['source_quality_valid']=False
        b=self.score(f,ends=[80]);assert_frame_equal(a,b)

    def test_nonfinite_and_invalid_coordinates_rejected(self):
        for value in (np.nan,np.inf,91.):
            f=stream();f.loc[5,'latitude']=value
            with self.assertRaises(ValueError):self.score(f)
        with self.assertRaises(ValueError):audit.frechet_distance(np.array([[np.inf,0]]),np.zeros((1,2)))

    def test_noncontiguous_indices_and_time_rejected(self):
        f=stream();f.loc[5:,'source_point_index']+=1
        with self.assertRaises(ValueError):self.score(f)
        f=stream();f.loc[5,'timestamp']=f.timestamp.iloc[4]
        with self.assertRaises(ValueError):self.score(f)

    def test_availability_and_invalid_available_score(self):
        f=pd.DataFrame({'status':['available','no_history'],audit.SCORE:[1.,np.nan]})
        self.assertEqual(audit.availability(f)['availability_pct'],50)
        f.loc[0,audit.SCORE]=np.inf
        with self.assertRaises(ValueError):audit.availability(f)

    def test_source_pair_delta_and_missing_source_fails(self):
        n=pd.DataFrame({'source_trajectory_id':['a','a'],'source_point_index':[1,2],audit.SCORE:[2.,5.],'status':'available'})
        a=n.copy();a[audit.SCORE]=[3.,9.];r=audit.paired_audit(n,a)
        self.assertEqual(r.window_delta_m.tolist(),[1,4])
        with self.assertRaises(ValueError):audit.paired_audit(n.iloc[:1],a)

    def test_hard_selection_and_delta_cutoffs(self):
        f=pd.DataFrame({'all_8_inside_train_p01_p99':[True,False,True],'point':[1,2,3]})
        self.assertEqual(audit.hard_subset(f).point.tolist(),[1,3])
        s=audit.delta_summary([-1,0,10,25,50,100]);self.assertEqual(s['count'],6);self.assertEqual(s['ge_25m_pct'],50)

    def test_pearson_spearman_and_equality_tolerance(self):
        r=audit.correlation([1,2,3],[3,2,1]);self.assertAlmostEqual(r['pearson'],-1);self.assertAlmostEqual(r['spearman'],-1)
        self.assertEqual(audit.correlation([1,2],[1+1e-6,2])['equal_fraction'],1)
        self.assertIsNone(audit.correlation([1,1],[2,3])['pearson'])

    def test_progression_and_user_aggregation(self):
        f=pd.DataFrame({'status':'available',audit.SCORE:[1.,2.,4.],'window_delta_m':[0.,1.,3.],
            'progression_bucket':['early','middle','late'],'source_trajectory_id':'a','source_point_index':[1,2,3],
            'distance_m':[1.,2.,3.],'causal_prefix_distance_m':[1.,2.,3.],'user_id':'001','all_8_inside_train_p01_p99':True})
        p=audit.progression_audit(f,{}).set_index('progression_bucket');self.assertEqual(p.loc['late','score_median'],4)
        u=audit.user_metrics(f,f,{});self.assertEqual(u.user_id.tolist(),['001']);self.assertEqual(u.hard_route_total_count.iloc[0],3)

    def test_csv_roundtrip_selected_metadata(self):
        f=self.score(ends=[80])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'score.csv';f.to_csv(p,index=False,na_rep='NA');g=pd.read_csv(p,dtype={'reference_user_id':'string'})
            np.testing.assert_allclose(f[audit.SCORE],g[audit.SCORE],equal_nan=True)
            self.assertEqual(g.reference_user_id.iloc[1],'001')

    def test_output_overwrite_refused_before_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'outputs/metrics/stage6/window_similarity').mkdir(parents=True)
            with self.assertRaises(FileExistsError):audit.run_audit(root/'data',root)


class WindowArtifactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_context_reference import ContextReferenceTests
        cls.fixture=ContextReferenceTests
        with contextlib.redirect_stdout(io.StringIO()):cls.fixture.setUpClass()
        cls.root=cls.fixture.root;cls.data=cls.fixture.data
        route=cls.fixture.result['tables']['route_deviation_context_distances']
        pairs=cls.fixture.result['tables']['source_matched_context_delta']
        flags=route[audit.SCORE_KEY+['dataset_split']].merge(pairs[audit.SCORE_KEY+['zero_displacement','feature_near_identical']],on=audit.SCORE_KEY,validate='one_to_one')
        flags['evaluation_included']=True;flags['all_8_inside_train_p01_p99']=True
        path=cls.root/'outputs/metrics/stage6'/cls.data.name/'route_audit';flags.to_csv(path/'route_deviation_samples.csv',index=False)
        expected={k:multi.file_snapshot(cls.root/k) for k in cls.fixture.before}
        for p in ('outputs/metrics/stage6/context_reference','outputs/figures/stage6/context_reference'):expected[p]=multi.file_snapshot(cls.root/p)
        cls.expected=expected;cls.snapshot=cls.root/'stage64_snapshot.json';single.save_json(cls.snapshot,expected)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(single,'fit_and_select',side_effect=AssertionError('no training')), \
             patch.object(single,'reconstruction_errors',side_effect=AssertionError('no inference')), \
             patch.object(single,'calculate_threshold',side_effect=AssertionError('no threshold')):
            cls.result=audit.run_audit(cls.data,cls.root,cls.snapshot)

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def test_all_configurations_all_endpoints_and_no_training(self):
        r=self.result['report'];self.assertEqual(r['counts']['configurations'],36)
        c=self.result['tables']['window_config_comparison']
        self.assertTrue(c.normal_total_count.eq(r['counts']['normal_endpoints']).all())
        self.assertFalse(r['model_trained_or_inferred']);self.assertFalse(r['test_parameter_tuning'])

    def test_protected_files_and_export_checksums(self):
        previous.assert_protected(self.root,self.expected)
        for name,digest in self.result['verification']['output_checksums'].items():self.assertEqual(single.sha256_file(self.result['metrics_dir']/name),digest)
        self.assertEqual(len(self.result['verification']['figure_checksums']),7)

    def test_summaries_and_source_pair_total(self):
        r=self.result;self.assertEqual(len(r['tables']['window_similarity_summary']),72)
        self.assertEqual(len(r['tables']['source_matched_window_delta']),r['report']['counts']['route_endpoints']*36)
        self.assertEqual(len(r['tables']['low_distance_challenge_metrics']),72)

    def test_incomplete_protection_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'snapshot.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Incomplete'):audit.run_audit(root/'data',root,p)


if __name__=='__main__':unittest.main()
