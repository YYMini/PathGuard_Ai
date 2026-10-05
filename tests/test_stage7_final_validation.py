"""Confirmatory contracts tested on tiny fixtures, never users 020-039 raw GPS."""
import copy, json, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from src import stage7_protocol as p
from src import prepare_final_cohort as cohort
from src import evaluate_final_validation as final
from src import compare_stage5_baselines as baseline
from src.load_multiuser_geolife import trajectory_fingerprint
from src.feature_engineering import generate_trajectory_features
from src.prepare_dataset import add_normal_metadata
from src.prepare_multiuser_dataset import generate_evaluation_synthetic, evaluation_rows
from src.data_quality import add_quality_flags, summarize_trajectories
from src.train_autoencoder import add_bearing_features, FEATURE_COLUMNS
from src.alert_policy import AlertPolicy, apply_policy
from src.audit_alert_aggregation import build_catalog
from src.audit_event_evaluation_protocol import positive_runs

def valid_protocol():
    return dict(stage=7,purpose='confirmatory_final_validation',final_users=list(p.USERS),
        max_trajectories_per_user=5,anomaly_types=list(p.ANOMALY_TYPES),model_seeds=list(p.SEEDS),
        feature_list=FEATURE_COLUMNS,alert_policy='G0_C60',no_tuning_after_evaluation=True,synthetic_seed=42,
        min_trajectory_points=100,max_low_quality_ratio=.05,threshold_comparison='score > threshold',
        evaluation_semantics='quality-valid original normal once + synthetic label=1 only',
        std_ddof=1,missing_user_policy='report_and_fail_no_replacement')

def fixture(user='020', n=130):
    source=user+'__fixture'
    gps=pd.DataFrame(dict(user_id=user,trajectory_id=source,
        timestamp=pd.date_range('2008-01-01',periods=n,freq='3s'),
        latitude=39.9+np.arange(n)*.00001,longitude=116.3+np.sin(np.arange(n)/10)*.00001,altitude=0.))
    features,_=generate_trajectory_features(gps)
    features=add_quality_flags(features)
    features['source_point_index']=np.arange(n)
    features['original_trajectory_id']='fixture'
    normal=add_normal_metadata(features)
    normal['source_quality_valid']=normal.is_training_eligible & ~normal.is_low_quality
    normal['synthetic_value_valid']=True
    normal['dataset_split']='test'
    normal=add_bearing_features(normal)
    synthetic,manifest=generate_evaluation_synthetic(normal,{user:'test'},42)
    synthetic=add_bearing_features(synthetic)
    evaluated=evaluation_rows(pd.concat([normal,synthetic],ignore_index=True))
    inputs=pd.DataFrame([dict(user_id=user,trajectory_id=source,original_trajectory_id='fixture',
        content_fingerprint=trajectory_fingerprint(normal),eligible_for_dataset=True,
        exclusion_reason='',cleaned_row_count=n,quality_eligible_for_dataset=True)])
    prior=pd.DataFrame(columns=['content_fingerprint','trajectory_id','original_trajectory_id','dataset_split'])
    return normal,synthetic,manifest,evaluated,inputs,prior

class ProtocolTests(unittest.TestCase):
    def test_exact_final_users(self): self.assertEqual(p.USERS,tuple(f'{i:03}' for i in range(20,40)))
    def test_no_prior_user_overlap(self): self.assertFalse(set(p.USERS)&{f'{i:03}' for i in range(20)})
    def test_protocol_valid(self): self.assertEqual(p.validate_protocol(valid_protocol())['stage'],7)
    def test_protocol_user_mutation(self):
        x=valid_protocol();x['final_users'][0]='019'
        with self.assertRaises(ValueError):p.validate_protocol(x)
    def test_feature_immutability(self):
        x=valid_protocol();x['feature_list']=FEATURE_COLUMNS+['user_id']
        with self.assertRaises(ValueError):p.validate_protocol(x)
    def test_seed_mutation(self):
        x=valid_protocol();x['model_seeds']=[42]
        with self.assertRaises(ValueError):p.validate_protocol(x)
    def test_alert_mutation(self):
        x=valid_protocol();x['alert_policy']='G0_C300'
        with self.assertRaises(ValueError):p.validate_protocol(x)
    def test_quality_policy_mutation(self):
        x=valid_protocol();x['max_low_quality_ratio']=.1
        with self.assertRaises(ValueError):p.validate_protocol(x)
    def test_generator_seed_mutation(self):
        x=valid_protocol();x['synthetic_seed']=7
        with self.assertRaises(ValueError):p.validate_protocol(x)
    def test_semantics_mutation(self):
        x=valid_protocol();x['evaluation_semantics']='all copied rows'
        with self.assertRaises(ValueError):p.validate_protocol(x)
    def test_protocol_hash_detects_modification(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'p.json';p.save(f,valid_protocol());h=p.sha(f);p.assert_hashes(t,{'p.json':h})
            f.write_text('{}')
            with self.assertRaises(ValueError):p.assert_hashes(t,{'p.json':h})
    def test_freeze_file_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'p.json';p.save(f,{'a':1})
            with self.assertRaises(FileExistsError):p.save(f,{'a':2})
    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):p.save(Path(t)/'p.json',{'x':float('inf')})
    def test_guard_scaler_fit(self):
        with p.no_fit_guard():
            with self.assertRaises(RuntimeError):StandardScaler().fit(np.zeros((2,8)))
    def test_guard_scaler_partial_fit(self):
        with p.no_fit_guard():
            with self.assertRaises(RuntimeError):StandardScaler().partial_fit(np.zeros((2,8)))
    def test_guard_if_fit(self):
        with p.no_fit_guard():
            with self.assertRaises(RuntimeError):IsolationForest().fit(np.zeros((2,8)))
    def test_guard_rule_fit(self):
        with p.no_fit_guard():
            with self.assertRaises(RuntimeError):baseline.StatisticalRule().fit(pd.DataFrame())
    def test_guard_threshold(self):
        from src.train_autoencoder import calculate_threshold
        # Guard all aliases used by prior modules, not a detached pre-patch local binding.
        from src import train_autoencoder as train
        with p.no_fit_guard():
            with self.assertRaises(RuntimeError):train.calculate_threshold(np.ones(3),pd.Series([0]*3),95)
    def test_guard_single_training_alias(self):
        with p.no_fit_guard():
            with self.assertRaises(RuntimeError):p.single.fit_train_scaler(pd.DataFrame())
    def test_model_manifest_all_11(self):
        m=p.build_model_manifest(p.ROOT)
        self.assertEqual(len(m['models']),11)
        self.assertFalse(m['model_training'])
        self.assertTrue(all(r['training_row_count']==86067 for r in m['models']))
    def test_generator_frozen_parameters(self):
        from src.synthetic_anomalies import _segment
        a=_segment(130,np.random.default_rng(42))
        b=_segment(130,np.random.default_rng(42))
        self.assertEqual(a,b);self.assertTrue(10<=a[1]-a[0]<=30)

class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.normal,cls.synthetic,cls.manifest,cls.evaluated,cls.inputs,cls.prior=fixture()
    def test_first5_deterministic(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'020/Trajectory';d.mkdir(parents=True)
            for n in ['z','b','a','f','e','d','c']:(d/(n+'.plt')).touch()
            r=cohort.select_first5(t,['020'])[0]
            self.assertEqual(r['selected_filenames'],['a.plt','b.plt','c.plt','d.plt','e.plt'])
    def test_fewer_than5_kept(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'020/Trajectory';d.mkdir(parents=True);(d/'a.plt').touch()
            self.assertEqual(cohort.select_first5(t,['020'])[0]['selected_filenames'],['a.plt'])
    def test_missing_user_recorded_no_replacement(self):
        with tempfile.TemporaryDirectory() as t:
            r=cohort.select_first5(t)
            self.assertEqual(len(r),20);self.assertTrue(all(x['missing_user'] for x in r))
    def test_same_final_duplicate_deterministic(self):
        b=self.inputs.copy();b['trajectory_id']='021__z';b['user_id']='021';b['original_trajectory_id']='z'
        r,e=cohort.exclude_duplicates(pd.concat([b,self.inputs],ignore_index=True),self.prior)
        self.assertEqual(r.loc[r.eligible_for_dataset,'trajectory_id'].tolist(),['020__fixture'])
        self.assertEqual(e.exclusion_reason.iloc[0],'exact_duplicate_trajectory')
    def test_cross_stage_duplicate_excluded(self):
        prior=pd.DataFrame([dict(content_fingerprint=self.inputs.content_fingerprint.iloc[0],trajectory_id='000__old',
                                 original_trajectory_id='old',dataset_split='train')])
        r,e=cohort.exclude_duplicates(self.inputs,prior)
        self.assertFalse(r.eligible_for_dataset.any());self.assertEqual(e.prior_split.iloc[0],'train')
        self.assertEqual(e.excluded_rows.iloc[0],130)
    def test_cross_stage_test_duplicate(self):
        prior=pd.DataFrame([dict(content_fingerprint=self.inputs.content_fingerprint.iloc[0],trajectory_id='008__old',
                                 original_trajectory_id='old',dataset_split='test')])
        r,e=cohort.exclude_duplicates(self.inputs,prior)
        self.assertEqual(e.exclusion_reason.iloc[0],'cross_stage_exact_duplicate_trajectory')
    def test_quality_filter_first_row(self):self.assertFalse(self.normal.source_quality_valid.iloc[0])
    def test_quality_trajectory_minimum(self):
        x=self.normal.iloc[:99];self.assertFalse(summarize_trajectories(x).eligible_for_dataset.iloc[0])
    def test_four_synthetic_samples(self):self.assertEqual(self.synthetic.sample_id.nunique(),4)
    def test_synthetic_seed_same_stage5(self):
        from src.prepare_multiuser_dataset import synthetic_seed_for
        for r in self.manifest.itertuples():self.assertEqual(int(r.random_seed),synthetic_seed_for(r.source_trajectory_id,r.anomaly_type,42))
    def test_copied_normals_not_evaluated(self):
        self.assertTrue(self.evaluated.loc[self.evaluated.is_synthetic,'anomaly_label'].eq(1).all())
    def test_lineage_valid(self):self.assertEqual(cohort.validate_cohort(self.normal,self.synthetic,self.inputs,self.prior)['lineage_errors'],0)
    def test_lineage_corruption_rejected(self):
        x=self.synthetic.copy();x.loc[0,'source_point_index']=1000
        with self.assertRaises(ValueError):cohort.validate_cohort(self.normal,x,self.inputs,self.prior)
    def test_source_quality_corruption_rejected(self):
        x=self.synthetic.copy();x.loc[0,'source_quality_valid']=True
        with self.assertRaises(ValueError):cohort.validate_cohort(self.normal,x,self.inputs,self.prior)
    def test_fingerprint_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'n.csv';self.normal.to_csv(f,index=False)
            self.assertEqual(trajectory_fingerprint(cohort.read_final_csv(f)),trajectory_fingerprint(self.normal))
    def test_no_nan_inf_features(self):self.assertTrue(np.isfinite(self.evaluated[FEATURE_COLUMNS]).all().all())
    def test_excluded_duplicate_cannot_enter(self):
        x=self.inputs.copy();x['eligible_for_dataset']=False
        with self.assertRaises(ValueError):cohort.validate_cohort(self.normal,self.synthetic,x,self.prior)

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normal,cls.synthetic,cls.manifest,cls.evaluated,cls.inputs,cls.prior=fixture()
        cls.model_manifest=p.build_model_manifest(p.ROOT)
        cls.rule=final.load_frozen_detector(cls.model_manifest['models'][0])
    def test_model_hash_mutation_rejected(self):
        record=copy.deepcopy(self.model_manifest['models'][0]);record['model_sha256']='0'*64
        with self.assertRaises(ValueError):final.load_frozen_detector(record)
    def test_scaler_hash_mutation_rejected(self):
        r=next(copy.deepcopy(m) for m in self.model_manifest['models'] if m['detector']=='autoencoder')
        r['scaler_sha256']='0'*64
        with self.assertRaises(ValueError):final.load_frozen_detector(r)
    def test_threshold_immutability(self):
        r=copy.deepcopy(self.model_manifest['models'][0]);r['threshold']+=1
        with self.assertRaises(ValueError):final.load_frozen_detector(r)
    def test_rule_infer_no_threshold_recompute(self):
        with patch('numpy.percentile',side_effect=AssertionError('No threshold percentile during inference')):
            out=final.infer(self.rule,self.evaluated)
        self.assertEqual(len(out),len(self.evaluated));self.assertTrue(np.isfinite(out.anomaly_score).all())
    def test_all_models_frozen_inference(self):
        for record in self.model_manifest['models']:
            detector=final.load_frozen_detector(record)
            out=final.infer(detector,self.evaluated.iloc[:20])
            self.assertTrue(np.array_equal(out.predicted_anomaly,(out.anomaly_score>record['threshold']).astype(int)))
    def test_inference_deterministic(self):
        a=final.infer(self.rule,self.evaluated);b=final.infer(self.rule,self.evaluated)
        pd.testing.assert_frame_equal(a,b)
    def test_point_metrics_hand_computed(self):
        x=pd.DataFrame(dict(anomaly_label=[0,0,1,1],anomaly_score=[0.,2.,1.,3.],predicted_anomaly=[0,1,0,1]))
        m=baseline.evaluate if False else p.single.calculate_metrics(baseline.metrics_adapter(x),1.5)
        self.assertEqual(m['f1_score'],.5);self.assertEqual(m['false_positive_rate'],.5)
    def test_user_macro_and_anomaly_types(self):
        users=pd.DataFrame([dict(user_id='020',dataset_split='test',eligible_trajectory_count=1)])
        tables=final.metric_tables([(self.rule,final.infer(self.rule,self.evaluated))],users)
        macro=tables['final_user_macro_metrics']
        self.assertTrue(macro.defined_users.eq(1).all())
        self.assertEqual(set(tables['final_anomaly_type_seed_metrics'].anomaly_type),set(p.ANOMALY_TYPES))
        self.assertTrue(tables['final_anomaly_type_seed_metrics'].comparison_normal_row_count.eq(129).all())
    def test_sample_std_ddof1(self):
        rows=pd.DataFrame(dict(model=['isolation_forest']*5,score=[0,1,2,3,4]))
        s=baseline.summarize(rows,['model'],['score']).iloc[0]
        self.assertAlmostEqual(s.sample_std,np.std([0,1,2,3,4],ddof=1))
    def test_rule_std_undefined_reason(self):
        s=baseline.summarize(pd.DataFrame(dict(model=['statistical_rule'],score=[1])),['model'],['score']).iloc[0]
        self.assertTrue(pd.isna(s.sample_std));self.assertIn('single',s.na_reason)
    def test_route_event_grouping(self):
        route=self.synthetic.loc[self.synthetic.anomaly_type.eq('route_deviation')]
        events=build_catalog(route,self.manifest.loc[self.manifest.anomaly_type.eq('route_deviation')],
                             self.evaluated.loc[self.evaluated.anomaly_type.eq('route_deviation')])
        self.assertEqual(len(events),1)
        self.assertEqual(events.event_point_count.iloc[0],self.manifest.loc[self.manifest.anomaly_type.eq('route_deviation'),'anomaly_point_count'].iloc[0])
    def test_event_and_alert_integration(self):
        tables=final.event_alert_tables([(self.rule,final.infer(self.rule,self.evaluated))],
                                       self.normal,self.synthetic,self.manifest)
        self.assertEqual(len(tables['final_event_metrics']),1)
        self.assertEqual(set(tables['final_alert_metrics'].policy_id),{'G0_C0','G0_C60'})
        self.assertEqual(len(tables['final_policy_event_metrics']),2)
    def test_early_and_cooldown_prefix_suppression(self):
        f=pd.DataFrame(dict(user_id='020',source_trajectory_id='a',sample_id='a',
            source_point_index=[0,1,2,3],timestamp=pd.date_range('2008',periods=4,freq='10s'),
            predicted_anomaly=[1,0,1,0]))
        locked=apply_policy(f,AlertPolicy('G0',60))
        self.assertEqual(locked.notification_emitted.tolist(),[True,False,False,False])
        self.assertTrue(locked.suppressed_by_cooldown.iloc[2])
    def test_false_alert_runs_quality_holes(self):
        f=pd.DataFrame(dict(user_id='020',source_trajectory_id='a',sample_id='a',
            source_point_index=[0,1,3],timestamp=pd.date_range('2008',periods=3,freq='3s'),
            predicted_anomaly=[1,1,1]))
        self.assertEqual(len(positive_runs(f)),2)
    def test_nonfinite_final_metrics_rejected(self):
        with self.assertRaises(ValueError):final.check_tables({'x':pd.DataFrame({'a':[float('inf')]})})
    def test_undefined_statistics_recorded(self):
        report=final.check_tables({'x':pd.DataFrame({'sample_std':[np.nan]})})
        self.assertEqual(report.undefined_cells.iloc[0],1)
    def test_final_overwrite_guard_before_cohort_read(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/p.METRICS;out.mkdir(parents=True);(out/'final_execution_attempt.json').write_text('{}')
            with patch.object(p,'verify_freeze',return_value=(valid_protocol(),{})):
                with self.assertRaises(FileExistsError):final.run(t)

class PreparationFixtureTests(unittest.TestCase):
    def test_full_cohort_preparation_all20_with_duplicates_and_quality_exclusion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); raw=root/'raw'
            for number,user in enumerate(p.USERS):
                directory=raw/user/'Trajectory';directory.mkdir(parents=True)
                count=99 if user=='020' else 130
                lines=[]
                for i,time in enumerate(pd.date_range('2008-01-01',periods=count,freq='3s')):
                    lines.append(f'{39.0+number*.01+i*.00001:.7f},116.0,0,0,0,{time:%Y-%m-%d},{time:%H:%M:%S}')
                text='header\n'*6+'\n'.join(lines)+'\n'
                for filename in ['a.plt','b.plt']:(directory/filename).write_text(text)
            prior=root/cohort.single_data_path();prior.mkdir(parents=True)
            pd.DataFrame([dict(content_fingerprint='not_a_copy',trajectory_id='000__old',
                               original_trajectory_id='old',dataset_split='train')]).to_csv(prior/'input_manifest.csv',index=False)
            with patch.object(p,'verify_freeze',return_value=(valid_protocol(),{'protocol_sha256':'0'*64,'protocol_freeze_commit_sha':'f'*40})):
                stats=cohort.prepare(root,raw)
            self.assertEqual(stats['trajectory_count'],40)
            self.assertEqual(stats['eligible_trajectory_count'],19)
            self.assertEqual(stats['excluded_duplicate_count'],20)
            self.assertEqual(stats['excluded_quality_trajectories'],2)
            self.assertEqual(stats['synthetic_samples'],76)
            self.assertEqual(len(stats['requested_users']),20)
            self.assertEqual(stats['per_user'][0]['normal_rows'],0)
            self.assertEqual(stats['lineage_errors'],0)
            with patch.object(p,'verify_freeze',return_value=(valid_protocol(),{})):
                with self.assertRaises(FileExistsError):cohort.prepare(root,raw)


class EndToEndFixtureTests(unittest.TestCase):
    def test_complete_final_run_and_result_tamper(self):
        # All eleven real frozen models on artificial fixture coordinates; no final cohort access.
        normal,synthetic,manifest,evaluated,inputs,prior=fixture()
        protocol=valid_protocol();protocol['model_manifest']=p.build_model_manifest(p.ROOT)
        protocol['evaluator_source_hashes']={n:p.sha(p.ROOT/n) for n in p.SOURCE_FILES}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/p.METRICS;out.mkdir(parents=True);data=root/p.DATA;data.mkdir(parents=True)
            for path in p.model_hashes(protocol['model_manifest']):
                target=root/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p.ROOT/path,target)
            for path in p.SOURCE_FILES:
                target=root/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p.ROOT/path,target)
            prior_dir=root/cohort.single_data_path();prior_dir.mkdir(parents=True);prior.to_csv(prior_dir/'input_manifest.csv',index=False)
            baseline_dir=root/'outputs/metrics/stage5'/p.single.DEFAULT_DATA_DIR.name/'baseline_comparison'
            baseline_dir.mkdir(parents=True)
            shutil.copyfile(p.ROOT/'outputs/metrics/stage5'/p.single.DEFAULT_DATA_DIR.name/'baseline_comparison/baseline_summary.csv',
                            baseline_dir/'baseline_summary.csv')
            for name,f in [('final_normal',normal),('final_synthetic',synthetic),
                           ('final_evaluation',evaluated),('input_manifest',inputs),('synthetic_anomaly_manifest',manifest)]:
                f.to_csv(data/(name+'.csv'),index=False)
            hashes={f.name:p.sha(f) for f in data.glob('*.csv')}
            p.save(data/'final_manifest.json',{'file_hashes':hashes})
            p.save(root/p.PROTOCOL,protocol)
            receipt=dict(protocol_sha256=p.sha(root/p.PROTOCOL),protocol_freeze_commit_sha='f'*40)
            p.save(out/'final_protocol_manifest.json',receipt)
            with patch.object(p,'verify_freeze',return_value=(protocol,receipt)),patch.object(p,'git',return_value='f'*40):
                result=final.run(root)
            self.assertTrue(result['confirmatory']);self.assertEqual(result['detector_runs'],11)
            final.verify_results(root)
            # Results and model inference are never silently repeated.
            with patch.object(p,'verify_freeze',return_value=(protocol,receipt)):
                with self.assertRaises(FileExistsError):final.run(root)
            target=out/'final_seed_metrics.csv';target.write_text('corrupt')
            with self.assertRaises(ValueError):final.verify_results(root)

if __name__=='__main__':unittest.main()
