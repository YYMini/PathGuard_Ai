"""Stage 5-B data boundaries, fixed model, metrics and artifact integrity."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from src.autoencoder import Autoencoder
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import assign_user_splits, prepare_multiuser_dataset
from src.train_autoencoder import FEATURE_COLUMNS, TrainingOptions, set_seed, transform_with_scaler
from src import train_multiuser_autoencoder as stage5
from tests.test_multiuser_dataset import write_plt


class Stage5AutoencoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.users = [f'{i:03}' for i in range(5)]
        cls.assignments = assign_user_splits(cls.users)
        raw = cls.root/'raw'
        for user in cls.users:
            source = raw/user/'Trajectory'/'20240101000000.plt'
            write_plt(source, user, gap=True)
        train_user = next(u for u,s in cls.assignments.items() if s=='train')
        source = raw/train_user/'Trajectory'/'20240101000000.plt'
        source.with_name('20240102000000.plt').write_bytes(source.read_bytes())
        result = prepare_multiuser_dataset(raw,cls.root/'datasets','fixture_stage5',cls.users,2)
        cls.data = result['output_dir']
        cls.summary, cls.user_manifest = stage5.read_stage5_metadata(cls.data)
        cls.frames = {s:stage5.load_stage5_split(cls.data,s,cls.summary,cls.user_manifest)
                      for s in ('train','validation','test')}
        cls.options = TrainingOptions(epochs=2,seed=42)
        cls.output = cls.root/'run'
        stage4_model = cls.output/'models'/'stage4'/'model.pt'
        stage4_model.parent.mkdir(parents=True)
        stage4_model.write_bytes(b'Stage 4 model sentinel')
        cls.stage4_metrics = cls.output/'outputs'/'metrics'/'stage4_test_metrics.json'
        cls.stage4_metrics.parent.mkdir(parents=True)
        cls.stage4_metrics.write_text(json.dumps({'accuracy':.9189,'precision':.5310,'recall':.6897,
                                                'f1_score':.6,'roc_auc':.8183,'average_precision':.5148,
                                                'true_negative':847,'false_positive':53,'row_count':987}),encoding='utf-8')
        cls.protected = {p:stage5.sha256_file(p) for p in [stage4_model,cls.stage4_metrics]}
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = stage5.run_pipeline(cls.data,cls.output,cls.options,cls.stage4_metrics)
        cls.directories = {k:Path(v) for k,v in cls.result['output_directories'].items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def select(self,validation=None):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            set_seed(42)
            return stage5.fit_and_select(self.frames['train'],
                    self.frames['validation'] if validation is None else validation,
                    self.options,torch.device('cpu'),Path(tmp)/'model.pt')

    def test_stage5_dataset_loading_and_counts(self):
        self.assertEqual(set(self.frames),{'train','validation','test'})
        self.assertEqual(len(self.frames['train']),self.summary['splits']['train']['evaluation_rows'])
        self.assertEqual(len(self.frames['test']),self.summary['splits']['test']['evaluation_rows'])
        self.assertTrue(self.result['dataset_unchanged'])
        self.assertEqual(self.summary['deduplication']['excluded_duplicate_trajectory_count'],1)

    def test_exact_eight_feature_schema_and_stage4_architecture(self):
        self.assertEqual(FEATURE_COLUMNS,['time_diff_sec','distance_m','speed_mps','acceleration_mps2',
            'direction_change_deg','stop_duration_sec','bearing_sin','bearing_cos'])
        self.assertEqual(self.result['config']['model_architecture'],stage5.MODEL_ARCHITECTURE)
        model=self.result['selection']['model']
        self.assertIsInstance(model,Autoencoder)
        linear=[(m.in_features,m.out_features) for m in model.modules() if isinstance(m,torch.nn.Linear)]
        self.assertEqual(linear,[(8,16),(16,8),(8,4),(4,8),(8,16),(16,8)])
        self.assertEqual(model(torch.zeros(3,8)).shape,(3,8))

    def test_train_normal_only(self):
        train=self.frames['train']
        self.assertTrue(train.anomaly_label.eq(0).all())
        self.assertTrue(train.source_quality_valid.all())
        self.assertFalse(train.is_low_quality.any())

    def test_train_synthetic_zero_and_duplicate_stays_excluded(self):
        self.assertEqual(self.frames['train'].is_synthetic.sum(),0)
        excluded=self.summary['deduplication']['duplicate_exclusions'][0]['excluded_trajectory_id']
        self.assertNotIn(excluded,set(self.frames['train'].source_trajectory_id))

    def test_scaler_fit_count_mean_and_test_transform_does_not_fit(self):
        scaler=self.result['selection']['scaler']
        train=self.frames['train']
        self.assertEqual(self.result['config']['scaler_fit_row_count'],len(train))
        self.assertEqual(int(scaler.n_samples_seen_),len(train))
        expected=train[FEATURE_COLUMNS].to_numpy(dtype=np.float32).mean(axis=0,dtype=np.float64)
        np.testing.assert_allclose(scaler.mean_,expected)
        mean=scaler.mean_.copy()
        changed=self.frames['test'].copy()
        changed[FEATURE_COLUMNS]=changed[FEATURE_COLUMNS]*10000+10000
        transform_with_scaler(scaler,changed)
        np.testing.assert_array_equal(mean,scaler.mean_)
        self.assertEqual(int(scaler.n_samples_seen_),len(train))

    def test_validation_normal_only_passed_to_early_stopping(self):
        captured={}
        original=stage5.train_model
        def spy(model,train_values,normal_values,*args):
            captured['values']=normal_values.copy()
            return original(model,train_values,normal_values,*args)
        with patch.object(stage5,'train_model',side_effect=spy):
            selected=self.select()
        expected=transform_with_scaler(selected['scaler'],stage5.validation_normal_rows(self.frames['validation']))
        np.testing.assert_array_equal(captured['values'],expected)
        self.assertEqual(len(captured['values']),self.summary['splits']['validation']['normal_original_rows'])

    def test_validation_anomaly_changes_do_not_change_checkpoint_or_threshold(self):
        changed=self.frames['validation'].copy()
        mask=changed.is_synthetic
        changed.loc[mask,FEATURE_COLUMNS]=changed.loc[mask,FEATURE_COLUMNS]*1e5+1e5
        first=self.select(); second=self.select(changed)
        self.assertEqual(first['threshold'],second['threshold'])
        self.assertEqual(first['best_epoch'],second['best_epoch'])
        assert_frame_equal(first['history'],second['history'])
        for name,value in first['model'].state_dict().items():
            self.assertTrue(torch.equal(value,second['model'].state_dict()[name]))

    def test_changed_test_does_not_change_threshold_or_checkpoint(self):
        original=stage5.load_stage5_split
        events=[]
        def altered(data,split,summary,users):
            frame=original(data,split,summary,users)
            events.append(split)
            if split=='test':
                config_path=stage5.output_directories(self.root/'changed_test',summary['dataset_id'],42)['model']/'training_config.json'
                saved=json.loads(config_path.read_text(encoding='utf-8'))
                self.assertIn('threshold',saved)
                self.assertIn('best_epoch',saved)
                frame[FEATURE_COLUMNS]=frame[FEATURE_COLUMNS]*1000+1000
            return frame
        with patch.object(stage5,'load_stage5_split',side_effect=altered), contextlib.redirect_stdout(io.StringIO()):
            changed=stage5.run_pipeline(self.data,self.root/'changed_test',self.options,self.stage4_metrics)
        self.assertEqual(events,['train','validation','test'])
        self.assertEqual(changed['config']['threshold'],self.result['config']['threshold'])
        self.assertEqual(changed['config']['best_epoch'],self.result['config']['best_epoch'])
        self.assertFalse(np.allclose(changed['test_predictions'].reconstruction_error,
                                     self.result['test_predictions'].reconstruction_error))

    def test_threshold_is_validation_normal_95th_percentile(self):
        predictions=self.result['validation_predictions']
        normal=predictions.loc[~predictions.is_synthetic & predictions.anomaly_label.eq(0)]
        self.assertEqual(self.result['config']['threshold_percentile'],95)
        self.assertAlmostEqual(self.result['config']['threshold'],np.percentile(normal.reconstruction_error,95),places=7)
        self.assertEqual(self.result['config']['validation_normal_row_count'],len(normal))

    def test_test_users_are_unseen_and_disjoint(self):
        train=set(self.frames['train'].user_id); val=set(self.frames['validation'].user_id)
        test=set(self.frames['test'].user_id)
        self.assertFalse(train&val); self.assertFalse(train&test); self.assertFalse(val&test)
        expected={u for u,s in self.assignments.items() if s=='test'}
        self.assertEqual(test,expected)

    def test_user_confusion_counts_sum_to_overall(self):
        for name in ('tn','fp','fn','tp','normal_row_count','anomaly_row_count'):
            self.assertEqual(int(self.result['per_user'][name].sum()),self.result['test_metrics'][name])

    def test_user_macro_mean_and_defined_support(self):
        macro=self.result['user_macro_metrics']
        for name in stage5.METRIC_NAMES:
            self.assertAlmostEqual(macro['metric_means'][name],self.result['per_user'][name].mean())
            self.assertEqual(macro['defined_user_counts'][name],len(self.result['per_user']))

    def test_anomaly_type_row_counts_match(self):
        predictions=self.result['test_predictions']; per_type=self.result['per_type']
        self.assertEqual(int(per_type.anomaly_row_count.sum()),self.result['test_metrics']['anomaly_row_count'])
        for row in per_type.itertuples():
            selected=predictions.loc[predictions.anomaly_label.eq(1)&predictions.anomaly_type.eq(row.anomaly_type)]
            self.assertEqual(row.anomaly_row_count,len(selected))
            self.assertEqual(row.detected_anomaly_row_count,int(selected.predicted_anomaly.sum()))
            self.assertAlmostEqual(row.reconstruction_error_p95,np.percentile(selected.reconstruction_error,95))

    def test_type_auc_uses_original_normal_not_other_anomalies(self):
        predictions=self.result['test_predictions']
        for row in self.result['per_type'].itertuples():
            chosen=predictions.loc[(~predictions.is_synthetic & predictions.anomaly_label.eq(0))|
                                   (predictions.anomaly_label.eq(1)&predictions.anomaly_type.eq(row.anomaly_type))]
            self.assertEqual(row.comparison_normal_row_count,int(chosen.anomaly_label.eq(0).sum()))
            self.assertAlmostEqual(row.roc_auc,roc_auc_score(chosen.anomaly_label,chosen.reconstruction_error))
            self.assertAlmostEqual(row.average_precision,average_precision_score(chosen.anomaly_label,chosen.reconstruction_error))

    def test_prediction_csv_preserves_lineage_and_nanosecond_timestamps(self):
        for split in ('validation','test'):
            saved=read_dataset_csv(self.directories['metrics']/f'{split}_predictions.csv')
            assert_frame_equal(saved[stage5.LINEAGE_COLUMNS],self.frames[split][stage5.LINEAGE_COLUMNS],check_dtype=False)
            self.assertTrue(all(len(u)==3 for u in saved.user_id))

    def test_stage4_files_not_overwritten(self):
        for path,digest in self.protected.items():
            self.assertEqual(stage5.sha256_file(path),digest)
        for path in self.directories.values():
            self.assertIn('stage5',path.parts)
            self.assertNotIn('stage4',path.parts)

    def test_checkpoint_restores_identical_predictions(self):
        restored=Autoencoder(8)
        restored.load_state_dict(torch.load(self.directories['model']/'model.pt',weights_only=True))
        values=torch.tensor(transform_with_scaler(self.result['selection']['scaler'],self.frames['test']))
        with torch.no_grad():
            np.testing.assert_array_equal(restored(values).numpy(),self.result['selection']['model'](values).numpy())
        scaler=joblib.load(self.directories['model']/'scaler.joblib')
        np.testing.assert_array_equal(scaler.mean_,self.result['selection']['scaler'].mean_)

    def test_single_class_metrics_are_na_with_reasons(self):
        normal=self.result['test_predictions'].loc[lambda f:f.anomaly_label.eq(0)]
        metrics=stage5.calculate_metrics(normal,self.result['config']['threshold'])
        for key in ('roc_auc','average_precision','recall','f1_score'):
            self.assertIsNone(metrics[key]); self.assertIn(key,metrics['metric_na_reasons'])
        json.dumps(metrics,allow_nan=False)

    def test_no_predicted_positive_has_na_precision_but_defined_zero_recall(self):
        frame=self.result['test_predictions'].copy(); frame['predicted_anomaly']=0
        metrics=stage5.calculate_metrics(frame,1000)
        self.assertIsNone(metrics['precision'])
        self.assertEqual(metrics['metric_na_reasons']['precision'],'no_predicted_anomaly_rows')
        self.assertEqual(metrics['recall'],0)
        self.assertEqual(metrics['f1_score'],0)

    def test_missing_anomaly_type_and_macro_na_are_safe(self):
        frame=self.result['test_predictions'].loc[lambda f:f.anomaly_label.eq(0)].copy()
        types=stage5.per_anomaly_type_metrics(frame,1)
        self.assertTrue(types.recall.isna().all())
        self.assertTrue(types.average_precision.isna().all())
        users,macro=stage5.per_user_metrics(frame,1,self.user_manifest)
        self.assertIsNone(macro['metric_means']['roc_auc'])
        self.assertEqual(macro['defined_user_counts']['roc_auc'],0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metrics.csv'; users.to_csv(path,index=False,na_rep='NA')
            self.assertIn('NA',path.read_text(encoding='utf-8'))
        self.assertIn('requires_normal_and_anomaly_classes',users.na_reasons.iloc[0])

    def test_all_required_artifacts_and_nine_figures(self):
        for name in ['model.pt','scaler.joblib','training_config.json','feature_columns.json']:
            self.assertTrue((self.directories['model']/name).is_file(),name)
        for name in ['training_history.csv','validation_predictions.csv','test_predictions.csv',
                     'test_metrics.json','validation_metrics.json','per_user_metrics.csv',
                     'per_anomaly_type_metrics.csv','stage4_stage5_comparison.csv','user_macro_metrics.json']:
            self.assertTrue((self.directories['metrics']/name).is_file(),name)
        self.assertEqual({Path(p).name for p in self.result['figures']},
            {'training_loss.png','reconstruction_error_distribution.png','reconstruction_error_distribution_zoom.png',
             'confusion_matrix.png','roc_curve.png','precision_recall_curve.png','anomaly_score_by_type.png',
             'per_user_f1.png','per_user_fpr.png'})

    def test_comparison_delta_ap_naming_and_interpretation(self):
        report=json.loads((self.directories['metrics']/'stage4_stage5_comparison.json').read_text(encoding='utf-8'))
        ap=next(row for row in report['metrics'] if row['metric']=='average_precision')
        self.assertAlmostEqual(ap['delta_stage5_minus_stage4'],ap['stage5']-ap['stage4'])
        self.assertIn('not model superiority',report['interpretation'])
        self.assertNotIn('pr_auc',{row['metric'] for row in report['metrics']})

    def test_existing_outputs_are_refused(self):
        with self.assertRaises(FileExistsError):
            stage5.run_pipeline(self.data,self.output,self.options)

    def test_wrong_model_seed_or_percentile_is_refused(self):
        for options in [TrainingOptions(seed=7),TrainingOptions(threshold_percentile=90)]:
            with self.assertRaises(ValueError):
                stage5.run_pipeline(self.data,self.root/'invalid',options)

    def test_nonfinite_scores_are_rejected_in_metrics(self):
        frame=self.result['test_predictions'].copy()
        frame.loc[frame.index[0],'reconstruction_error']=np.nan
        with self.assertRaises(ValueError):
            stage5.calculate_metrics(frame,1)


if __name__=='__main__':
    unittest.main()