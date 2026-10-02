"""Fixed seed set, reuse compatibility, immutable inputs and sample aggregation."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import evaluate_multi_seed_stability as multi
from src import train_multiuser_autoencoder as single
from src.prepare_multiuser_dataset import prepare_multiuser_dataset
from src.train_autoencoder import TrainingOptions
from tests.test_multiuser_dataset import write_plt


class MultiSeedStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)
        users=[f'{i:03}' for i in range(5)]; raw=cls.root/'raw'
        for user in users:
            write_plt(raw/user/'Trajectory'/'20240101000000.plt',user,gap=True)
        dataset=prepare_multiuser_dataset(raw,cls.root/'datasets','stability_fixture',users,1)
        cls.data=dataset['output_dir'];cls.output=cls.root/'output'
        with contextlib.redirect_stdout(io.StringIO()):
            single.run_pipeline(cls.data,cls.output)
        cls.summary,cls.users=single.read_stage5_metadata(cls.data)
        cls.paths=single.output_directories(cls.output,cls.summary['dataset_id'],42)
        cls.dataset_before=multi.file_snapshot(cls.data)
        cls.seed42_before={key:multi.file_snapshot(path) for key,path in cls.paths.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result=multi.run_multi_seed(cls.data,cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_five_seed_list_and_order(self):
        self.assertEqual(multi.MODEL_SEEDS,(7,21,42,100,2026))
        self.assertEqual(self.result['tables']['seed_results'].seed.tolist(),list(multi.MODEL_SEEDS))
        self.assertEqual(self.result['report']['model_seeds'],list(multi.MODEL_SEEDS))

    def test_dataset_checksums_unchanged(self):
        self.assertEqual(multi.file_snapshot(self.data),self.dataset_before)
        self.assertEqual(self.result['report']['dataset_checksums'],self.dataset_before)
        self.assertTrue(self.result['report']['dataset_unchanged'])

    def test_user_split_same_for_all_seeds(self):
        for seed in multi.MODEL_SEEDS:
            paths=single.output_directories(self.output,self.summary['dataset_id'],seed)
            config=json.loads((paths['model']/'training_config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['user_splits'],{s:self.summary['splits'][s]['users'] for s in ('train','validation','test')})
        self.assertEqual(self.result['report']['user_split_manifest_sha256'],self.dataset_before['user_split_manifest.csv'])

    def test_synthetic_manifest_same(self):
        self.assertEqual(self.result['report']['synthetic_manifest_sha256'],self.dataset_before['synthetic_anomaly_manifest.csv'])
        self.assertEqual(single.sha256_file(self.data/'synthetic_anomaly_manifest.csv'),self.dataset_before['synthetic_anomaly_manifest.csv'])

    def test_seed_outputs_are_separate(self):
        directories=[single.output_directories(self.output,self.summary['dataset_id'],s) for s in multi.MODEL_SEEDS]
        self.assertEqual(len({p['model'] for p in directories}),5)
        for seed,paths in zip(multi.MODEL_SEEDS,directories):
            for path in paths.values():
                self.assertEqual(path.name,f'seed_{seed}')
            self.assertTrue((paths['model']/'model.pt').is_file())
            self.assertTrue((paths['metrics']/'test_metrics.json').is_file())

    def test_existing_result_overwrite_refused_before_training(self):
        with patch.object(single,'run_pipeline') as runner, self.assertRaises(FileExistsError):
            multi.run_multi_seed(self.data,self.output)
        runner.assert_not_called()

    def test_mean_computation(self):
        table=multi.summarize_values(pd.DataFrame({'f1_score':[.1,.2,.3,.4,.5]}),[],['f1_score'])
        self.assertAlmostEqual(table.iloc[0]['mean'],.3)
        self.assertEqual(table.iloc[0]['min'],.1); self.assertEqual(table.iloc[0]['max'],.5)

    def test_sample_standard_deviation_uses_ddof_one(self):
        values=np.array([1.,2.,3.,4.,5.])
        table=multi.summarize_values(pd.DataFrame({'threshold':values}),[],['threshold'])
        self.assertAlmostEqual(table.iloc[0]['sample_std'],np.std(values,ddof=1))
        self.assertNotAlmostEqual(table.iloc[0]['sample_std'],np.std(values,ddof=0))
        self.assertEqual(self.result['report']['aggregation']['std_ddof'],1)

    def test_user_summary_matches_seed_records(self):
        tables=self.result['tables']
        for row in tables['per_user_seed_summary'].itertuples():
            values=tables['per_user_seed_results'].loc[lambda f:f.user_id.eq(row.user_id),row.metric]
            self.assertEqual(row.seed_count,5)
            self.assertAlmostEqual(row.mean,values.mean())
            self.assertAlmostEqual(row.sample_std,values.std(ddof=1))

    def test_type_summary_matches_seed_records(self):
        tables=self.result['tables']
        for row in tables['per_anomaly_type_seed_summary'].itertuples():
            values=tables['per_anomaly_type_seed_results'].loc[lambda f:f.anomaly_type.eq(row.anomaly_type),'recall']
            self.assertEqual(row.seed_count,5)
            self.assertAlmostEqual(row.mean,values.mean())
            self.assertAlmostEqual(row.sample_std,values.std(ddof=1))

    def test_seed42_compatibility_and_reuse_record(self):
        self.assertEqual(self.result['report']['reused_seeds'],[42])
        self.assertEqual(self.result['report']['newly_trained_seeds'],[7,21,100,2026])
        for key,path in self.paths.items():
            self.assertEqual(multi.file_snapshot(path),self.seed42_before[key])
        self.assertTrue(self.result['report']['seed42_artifacts_unchanged'])
        verified=multi.verify_seed_artifacts(self.data,self.output,self.summary,self.users,42)
        self.assertEqual(verified['artifact_checksums'],self.seed42_before)

    def test_incompatible_hyperparameter_config_rejected(self):
        config=json.loads((self.paths['model']/'training_config.json').read_text(encoding='utf-8'))
        for key,value in [('seed',7),('batch_size',32),('threshold_percentile',90),('learning_rate',.01)]:
            changed={**config,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):
                multi.validate_config(changed,self.summary,self.data,42)

    def test_incompatible_dataset_and_split_config_rejected(self):
        config=json.loads((self.paths['model']/'training_config.json').read_text(encoding='utf-8'))
        for key,value in [('dataset_summary_sha256','other'),('dataset_file_checksums',{}),('user_splits',{})]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                multi.validate_config({**config,key:value},self.summary,self.data,42)

    def test_dataset_snapshot_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'user_split_manifest.csv';path.write_text('a',encoding='utf-8')
            snapshot=multi.file_snapshot(root);path.write_text('b',encoding='utf-8')
            with self.assertRaises(ValueError):multi.assert_snapshot(root,snapshot)

    def test_fixed_hyperparameters_in_each_run(self):
        for seed in multi.MODEL_SEEDS:
            paths=single.output_directories(self.output,self.summary['dataset_id'],seed)
            config=json.loads((paths['model']/'training_config.json').read_text(encoding='utf-8'))
            self.assertEqual({key:config[key] for key in asdict(TrainingOptions())},asdict(TrainingOptions(seed=seed)))
        with self.assertRaises(ValueError):
            single.run_pipeline(self.data,self.root/'invalid',TrainingOptions(epochs=2,seed=7),experiment_stage='5.3')

    def test_macro_summary_matches_each_seed(self):
        tables=self.result['tables']
        for name in multi.MACRO_METRICS:
            key=f'user_macro_{name}'
            row=tables['seed_summary'].loc[lambda f:f.metric.eq(key)].iloc[0]
            self.assertAlmostEqual(row['mean'],tables['seed_results'][key].mean())
            self.assertAlmostEqual(row['sample_std'],tables['seed_results'][key].std(ddof=1))

    def test_na_aggregation_keeps_support_and_undefined_std(self):
        summary=multi.summarize_values(pd.DataFrame({'metric':[None,.2,None]}),[],['metric']).iloc[0]
        self.assertEqual(summary.defined_seed_count,1)
        self.assertEqual(summary['mean'],.2)
        self.assertTrue(pd.isna(summary.sample_std))
        self.assertEqual(summary.na_reason,'sample_std_requires_two_seeds')
        empty=multi.summarize_values(pd.DataFrame({'metric':[None,None]}),[],['metric']).iloc[0]
        self.assertEqual(empty.defined_seed_count,0)
        self.assertTrue(pd.isna(empty['mean']))

    def test_required_aggregate_files_and_six_figures(self):
        for name in ['seed_results.csv','seed_summary.csv','per_user_seed_results.csv','per_user_seed_summary.csv',
                     'per_anomaly_type_seed_results.csv','per_anomaly_type_seed_summary.csv','multi_seed_summary.json']:
            self.assertTrue((self.result['metrics_dir']/name).is_file())
        self.assertEqual({Path(p).name for p in self.result['report']['figures']},
                         {'f1_by_seed.png','roc_auc_by_seed.png','average_precision_by_seed.png','threshold_by_seed.png',
                          'anomaly_recall_by_seed.png','per_user_f1_by_seed.png'})
        self.assertEqual(self.result['report']['metric_summary']['f1_score']['defined_seed_count'],5)

    def test_seed42_numeric_position_and_rank(self):
        values=self.result['tables']['seed_results']; report=self.result['report']
        for name in single.METRIC_NAMES:
            position=report['seed42_positions'][name]
            score=values.loc[values.seed.eq(42),name].iloc[0]
            self.assertEqual(position['value'],score)
            self.assertAlmostEqual(position['difference_from_mean'],score-values[name].mean())
            self.assertEqual(position['ascending_rank'],int(values[name].rank(method='min').loc[values.seed.eq(42)].iloc[0]))


if __name__=='__main__':
    unittest.main()