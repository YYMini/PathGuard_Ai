"""Stage 5.4 fairness, fit provenance, frozen AE reuse and meaningful aggregation."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import compare_stage5_baselines as baseline
from src import evaluate_multi_seed_stability as multi
from src import train_multiuser_autoencoder as single
from src.prepare_multiuser_dataset import prepare_multiuser_dataset
from src.load_multiuser_geolife import read_dataset_csv
from tests.test_multiuser_dataset import write_plt


class BaselineComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        ids = [f'{i:03}' for i in range(5)]; raw = cls.root / 'raw'
        for user in ids:
            write_plt(raw / user / 'Trajectory/20240101000000.plt', user, gap=True)
        dataset = prepare_multiuser_dataset(raw, cls.root / 'datasets', 'baseline_fixture', ids, 1)
        cls.data = dataset['output_dir']; cls.output = cls.root / 'output'
        with contextlib.redirect_stdout(io.StringIO()):
            multi.run_multi_seed(cls.data, cls.output)
        cls.summary, cls.users = single.read_stage5_metadata(cls.data)
        cls.frames = {s: single.load_stage5_split(cls.data, s, cls.summary, cls.users) for s in ('train', 'validation', 'test')}
        cls.before = multi.file_snapshot(cls.data)
        cls.ae_before = {seed: {key: multi.file_snapshot(path) for key, path in single.output_directories(
            cls.output, cls.summary['dataset_id'], seed).items()} for seed in baseline.MODEL_SEEDS}
        cls.fit_calls = []; cls.threshold_calls = []
        real_rule = baseline.StatisticalRule.fit; real_if = baseline.fit_isolation_forest
        real_percentile = baseline.np.percentile
        def rule_spy(model, train):
            cls.fit_calls.append(train.copy()); return real_rule(model, train)
        def if_spy(train, seed):
            cls.fit_calls.append(train.copy()); return real_if(train, seed)
        def percentile_spy(values, q, *args, **kwargs):
            if np.ndim(q) == 0 and q == 95 and np.ndim(values) == 1 and len(values) == len(single.validation_normal_rows(cls.frames['validation'])):
                cls.threshold_calls.append(np.asarray(values).copy())
            return real_percentile(values, q, *args, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(single, 'train_model', side_effect=AssertionError('AE training forbidden')), \
                patch.object(single, 'fit_and_select', side_effect=AssertionError('AE selection forbidden')), \
                patch.object(single, 'calculate_threshold', side_effect=AssertionError('AE threshold recalculation forbidden')), \
                patch.object(multi, 'verify_seed_artifacts', side_effect=AssertionError('AE re-inference forbidden')), \
                patch.object(single, 'reconstruction_errors', side_effect=AssertionError('AE inference forbidden')), \
                patch.object(baseline.IsolationForest, 'predict', side_effect=AssertionError('IF predict forbidden')), \
                patch.object(baseline.StatisticalRule, 'fit', rule_spy), \
                patch.object(baseline, 'fit_isolation_forest', if_spy), \
                patch.object(baseline.np, 'percentile', percentile_spy):
            cls.result = baseline.run_comparison(cls.data, cls.output)
        cls.tables = cls.result['tables']; cls.validation = cls.result['report']['comparison_validation']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rule(self):
        return baseline.StatisticalRule().fit(self.frames['train'])

    def test_same_frozen_dataset_rows_and_checksums(self):
        self.assertEqual(self.before, multi.file_snapshot(self.data))
        self.assertEqual(self.validation['dataset_checksums'], self.before)
        self.assertEqual(self.validation['train_normal_row_count'], len(self.frames['train']))
        for split, frame in self.frames.items():
            self.assertEqual(self.validation['row_identity_feature_lineage_sha256'][split], baseline.row_digest(frame))
        self.assertTrue(all(self.validation['checks'].values()))

    def test_exact_eight_feature_schema_all_models(self):
        self.assertEqual(self.validation['feature_columns'], single.FEATURE_COLUMNS)
        self.assertEqual(len(single.FEATURE_COLUMNS), 8)
        changed = dict(self.frames); changed['test'] = self.frames['test'].drop(columns=['bearing_cos'])
        with self.assertRaises(ValueError):
            baseline.load_existing_autoencoders(self.data, self.output, self.summary, self.users, changed)

    def test_rule_fit_receives_train_original_normal_only(self):
        self.assertEqual(len(self.fit_calls), 6)
        pd.testing.assert_frame_equal(self.fit_calls[0], self.frames['train'])
        self.assertEqual(self.rule().fit_row_count_, len(self.frames['train']))

    def test_isolation_forest_fit_receives_train_original_normal_only(self):
        for frame in self.fit_calls[1:]:
            pd.testing.assert_frame_equal(frame, self.frames['train'])
            self.assertFalse(frame.is_synthetic.any())
            self.assertTrue(frame.anomaly_label.eq(0).all())

    def test_invalid_fit_sources_rejected_by_both_models(self):
        for source in ('validation', 'test'):
            for method in (lambda f: baseline.StatisticalRule().fit(f), lambda f: baseline.fit_isolation_forest(f, 7)):
                with self.subTest(source=source), self.assertRaises(ValueError): method(self.frames[source])
        changed = self.frames['train'].copy(); changed.loc[changed.index[0], 'is_synthetic'] = True
        with self.assertRaises(ValueError): baseline.StatisticalRule().fit(changed)
        with self.assertRaises(ValueError): baseline.fit_isolation_forest(changed, 7)

    def test_test_not_accepted_for_fit_or_threshold(self):
        self.assertNotIn('test', inspect.signature(baseline.select_baseline).parameters)
        with self.assertRaises(ValueError):
            baseline.select_baseline(self.frames['train'], self.frames['test'], 'statistical_rule')
        for name in ('statistical_rule', 'isolation_forest/seed_7'):
            config = json.loads((self.output / 'models/stage5/baseline_fixture/baselines' / name / 'run_config.json').read_text())
            self.assertFalse(config['test_used_for_fit_or_threshold'])
            self.assertFalse(config['validation_anomaly_used_for_fit_or_threshold'])

    def test_validation_anomalies_do_not_affect_threshold(self):
        original = baseline.select_baseline(self.frames['train'], self.frames['validation'], 'statistical_rule')
        modified = self.frames['validation'].copy()
        modified.loc[modified.is_synthetic, single.FEATURE_COLUMNS] = 1e12
        second = baseline.select_baseline(self.frames['train'], modified, 'statistical_rule')
        self.assertEqual(original['threshold'], second['threshold'])
        self.assertEqual(original['validation_normal_row_count'], len(single.validation_normal_rows(modified)))
        self.assertEqual(len(self.threshold_calls), 6)

    def test_95th_percentile_and_strict_greater_decisions(self):
        root = self.result['metrics_dir']
        for name in ['statistical_rule'] + [f'isolation_forest/seed_{s}' for s in baseline.MODEL_SEEDS]:
            saved = baseline.read_baseline_predictions(root / name / 'validation_predictions.csv')
            normal = saved.loc[~saved.is_synthetic & saved.anomaly_label.eq(0)]
            metric = json.loads((root / name / 'validation_metrics.json').read_text())
            self.assertEqual(metric['threshold'], float(np.percentile(normal.anomaly_score, 95, method='linear')))
            np.testing.assert_array_equal(saved.predicted_anomaly, (saved.anomaly_score > metric['threshold']).astype(int))
        f = self.frames['test'].iloc[:2]
        scored = baseline.predictions(f, np.array([1., 2.]), 1.)
        self.assertEqual(scored.predicted_anomaly.tolist(), [0, 1])

    def test_rule_score_increases_with_deviation_and_uses_all_eight(self):
        train = self.frames['train'].copy(); train.loc[:, single.FEATURE_COLUMNS] = 0.
        rule = baseline.StatisticalRule().fit(train)
        point = train.iloc[:1].copy()
        self.assertEqual(rule.score(point)[0], 0.)
        for name in single.FEATURE_COLUMNS:
            changed = point.copy(); changed.loc[:, name] = 2.
            self.assertGreater(rule.score(changed)[0], 0.)
            changed.loc[:, name] = 4.
            self.assertAlmostEqual(rule.score(changed)[0], 4e9, delta=1e-6)

    def test_rule_center_scale_fallbacks_are_finite(self):
        train = self.frames['train'].copy(); n = len(train)
        train['time_diff_sec'] = 2.
        train['stop_duration_sec'] = [0.] * (n - 1) + [100.]
        train['distance_m'] = np.arange(n, dtype=float)
        rule = baseline.StatisticalRule().fit(train)
        self.assertEqual(rule.center_[0], 2.)
        self.assertEqual(rule.scale_sources_[0], 'constant_floor')
        self.assertEqual(rule.scale_sources_[5], 'population_std')
        self.assertEqual(rule.scale_sources_[1], 'iqr')
        self.assertTrue(np.isfinite(rule.score(train)).all())
        self.assertAlmostEqual(rule.scale_[5], train.stop_duration_sec.std(ddof=0))

    def test_if_score_direction_is_negative_score_samples(self):
        selection = baseline.select_baseline(self.frames['train'], self.frames['validation'], 'isolation_forest', 7)
        expected = -selection['model'].score_samples(baseline.transform_with_scaler(selection['scaler'], self.frames['test']))
        np.testing.assert_array_equal(baseline.anomaly_scores(selection, self.frames['test']), expected)
        self.assertEqual(int(selection['scaler'].n_samples_seen_), len(self.frames['train']))

    def test_if_parameters_and_exact_seed_set(self):
        rows = self.tables['baseline_results']
        self.assertEqual(rows.loc[rows.model.eq('isolation_forest'), 'seed'].astype(int).tolist(), list(baseline.MODEL_SEEDS))
        self.assertEqual(baseline.IF_PARAMETERS['n_estimators'], 200)
        self.assertEqual(baseline.IF_PARAMETERS['contamination'], 'auto')
        self.assertEqual(baseline.IF_PARAMETERS['max_samples'], 'auto')
        with self.assertRaises(ValueError): baseline.fit_isolation_forest(self.frames['train'], 8)
        with self.assertRaises(ValueError): baseline.select_baseline(self.frames['train'], self.frames['validation'], 'statistical_rule', 42)

    def test_rule_single_run_and_undefined_seed_std(self):
        rows = self.tables['baseline_results']
        self.assertEqual(len(rows.loc[rows.model.eq('statistical_rule')]), 1)
        summary = self.tables['baseline_summary'].loc[lambda f: f.model.eq('statistical_rule')]
        self.assertTrue(summary.run_count.eq(1).all())
        self.assertTrue(summary.sample_std.isna().all())
        self.assertTrue(summary.na_reason.eq('deterministic_single_run_std_undefined').all())

    def test_sample_seed_summary_matches_actual_runs(self):
        for row in self.tables['baseline_summary'].itertuples():
            values = self.tables['baseline_results'].loc[lambda f: f.model.eq(row.model), row.metric].dropna()
            if len(values): self.assertAlmostEqual(row.mean, values.mean())
            if len(values) > 1:
                self.assertAlmostEqual(row.sample_std, values.std(ddof=1))
                self.assertEqual(row.run_count, 5)
                self.assertEqual(row.min, values.min()); self.assertEqual(row.max, values.max())

    def test_per_user_metrics_and_macro_use_equal_user_weight(self):
        for row in self.tables['baseline_results'].itertuples():
            users = self.tables['per_user_baseline_metrics'].loc[lambda f: f.model.eq(row.model)]
            if pd.notna(row.seed): users = users.loc[users.seed.eq(row.seed)]
            self.assertEqual(set(users.user_id), set(self.frames['test'].user_id))
            for name in baseline.USER_METRICS:
                self.assertAlmostEqual(getattr(row, f'user_macro_{name}'), users[name].mean())

    def test_anomaly_type_uses_only_common_original_normal_negatives(self):
        frame = baseline.read_baseline_predictions(self.result['metrics_dir'] / 'statistical_rule/test_predictions.csv')
        normal = frame.loc[~frame.is_synthetic & frame.anomaly_label.eq(0)]
        rows = self.tables['per_anomaly_type_baseline_metrics'].loc[lambda f: f.model.eq('statistical_rule')]
        for row in rows.itertuples():
            anomaly = frame.loc[frame.anomaly_type.eq(row.anomaly_type) & frame.is_synthetic]
            combined = pd.concat([normal, anomaly])
            self.assertEqual(row.anomaly_row_count, len(anomaly))
            self.assertEqual(row.detected_anomaly_row_count, int(anomaly.predicted_anomaly.sum()))
            self.assertEqual(row.comparison_normal_row_count, len(normal))
            self.assertAlmostEqual(row.recall, anomaly.predicted_anomaly.mean())
            self.assertAlmostEqual(row.roc_auc, roc_auc_score(combined.anomaly_label, combined.anomaly_score))
            self.assertAlmostEqual(row.average_precision, average_precision_score(combined.anomaly_label, combined.anomaly_score))
            for suffix, expected in [('mean', anomaly.anomaly_score.mean()), ('median', anomaly.anomaly_score.median()),
                                     ('p95', np.percentile(anomaly.anomaly_score, 95)), ('p99', np.percentile(anomaly.anomaly_score, 99))]:
                np.testing.assert_allclose(getattr(row, 'anomaly_score_' + suffix), expected, rtol=1e-12, atol=1e-12)

    def test_autoencoder_all_artifacts_and_thresholds_unchanged(self):
        for seed in baseline.MODEL_SEEDS:
            for name, path in single.output_directories(self.output, self.summary['dataset_id'], seed).items():
                self.assertEqual(multi.file_snapshot(path), self.ae_before[seed][name])
            stored = json.loads((single.output_directories(self.output, 'baseline_fixture', seed)['metrics'] / 'test_metrics.json').read_text())
            row = self.tables['baseline_results'].loc[lambda f: f.model.eq('autoencoder') & f.seed.eq(seed)].iloc[0]
            self.assertEqual(row.threshold, stored['threshold'])
            for name in baseline.GLOBAL_METRICS: self.assertEqual(row[name], stored[name])
        reuse = self.validation['autoencoder_reuse']
        self.assertFalse(reuse['threshold_recalculated']); self.assertFalse(reuse['model_inference_executed'])
        self.assertFalse(reuse['model_training_executed'])

    def test_changed_test_lineage_or_features_rejected(self):
        for name, value in [('sample_id', 'different'), ('speed_mps', 9999.)]:
            changed = self.frames['test'].copy(); changed.loc[changed.index[0], name] = value
            saved = read_dataset_csv(single.output_directories(self.output, 'baseline_fixture', 7)['metrics'] / 'test_predictions.csv')
            with self.assertRaises(AssertionError): baseline.assert_same_rows(saved, changed)

    def test_ae_checksum_tamper_rejected_before_baseline_fit(self):
        p = single.output_directories(self.output, 'baseline_fixture', 7)['model'] / 'feature_columns.json'
        content = p.read_bytes()
        try:
            p.write_bytes(b'{}')
            with patch.object(baseline, 'select_baseline') as fitter, self.assertRaises(ValueError):
                baseline.load_existing_autoencoders(self.data, self.output, self.summary, self.users, self.frames)
            fitter.assert_not_called()
        finally: p.write_bytes(content)

    def test_no_overwrite_before_fit(self):
        with patch.object(baseline, 'select_baseline') as fitter, self.assertRaises(FileExistsError):
            baseline.run_comparison(self.data, self.output)
        fitter.assert_not_called()

    def test_score_csv_round_trip_keeps_threshold_ties_exact(self):
        score = .12345678901234568
        original = baseline.predictions(self.frames['test'].iloc[:2], np.array([score, score]), score)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.csv'; original.to_csv(path, index=False)
            saved = baseline.read_baseline_predictions(path)
            np.testing.assert_array_equal(saved.anomaly_score, original.anomaly_score)
            np.testing.assert_array_equal(saved.predicted_anomaly, (saved.anomaly_score > score).astype(int))
            self.assertTrue(saved.predicted_anomaly.eq(0).all())

    def test_nonfinite_features_or_scores_fail_fast(self):
        for value in (np.nan, np.inf):
            f = self.frames['train'].copy(); f.loc[f.index[0], 'distance_m'] = value
            with self.assertRaises(ValueError): baseline.StatisticalRule().fit(f)
            with self.assertRaises(ValueError): baseline.fit_isolation_forest(f, 7)
        with self.assertRaises(ValueError): baseline.predictions(self.frames['test'].iloc[:1], np.array([np.nan]), 0.)

    def test_undefined_metrics_remain_na_and_json_safe(self):
        f = self.frames['test'].loc[lambda x: ~x.is_synthetic].copy()
        f = baseline.predictions(f, np.zeros(len(f)), 1.)
        metrics = single.calculate_metrics(baseline.metrics_adapter(f), 1.)
        self.assertIsNone(metrics['precision']); self.assertIsNone(metrics['roc_auc'])
        self.assertIsNone(metrics['recall']); self.assertIsNone(metrics['average_precision'])
        self.assertTrue(metrics['metric_na_reasons'])
        json.dumps(metrics, allow_nan=False)
        frame = pd.DataFrame({'model': ['isolation_forest'] * 3, 'recall': [None, .2, None]})
        summary = baseline.summarize(frame, ['model'], ['recall']).iloc[0]
        self.assertEqual(summary.defined_run_count, 1)
        self.assertTrue(pd.isna(summary.sample_std))

    def test_required_outputs_separated_and_six_figures(self):
        root = self.result['metrics_dir']
        for name in ('baseline_results.csv', 'baseline_summary.csv', 'per_user_baseline_metrics.csv',
                     'per_anomaly_type_baseline_metrics.csv', 'baseline_comparison.json', 'comparison_validation.json'):
            self.assertTrue((root / name).is_file())
        for seed in baseline.MODEL_SEEDS:
            self.assertTrue((root / f'isolation_forest/seed_{seed}/test_predictions.csv').is_file())
        self.assertEqual({Path(p).name for p in self.result['report']['figures']},
                         {'model_f1_comparison.png', 'model_roc_auc_comparison.png', 'model_average_precision_comparison.png',
                          'model_fpr_comparison.png', 'anomaly_recall_by_model.png', 'per_user_f1_by_model.png'})
        for name in ('baseline_comparison.json', 'comparison_validation.json'):
            json.loads((root / name).read_text(), parse_constant=lambda s: self.fail('Nonstandard JSON: ' + s))


if __name__ == '__main__':
    unittest.main()