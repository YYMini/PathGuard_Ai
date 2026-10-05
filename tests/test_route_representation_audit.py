"""Stage 6.1 read-only route audit with small frozen historical-score fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import audit_route_representation as audit
from src import compare_stage5_baselines as baseline
from src import train_multiuser_autoencoder as single
from src import evaluate_multi_seed_stability as multi
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import prepare_multiuser_dataset
from tests.test_multiuser_dataset import write_plt


class RouteRepresentationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        raw = cls.root / 'raw'; ids = [f'{i:03}' for i in range(5)]
        for user in ids: write_plt(raw / user / 'Trajectory/20240101000000.plt', user, gap=True)
        result = prepare_multiuser_dataset(raw, cls.root / 'data/processed/stage5', 'audit_fixture', ids, 1)
        cls.data = result['output_dir']; cls.summary, cls.users = single.read_stage5_metadata(cls.data)
        cls.frames = {s: single.load_stage5_split(cls.data, s, cls.summary, cls.users) for s in ('train', 'validation', 'test')}
        base = cls.root / 'outputs/metrics/stage5/audit_fixture/baseline_comparison'
        historical = []
        for model, seeds in [('statistical_rule', [None]), ('isolation_forest', baseline.MODEL_SEEDS), ('autoencoder', baseline.MODEL_SEEDS)]:
            for seed in seeds:
                if model == 'autoencoder':
                    dirs = single.output_directories(cls.root, 'audit_fixture', seed)
                    model_dir = dirs['model']; metric_dir = dirs['metrics']; config_name = 'training_config.json'
                else:
                    suffix = Path(model) if seed is None else Path(model) / f'seed_{seed}'
                    model_dir = cls.root / 'models/stage5/audit_fixture/baselines' / suffix; metric_dir = base / suffix; config_name = 'run_config.json'
                model_dir.mkdir(parents=True); metric_dir.mkdir(parents=True)
                single.save_json(model_dir / config_name, {'threshold': .5})
                frame = cls.frames['test'].copy(); is_route = frame.is_synthetic & frame.anomaly_type.eq('route_deviation') & frame.anomaly_label.eq(1)
                frame['anomaly_score'] = np.where(is_route, .9, .1); frame['predicted_anomaly'] = is_route.astype(int)
                if model == 'autoencoder': frame = frame.rename(columns={'anomaly_score': 'reconstruction_error'})
                frame.to_csv(metric_dir / 'test_predictions.csv', index=False)
                historical.append({'model': model, 'seed': seed, 'anomaly_type': 'route_deviation', 'anomaly_row_count': int(is_route.sum()), 'detected_anomaly_row_count': int(is_route.sum()), 'recall': 1.})
        pd.DataFrame(historical).to_csv(base / 'per_anomaly_type_baseline_metrics.csv', index=False)
        fig = cls.root / 'outputs/figures/stage5/audit_fixture'; fig.mkdir(parents=True)
        roots = [cls.data, cls.root / 'models/stage5/audit_fixture', cls.root / 'outputs/metrics/stage5/audit_fixture', fig]
        cls.before = {str(p.relative_to(cls.root)): multi.file_snapshot(p) for p in roots}
        cls.snapshot = cls.root / 'protected.json'; single.save_json(cls.snapshot, cls.before)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(single, 'fit_and_select', side_effect=AssertionError('AE training forbidden')), \
             patch.object(single, 'reconstruction_errors', side_effect=AssertionError('AE inference forbidden')), \
             patch.object(single, 'calculate_threshold', side_effect=AssertionError('AE threshold forbidden')), \
             patch.object(baseline, 'fit_train_scaler', side_effect=AssertionError('Scaler creation forbidden')), \
             patch.object(baseline, 'select_baseline', side_effect=AssertionError('Baseline fitting forbidden')), \
             patch.object(baseline, 'anomaly_scores', side_effect=AssertionError('Model inference forbidden')):
            cls.result = audit.run_audit(cls.data, cls.root, cls.snapshot)
        cls.samples = cls.result['tables']['route_deviation_samples']; cls.report = cls.result['report']

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_dataset_immutable(self):
        self.assertEqual(multi.file_snapshot(self.data), self.before[str(self.data.relative_to(self.root))])
        self.assertTrue(self.result['verification']['dataset_immutable'])

    def test_stage5_artifacts_immutable(self):
        audit.assert_protected(self.root, self.before)
        self.assertTrue(self.result['verification']['all_stage5_artifacts_immutable'])

    def test_meter_displacement_known_equatorial_distance(self):
        from src.feature_engineering import haversine_distance, EARTH_RADIUS_M
        expected = EARTH_RADIUS_M * np.pi / 180
        self.assertAlmostEqual(haversine_distance(0, 0, 0, 1), expected, places=8)
        computed = haversine_distance(self.samples.original_latitude, self.samples.original_longitude, self.samples.latitude, self.samples.longitude)
        np.testing.assert_array_equal(computed, self.samples.displacement_m)

    def test_feature_deltas_are_paired_not_positional(self):
        for name in single.FEATURE_COLUMNS:
            np.testing.assert_array_equal(self.samples[name + '_delta'], self.samples[name] - self.samples['original_' + name])
        self.assertTrue(self.samples.time_diff_sec_delta.eq(0).all())

    def test_empirical_midrank_ties_and_outside_range(self):
        actual = audit.empirical_percentile([1, 2, 2, 4], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(actual, [0, 12.5, 50, 75, 87.5, 100])

    def test_percentiles_match_train_reference(self):
        rows = self.result['tables']['route_feature_percentiles']
        for name, frame in rows.groupby('feature'):
            expected = audit.empirical_percentile(self.frames['train'][name], frame.synthetic_value)
            np.testing.assert_array_equal(expected, frame.train_synthetic_percentile)
        self.assertEqual(len(rows), 8 * len(self.samples))

    def test_train_only_reference_and_test_fit_refused(self):
        for record in self.report['train_reference'].values():
            self.assertEqual(record['fit_source'], 'Train original normal only')
            self.assertEqual(record['fit_rows'], len(self.frames['train']))
        with self.assertRaises(ValueError): audit.robust_reference(self.frames['test'])

    def test_user_grouping_recall_matches_stored_predictions(self):
        scores = self.result['tables']['route_model_scores']; users = self.result['tables']['route_user_summary']
        self.assertEqual(set(users.user_id), set(self.frames['test'].user_id))
        for row in users.itertuples():
            rows = scores.loc[scores.user_id.eq(row.user_id) & scores.model.eq(row.model)]
            self.assertEqual(row.recall_mean, rows.predicted_anomaly.mean())
            self.assertEqual(row.route_row_count, rows[audit.SCORE_KEY].drop_duplicates().shape[0])

    def test_route_label_lineage_manifest_and_quality_scope(self):
        self.assertTrue(self.samples.anomaly_label.eq(1).all())
        self.assertTrue(self.samples.anomaly_type.eq('route_deviation').all())
        self.assertFalse(self.samples.duplicated(audit.SCORE_KEY).any())
        evaluation = self.samples.loc[self.samples.evaluation_included]
        self.assertTrue(evaluation.source_quality_valid.all())
        self.assertEqual(len(evaluation.loc[evaluation.dataset_split.eq('test')]), self.report['counts']['test_evaluation_route_points'])

    def test_alignment_survives_shuffled_originals(self):
        originals = single.add_bearing_features(read_dataset_csv(self.data / 'source_normal.csv'))
        sample = self.samples.drop(columns=[c for c in self.samples if c.startswith('original_') or c == 'paired_trajectory_id'])
        aligned = audit.align_source(sample, originals.sample(frac=1, random_state=3))
        np.testing.assert_array_equal(aligned.original_latitude, self.samples.original_latitude)
        np.testing.assert_array_equal(aligned.original_speed_mps, self.samples.original_speed_mps)

    def test_missing_duplicate_or_wrong_split_lineage_fails(self):
        originals = single.add_bearing_features(read_dataset_csv(self.data / 'source_normal.csv'))
        sample = self.samples.drop(columns=[c for c in self.samples if c.startswith('original_') or c == 'paired_trajectory_id'])
        with self.assertRaises(ValueError): audit.align_source(sample, pd.concat([originals, originals.iloc[:1]]))
        changed = sample.copy(); changed.loc[changed.index[0], 'source_point_index'] = 999999
        with self.assertRaises(ValueError): audit.align_source(changed, originals)
        changed = sample.copy(); changed.loc[changed.index[0], 'dataset_split'] = 'train'
        with self.assertRaises(ValueError): audit.align_source(changed, originals)

    def test_full_neighbors_do_not_skip_quality_excluded_points(self):
        for row in self.samples.itertuples():
            synthetic = read_dataset_csv(self.data / (row.dataset_split + '.csv'))
            g = synthetic.loc[synthetic.sample_id.eq(row.sample_id)].sort_values('source_point_index').reset_index(drop=True)
            position = int(row.trajectory_position)
            distance = audit.haversine_distance(g.latitude.iloc[position], g.longitude.iloc[position], g.latitude.iloc[position + 1], g.longitude.iloc[position + 1])
            self.assertAlmostEqual(row.next_distance_m, distance, places=10)

    def test_output_overwrite_refused_before_analysis(self):
        with patch.object(audit, 'full_route_audit') as analysis, self.assertRaises(FileExistsError):
            audit.run_audit(self.data, self.root, self.snapshot)
        analysis.assert_not_called()

    def test_checksum_tamper_fails_before_analysis(self):
        path = self.root / 'models/stage5/audit_fixture/baselines/statistical_rule/run_config.json'
        content = path.read_bytes()
        try:
            path.write_bytes(b'{}')
            with patch.object(audit, 'full_route_audit') as analysis, self.assertRaises(ValueError):
                audit.assert_protected(self.root, self.before)
            analysis.assert_not_called()
        finally: path.write_bytes(content)

    def test_no_model_fit_inference_threshold_scaler_or_test_tuning(self):
        self.assertEqual(self.report['model_reuse'], {'training': False, 'inference': False, 'threshold_recalculated': False,
            'scaler_created': False, 'model_seeds': list(baseline.MODEL_SEEDS), 'scores': 'Read existing Stage 5 CSVs only'})
        self.assertFalse(self.report['fit_or_tuning_on_test'])
        self.assertFalse(self.report['candidate_features_added_to_pipeline'])

    def test_no_split_leakage_and_same_feature_schema(self):
        sets = {s: set(f.user_id) for s, f in self.frames.items()}
        self.assertFalse(sets['train'] & sets['test']); self.assertFalse(sets['validation'] & sets['test'])
        self.assertEqual(self.report['feature_columns'], single.FEATURE_COLUMNS)
        self.assertEqual(self.result['verification']['cross_split_leakage'], 0)

    def test_200m_rotation_counterexample_preserves_eight_features(self):
        probe = self.report['spatial_information_probe']
        self.assertEqual(probe['source_split'], 'train')
        self.assertAlmostEqual(probe['anchor_displacement_m'], 200, places=6)
        self.assertTrue(probe['all_8_equal_within_numerical_tolerance'])
        self.assertFalse(probe['models_executed']); self.assertFalse(probe['dataset_written'])
        self.assertEqual(set(probe['max_absolute_feature_delta']), set(single.FEATURE_COLUMNS))

    def test_zero_offset_endpoint_labels_are_visible(self):
        segments = self.result['tables']['route_segment_summary']
        self.assertTrue(segments.endpoint_zero_displacement_count.eq(2).all())
        self.assertGreater(int(self.samples.zero_displacement.sum()), 0)
        self.assertGreater(int(self.samples.feature_near_identical.sum()), 0)

    def test_descriptive_fields_ddof_and_na_safety(self):
        record = audit.describe([1, 2, 3])
        self.assertEqual(record['std'], 1.)
        self.assertTrue(set(['median', 'mean', 'std', 'iqr', 'p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99', 'min', 'max']).issubset(record))
        self.assertIsNone(audit.describe([])['mean']); self.assertIsNone(audit.describe([1])['std'])
        with self.assertRaises(ValueError): audit.describe([np.nan])
        json.dumps(audit.describe([]), allow_nan=False)

    def test_required_artifacts_and_report_completeness(self):
        expected = ['route_deviation_samples.csv', 'route_feature_summary.csv', 'route_feature_percentiles.csv',
                    'route_user_summary.csv', 'synthetic_route_audit.json', 'stage6_1_verification.json']
        for name in expected: self.assertTrue((self.result['metrics_dir'] / name).is_file())
        required = ['generation_rule', 'counts', 'physical_review', 'normal_noise_proxy', 'spatial_information_probe',
                    'train_reference', 'model_reuse', 'source_code_sha256', 'limitations', 'figures']
        self.assertTrue(set(required).issubset(self.report)); self.assertEqual(len(self.report['figures']), 6)
        self.assertEqual(self.report['generation_rule']['timestamps_modified'], False)
        json.dumps(self.report, allow_nan=False)
        for name, digest in self.result['verification']['output_checksums'].items():
            self.assertEqual(single.sha256_file(self.result['metrics_dir'] / name), digest)


if __name__ == '__main__': unittest.main()
