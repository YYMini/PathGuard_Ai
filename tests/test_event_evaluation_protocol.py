"""Stage 6.6 event semantics, boundaries, exposure and frozen-input contracts."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import audit_event_evaluation_protocol as audit
from src import audit_synthetic_route_labels as labels
from src import train_multiuser_autoencoder as single


def stream(decisions=(0, 1, 1, 0, 1, 0, 0), sample='sample', source='source', user='001'):
    n = len(decisions)
    return pd.DataFrame({'sample_id': [sample] * n, 'source_trajectory_id': [source] * n,
                         'user_id': [user] * n, 'source_point_index': np.arange(10, 10 + n),
                         'timestamp': pd.date_range('2020-01-01', periods=n, freq='3s'),
                         'time_diff_sec': np.full(n, 3.), 'predicted_anomaly': decisions})


def event(frame):
    g = frame.sort_values('source_point_index')
    return {'sample_id': g.sample_id.iloc[0], 'event_id': g.sample_id.iloc[0] + '__run_01',
            'user_id': g.user_id.iloc[0], 'source_trajectory_id': g.source_trajectory_id.iloc[0],
            'event_start_index': int(g.source_point_index.iloc[0]),
            'event_end_index': int(g.source_point_index.iloc[-1]),
            'event_start_timestamp': g.timestamp.iloc[0], 'event_end_timestamp': g.timestamp.iloc[-1],
            'event_point_count': len(g), 'evaluated_point_count': len(g),
            'excluded_quality_point_count': 0}


def catalog_input():
    full = stream()
    full['anomaly_label'] = [0, 1, 1, 1, 1, 1, 0]
    full['anomaly_segment_id'] = 'segment'
    tiers = full.loc[full.anomaly_label.eq(1)].copy()
    tiers['identifiability_tier'] = ['tier4', 'tier1', 'tier1', 'tier1', 'tier2']
    manifest = pd.DataFrame([{'sample_id': 'sample', 'source_trajectory_id': 'source', 'user_id': '001',
                              'anomaly_point_count': 5, 'generated_point_count': 7,
                              'anomaly_start_time': full.timestamp.iloc[1], 'anomaly_end_time': full.timestamp.iloc[5]}])
    return full, manifest, tiers


class EventEvaluationTests(unittest.TestCase):
    def test_sample_grouping(self):
        f, m, t = catalog_input()
        f2, m2, t2 = f.copy(), m.copy(), t.copy()
        for frame in (f2, m2, t2):
            frame['sample_id'] = 'second'
        events = audit.build_events(pd.concat([f2, f]), pd.concat([m, m2]), pd.concat([t, t2]))
        self.assertEqual(len(events), 2)
        self.assertEqual(set(events.synthetic_sample_id), {'sample', 'second'})

    def test_full_boundaries(self):
        events = audit.build_events(*catalog_input())
        self.assertEqual(events.event_start_index.iloc[0], 11)
        self.assertEqual(events.event_end_index.iloc[0], 15)
        self.assertEqual(events.event_point_count.iloc[0], 5)
        self.assertEqual(events.event_duration_sec.iloc[0], 12)

    def test_metadata_boundary_mismatch_fails(self):
        f, m, t = catalog_input()
        m.loc[0, 'anomaly_end_time'] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            audit.build_events(f, m, t)

    def test_any_positive_detects(self):
        f = stream((0, 0, 0, 1, 0, 0, 0))
        self.assertTrue(audit.evaluate_event(event(f), f)['event_detected'])

    def test_missed_event_na(self):
        f = stream((0,) * 7)
        r = audit.evaluate_event(event(f), f)
        self.assertEqual(r['event_status'], 'missed_event')
        self.assertTrue(np.isnan(r['delay_points']))
        self.assertTrue(np.isnan(r['delay_seconds']))
        self.assertEqual(r['event_point_coverage'], 0)
        self.assertFalse(r['early_detection_50'])

    def test_first_index_sorted_not_input_order(self):
        f = stream().sample(frac=1, random_state=1)
        self.assertEqual(audit.evaluate_event(event(f), f)['first_detection_source_index'], 11)

    def test_relative_position(self):
        f = stream()
        self.assertAlmostEqual(audit.evaluate_event(event(f), f)['relative_detection_position'], 1 / 6)

    def test_delay_points(self):
        f = stream()
        self.assertEqual(audit.evaluate_event(event(f), f)['delay_points'], 1)

    def test_delay_seconds(self):
        f = stream()
        self.assertEqual(audit.evaluate_event(event(f), f)['delay_seconds'], 3)

    def test_early_fixed_inclusive_cutoffs(self):
        f = stream((0, 1, 0, 0, 0))
        r = audit.evaluate_event(event(f), f)
        self.assertFalse(r['early_detection_10'])
        self.assertTrue(r['early_detection_25'])
        self.assertTrue(r['early_detection_50'])
        self.assertEqual(audit.EARLY_CUTOFFS, (.1, .25, .5))

    def test_singleton_relative_zero(self):
        f = stream((1,))
        self.assertEqual(audit.evaluate_event(event(f), f)['relative_detection_position'], 0)

    def test_coverage_full_and_available_denominators(self):
        f = stream()
        e = event(f)
        observed = f.drop(index=2)
        e['evaluated_point_count'] = 6
        e['excluded_quality_point_count'] = 1
        r = audit.evaluate_event(e, observed)
        self.assertAlmostEqual(r['event_point_coverage'], 2 / 7)
        self.assertAlmostEqual(r['evaluated_point_coverage'], 2 / 6)
        self.assertAlmostEqual(r['coverage_available_ratio'], 6 / 7)

    def test_longest_run(self):
        f = stream()
        r = audit.evaluate_event(event(f), f)
        self.assertEqual(r['longest_positive_run_points'], 2)
        self.assertAlmostEqual(r['longest_positive_run_ratio'], 2 / 7)

    def test_fragmentation_and_average(self):
        f = stream()
        r = audit.evaluate_event(event(f), f)
        self.assertEqual(r['positive_run_count'], 2)
        self.assertEqual(r['average_run_length'], 1.5)

    def test_missing_source_index_breaks_run(self):
        f = stream((1, 1, 1)).drop(index=1)
        self.assertEqual(len(audit.positive_runs(f)), 2)

    def test_existing_long_gap_breaks_run(self):
        f = stream((1, 1, 1))
        f.loc[2, 'timestamp'] += pd.Timedelta(seconds=400)
        f.loc[2, 'time_diff_sec'] = 403
        self.assertEqual(len(audit.positive_runs(f)), 2)

    def test_tier_counts_and_observable_ratio(self):
        r = audit.build_events(*catalog_input()).iloc[0]
        self.assertEqual([r[t + '_count'] for t in labels.TIERS], [3, 1, 0, 1])
        self.assertEqual(r.observable_ratio, .8)
        self.assertTrue(r.contains_tier4)

    def test_tier_duplicate_exact_join_fails(self):
        f, m, t = catalog_input()
        with self.assertRaisesRegex(ValueError, 'Duplicate tier'):
            audit.build_events(f, m, pd.concat([t, t.iloc[:1]]))

    def test_prediction_exact_join_missing_and_duplicates_fail(self):
        left = stream()[audit.SCORE_KEY]
        for right in (left.iloc[:-1], pd.concat([left, left.iloc[:1]])):
            with self.assertRaises(ValueError):
                labels.exact_join(left, right, audit.SCORE_KEY)

    def test_prediction_outside_boundary_fails(self):
        f = stream()
        e = event(f)
        e['event_end_index'] = 14
        with self.assertRaisesRegex(ValueError, 'outside'):
            audit.evaluate_event(e, f)

    def test_prediction_identity_mismatch_fails(self):
        f = stream()
        e = event(f)
        e['user_id'] = '008'
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit.evaluate_event(e, f)

    def test_seed_separation(self):
        data = pd.DataFrame({'model': ['isolation_forest'] * 5, 'seed': [7, 21, 42, 100, 2026], 'rate': [0, .25, .5, .75, 1]})
        r = audit.seed_summary(data, ['rate']).iloc[0]
        self.assertEqual(r.seed_count, 5)
        self.assertEqual(r['min'], 0)
        self.assertEqual(r['max'], 1)

    def test_seed_mean_sample_std(self):
        data = pd.DataFrame({'model': ['autoencoder'] * 5, 'seed': [7, 21, 42, 100, 2026], 'rate': [0, .25, .5, .75, 1]})
        r = audit.seed_summary(data, ['rate']).iloc[0]
        self.assertEqual(r['mean'], .5)
        self.assertAlmostEqual(r.sample_std, np.std(data.rate, ddof=1))

    def test_missing_seed_rejected(self):
        data = pd.DataFrame({'model': ['autoencoder'], 'seed': [7], 'rate': [1]})
        with self.assertRaisesRegex(ValueError, 'seeds'):
            audit.seed_summary(data, ['rate'])

    def test_rule_deterministic_std_na(self):
        r = audit.seed_summary(pd.DataFrame({'model': ['statistical_rule'], 'seed': [None], 'rate': [.9]}), ['rate']).iloc[0]
        self.assertTrue(np.isnan(r.sample_std))
        self.assertEqual(r.std_na_reason, 'deterministic_single_result')

    def test_normal_runs_and_duration(self):
        r = audit.positive_runs(stream())
        self.assertEqual(r.length_points.tolist(), [2, 1])
        self.assertEqual(r.duration_seconds.tolist(), [3., 0.])

    def test_trajectory_boundary_runs_separated(self):
        f = pd.concat([stream((1, 1), source='a'), stream((1, 1), source='b')], ignore_index=True)
        f['model'], f['seed'] = 'statistical_rule', np.nan
        runs, trajectories = audit.evaluate_normal_predictions(f)
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(trajectories), 2)
        with self.assertRaises(ValueError):
            audit.positive_runs(f)

    def test_fp_per_1000_and_any_alert(self):
        f = pd.concat([stream((1, 0), source='a'), stream((0, 0), source='b')], ignore_index=True)
        f['model'], f['seed'] = 'statistical_rule', np.nan
        _, trajectories = audit.evaluate_normal_predictions(f)
        s = audit.normal_summary(trajectories)
        self.assertEqual(s['fp_per_1000_normal_points'], 250)
        self.assertEqual(s['normal_trajectories_with_alert_percent'], 50)
        self.assertEqual(s['false_alert_runs_per_trajectory'], .5)

    def test_user_aggregation_preserves_ids(self):
        f = pd.concat([stream((1, 0), user='001', source='a'), stream((0, 0), user='008', source='b')], ignore_index=True)
        f['model'], f['seed'] = 'statistical_rule', np.nan
        _, t = audit.evaluate_normal_predictions(f)
        s = audit.grouped_summary(t, ['model', 'seed', 'user_id'], audit.normal_summary)
        self.assertEqual(set(s.user_id), {'001', '008'})
        self.assertEqual(s.fp_point_count.tolist(), [1, 0])

    def test_threshold_validation_no_mutation(self):
        f = pd.DataFrame({'anomaly_score': [1., 2., 3.], 'predicted_anomaly': [0, 0, 1]})
        before = f.copy(deep=True)
        threshold = 2.
        audit.validate_fixed_decisions(f, threshold)
        pd.testing.assert_frame_equal(f, before)
        self.assertEqual(threshold, 2.)
        with self.assertRaises(ValueError):
            audit.validate_fixed_decisions(f, 1.)

    def test_prediction_and_official_metric_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('test_predictions.csv', 'test_metrics.json'):
                (root / name).write_text('frozen', encoding='utf-8')
            snapshot = {p.name: single.sha256_file(p) for p in root.iterdir()}
            audit.validate_fixed_decisions(pd.DataFrame({'anomaly_score': [1], 'predicted_anomaly': [0]}), 2)
            labels.assert_protected(root, snapshot)
            self.assertEqual((root / 'test_metrics.json').read_text(), 'frozen')

    def test_nan_and_infinity_safety(self):
        safe = labels.json_safe({'missed': np.nan, 'timestamp': pd.NaT})
        self.assertEqual(json.dumps(safe, allow_nan=False), '{"missed": null, "timestamp": null}')
        with self.assertRaises(ValueError):
            labels.json_safe(np.inf)
        with self.assertRaises(ValueError):
            audit.validate_fixed_decisions(pd.DataFrame({'anomaly_score': [np.nan], 'predicted_anomaly': [0]}), 2)

    def test_exposure_excludes_long_gap(self):
        f = stream((0, 0, 0))
        f.loc[2, 'timestamp'] += pd.Timedelta(seconds=400)
        f.loc[2, 'time_diff_sec'] = 403
        r = audit.normal_exposure(f)
        self.assertEqual(r['observation_exposure_sec'], 3)
        self.assertEqual(r['excluded_observation_links'], 1)

    def test_exposure_excludes_quality_holes(self):
        f = stream((0, 0, 0)).drop(index=1)
        r = audit.normal_exposure(f)
        self.assertEqual(r['observation_exposure_sec'], 0)
        self.assertTrue(r['time_exposure_na_reason'])

    def test_exposure_validates_existing_interval(self):
        f = stream((0, 0))
        f.loc[1, 'time_diff_sec'] = 4
        with self.assertRaisesRegex(ValueError, 'interval'):
            audit.normal_exposure(f)

    def test_event_duration_long_gap_is_na(self):
        f, m, t = catalog_input()
        for frame in (f, t):
            frame.loc[frame.source_point_index.ge(13), 'timestamp'] += pd.Timedelta(seconds=400)
        m.loc[0, 'anomaly_end_time'] += pd.Timedelta(seconds=400)
        r = audit.build_events(f, m, t).iloc[0]
        self.assertTrue(np.isnan(r.event_duration_sec))
        self.assertEqual(r.duration_group, 'not_available')

    def test_all_events_denominator_and_detected_only_delay(self):
        hit, miss = stream(), stream((0,) * 7, sample='miss')
        rows = pd.DataFrame([audit.evaluate_event(event(hit), hit), audit.evaluate_event(event(miss), miss)])
        r = audit.event_summary(rows)
        self.assertEqual(r['event_detection_rate'], .5)
        self.assertEqual(r['early_detection_25_rate'], .5)
        self.assertEqual(r['delay_points_median'], 1)
        self.assertEqual(r['delay_conditioning'], 'detected_events_only')

    def test_recall_is_evaluated_length_weighted_coverage(self):
        a, b = stream(), stream((1, 0, 0), sample='b')
        rows = pd.DataFrame([audit.evaluate_event(event(a), a), audit.evaluate_event(event(b), b)])
        r = audit.event_summary(rows)
        self.assertAlmostEqual(r['point_recall'], 4 / 10)
        self.assertNotAlmostEqual(r['point_recall'], rows.event_point_coverage.mean())

    def test_no_training_inference_or_alert_policy_calls(self):
        tree = ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))}
        forbidden = {'fit', 'fit_transform', 'predict', 'train_model', 'reconstruction_errors', 'calculate_threshold', 'fit_and_select', 'load_state_dict', 'generate_anomalies', 'rolling'}
        self.assertFalse(calls & forbidden)

    def test_checksum_change_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / 'frozen.csv'
            p.write_text('original')
            snapshot = {'frozen.csv': single.sha256_file(p)}
            labels.assert_protected(root, snapshot)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                labels.assert_protected(root, snapshot)

    def test_existing_stage65_protected_files(self):
        snapshot = single.ROOT / 'outputs/metrics/stage66_protected_snapshot.json'
        if not snapshot.exists():
            self.skipTest('Local frozen artifacts not distributed in repository.')
        labels.assert_protected(single.ROOT, json.loads(snapshot.read_text(encoding='utf-8')))

    def test_duplicate_indices_and_non_increasing_time_fail(self):
        f = stream()
        f.loc[1, 'source_point_index'] = 10
        with self.assertRaises(ValueError):
            audit.positive_runs(f)
        f = stream()
        f.loc[1, 'timestamp'] = f.timestamp.iloc[0]
        with self.assertRaises(ValueError):
            audit.normal_exposure(f)

    def test_overwrite_refused_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'outputs/metrics/stage6/event_evaluation').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                audit.run_audit(root, root)


if __name__ == '__main__':
    unittest.main()
