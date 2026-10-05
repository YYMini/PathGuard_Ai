"""Stage 6.2 distance, leakage, lineage and read-only integration checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import audit_route_reference_coverage as audit
from src import audit_route_representation as previous
from src import compare_stage5_baselines as baseline
from src import evaluate_multi_seed_stability as multi
from src import train_multiuser_autoencoder as single
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import prepare_multiuser_dataset
from tests.test_multiuser_dataset import write_plt


class RouteReferenceCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        raw = cls.root / 'raw'; ids = [f'{i:03}' for i in range(5)]
        for user in ids: write_plt(raw / user / 'Trajectory/20240101000000.plt', user, gap=True)
        result = prepare_multiuser_dataset(raw, cls.root / 'data/processed/stage5', 'coverage_fixture', ids, 1)
        cls.data = result['output_dir']; summary, cls.users = single.read_stage5_metadata(cls.data)
        cls.train = single.load_stage5_split(cls.data, 'train', summary, cls.users)
        cls.test = single.load_stage5_split(cls.data, 'test', summary, cls.users)
        cls.trajectories = read_dataset_csv(cls.data / 'trajectory_quality_summary.csv')
        cls.ref = audit.SpatialReference.build(cls.train, cls.users, cls.trajectories)
        cls.normal = cls.ref.query(cls.test.loc[~cls.test.is_synthetic])
        cls.route = cls.test.loc[cls.test.is_synthetic & cls.test.anomaly_type.eq('route_deviation')]
        cls.pairs = audit.source_matched(cls.route, cls.normal, cls.ref)
        stage61 = cls.root / 'outputs/metrics/stage6/coverage_fixture/route_audit'; stage61.mkdir(parents=True)
        b = cls.pairs[audit.SCORE_KEY].copy()
        b['dataset_split'] = 'test'; b['evaluation_included'] = True
        b['displacement_m'] = audit.haversine_distance(cls.pairs.original_latitude,
            cls.pairs.original_longitude, cls.pairs.latitude, cls.pairs.longitude)
        b['zero_displacement'] = b.displacement_m.le(previous.NUMERICAL_ATOL)
        b['feature_near_identical'] = np.isclose(cls.route[single.FEATURE_COLUMNS].to_numpy(),
            cls.normal.set_index(audit.KEY).loc[pd.MultiIndex.from_frame(cls.route[audit.KEY]), single.FEATURE_COLUMNS].to_numpy(),
            atol=previous.NUMERICAL_ATOL, rtol=previous.NUMERICAL_RTOL).all(axis=1)
        b.to_csv(stage61 / 'route_deviation_samples.csv', index=False)
        folders = [cls.data, cls.root / 'models/stage5/coverage_fixture',
            cls.root / 'outputs/metrics/stage5/coverage_fixture', cls.root / 'outputs/figures/stage5/coverage_fixture',
            stage61, cls.root / 'outputs/figures/stage6/coverage_fixture/route_audit']
        for p in folders: p.mkdir(parents=True, exist_ok=True)
        cls.before = {str(p.relative_to(cls.root)): multi.file_snapshot(p) for p in folders}
        cls.snapshot = cls.root / 'protected.json'; single.save_json(cls.snapshot, cls.before)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(single, 'fit_and_select', side_effect=AssertionError('no training')), \
             patch.object(single, 'reconstruction_errors', side_effect=AssertionError('no inference')), \
             patch.object(single, 'calculate_threshold', side_effect=AssertionError('no threshold')), \
             patch.object(baseline, 'fit_train_scaler', side_effect=AssertionError('no scaler')), \
             patch.object(baseline, 'select_baseline', side_effect=AssertionError('no IF fitting')):
            cls.result = audit.run_audit(cls.data, cls.root, cls.snapshot)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def rejected(self, frame=None, trajectories=None):
        with self.assertRaises(ValueError):
            audit.SpatialReference.build(self.train if frame is None else frame,
                self.users, self.trajectories if trajectories is None else trajectories)

    def test_train_original_normal_reference_only(self):
        self.assertEqual(self.ref.audit['row_count'], len(self.train))
        self.assertTrue(self.ref.rows.dataset_split.eq('train').all())
        self.assertFalse(self.ref.rows.is_synthetic.any())
        self.assertTrue(self.ref.rows.source_quality_valid.all())

    def test_validation_user_reference_fails(self):
        f = self.train.copy(); f.loc[f.index[0], 'user_id'] = self.users.loc[self.users.dataset_split.eq('validation'), 'user_id'].iloc[0]
        self.rejected(f)

    def test_test_user_reference_fails(self):
        f = self.train.copy(); f.loc[f.index[0], 'user_id'] = self.users.loc[self.users.dataset_split.eq('test'), 'user_id'].iloc[0]
        self.rejected(f)

    def test_synthetic_reference_fails(self):
        f = self.train.copy(); f.loc[f.index[0], 'is_synthetic'] = True; self.rejected(f)

    def test_duplicate_trajectory_blocked(self):
        t = self.trajectories.copy(); idx = t.index[t.dataset_split.eq('train') & t.eligible_for_dataset][0]
        t.loc[idx, 'is_exact_duplicate'] = True; self.rejected(trajectories=t)

    def test_duplicate_fingerprint_blocked(self):
        t = self.trajectories.copy(); idx = t.index[t.dataset_split.eq('train') & t.eligible_for_dataset]
        t.loc[idx, 'content_fingerprint'] = 'same'; self.rejected(trajectories=t)

    def test_low_quality_row_fails(self):
        f = self.train.copy(); f.loc[f.index[0], 'is_low_quality'] = True; self.rejected(f)

    def test_low_quality_trajectory_fails(self):
        t = self.trajectories.copy(); idx = t.index[t.dataset_split.eq('train') & t.eligible_for_dataset][0]
        t.loc[idx, 'quality_eligible_for_dataset'] = False; self.rejected(trajectories=t)

    def test_identical_coordinate_distance_zero(self):
        self.assertAlmostEqual(self.ref.query(self.train.iloc[:1])[audit.DISTANCE].iloc[0], 0., places=8)

    def test_known_haversine_distance(self):
        rows = self.ref.rows.copy(); rows['latitude'] = 0.; rows['longitude'] = 0.
        tree = audit.SpatialReference(rows, audit.BallTree(audit.coordinates(rows), metric='haversine'), np.zeros(len(rows), dtype=int), {})
        q = pd.DataFrame({'latitude': [0.], 'longitude': [1.]})
        self.assertAlmostEqual(tree.query(q)[audit.DISTANCE].iloc[0], audit.EARTH_RADIUS_M * np.pi / 180., places=7)

    def test_lat_lon_radian_conversion(self):
        np.testing.assert_allclose(audit.coordinates(pd.DataFrame({'latitude': [45.], 'longitude': [90.]})), [[np.pi/4, np.pi/2]])
        q = self.normal.iloc[:2]; near = self.ref.query(q)
        idx = self.ref.rows.set_index(['user_id', 'trajectory_id', 'source_point_index'])
        selected = idx.loc[pd.MultiIndex.from_frame(near[['nearest_train_' + c for c in audit.META]].rename(columns=lambda c: c.removeprefix('nearest_train_')))]
        d = audit.haversine_distance(q.latitude.to_numpy(), q.longitude.to_numpy(), selected.latitude.to_numpy(), selected.longitude.to_numpy())
        np.testing.assert_allclose(d, near[audit.DISTANCE], atol=1e-6)

    def test_source_lineage_matching_shuffled(self):
        paired = audit.source_matched(self.route, self.normal.sample(frac=1, random_state=9), self.ref)
        np.testing.assert_allclose(paired.delta_reference_distance_m, self.pairs.delta_reference_distance_m)

    def test_source_matched_delta_exact(self):
        np.testing.assert_array_equal(self.pairs.delta_reference_distance_m,
            self.pairs.synthetic_reference_distance_m - self.pairs.original_reference_distance_m)
        self.assertTrue((self.pairs.delta_reference_distance_m.abs() <= self.pairs.actual_displacement_m + 1e-5).all())

    def test_coverage_inclusive_radius(self):
        f = pd.DataFrame({audit.DISTANCE: [0., 25., 26., 50.]})
        self.assertEqual(audit.coverage(f, (25,)).iloc[0].normal_covered_count, 2)

    def test_coverage_monotonic(self):
        self.assertTrue(audit.coverage(self.normal).normal_covered_count.diff().dropna().ge(0).all())

    def test_per_user_coverage(self):
        users, covered = audit.user_analysis(self.normal, self.pairs)
        self.assertEqual(users.normal_count.sum(), len(self.normal))
        self.assertEqual(set(users.user_id), set(self.normal.user_id))
        self.assertEqual(len(covered), len(users) * len(audit.ABSTENTION_RADII_M))

    def test_covered_subset_uses_original_source(self):
        p = self.pairs.iloc[:1].copy(); p['original_reference_distance_m'] = 0.
        p['synthetic_reference_distance_m'] = 100000.
        a = audit.abstention(self.normal, p)
        self.assertTrue(a.route_source_covered_count.eq(1).all())

    def test_roc_auc_high_distance_direction(self):
        self.assertEqual(audit.separation([0., 1.], [2., 3.])['roc_auc'], 1.)
        self.assertEqual(audit.separation([2., 3.], [0., 1.])['roc_auc'], 0.)

    def test_average_precision(self):
        self.assertAlmostEqual(audit.separation([1., 3.], [2., 4.])['average_precision'], 5./6.)

    def test_zero_displacement_reference_delta(self):
        b = self.pairs.loc[self.pairs.zero_displacement]
        self.assertGreater(len(b), 0); np.testing.assert_allclose(b.delta_reference_distance_m, 0., atol=1e-6)

    def test_identical_features_do_not_imply_zero_coordinates(self):
        q = self.route.iloc[:1].copy(); n = self.normal.set_index(audit.KEY).loc[tuple(q[audit.KEY].iloc[0])]
        q[single.FEATURE_COLUMNS] = n[single.FEATURE_COLUMNS].to_numpy()
        q['latitude'] = float(n.latitude) + .001
        p = audit.source_matched(q, self.normal, self.ref)
        self.assertTrue(p.feature_near_identical.iloc[0]); self.assertFalse(p.zero_displacement.iloc[0])

    def test_nan_inf_blocked(self):
        for value in (np.nan, np.inf):
            f = self.train.copy(); f.loc[f.index[0], 'latitude'] = value; self.rejected(f)
            with self.assertRaises(ValueError): audit.separation([value], [1.])

    def test_missing_original_source_fails(self):
        with self.assertRaises(ValueError): audit.source_matched(self.route, self.normal.iloc[0:0], self.ref)

    def test_source_identity_mismatch_fails(self):
        f = self.route.copy(); f['user_id'] = '999'
        with self.assertRaises(ValueError): audit.source_matched(f, self.normal, self.ref)

    def test_duplicate_lineage_fails(self):
        self.rejected(pd.concat([self.train, self.train.iloc[:1]]))
        with self.assertRaises(ValueError): audit.source_matched(pd.concat([self.route, self.route.iloc[:1]]), self.normal, self.ref)

    def test_nearest_user_frequency_totals(self):
        counts = audit.nearest_user_frequencies(self.normal)
        self.assertEqual(counts.point_count.sum(), len(self.normal))
        np.testing.assert_allclose(counts.groupby('user_id').fraction.sum(), 1.)

    def test_empty_subset_na_reason(self):
        self.assertIsNone(audit.separation([], [1.])['roc_auc'])
        self.assertEqual(audit.separation([], [1.])['na_reason'], 'both_classes_required')

    def test_protected_outputs_unchanged(self):
        previous.assert_protected(self.root, self.before)
        self.assertTrue(self.result['verification']['protected_artifacts_unchanged'])

    def test_required_artifacts_and_no_test_tuning(self):
        tables = self.result['tables']
        for name in ('normal_reference_distances', 'route_deviation_reference_distances',
            'source_matched_distance_delta', 'coverage_summary', 'per_user_coverage',
            'abstention_analysis', 'nearest_reference_user_summary', 'label_boundary_spatial_audit'):
            self.assertIn(name, tables); self.assertTrue((self.result['metrics_dir'] / (name + '.csv')).exists())
        self.assertEqual(len(list(self.result['figure_dir'].glob('*.png'))), 6)
        self.assertFalse(self.result['report']['test_parameter_tuning'])
        self.assertFalse(self.result['report']['decision_threshold_selected'])
        self.assertFalse(self.result['report']['model_trained_or_inferred'])

    def test_output_overwrite_refused(self):
        with self.assertRaises(FileExistsError): audit.run_audit(self.data, self.root, self.snapshot)

    def test_stage61_boundary_crosscheck_fails_on_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            f = read_dataset_csv(self.root / 'outputs/metrics/stage6/coverage_fixture/route_audit/route_deviation_samples.csv')
            f['displacement_m'] = 99999.; p = Path(directory) / 'bad.csv'; f.to_csv(p, index=False)
            with self.assertRaises(AssertionError): audit.verify_stage61_boundary(self.pairs, p)

    def test_amplitude_unavailable_explicit(self):
        self.assertTrue(self.pairs.configured_amplitude_m.isna().all())
        self.assertFalse(self.result['report']['amplitude']['per_sample_amplitude_available'])


if __name__ == '__main__': unittest.main()
