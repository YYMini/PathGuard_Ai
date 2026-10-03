"""Stage 5-A identity, leakage, lineage, disk round-trip and duplicate tests."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.load_multiuser_geolife import (
    load_multiuser_features, read_dataset_csv, trajectory_fingerprint, validate_user_ids,
)
from src.prepare_multiuser_dataset import (
    MODEL_FEATURE_COLUMNS, assign_user_splits, attach_source_quality,
    duplicate_fingerprint_report, evaluation_rows, prepare_multiuser_dataset,
    synthetic_seed_for, validate_dataset, validate_numeric_values, validate_saved_dataset,
)


def write_plt(path, user, points=120, gap=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp('2024-01-01')
    lines = ['header'] * 6
    for i in range(points):
        stamp = start + pd.Timedelta(seconds=10*i + (600 if gap and i >= 30 else 0))
        lat = 39.0 + int(user)*.02 + i*.00001
        lon = 116.0 + i*.00002
        lines.append(f'{lat:.7f},{lon:.7f},0,10,0,{stamp:%Y-%m-%d},{stamp:%H:%M:%S}')
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


class MultiuserDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.raw = cls.root / 'raw'
        cls.users = [f'{i:03}' for i in range(5)]
        for user in cls.users:
            write_plt(cls.raw/user/'Trajectory'/'20240101000000.plt', user, gap=True)
            write_plt(cls.raw/user/'Trajectory'/'20240102000000.plt', user,
                      points=50 if user == '000' else 120)
        cls.result = prepare_multiuser_dataset(cls.raw, cls.root/'out', 'fixture', cls.users, 2)
        cls.output = cls.result['output_dir']
        cls.frames = {s: read_dataset_csv(cls.output/f'{s}.csv') for s in ('train','validation','test')}
        cls.originals = read_dataset_csv(cls.output/'source_normal.csv')
        cls.assignments = assign_user_splits(cls.users)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copies(self):
        return {s: f.copy(deep=True) for s, f in self.frames.items()}

    def sample(self):
        frame = self.frames['validation']
        sample_id = frame.loc[frame.is_synthetic & frame.anomaly_type.eq('abnormal_speed'), 'sample_id'].iloc[0]
        return frame.loc[frame.sample_id.eq(sample_id)].copy().reset_index(drop=True)

    def test_user_ids_are_strings_with_leading_zero(self):
        users = set(self.originals.user_id)
        self.assertEqual(users, set(self.users))
        self.assertTrue(all(isinstance(u, str) and len(u)==3 for u in users))

    def test_loader_distinguishes_same_filenames_across_users(self):
        featured, manifest = load_multiuser_features(self.raw, self.users, 2)
        self.assertEqual(manifest.trajectory_id.nunique(), 10)
        self.assertEqual(featured.trajectory_id.nunique(), 10)
        self.assertIn('000__20240101000000', set(featured.trajectory_id))
        self.assertIn('001__20240101000000', set(featured.trajectory_id))

    def test_original_filename_is_preserved(self):
        expected = self.originals.user_id + '__' + self.originals.original_trajectory_id
        self.assertTrue(self.originals.trajectory_id.eq(expected).all())

    def test_source_point_indices_are_contiguous_per_original(self):
        for _, group in self.originals.groupby('source_trajectory_id'):
            self.assertEqual(group.source_point_index.tolist(), list(range(len(group))))

    def test_feature_state_resets_at_each_user_and_route_boundary(self):
        featured, _ = load_multiuser_features(self.raw, self.users, 2)
        first = featured.groupby('trajectory_id').head(1)
        self.assertTrue(first[['time_diff_sec','distance_m','speed_mps','stop_duration_sec']].eq(0).all().all())
        second = featured.groupby('trajectory_id').nth(1)
        self.assertTrue(second.distance_m.between(1,5).all())

    def test_user_split_is_reproducible_and_input_order_independent(self):
        self.assertEqual(self.assignments, assign_user_splits(self.users[::-1]))
        counts = pd.Series(assign_user_splits([f'{i:03}' for i in range(20)])).value_counts().to_dict()
        self.assertEqual(counts, {'train':12,'validation':4,'test':4})

    def test_splits_are_user_disjoint(self):
        report = validate_dataset(self.frames, self.originals, self.assignments)
        self.assertEqual(report['user_overlap'], 0)
        self.assertEqual(report['source_trajectory_overlap'], 0)

    def test_user_overlap_is_rejected(self):
        frames = self.copies()
        frames['test'].loc[:, 'user_id'] = frames['train'].user_id.iloc[0]
        with self.assertRaisesRegex(ValueError, 'User overlap'):
            validate_dataset(frames, self.originals, self.assignments)

    def test_source_overlap_is_rejected(self):
        frames = self.copies()
        frames['test'].loc[:, 'source_trajectory_id'] = frames['train'].source_trajectory_id.iloc[0]
        with self.assertRaisesRegex(ValueError, 'Source trajectory overlap'):
            validate_dataset(frames, self.originals, self.assignments)

    def test_train_has_only_valid_original_normal_rows(self):
        train = self.frames['train']
        self.assertFalse(train.is_synthetic.any())
        self.assertTrue(train.anomaly_label.eq(0).all())
        self.assertTrue(train.source_quality_valid.all())

    def test_train_synthetic_contamination_is_rejected(self):
        frames = self.copies()
        frames['train'].loc[0, 'is_synthetic'] = True
        with self.assertRaises(ValueError):
            validate_dataset(frames, self.originals, self.assignments)

    def test_train_anomaly_contamination_is_rejected(self):
        frames = self.copies()
        frames['train'].loc[0, 'anomaly_label'] = 1
        with self.assertRaises(ValueError):
            validate_dataset(frames, self.originals, self.assignments)

    def test_full_synthetic_lineage_and_four_types_are_preserved(self):
        for split in ('validation','test'):
            frame = self.frames[split]
            for source, group in frame.groupby('source_trajectory_id'):
                expected = set(self.originals.loc[self.originals.source_trajectory_id.eq(source), 'source_point_index'])
                self.assertEqual(set(group.loc[group.is_synthetic,'anomaly_type']),
                                 {'route_deviation','abnormal_speed','long_stop','direction_change'})
                for _, sample in group.loc[group.is_synthetic].groupby('sample_id'):
                    self.assertEqual(set(sample.source_point_index), expected)

    def test_changed_timestamps_use_source_point_index_for_quality(self):
        sample = self.sample()
        source = self.originals.loc[self.originals.source_trajectory_id.eq(sample.source_trajectory_id.iloc[0])]
        self.assertFalse(sample.timestamp.reset_index(drop=True).equals(source.timestamp.reset_index(drop=True)))
        merged = attach_source_quality(sample, self.originals)
        assert_frame_equal(sample[['source_point_index','source_quality_valid','quality_reason']],
                           merged[['source_point_index','source_quality_valid','quality_reason']], check_dtype=False)

    def test_shuffled_samples_still_match_quality_by_lineage(self):
        sample = self.sample().sample(frac=1, random_state=7).reset_index(drop=True)
        merged = attach_source_quality(sample, self.originals)
        self.assertEqual(sample.source_point_index.tolist(), merged.source_point_index.tolist())
        self.assertEqual(sample.source_quality_valid.tolist(), merged.source_quality_valid.tolist())

    def test_large_synthetic_values_do_not_change_source_quality(self):
        sample = self.sample()
        position = sample.index[sample.source_quality_valid][0]
        sample.loc[position, 'speed_mps'] = 5000.0
        merged = attach_source_quality(sample, self.originals)
        self.assertTrue(merged.loc[merged.source_point_index.eq(sample.loc[position,'source_point_index']), 'source_quality_valid'].iloc[0])
        self.assertTrue(merged.synthetic_value_valid.all())

    def test_missing_lineage_is_rejected(self):
        sample = self.sample()
        sample.loc[sample.index[0], 'source_point_index'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Missing lineage'):
            attach_source_quality(sample, self.originals)

    def test_unknown_source_point_is_rejected(self):
        sample = self.sample()
        sample.loc[sample.index[0], 'source_point_index'] = 999999
        with self.assertRaisesRegex(ValueError, 'missing original'):
            attach_source_quality(sample, self.originals)

    def test_duplicate_original_lineage_is_rejected(self):
        originals = pd.concat([self.originals, self.originals.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate lineage'):
            attach_source_quality(self.sample(), originals)

    def test_duplicate_lineage_within_sample_is_rejected(self):
        sample = self.sample()
        sample = pd.concat([sample,sample.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate lineage'):
            attach_source_quality(sample, self.originals)

    def test_fractional_and_negative_lineage_indices_are_rejected(self):
        for value in (-1, .5):
            with self.subTest(value=value):
                sample = self.sample()
                sample['source_point_index'] = sample.source_point_index.astype(float)
                sample.loc[sample.index[0], 'source_point_index'] = value
                with self.assertRaises(ValueError):
                    attach_source_quality(sample, self.originals)

    def test_csv_roundtrip_preserves_ids_and_subsecond_timestamps(self):
        self.assertTrue(self.result['leakage']['csv_roundtrip_passed'])
        self.assertIn('000', set(read_dataset_csv(self.output/'input_manifest.csv').user_id))
        sample = self.sample()
        self.assertTrue((sample.timestamp.dt.microsecond.ne(0) | sample.timestamp.dt.nanosecond.ne(0)).any())

    def test_nan_and_inf_features_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                frame = self.frames['train'].copy()
                frame.loc[0,'speed_mps'] = value
                with self.assertRaisesRegex(ValueError, 'NaN/inf'):
                    validate_numeric_values(frame)

    def test_evaluation_excludes_synthetic_normal_and_bad_source_rows(self):
        for split in ('validation','test'):
            selected = evaluation_rows(self.frames[split])
            saved = read_dataset_csv(self.output/f'{split}_evaluation.csv')
            assert_frame_equal(selected, saved, check_dtype=False)
            self.assertFalse((selected.is_synthetic & selected.anomaly_label.eq(0)).any())
            self.assertTrue(selected.source_quality_valid.all())

    def test_invalid_quality_claim_is_rejected(self):
        frames = self.copies()
        idx = frames['validation'].index[~frames['validation'].source_quality_valid][0]
        frames['validation'].loc[idx,'source_quality_valid'] = True
        with self.assertRaisesRegex(ValueError, 'Source quality mismatch'):
            validate_dataset(frames, self.originals, self.assignments)

    def test_all_required_artifacts_exist(self):
        names = ['input_manifest.csv','user_quality_summary.csv','trajectory_quality_summary.csv',
                 'user_split_manifest.csv','synthetic_anomaly_manifest.csv','train.csv','validation.csv',
                 'test.csv','leakage_report.json','dataset_summary.json']
        for name in names:
            self.assertTrue((self.output/name).is_file(), name)
        self.assertEqual(self.result['summary']['excluded_trajectory_count'], 1)

    def test_existing_dataset_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            prepare_multiuser_dataset(self.raw,self.root/'out','fixture',self.users,2)

    def test_saved_dataset_revalidation_verifies_checksums(self):
        report = validate_saved_dataset(self.output)
        self.assertTrue(report['checksums_verified'])
        self.assertTrue(report['structural_checks_passed'])

    def test_tampered_csv_is_rejected_by_checksum(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'copy'
            shutil.copytree(self.output, copy)
            with (copy / 'train.csv').open('a', encoding='utf-8') as stream:
                stream.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                validate_saved_dataset(copy)

    def test_global_id_collision_is_rejected(self):
        originals = self.originals.copy()
        first = originals.source_trajectory_id.iloc[0]
        other = originals.source_trajectory_id.ne(first)
        originals.loc[other, 'trajectory_id'] = originals.trajectory_id.iloc[0]
        with self.assertRaisesRegex(ValueError, 'Global trajectory ID collision'):
            validate_dataset(self.frames, originals, self.assignments)

    def test_lineage_cannot_borrow_another_users_source(self):
        sample = self.sample()
        sample['user_id'] = '999'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            attach_source_quality(sample, self.originals)

    def test_each_type_has_exactly_one_synthetic_sample(self):
        manifest = read_dataset_csv(self.output / 'synthetic_anomaly_manifest.csv')
        self.assertFalse(manifest.duplicated(['source_trajectory_id','anomaly_type']).any())
        self.assertTrue(manifest.groupby('source_trajectory_id').size().eq(4).all())

    def test_model_feature_schema_excludes_metadata(self):
        self.assertEqual(len(MODEL_FEATURE_COLUMNS),8)
        self.assertFalse(set(MODEL_FEATURE_COLUMNS) & {'user_id','dataset_split','anomaly_label','source_point_index'})

    def test_synthetic_seed_is_stable_and_source_specific(self):
        one = synthetic_seed_for('000__a','abnormal_speed',42)
        self.assertEqual(one, synthetic_seed_for('000__a','abnormal_speed',42))
        self.assertNotEqual(one, synthetic_seed_for('001__a','abnormal_speed',42))


class FingerprintAndInputTests(unittest.TestCase):
    def test_invalid_or_duplicate_user_ids_are_rejected(self):
        for ids in (['0'], [0], ['000','000'], ['../']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_user_ids(ids)

    def test_too_few_users_are_rejected(self):
        with self.assertRaises(ValueError):
            assign_user_splits(['000','001'])

    def test_fingerprint_detects_copies_with_shifted_absolute_date(self):
        frame = pd.DataFrame({'latitude':[39.,39.01], 'longitude':[116.,116.01],
                              'timestamp':pd.date_range('2024-01-01',periods=2,freq='10s')})
        shifted = frame.copy()
        shifted['timestamp'] += pd.Timedelta(days=365)
        self.assertEqual(trajectory_fingerprint(frame),trajectory_fingerprint(shifted))
        shifted.loc[1,'longitude'] += .1
        self.assertNotEqual(trajectory_fingerprint(frame),trajectory_fingerprint(shifted))

    def test_cross_split_duplicates_are_reported_without_deleting_records(self):
        inputs = pd.DataFrame({'user_id':['000','001','002'], 'trajectory_id':['000__a','001__a','002__b'],
                               'content_fingerprint':['copy','copy','other'], 'eligible_for_dataset':[True]*3})
        original = inputs.copy(deep=True)
        report = duplicate_fingerprint_report(inputs,{'000':'train','001':'test','002':'validation'})
        self.assertEqual(report['duplicate_fingerprint_count'],1)
        self.assertEqual(report['cross_split_duplicate_fingerprint_count'],1)
        self.assertEqual(report['duplicate_trajectory_count'],2)
        assert_frame_equal(inputs,original)

    def test_duplicate_within_split_has_no_cross_split_flag(self):
        inputs = pd.DataFrame({'user_id':['000','000'], 'trajectory_id':['000__a','000__b'],
                               'content_fingerprint':['same','same'], 'eligible_for_dataset':[True,True]})
        report = duplicate_fingerprint_report(inputs,{'000':'train'})
        self.assertEqual(report['duplicate_fingerprint_count'],1)
        self.assertEqual(report['cross_split_duplicate_fingerprint_count'],0)


if __name__ == '__main__':
    unittest.main()
