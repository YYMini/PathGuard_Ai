"""Trajectory duplicate policy, deterministic retention and pre-synthetic failure."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.load_multiuser_geolife import read_dataset_csv, trajectory_fingerprint
from src.prepare_multiuser_dataset import assign_user_splits, prepare_multiuser_dataset, validate_saved_dataset
from src.trajectory_deduplication import FingerprintLeakageError, deduplicate_trajectory_manifest
from tests.test_multiuser_dataset import write_plt


def manifest():
    return pd.DataFrame({
        'user_id':['000','000','001'],
        'trajectory_id':['000__b','000__a','001__c'],
        'original_trajectory_id':['b','a','c'],
        'content_fingerprint':['duplicate','duplicate','unique'],
        'eligible_for_dataset':[True,True,True],
        'exclusion_reason':['','',''],
        'cleaned_row_count':[120,120,100],
        'eligible_normal_row_count':[118,118,98],
    })


class TrajectoryDeduplicationTests(unittest.TestCase):
    def test_same_split_duplicate_detection_and_row_counts(self):
        result, report = deduplicate_trajectory_manifest(manifest(), {'000':'train','001':'validation'})
        self.assertEqual(report['duplicate_fingerprint_count'],1)
        self.assertEqual(report['excluded_duplicate_trajectory_count'],1)
        self.assertEqual(report['excluded_duplicate_rows'],120)
        self.assertEqual(report['excluded_duplicate_normal_rows'],118)
        self.assertEqual(set(result.loc[result.eligible_for_dataset,'trajectory_id']), {'000__a','001__c'})

    def test_retention_is_independent_of_input_order(self):
        for seed in range(5):
            _, report = deduplicate_trajectory_manifest(manifest().sample(frac=1,random_state=seed),
                                                       {'000':'train','001':'test'})
            record = report['duplicate_exclusions'][0]
            self.assertEqual(record['retained_trajectory_id'],'000__a')
            self.assertEqual(record['excluded_trajectory_id'],'000__b')

    def test_filename_order_precedes_global_id_order_with_id_tiebreak(self):
        data = manifest().iloc[:2].copy()
        data['user_id'] = ['000','001']
        data['trajectory_id'] = ['000__z','001__a']
        data['original_trajectory_id'] = ['z','a']
        _, report = deduplicate_trajectory_manifest(data,{'000':'train','001':'train'})
        self.assertEqual(report['groups'][0]['retained_trajectory_id'],'001__a')
        data['original_trajectory_id'] = ['a','a']
        _, report = deduplicate_trajectory_manifest(data,{'000':'train','001':'train'})
        self.assertEqual(report['groups'][0]['retained_trajectory_id'],'000__z')

    def test_reason_and_complete_audit_record_are_preserved(self):
        data = manifest()
        data.loc[0,'exclusion_reason'] = 'earlier_reason'
        result, report = deduplicate_trajectory_manifest(data,{'000':'train','001':'test'})
        excluded = result.loc[result.trajectory_id.eq('000__b')].iloc[0]
        self.assertEqual(excluded.exclusion_reason,'earlier_reason;exact_duplicate_trajectory')
        self.assertTrue(excluded.is_exact_duplicate)
        self.assertFalse(excluded.eligible_for_dataset)
        self.assertTrue(excluded.quality_eligible_for_dataset)
        for key in ('duplicate_fingerprint','user_id','trajectory_id','retained_trajectory_id',
                    'excluded_trajectory_id','split','excluded_rows'):
            self.assertIn(key,report['duplicate_exclusions'][0])

    def test_policy_applies_to_validation_and_test(self):
        for split in ('validation','test'):
            with self.subTest(split=split):
                result, report = deduplicate_trajectory_manifest(manifest(),{'000':split,'001':'train'})
                self.assertEqual(report['duplicate_exclusions'][0]['split'],split)
                self.assertEqual(result.loc[result.user_id.eq('000'),'eligible_for_dataset'].sum(),1)

    def test_cross_split_duplicates_raise_before_any_exclusion(self):
        data = manifest()
        data.loc[0,'user_id'] = '002'
        original = data.copy(deep=True)
        with self.assertRaises(FingerprintLeakageError) as caught:
            deduplicate_trajectory_manifest(data,{'000':'train','001':'validation','002':'test'})
        self.assertEqual(caught.exception.report['cross_split_duplicate_fingerprint_count'],1)
        self.assertEqual(caught.exception.report['duplicate_exclusions'],[])
        assert_frame_equal(data,original)

    def test_cross_split_check_includes_quality_ineligible_input(self):
        data = manifest()
        data.loc[0,'user_id'] = '002'
        data.loc[0,'eligible_for_dataset'] = False
        with self.assertRaises(FingerprintLeakageError):
            deduplicate_trajectory_manifest(data,{'000':'train','001':'validation','002':'test'})

    def test_input_manifest_is_not_mutated(self):
        data=manifest(); original=data.copy(deep=True)
        deduplicate_trajectory_manifest(data,{'000':'train','001':'test'})
        assert_frame_equal(data,original)

    def test_multiple_duplicate_members_keep_only_one(self):
        data=manifest(); data.loc[2,'content_fingerprint']='duplicate'
        result, report=deduplicate_trajectory_manifest(data,{'000':'train','001':'train'})
        self.assertEqual(result.eligible_for_dataset.sum(),1)
        self.assertEqual(report['excluded_duplicate_trajectory_count'],2)

    def test_csv_roundtrip_preserves_fingerprint_and_dedup_annotation(self):
        result,_=deduplicate_trajectory_manifest(manifest(),{'000':'train','001':'test'})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input_manifest.csv'; result.to_csv(path,index=False)
            restored=read_dataset_csv(path)
            assert_frame_equal(result,restored,check_dtype=False)
            # Recompute from serialized GPS values, rather than only checking
            # that a precomputed hash string survives the manifest CSV.
            gps = pd.DataFrame({
                'latitude': [39.12345671, 39.12345681, 39.12345691],
                'longitude': [116.12345671, 116.12345681, 116.12345691],
                'timestamp': pd.to_datetime([
                    '2024-01-01 00:00:00.000000001',
                    '2024-01-01 00:00:10.123456789',
                    '2024-01-01 00:00:20.987654321']),
            })
            fingerprint = trajectory_fingerprint(gps)
            gps_path = Path(tmp)/'gps.csv'
            gps.to_csv(gps_path,index=False)
            self.assertEqual(trajectory_fingerprint(read_dataset_csv(gps_path)),fingerprint)

    def test_missing_fingerprint_is_rejected(self):
        data=manifest(); data.loc[0,'content_fingerprint']=None
        with self.assertRaises(ValueError):
            deduplicate_trajectory_manifest(data,{'000':'train','001':'test'})


class DeduplicationPipelineTests(unittest.TestCase):
    def make_raw(self,root):
        users=[f'{i:03}' for i in range(5)]
        for user in users:
            write_plt(root/user/'Trajectory'/'20240101000000.plt',user,gap=True)
        return users,assign_user_splits(users)

    def test_dedup_precedes_synthetic_and_preserves_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'raw'; users,assignments=self.make_raw(raw)
            user=next(u for u,s in assignments.items() if s=='validation')
            source=raw/user/'Trajectory'/'20240101000000.plt'
            duplicate=source.with_name('20240102000000.plt'); duplicate.write_bytes(source.read_bytes())
            result=prepare_multiuser_dataset(raw,root/'out','dedup',users,2)
            summary=result['summary']; self.assertEqual(summary['quality_eligible_trajectory_count_before_dedup'],6)
            self.assertEqual(summary['eligible_trajectory_count'],5)
            self.assertEqual(summary['deduplication']['excluded_duplicate_trajectory_count'],1)
            split_manifest=read_dataset_csv(result['output_dir']/'user_split_manifest.csv')
            self.assertEqual(dict(zip(split_manifest.user_id,split_manifest.dataset_split)),assignments)
            synthetic=read_dataset_csv(result['output_dir']/'synthetic_anomaly_manifest.csv')
            self.assertNotIn(f'{user}__20240102000000',set(synthetic.source_trajectory_id))
            self.assertEqual(summary['splits']['validation']['synthetic_sample_count'],4)
            self.assertTrue(validate_saved_dataset(result['output_dir'])['structural_checks_passed'])
            for name in ('input_manifest','trajectory_quality_summary'):
                audit=read_dataset_csv(result['output_dir']/f'{name}.csv')
                row=audit.loc[audit.trajectory_id.eq(f'{user}__20240102000000')].iloc[0]
                self.assertEqual(row.exclusion_reason,'exact_duplicate_trajectory')
                self.assertEqual(row.retained_trajectory_id,f'{user}__20240101000000')
                self.assertEqual(row.duplicate_excluded_rows,120)
            for name in ('duplicate_report','leakage_report','dataset_summary'):
                record=json.loads((result['output_dir']/f'{name}.json').read_text(encoding='utf-8'))
                if name=='dataset_summary': record=record['deduplication']
                self.assertEqual(record['duplicate_exclusions'][0]['excluded_rows'],120)

    def test_cross_split_copies_fail_without_generating_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'raw'; users,assignments=self.make_raw(raw)
            train=next(u for u,s in assignments.items() if s=='train')
            test=next(u for u,s in assignments.items() if s=='test')
            (raw/test/'Trajectory'/'20240101000000.plt').write_bytes(
                (raw/train/'Trajectory'/'20240101000000.plt').read_bytes())
            with patch('src.prepare_multiuser_dataset.generate_evaluation_synthetic') as generate:
                with self.assertRaises(FingerprintLeakageError):
                    prepare_multiuser_dataset(raw,root/'out','leakage',users,1)
                generate.assert_not_called()
            output=root/'out'/'leakage'
            report=json.loads((output/'leakage_report.json').read_text(encoding='utf-8'))
            self.assertFalse(report['ready_for_stage5_b'])
            self.assertEqual(report['duplicate_exclusions'],[])
            self.assertFalse((output/'train.csv').exists())


if __name__=='__main__':
    unittest.main()
