"""Stage 6.5 exact audit joins, observability tiers and frozen prediction metrics."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src import audit_synthetic_route_labels as audit
from src.train_autoencoder import FEATURE_COLUMNS


def fixture():
    originals=[];copies=[]
    for number in range(19):
        user=('001','008','013','017')[number%4];source=f'{user}__fixture{number:02d}';length=20 if number<18 else 18
        n=length+2;sample=source+'__route_deviation__01'
        f=pd.DataFrame({'user_id':user,'trajectory_id':source,'source_trajectory_id':source,'source_point_index':range(n),
            'dataset_split':'test','timestamp':pd.date_range('2024-01-01',periods=n,freq='10s'),
            'latitude':39.+np.arange(n)*.0001,'longitude':116.+number*.001,'bearing_deg':0.,
            'time_diff_sec':10.,'distance_m':11.,'speed_mps':1.1,'acceleration_mps2':0.,
            'direction_change_deg':0.,'stop_duration_sec':0.})
        originals.append(f);g=f.copy();g['sample_id']=sample;g['anomaly_label']=0;g.loc[1:length,'anomaly_label']=1
        g.loc[2:length-1,'latitude']+=.001;g.loc[2:length,'distance_m']+=2.;copies.append(g)
    originals=pd.concat(originals,ignore_index=True);copies=pd.concat(copies,ignore_index=True)
    full,runs=audit.segment_structure(audit.point_deltas(copies,originals))
    points=full.loc[full.anomaly_label.eq(1)].reset_index(drop=True).copy()
    for signal in audit.CONTEXT_SIGNALS:
        points[signal+'_original_score_m']=10.;points[signal+'_synthetic_score_m']=np.where(points.feature_changed_count.gt(0),30.,10.)
    points['synthetic_reference_distance_m']=50.;points['original_reference_distance_m']=40.
    points=audit.classify_tiers(points)
    preds=[]
    for model,seeds in [('statistical_rule',[None]),('isolation_forest',[7,21,42,100,2026]),('autoencoder',[7,21,42,100,2026])]:
        for seed in seeds:
            p=points[audit.SCORE_KEY+['user_id']].copy();p['model']=model;p['seed']=seed
            p['predicted_anomaly']=(np.arange(len(p))%3==0).astype(int);p['anomaly_score']=p.predicted_anomaly+1.
            p['original_predicted_anomaly']=0;p['original_anomaly_score']=0.;preds.append(p)
    return originals,copies,full,runs,points,pd.concat(preds,ignore_index=True)


class RouteLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.originals,cls.copies,cls.full,cls.runs,cls.points,cls.predictions=fixture()

    def test_exact_lineage_join_preserves_ids(self):
        p=audit.point_deltas(self.copies,self.originals)
        self.assertEqual(len(p),len(self.copies));self.assertIn('001',set(p.user_id));self.assertTrue(p.user_id.eq(p.original_user_id).all())

    def test_original_filename_metadata_does_not_collide(self):
        copies=self.copies.copy();originals=self.originals.copy()
        copies['original_trajectory_id']='20240101000000';originals['original_trajectory_id']='20240101000000'
        result=audit.point_deltas(copies,originals)
        self.assertTrue(result.original_trajectory_id.eq('20240101000000').all())
        self.assertTrue(result.paired_trajectory_id.eq(result.trajectory_id).all())

    def test_missing_source_fails(self):
        with self.assertRaises(ValueError):audit.point_deltas(self.copies,self.originals.iloc[1:])

    def test_duplicate_source_and_sample_fail(self):
        with self.assertRaises(ValueError):audit.point_deltas(self.copies,pd.concat([self.originals,self.originals.iloc[:1]]))
        with self.assertRaises(ValueError):audit.point_deltas(pd.concat([self.copies,self.copies.iloc[:1]]),self.originals)

    def test_source_user_mismatch_fails(self):
        f=self.copies.copy();f.loc[0,'user_id']='999'
        with self.assertRaises(ValueError):audit.point_deltas(f,self.originals)

    def test_haversine_displacement_and_coordinate_deltas(self):
        p=self.points.loc[self.points.directly_displaced].iloc[0]
        self.assertAlmostEqual(p.latitude_delta_deg,.001,places=10);self.assertAlmostEqual(p.actual_displacement_m,111.1949266,places=5)
        self.assertEqual(p.timestamp_delta_sec,0)

    def test_zero_displacement_38_regression(self):
        self.assertEqual(len(self.points),378);self.assertEqual(int(self.points.zero_displacement.sum()),38)
        self.assertEqual(int(self.points.actual_displacement_m.eq(0).sum()),38)

    def test_identical_feature_19_regression(self):
        self.assertEqual(int(self.points.feature_near_identical.sum()),19)
        self.assertTrue(self.points.loc[self.points.feature_near_identical,'identifiability_tier'].eq('tier4').all())

    def test_native_feature_tolerance(self):
        f=self.copies.copy();f.loc[0,'distance_m']=self.originals.distance_m.iloc[0]+1e-7
        result=audit.point_deltas(f,self.originals);self.assertFalse(bool(result.distance_m_changed.iloc[0]))
        f.loc[0,'distance_m']+=1e-3;result=audit.point_deltas(f,self.originals);self.assertTrue(bool(result.distance_m_changed.iloc[0]))

    def test_feature_changed_count_categories(self):
        np.testing.assert_array_equal(audit.feature_category([0,1,2,3,4,8]),['0','1','2-3','2-3','4+','4+'])
        with self.assertRaises(ValueError):audit.feature_category([9])

    def test_per_feature_delta_quantiles(self):
        s=audit.feature_summary(self.points).set_index('feature')
        self.assertEqual(s.loc['distance_m','changed_count'],359);self.assertEqual(s.loc['time_diff_sec','changed_count'],0)
        self.assertEqual(s.loc['distance_m','absolute_delta_p90'],2.)

    def tier_fixture(self):
        p=self.points.iloc[:4].copy();p['directly_displaced']=[True,False,False,False];p['feature_changed_count']=[0,1,0,0]
        for signal in audit.CONTEXT_SIGNALS:p[signal+'_original_score_m']=10.;p[signal+'_synthetic_score_m']=[10.,10.,11.,10.]
        return audit.classify_tiers(p)

    def test_tier1_direct_priority(self):self.assertEqual(self.tier_fixture().identifiability_tier.iloc[0],'tier1')
    def test_tier2_derived_only(self):self.assertEqual(self.tier_fixture().identifiability_tier.iloc[1],'tier2')
    def test_tier3_context_only(self):self.assertEqual(self.tier_fixture().identifiability_tier.iloc[2],'tier3')
    def test_tier4_unidentifiable(self):self.assertEqual(self.tier_fixture().identifiability_tier.iloc[3],'tier4')

    def test_tiers_mutually_exclusive_exhaustive(self):
        p=self.tier_fixture();self.assertEqual(p.identifiability_tier.tolist(),list(audit.TIERS));self.assertEqual(int(audit.tier_summary(self.points).row_count.sum()),378)

    def test_context_equality_tolerance_and_missing_guard(self):
        p=self.tier_fixture();p['directly_displaced']=False;p['feature_changed_count']=0
        for signal in audit.CONTEXT_SIGNALS:p[signal+'_synthetic_score_m']=p[signal+'_original_score_m']+1e-6
        self.assertTrue(audit.classify_tiers(p).identifiability_tier.eq('tier4').all())
        p.loc[p.index[0],'w10g0_prefix_synthetic_score_m']=np.nan
        with self.assertRaisesRegex(ValueError,'Incomplete'):audit.classify_tiers(p)

    def test_infinite_and_invalid_context_rejected(self):
        for value in (np.inf,'invalid'):
            p=self.points.iloc[:2].copy();p['stage63_prefix_synthetic_score_m']=value
            with self.assertRaises(ValueError):audit.classify_tiers(p)

    def test_segment_start_interior_end(self):
        b=audit.position_summary(self.points,'segment_boundary').set_index('segment_boundary')
        self.assertEqual(b.loc['start','labelled_rows'],19);self.assertEqual(b.loc['end','labelled_rows'],19)
        self.assertEqual(b.loc['interior','labelled_rows'],340);self.assertEqual(b.loc['start','tier4_count'],19)

    def test_progression_positions_full_copy(self):
        g=self.full.loc[self.full.sample_id.eq(self.full.sample_id.iloc[0])&self.full.anomaly_label.eq(1)]
        self.assertEqual(g.segment_position.tolist(),list(range(20)));self.assertEqual(g.segment_relative_position.iloc[-1],1.)
        self.assertEqual(set(g.progression_bucket),{'early','middle','late'})

    def test_disjoint_label_runs_and_singleton(self):
        f=self.full.loc[self.full.sample_id.eq(self.full.sample_id.iloc[0])].copy();f['anomaly_label']=0;f.loc[f.index[2:5],'anomaly_label']=1;f.loc[f.index[9],'anomaly_label']=1
        structured,runs=audit.segment_structure(f);self.assertEqual(runs.full_labelled_length.tolist(),[3,1])
        self.assertEqual(structured.loc[structured.segment_boundary.eq('singleton'),'segment_length'].tolist(),[1])

    def test_noncontiguous_or_nonincreasing_stream_rejected(self):
        f=self.full.loc[self.full.sample_id.eq(self.full.sample_id.iloc[0])].copy();f.loc[f.index[4:],'source_point_index']+=1
        with self.assertRaises(ValueError):audit.segment_structure(f)
        f=self.full.loc[self.full.sample_id.eq(self.full.sample_id.iloc[0])].copy();f.loc[f.index[3],'timestamp']=f.timestamp.iloc[2]
        with self.assertRaises(ValueError):audit.segment_structure(f)

    def test_previous_point_boundary_propagation(self):
        end=self.points.loc[self.points.segment_boundary.eq('end')]
        self.assertTrue(end.previous_source_modified.all());self.assertTrue(end.feature_propagation_only.all())
        start=self.points.loc[self.points.segment_boundary.eq('start')];self.assertFalse(start.causal_raw_history_changed.any())

    def test_displacement_bin_edges_and_nonfinite(self):
        np.testing.assert_array_equal(audit.displacement_bin([0,1,10,25,50,100,200]),list(audit.BIN_ORDER))
        for values in ([-1],[np.nan],[np.inf]):
            with self.assertRaises(ValueError):audit.displacement_bin(values)

    def test_prediction_exact_join(self):
        route=self.predictions.loc[self.predictions.model.eq('statistical_rule')].copy()
        normal=route[audit.KEY+['user_id','anomaly_score','predicted_anomaly']]
        r=audit.prediction_join(self.points,route,normal);self.assertEqual(len(r),378)
        with self.assertRaises(ValueError):audit.prediction_join(self.points,route.iloc[:-1],normal)

    def test_prediction_user_and_binary_validation(self):
        route=self.predictions.loc[self.predictions.model.eq('statistical_rule')].copy();normal=route[audit.KEY+['user_id','anomaly_score','predicted_anomaly']].copy()
        route.loc[route.index[0],'predicted_anomaly']=2
        with self.assertRaises(ValueError):audit.prediction_join(self.points,route,normal)
        route=self.predictions.loc[self.predictions.model.eq('statistical_rule')].copy();route.loc[route.index[0],'user_id']='999'
        with self.assertRaises(ValueError):audit.prediction_join(self.points,route,normal)

    def test_detector_recall_by_tier_and_empty_tier(self):
        rows=audit.recall_rows(self.points,self.predictions,'identifiability_tier',audit.TIERS);summary=audit.recall_summary(rows,'identifiability_tier')
        self.assertEqual(len(rows),44);self.assertEqual(len(summary),12)
        self.assertTrue(summary.loc[summary.identifiability_tier.eq('tier3'),'recall_mean'].isna().all())
        for _,g in rows.groupby(['model','seed'],dropna=False):self.assertEqual(int(g.anomaly_row_count.sum()),378)

    def test_counterfactual_scope_and_counts(self):
        c=audit.counterfactual_metrics(self.points,self.predictions);self.assertTrue(c.excluded_tier4_rows.eq(19).all())
        self.assertTrue(c.observable_rows.eq(359).all());self.assertFalse(c.official_metric_changed.any())
        self.assertTrue(c.metric_scope.eq('counterfactual observable-subset recall').all())
        row=c.iloc[0];self.assertAlmostEqual(row.counterfactual_observable_subset_recall,row.observable_detected/359)

    def test_event_count_and_any_detection(self):
        e=audit.event_metrics(self.points,self.predictions,self.runs);self.assertEqual(len(e),19*11)
        self.assertTrue(e.event_detected.all());s=audit.event_summary(e);self.assertTrue(s.loc[s.user_id.eq('ALL'),'event_count'].eq(19).all())

    def test_first_detection_delay_full_index_and_time(self):
        p=self.predictions.copy();p['predicted_anomaly']=0;point=self.points.loc[self.points.sample_id.eq(self.points.sample_id.iloc[0])].iloc[3]
        select=p.sample_id.eq(point.sample_id)&p.source_point_index.eq(point.source_point_index);p.loc[select,'predicted_anomaly']=1
        e=audit.event_metrics(self.points,p,self.runs);hit=e.loc[e.event_detected]
        self.assertTrue(hit.detection_delay_points.eq(3).all());self.assertTrue(hit.detection_delay_seconds.eq(30).all())
        self.assertTrue(e.loc[~e.event_detected,'detection_delay_seconds'].isna().all())

    def test_event_delay_preserves_quality_omitted_positions(self):
        point=self.points.loc[self.points.sample_id.eq(self.points.sample_id.iloc[0])].iloc[3]
        points=self.points.loc[~(self.points.sample_id.eq(point.sample_id)&self.points.segment_position.eq(2))]
        p=self.predictions.copy();p['predicted_anomaly']=0;p.loc[p.sample_id.eq(point.sample_id)&p.source_point_index.eq(point.source_point_index),'predicted_anomaly']=1
        # Predictions are matched to the retained evaluation universe only.
        p=p.merge(points[audit.SCORE_KEY],on=audit.SCORE_KEY,how='inner',validate='many_to_one')
        e=audit.event_metrics(points,p,self.runs);self.assertTrue(e.loc[e.event_detected,'detection_delay_points'].eq(3).all())

    def test_user_aggregate(self):
        e=audit.event_metrics(self.points,self.predictions,self.runs);u=audit.user_summary(self.points,self.predictions,e)
        self.assertEqual(set(u.user_id),{'001','008','013','017'});self.assertEqual(int(u.route_label_count.sum()),378)
        self.assertTrue(u.tier3_count.eq(0).all())

    def test_sample_aggregate_and_policy_simulation(self):
        s=audit.sample_summary(self.points,self.full,self.runs);self.assertEqual(len(s),19);self.assertEqual(int(s.evaluated_labelled_rows.sum()),378)
        policy=audit.policy_simulation(self.points);self.assertEqual(policy.loc[policy.count_unit.eq('points'),['positive','boundary_ignore','negative_candidate']].sum(axis=1).tolist(),[378,378,378])
        self.assertEqual(policy.loc[policy.policy.str.startswith('C'),'positive'].iloc[0],19)

    def test_label_and_input_frames_unchanged(self):
        originals=self.originals.copy(deep=True);copies=self.copies.copy(deep=True)
        audit.point_deltas(self.copies,self.originals);audit.classify_tiers(self.points)
        assert_frame_equal(originals,self.originals);assert_frame_equal(copies,self.copies)

    def test_raw_nan_inf_and_coordinate_bounds_rejected(self):
        for column,value in [('latitude',np.nan),('longitude',np.inf),('latitude',91.),('distance_m',np.inf)]:
            f=self.copies.copy();f.loc[0,column]=value
            with self.assertRaises(ValueError):audit.point_deltas(f,self.originals)

    def test_checksum_preservation_and_mutation_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'synthetic.csv';p.write_text('original')
            snapshot={'synthetic.csv':audit.single.sha256_file(p)};audit.assert_protected(root,snapshot)
            p.write_text('changed')
            with self.assertRaises(ValueError):audit.assert_protected(root,snapshot)
            with self.assertRaises(ValueError):audit.assert_protected(root,{'../outside':'bad'})

    def test_json_na_defined_and_infinite_rejected(self):
        self.assertEqual(audit.json_safe({'empty':np.nan,'int':np.int64(3),'bool':np.bool_(True)}),{'empty':None,'int':3,'bool':True})
        with self.assertRaises(ValueError):audit.json_safe(np.inf)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'outputs/metrics/stage6/route_label_audit').mkdir(parents=True)
            with self.assertRaises(FileExistsError):audit.run_audit(root/'data',root)

    def test_incomplete_snapshot_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'snapshot.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Incomplete'):audit.run_audit(root/'data',root,p)

    def test_full_table_export_and_figures(self):
        tables=audit.build_tables(self.points,self.full,self.runs,self.predictions)
        self.assertEqual(len(tables['route_label_point_audit']),378);self.assertEqual(int(tables['label_intervention_confusion'].row_count.sum()),len(self.full))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name,frame in tables.items():frame.to_csv(root/(name+'.csv'),index=False,na_rep='NA')
            paths=audit.create_figures(tables,root);self.assertEqual(len(paths),7);self.assertTrue(all(p.stat().st_size>1000 for p in paths))
            saved=pd.read_csv(root/'route_label_point_audit.csv',dtype={'user_id':'string'});self.assertEqual(saved.identifiability_tier.value_counts().to_dict(),{'tier1':340,'tier2':19,'tier4':19})


if __name__=='__main__':unittest.main()
