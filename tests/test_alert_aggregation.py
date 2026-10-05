"""Stage 6.7 causal gates, Validation selection, lock and Test isolation."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import audit_alert_aggregation as audit
from src import audit_synthetic_route_labels as labels
from src import audit_event_evaluation_protocol as events
from src import train_multiuser_autoencoder as single
from src.alert_policy import AlertPolicy, LockedPolicies, apply_policy, GATE_DEFINITIONS


def stream(bits, step=3, source='s', sample='copy', user='001'):
    return pd.DataFrame({'source_point_index':np.arange(len(bits)), 'source_trajectory_id':source,
                         'sample_id':sample,'user_id':user,'timestamp':pd.date_range('2020-01-01',periods=len(bits),freq=f'{step}s'),
                         'time_diff_sec':step,'predicted_anomaly':bits})


def event_record(frame,start=0,end=None):
    end=len(frame)-1 if end is None else end
    return {'event_id':'copy__run_01','sample_id':'copy','user_id':'001','source_trajectory_id':'s',
            'event_start_index':start,'event_end_index':end,'event_start_timestamp':frame.timestamp.iloc[start],
            'event_end_timestamp':frame.timestamp.iloc[end],'event_point_count':end-start+1,
            'evaluated_point_count':end-start+1,'excluded_quality_point_count':0}


def config():
    return json.loads((single.ROOT/audit.CONFIG).read_text(encoding='utf-8'))


def fake_grid():
    records=[]
    for model in audit.FAMILIES:
        for seed in ([np.nan] if model=='statistical_rule' else [7,21,42,100,2026]):
            for p in config()['policies']:
                records.append({'model':model,'seed':seed,'policy_id':p['policy_id'],'dataset_split':'validation',
                    'notification_event_detection_rate':.8,'notification_early25_rate':.6,
                    'notification_delay_points_median':1.,'false_notifications_per_1000_normal_points':10.,
                    'false_notifications_per_trajectory':10.,'false_notification_count':100})
    return pd.DataFrame(records)


def favorable(grid,policy='G1_C0',model=None):
    result=grid.copy()
    mask=result.policy_id.eq(policy)
    if model:mask&=result.model.eq(model)
    result.loc[mask,'false_notifications_per_1000_normal_points']=4.
    result.loc[mask,'false_notifications_per_trajectory']=4.
    return result


class AlertPolicyTests(unittest.TestCase):
    def test_raw_identity(self):
        f=stream([0,1,1,0,1])
        r=apply_policy(f,AlertPolicy('G0',0))
        self.assertEqual(r.gate_positive.astype(int).tolist(),f.predicted_anomaly.tolist())

    def test_consecutive_two(self):
        r=apply_policy(stream([1,1,0,1,1]),AlertPolicy('G1',0))
        self.assertEqual(r.gate_positive.tolist(),[False,True,False,False,True])

    def test_consecutive_three(self):
        r=apply_policy(stream([1,1,1,0,1,1]),AlertPolicy('G2',0))
        self.assertEqual(r.gate_positive.tolist(),[False,False,True,False,False,False])

    def test_consecutive_five(self):
        r=apply_policy(stream([1,1,1,1,1,1,0]),AlertPolicy('G3',0))
        self.assertEqual(r.gate_positive.tolist(),[False,False,False,False,True,True,False])

    def test_vote_two_of_three(self):
        r=apply_policy(stream([1,1,0,0,1]),AlertPolicy('G4',0))
        self.assertEqual(r.gate_positive.tolist(),[False,False,True,False,False])

    def test_vote_three_of_five(self):
        r=apply_policy(stream([1,0,1,0,1,0]),AlertPolicy('G5',0))
        self.assertEqual(r.gate_positive.tolist(),[False,False,False,False,True,False])

    def test_vote_five_of_ten(self):
        r=apply_policy(stream([1,0]*5+[0]),AlertPolicy('G6',0))
        self.assertEqual(r.gate_positive.tolist(),[False]*9+[True,False])

    def test_no_future_access_prefix_invariance_all_policies(self):
        first=stream([0,1,0,1,1,0,1,1,1,0,1,0])
        full=stream(first.predicted_anomaly.tolist()+[1]*15)
        for p in config()['policies']:
            policy=AlertPolicy.from_dict(p)
            a,b=apply_policy(first,policy),apply_policy(full,policy).iloc[:len(first)]
            self.assertEqual(a.gate_positive.tolist(),b.gate_positive.tolist())
            self.assertEqual(a.notification_emitted.tolist(),b.notification_emitted.tolist())

    def test_no_centered_rolling_or_label_input(self):
        f=stream([1,0,1,1,0])
        a=apply_policy(f,AlertPolicy('G4',60))
        f['anomaly_label']=[1,1,1,1,1]
        b=apply_policy(f,AlertPolicy('G4',60))
        self.assertEqual(a.notification_emitted.tolist(),b.notification_emitted.tolist())

    def test_trajectory_gate_reset(self):
        a=apply_policy(stream([1,1],source='a'),AlertPolicy('G2',0))
        b=apply_policy(stream([1],source='b'),AlertPolicy('G2',0))
        self.assertFalse(a.gate_positive.any())
        self.assertFalse(b.gate_positive.any())
        with self.assertRaises(ValueError):apply_policy(pd.concat([stream([1],source='a'),stream([1],source='b')]),AlertPolicy('G1',0))

    def test_warmup_full_window_all_gates(self):
        for gate,definition in GATE_DEFINITIONS.items():
            w=definition[1]
            r=apply_policy(stream([1]*w),AlertPolicy(gate,0))
            self.assertEqual(r.gate_positive.tolist(),[False]*(w-1)+[True])

    def test_gate_run_candidates(self):
        r=apply_policy(stream([0,1,1,0,1]),AlertPolicy('G0',0))
        self.assertEqual(r.notification_candidate.tolist(),[False,True,False,False,True])

    def test_cooldown_zero(self):
        r=apply_policy(stream([1,0,1,0,1]),AlertPolicy('G0',0))
        self.assertEqual(r.notification_emitted.tolist(),[True,False,True,False,True])

    def test_cooldown_sixty_inclusive(self):
        r=apply_policy(stream([1,0,1,0,1],step=30),AlertPolicy('G0',60))
        self.assertEqual(r.notification_emitted.tolist(),[True,False,True,False,True])

    def test_cooldown_three_hundred(self):
        r=apply_policy(stream([1,0,1,0,1,0,1],step=50),AlertPolicy('G0',300))
        self.assertEqual(np.flatnonzero(r.notification_emitted).tolist(),[0,6])

    def test_cooldown_trajectory_reset(self):
        for source in ['a','b']:
            self.assertTrue(apply_policy(stream([1],source=source),AlertPolicy('G0',300)).notification_emitted.iloc[0])

    def test_suppressed_does_not_restart_clock(self):
        f=stream([1,0,1,0,1],step=20)
        f.loc[3,'timestamp']=pd.Timestamp('2020-01-01 00:00:50')
        f.loc[4,'timestamp']=pd.Timestamp('2020-01-01 00:01:00')
        r=apply_policy(f,AlertPolicy('G0',60))
        self.assertEqual(np.flatnonzero(r.notification_emitted).tolist(),[0,4])
        self.assertTrue(r.suppressed_by_cooldown.iloc[2])
        self.assertEqual(r.suppressing_notification_source_index.iloc[2],0)

    def test_no_deferred_notification_inside_same_run(self):
        r=apply_policy(stream([1,0,1,1,1],step=20),AlertPolicy('G0',60))
        self.assertEqual(r.notification_emitted.tolist(),[True,False,False,False,False])

    def test_cooldown_does_not_change_gate(self):
        f=stream([1,0,1,1,0,1])
        a,b=apply_policy(f,AlertPolicy('G4',0)),apply_policy(f,AlertPolicy('G4',300))
        self.assertEqual(a.gate_positive.tolist(),b.gate_positive.tolist())

    def test_quality_hole_resets_gate_retains_cooldown(self):
        f=stream([1,1],step=30)
        f.loc[1,'source_point_index']=2
        r=apply_policy(f,AlertPolicy('G0',60))
        self.assertEqual(r.notification_candidate.tolist(),[True,True])
        self.assertEqual(r.notification_emitted.tolist(),[True,False])
        self.assertFalse(apply_policy(f,AlertPolicy('G1',0)).gate_positive.any())

    def test_long_gap_resets_history(self):
        f=stream([1,1],step=400)
        self.assertFalse(apply_policy(f,AlertPolicy('G1',0)).gate_positive.any())
        self.assertEqual(apply_policy(f,AlertPolicy('G0',300)).notification_emitted.tolist(),[True,True])

    def test_gate_suppression_reason(self):
        f=stream([0,0,1,0,0])
        r=audit.event_metrics(event_record(f,2,4),apply_policy(f,AlertPolicy('G1',0)))
        self.assertTrue(r['gate_induced_miss'])
        self.assertIn('insufficient_consecutive_positives',r['gate_suppression_reasons'])

    def test_warmup_suppression_reason(self):
        f=stream([1])
        r=audit.event_metrics(event_record(f),apply_policy(f,AlertPolicy('G2',0)))
        self.assertIn('warm_up',r['gate_suppression_reasons'])

    def test_pre_event_false_cooldown_suppression(self):
        f=stream([1,0,1,1,0,1,0])
        r=audit.event_metrics(event_record(f,2,6),apply_policy(f,AlertPolicy('G0',60)))
        self.assertTrue(r['raw_detected'])
        self.assertTrue(r['gate_detected'])
        self.assertFalse(r['notification_detected'])
        self.assertTrue(r['cooldown_induced_miss'])
        self.assertTrue(r['suppressed_by_pre_event_false_notification'])

    def test_cooldown_delay_and_reason(self):
        f=stream([1,0,1,0,0,1,0],step=20)
        r=audit.event_metrics(event_record(f,2,6),apply_policy(f,AlertPolicy('G0',60)))
        self.assertTrue(r['notification_detected'])
        self.assertEqual(r['notification_delay_points'],3)
        self.assertEqual(r['notification_delay_seconds'],60)
        self.assertTrue(r['cooldown_induced_delay'])

    def test_event_detection_notification_vs_gate(self):
        f=stream([1,1,1])
        r=audit.event_metrics(event_record(f,1,2),apply_policy(f,AlertPolicy('G0',0)))
        self.assertTrue(r['gate_detected'])
        self.assertFalse(r['notification_detected'])
        self.assertTrue(r['carried_gate_run_without_event_notification'])
        self.assertFalse(r['suppressed_by_cooldown'])

    def test_early_detection_fixed_inclusive(self):
        f=stream([0,1,0,0,0])
        r=audit.event_metrics(event_record(f),apply_policy(f,AlertPolicy('G0',0)))
        self.assertFalse(r['notification_early10'])
        self.assertTrue(r['notification_early25'])
        self.assertTrue(r['notification_early50'])

    def test_delay_and_coverage(self):
        f=stream([0,1,1,0,1])
        r=audit.event_metrics(event_record(f),apply_policy(f,AlertPolicy('G0',0)))
        self.assertEqual(r['notification_delay_points'],1)
        self.assertEqual(r['notification_delay_seconds'],3)
        self.assertEqual(r['raw_coverage'],3/5)
        self.assertEqual(r['gate_coverage'],3/5)
        self.assertEqual(r['notification_coverage'],2/5)

    def test_missed_na_not_imputed(self):
        f=stream([0,0,0])
        r=audit.event_metrics(event_record(f),apply_policy(f,AlertPolicy('G0',0)))
        self.assertTrue(np.isnan(r['notification_delay_points']))
        self.assertEqual(r['event_status'],'missed_event')

    def test_false_notification_count_per1000_any(self):
        f=stream([1,0,1,0,1])
        detector={'model':'statistical_rule','seed':np.nan,'normal_trajectories':[('s',f)],'event_contexts':[(event_record(f),f)]}
        m,e,n,t=audit.evaluate_detector_policy(detector,AlertPolicy('G0',0),'validation')
        self.assertEqual(m['false_notification_count'],3)
        self.assertEqual(m['false_notifications_per_1000_normal_points'],600)
        self.assertEqual(m['any_notification_trajectory_percent'],100)

    def test_invalid_policy_definition_and_id(self):
        with self.assertRaises(ValueError):AlertPolicy('G7',0)
        with self.assertRaises(ValueError):AlertPolicy('G0',30)
        with self.assertRaises(ValueError):AlertPolicy.from_dict({'gate_id':'G0','cooldown_seconds':0,'policy_id':'G1_C0'})


class ValidationSelectionTests(unittest.TestCase):
    def test_rule_selection(self):
        grid,selected=audit.select_validation_policies(favorable(fake_grid(),model='statistical_rule'),config())
        self.assertEqual(selected.loc[selected.model.eq('statistical_rule'),'policy_id'].iloc[0],'G1_C0')

    def test_if_one_family_policy_across_seeds(self):
        grid,selected=audit.select_validation_policies(favorable(fake_grid(),model='isolation_forest'),config())
        row=selected.loc[selected.model.eq('isolation_forest')].iloc[0]
        self.assertEqual(row.policy_id,'G1_C0')
        self.assertEqual(row.seed_count,5)

    def test_ae_seed_aggregation(self):
        g=favorable(fake_grid(),model='autoencoder')
        g.loc[g.model.eq('autoencoder')&g.policy_id.eq('G1_C0')&g.seed.eq(7),'false_notifications_per_1000_normal_points']=6
        grid,selected=audit.select_validation_policies(g,config())
        row=selected.loc[selected.model.eq('autoencoder')].iloc[0]
        self.assertAlmostEqual(row.false_notifications_per_1000_normal_points,4.4)

    def test_ddof_one_and_rule_na(self):
        g=fake_grid()
        mask=g.model.eq('autoencoder')&g.policy_id.eq('G0_C0')
        g.loc[mask,'false_notifications_per_1000_normal_points']=[1,2,3,4,5]
        grid,_=audit.select_validation_policies(g,config())
        row=grid.loc[grid.model.eq('autoencoder')&grid.policy_id.eq('G0_C0')].iloc[0]
        self.assertAlmostEqual(row.false_notifications_per_1000_normal_points_sample_std,np.std([1,2,3,4,5],ddof=1))
        self.assertTrue(grid.loc[grid.model.eq('statistical_rule'),'notification_event_detection_rate_sample_std'].isna().all())

    def test_edr_guard_rejection(self):
        g=favorable(fake_grid())
        g.loc[g.policy_id.eq('G1_C0'),'notification_event_detection_rate']=.7
        grid,selected=audit.select_validation_policies(g,config())
        self.assertFalse(grid.loc[grid.policy_id.eq('G1_C0'),'eligible'].any())
        self.assertTrue(selected.policy_id.eq('G0_C0').all())

    def test_early25_guard_rejection(self):
        g=favorable(fake_grid())
        g.loc[g.policy_id.eq('G1_C0'),'notification_early25_rate']=.47
        grid,_=audit.select_validation_policies(g,config())
        self.assertFalse(grid.loc[grid.policy_id.eq('G1_C0'),'eligible'].any())

    def test_four_of_five_seed_guard(self):
        g=favorable(fake_grid(),model='isolation_forest')
        mask=g.model.eq('isolation_forest')&g.policy_id.eq('G1_C0')
        g.loc[mask,'notification_event_detection_rate']=[.6,.6,.9,.9,.9]
        grid,_=audit.select_validation_policies(g,config())
        row=grid.loc[grid.model.eq('isolation_forest')&grid.policy_id.eq('G1_C0')].iloc[0]
        self.assertTrue(row.edr_guard_passed)
        self.assertEqual(row.seed_guard_pass_count,3)
        self.assertFalse(row.eligible)

    def test_guard_inclusive_boundary(self):
        g=favorable(fake_grid())
        g.loc[g.policy_id.eq('G1_C0'),'notification_event_detection_rate']=.72
        g.loc[g.policy_id.eq('G1_C0'),'notification_early25_rate']=.48
        grid,_=audit.select_validation_policies(g,config())
        self.assertTrue(grid.loc[grid.policy_id.eq('G1_C0'),'eligible'].all())

    def test_lexicographic_primary_then_trajectory(self):
        g=favorable(favorable(fake_grid(),'G1_C0'),'G4_C0')
        g.loc[g.policy_id.eq('G4_C0'),'false_notifications_per_trajectory']=3
        _,selected=audit.select_validation_policies(g,config())
        self.assertTrue(selected.policy_id.eq('G4_C0').all())

    def test_lexicographic_delay_before_edr(self):
        g=favorable(favorable(fake_grid(),'G1_C0'),'G4_C0')
        g.loc[g.policy_id.eq('G4_C0'),'notification_delay_points_median']=.5
        _,selected=audit.select_validation_policies(g,config())
        self.assertTrue(selected.policy_id.eq('G4_C0').all())

    def test_edr_tiebreak_before_complexity(self):
        g=favorable(favorable(fake_grid(),'G1_C0'),'G4_C0')
        g.loc[g.policy_id.eq('G4_C0'),'notification_event_detection_rate']=.9
        _,selected=audit.select_validation_policies(g,config())
        self.assertTrue(selected.policy_id.eq('G4_C0').all())

    def test_complexity_then_shorter_cooldown_tiebreak(self):
        g=favorable(favorable(fake_grid(),'G1_C0'),'G1_C60')
        g=favorable(g,'G4_C0')
        _,selected=audit.select_validation_policies(g,config())
        self.assertTrue(selected.policy_id.eq('G1_C0').all())

    def test_raw_can_be_selected(self):
        _,selected=audit.select_validation_policies(fake_grid(),config())
        self.assertTrue(selected.policy_id.eq('G0_C0').all())

    def test_selector_rejects_test(self):
        g=fake_grid();g['dataset_split']='test'
        with self.assertRaisesRegex(ValueError,'rejects Test'):audit.select_validation_policies(g,config())

    def test_selector_rejects_mixed_or_untagged(self):
        g=fake_grid();g.loc[0,'dataset_split']='test'
        with self.assertRaises(ValueError):audit.select_validation_policies(g,config())
        with self.assertRaises(ValueError):audit.select_validation_policies(g.drop(columns='dataset_split'),config())

    def test_seed_and_candidate_completeness(self):
        with self.assertRaises(ValueError):audit.select_validation_policies(fake_grid().iloc[1:],config())

    def test_selection_input_immutable(self):
        g=fake_grid();before=g.copy(deep=True)
        audit.select_validation_policies(g,config())
        pd.testing.assert_frame_equal(g,before)

    def test_nan_delay_last_and_json_safety(self):
        g=favorable(favorable(fake_grid(),'G1_C0'),'G4_C0')
        g.loc[g.policy_id.eq('G1_C0'),'notification_delay_points_median']=np.nan
        _,selected=audit.select_validation_policies(g,config())
        self.assertTrue(selected.policy_id.eq('G4_C0').all())
        self.assertEqual(json.dumps(labels.json_safe({'delay':np.nan}),allow_nan=False),'{"delay": null}')


class LockAndProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/audit.CONFIG).parent.mkdir(parents=True)
        shutil.copyfile(single.ROOT/audit.CONFIG,self.root/audit.CONFIG)
        (self.root/audit.METRICS).mkdir(parents=True)
        digest=single.sha256_file(self.root/audit.CONFIG)
        self.manifest={'candidate_config_sha256':digest}
        audit.save(self.root/audit.METRICS/'candidate_policy_manifest.json',self.manifest)
        _,self.selected=audit.select_validation_policies(fake_grid(),config())
        with patch.object(audit.subprocess,'check_output',return_value='checkpoint\n'):
            self.path=audit.write_lock(self.root,self.selected,{'prediction_config_checksums':{},'source_checksums':{}},self.manifest)

    def test_locked_serialization_one_per_family(self):
        locked=audit.load_locked_policies(self.root,self.path)
        self.assertEqual(len(locked.family_policies),3)
        self.assertTrue(all(p.policy_id=='G0_C0' for m,p in locked.family_policies))

    def test_candidate_config_hash(self):
        with (self.root/audit.CONFIG).open('a') as file:file.write(' ')
        with self.assertRaisesRegex(ValueError,'Candidate config hash'):audit.load_locked_policies(self.root,self.path)

    def test_lock_hash(self):
        with self.path.open('a') as file:file.write(' ')
        with self.assertRaisesRegex(ValueError,'Locked policy hash'):audit.load_locked_policies(self.root,self.path)

    def test_manifest_hash(self):
        with (self.root/audit.METRICS/'candidate_policy_manifest.json').open('a') as file:file.write(' ')
        with self.assertRaisesRegex(ValueError,'manifest changed'):audit.load_locked_policies(self.root,self.path)

    def test_lock_overwrite_refused(self):
        with self.assertRaises(FileExistsError):audit.write_lock(self.root,self.selected,{},self.manifest)

    def test_test_evaluator_rejects_grid(self):
        with self.assertRaises(TypeError):audit.evaluate_locked_test({'dataset_split':'test'},fake_grid())

    def test_test_only_accepts_lock(self):
        with self.assertRaises(TypeError):audit.load_locked_policies(self.root,fake_grid())
        with self.assertRaises(ValueError):audit.load_locked_policies(self.root,self.root/audit.CONFIG)
        with self.assertRaises(ValueError):audit.run_test_evaluation(self.root,self.root,None)

    def test_test_rejects_validation_input(self):
        locked=audit.load_locked_policies(self.root,self.path)
        with self.assertRaises(ValueError):audit.evaluate_locked_test({'dataset_split':'validation'},locked)
        with self.assertRaises(ValueError):audit.evaluate_locked_test({'dataset_split':'test','candidate_grid':[]},locked)

    def test_no_test_metrics_in_lock(self):
        record=audit.read_json(self.path)
        self.assertEqual(record['selection_split'],'validation')
        self.assertFalse(record['test_metrics_used'])
        self.assertTrue(all(p['selection_metrics']['dataset_split']=='validation' for p in record['policies'].values()))

    def test_frozen_threshold_predictions_identity(self):
        f=pd.DataFrame({'anomaly_score':[1.,2.,3.],'predicted_anomaly':[0,0,1]})
        before=f.copy(deep=True)
        events.validate_fixed_decisions(f,2.)
        pd.testing.assert_frame_equal(f,before)
        with self.assertRaises(ValueError):events.validate_fixed_decisions(f,1.)

    def test_no_fit_model_or_label_modification_calls(self):
        calls=set()
        for path in [audit.__file__,single.ROOT/'src/alert_policy.py']:
            tree=ast.parse(Path(path).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,(ast.Name,ast.Attribute)):
                    calls.add(node.func.id if isinstance(node.func,ast.Name) else node.func.attr)
        self.assertFalse(calls&{'fit','partial_fit','fit_transform','train_model','reconstruction_errors','calculate_threshold','load_state_dict','generate_anomalies'})

    def test_test_function_has_no_candidate_search(self):
        import inspect
        source=inspect.getsource(audit.evaluate_locked_test)
        self.assertNotIn('select_validation_policies(',source)
        self.assertNotIn('frozen_candidates(',source)
        self.assertNotIn('min(',source)
        self.assertNotIn('sort_values(',source)

    def test_checksum_preservation_model_label_metric(self):
        for name in ['weights.pt','scaler.pkl','labels.csv','official_metrics.json']:
            (self.root/name).write_bytes(b'frozen')
        snapshot={name:single.sha256_file(self.root/name) for name in ['weights.pt','scaler.pkl','labels.csv','official_metrics.json']}
        apply_policy(stream([1,0,1]),AlertPolicy('G0',60))
        labels.assert_protected(self.root,snapshot)
        (self.root/'labels.csv').write_bytes(b'changed')
        with self.assertRaises(ValueError):labels.assert_protected(self.root,snapshot)

    def test_local_stage1_to66_snapshot(self):
        path=single.ROOT/'outputs/metrics/stage67_protected_snapshot.json'
        if not path.exists():self.skipTest('Local frozen artifacts not distributed.')
        labels.assert_protected(single.ROOT,json.loads(path.read_text(encoding='utf-8')))

    def test_nan_in_raw_prediction_fails(self):
        with self.assertRaises(ValueError):apply_policy(stream([1,np.nan]),AlertPolicy('G0',0))

    def test_candidate_config_21_fixed_complexity(self):
        self.assertEqual(len(config()['policies']),21)
        self.assertEqual(config()['cooldown_seconds'],[0,60,300])
        self.assertEqual(len(config()['gates']),7)
        self.assertEqual(config()['selection']['mean_edr_retention_min'],.9)
        self.assertEqual(AlertPolicy('G1',0).complexity,(1,0))
        self.assertLess(AlertPolicy('G4',0).complexity,AlertPolicy('G2',0).complexity)


if __name__=='__main__':unittest.main()

class ContextAndRegressionTests(unittest.TestCase):
    def test_existing_unmodified_prefix_exact_reuse(self):
        full=stream([0,0,0,1,0])
        full['latitude']=40.;full['longitude']=116.
        for feature in audit.FEATURE_COLUMNS:full[feature]=1.
        full['source_quality_valid']=True;full['is_low_quality']=False
        full['is_training_eligible']=True;full['synthetic_value_valid']=True
        originals=full.copy()
        normal=full.copy();normal['sample_id']='normal'
        copied=full.iloc[2:].copy()
        e=event_record(full,2,4)
        context=audit.make_context_stream(e,full,originals,normal,copied)
        self.assertEqual(context.source_point_index.tolist(),[0,1,2,3,4])
        self.assertTrue(context.sample_id.eq('copy').all())

    def test_changed_prefix_decision_reuse_rejected(self):
        full=stream([0,0,0,1,0]);full['latitude']=40.;full['longitude']=116.
        for feature in audit.FEATURE_COLUMNS:full[feature]=1.
        full['source_quality_valid']=True;full['is_low_quality']=False
        full['is_training_eligible']=True;full['synthetic_value_valid']=True
        originals=full.copy();normal=full.copy();copied=full.iloc[2:].copy()
        full.loc[0,'speed_mps']=1.001
        with self.assertRaisesRegex(ValueError,'Changed copied prefix'):
            audit.make_context_stream(event_record(full,2,4),full,originals,normal,copied)

    def test_stage66_raw_regression_matches_and_mismatch_fails(self):
        raw=pd.DataFrame([{'model':'statistical_rule','seed':np.nan,'evaluation':'raw',
            'raw_event_detection_rate':.8,'gate_event_detection_rate':.8,'gate_early25_rate':.6,
            'gate_delay_points_median':1.,'gate_delay_seconds_median':3.,'gate_coverage_median':.2,
            'raw_point_recall':.2,'raw_fp_points':4,'raw_fpr':.1,'false_gate_run_count':3,'observation_exposure_sec':100.}])
        old=pd.DataFrame([{'model':'statistical_rule','seed':np.nan,'event_detection_rate':.8,'early_detection_25_rate':.6,
            'delay_points_median':1.,'delay_seconds_median':3.,'event_point_coverage_median':.2,
            'point_recall':.2,'fp_point_count':4,'fpr':.1,'false_alert_run_count':3,'normal_observation_exposure_sec':100.}])
        current=pd.DataFrame([{'model':'statistical_rule','seed':np.nan,'evaluation':'raw','event_id':'x',
            'raw_detected':True,'gate_detected':True,'raw_delay_points':1.,'gate_delay_seconds':3.,'raw_coverage':.2}])
        previous=pd.DataFrame([{'model':'statistical_rule','seed':np.nan,'event_id':'x','event_detected':True,
            'delay_points':1.,'delay_seconds':3.,'event_point_coverage':.2}])
        with patch.object(audit.labels,'read_frozen',side_effect=[old,previous]):
            self.assertTrue(audit.validate_test_raw_regression(Path('.'),raw,current)['passed'])
        raw.loc[0,'false_gate_run_count']=4
        with patch.object(audit.labels,'read_frozen',side_effect=[old,previous]):
            with self.assertRaisesRegex(ValueError,'regression mismatch'):
                audit.validate_test_raw_regression(Path('.'),raw,current)
