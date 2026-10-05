"""Frozen confirmatory point, event and alert evaluation. One successful run only."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src import stage7_protocol as contract
from src import compare_stage5_baselines as baseline
from src import train_multiuser_autoencoder as single
from src import train_autoencoder as training
from src.autoencoder import Autoencoder
from src import audit_event_evaluation_protocol as event_audit
from src import audit_alert_aggregation as alert_audit
from src.alert_policy import AlertPolicy
from src.prepare_final_cohort import read_final_csv, validate_cohort, single_data_path

def load_frozen_detector(record, root=contract.ROOT):
    root=Path(root)
    contract.assert_hashes(root,contract.model_hashes({'models':[record]}))
    if record['feature_list']!=training.FEATURE_COLUMNS or record['training_row_count']!=86067:
        raise ValueError('Frozen detector schema/training provenance changed.')
    config=contract.read(root/record['config_path'])
    if float(config['threshold'])!=record['threshold']:
        raise ValueError('Frozen threshold changed.')
    family=record['detector']
    if family=='statistical_rule':
        params=contract.read(root/record['model_path'])
        if params['definition']!=baseline.RULE_DEFINITION:
            raise ValueError('Frozen Rule definition changed.')
        model=baseline.StatisticalRule()
        model.center_=np.asarray(params['center'],dtype=float)
        model.scale_=np.asarray(params['scale'],dtype=float)
        if model.center_.shape!=(8,) or model.scale_.shape!=(8,) or (model.scale_<=0).any():
            raise ValueError('Invalid frozen Rule state.')
        scaler=None
    else:
        scaler=joblib.load(root/record['scaler_path'])
        if not isinstance(scaler,StandardScaler) or scaler.n_features_in_!=8 or int(scaler.n_samples_seen_)!=86067:
            raise ValueError('Frozen scaler schema/fit row count differs.')
        if family=='isolation_forest':
            model=joblib.load(root/record['model_path'])
            params=model.get_params()
            if model.n_features_in_!=8 or params['random_state']!=record['seed'] or any(params[k]!=v for k,v in baseline.IF_PARAMETERS.items()):
                raise ValueError('Frozen IF configuration changed.')
        elif family=='autoencoder':
            model=Autoencoder(8)
            model.load_state_dict(torch.load(root/record['model_path'],map_location='cpu',weights_only=True))
            model.eval()
        else:
            raise ValueError('Unknown frozen detector.')
    return {'record':record,'model':model,'scaler':scaler}

def infer(detector, frame):
    if not len(frame):
        raise ValueError('Nonempty inference input required.')
    values=baseline.feature_values(frame)
    record=detector['record']
    with contract.no_fit_guard():
        if record['detector']=='statistical_rule':
            scores=detector['model'].score(frame)
        else:
            scaled=training.transform_with_scaler(detector['scaler'],frame)
            if record['detector']=='isolation_forest':
                scores=-detector['model'].score_samples(scaled)
            else:
                scores=training.reconstruction_errors(detector['model'],scaled,torch.device('cpu'),batch_size=1024)
    return baseline.predictions(frame,np.asarray(scores,dtype=float),record['threshold'])

def evaluate_points(pred, threshold, user_manifest):
    """Preserve every requested user, including users with no eligible rows."""
    adapted=baseline.metrics_adapter(pred)
    overall=single.calculate_metrics(adapted,threshold)
    users=[]
    for user in user_manifest.itertuples():
        frame=adapted.loc[adapted.user_id.eq(user.user_id)]
        if len(frame):
            metric=single.calculate_metrics(frame,threshold)
            reasons=metric.pop('metric_na_reasons')
        else:
            metric=dict(row_count=0,normal_row_count=0,anomaly_row_count=0,
                        tn=0,fp=0,fn=0,tp=0,threshold=float(threshold),
                        **{name:None for name in single.METRIC_NAMES})
            reasons={name:'no_quality_eligible_evaluation_rows' for name in single.METRIC_NAMES}
        users.append(dict(user_id=str(user.user_id),eligible_trajectory_count=int(user.eligible_trajectory_count),
                          **metric,na_reasons=json.dumps(reasons,sort_keys=True)))
    per_user=pd.DataFrame(users)
    macro=dict(user_count=len(per_user),metric_means={},defined_user_counts={},undefined_user_counts={})
    for metric in single.METRIC_NAMES:
        values=pd.to_numeric(per_user[metric],errors='coerce').dropna()
        macro['metric_means'][metric]=float(values.mean()) if len(values) else None
        macro['defined_user_counts'][metric]=len(values)
        macro['undefined_user_counts'][metric]=len(per_user)-len(values)
    types=single.per_anomaly_type_metrics(adapted,threshold).rename(
        columns={f'reconstruction_error_{suffix}':f'anomaly_score_{suffix}'
                 for suffix in ['mean','median','p95','p99']})
    return dict(metrics=overall,per_user=per_user,macro=macro,per_type=types)


def metric_tables(records, user_manifest):
    results=[]
    macro_rows=[]
    for detector,pred in records:
        m=detector['record']
        result=evaluate_points(pred,m['threshold'],user_manifest)
        result.update(threshold=m['threshold'],elapsed_seconds=0.,reused_existing=True)
        results.append((m['detector'],m['seed'],result))
        for metric in baseline.USER_METRICS:
            values=pd.to_numeric(result['per_user'][metric],errors='coerce').dropna()
            macro_rows.append(dict(model=m['detector'],seed=m['seed'],metric=metric,
                mean=float(values.mean()) if len(values) else None,
                user_sample_std=float(values.std(ddof=1)) if len(values)>1 else None,
                defined_users=len(values),requested_users=len(user_manifest),
                undefined_users=len(user_manifest)-len(values),
                na_reason='no_defined_users' if not len(values) else ''))
    tables=baseline.collect_tables(results)
    tables=dict(final_seed_metrics=tables['baseline_results'],
        final_point_metrics=tables['baseline_summary'],
        final_user_seed_metrics=tables['per_user_baseline_metrics'],
        final_user_metrics=baseline.summarize(tables['per_user_baseline_metrics'],['model','user_id'],
                       baseline.USER_METRICS+['normal_row_count','anomaly_row_count']),
        final_anomaly_type_seed_metrics=tables['per_anomaly_type_baseline_metrics'],
        final_anomaly_type_metrics=tables['per_anomaly_type_baseline_summary'],
        final_user_macro_metrics=pd.DataFrame(macro_rows))
    return tables

def event_alert_tables(records, normal_full, synthetic, manifest):
    route=synthetic.loc[synthetic.anomaly_type.eq('route_deviation')]
    route_manifest=manifest.loc[manifest.anomaly_type.eq('route_deviation')]
    first=records[0][1]
    catalog=alert_audit.build_catalog(route,route_manifest,first.loc[first.anomaly_type.eq('route_deviation')])
    raw_events=[]; summary=[]; policy_events=[]; false=[]; traces=[]
    for detector,pred in records:
        m=detector['record']; identity={'model':m['detector'],'seed':m['seed']}
        normal=pred.loc[~pred.is_synthetic].copy()
        anomalies=pred.loc[pred.anomaly_type.eq('route_deviation')].copy()
        context=dict(model=m['detector'],seed=m['seed'],normal=normal,route=anomalies)
        inputs=dict(detectors=[context],events=catalog,full=route,originals=normal_full)
        alert_audit.prepare_contexts(inputs)
        for event in catalog.to_dict('records'):
            g=anomalies.loc[anomalies.sample_id.eq(event['sample_id'])]
            raw_events.append(dict(identity,**event_audit.evaluate_event(event,g)))
        for evaluation,policy in [('RAW',AlertPolicy('G0',0)),('LOCKED',AlertPolicy('G0',60))]:
            # RAW comparator is fixed; never search or choose among final policies.
            metrics,events,trajectories,trace=alert_audit.evaluate_detector_policy(context,policy,'final',capture=True)
            metrics['evaluation']=evaluation
            metrics['raw_fp_per_1000']=1000*metrics['raw_fpr']
            metrics['raw_any_alert_percent']=100*float(trajectories.raw_fp_points.gt(0).mean())
            summary.append(metrics)
            for frame,dest in [(events,policy_events),(trajectories,false),(trace,traces)]:
                frame['evaluation']=evaluation
                dest.append(frame)
    raw=pd.DataFrame(raw_events)
    seed_raw=event_audit.grouped_summary(raw,['model','seed'],event_audit.event_summary)
    policy=pd.DataFrame(summary)
    stability_metrics=[c for c in policy if c not in ['model','seed','policy_id','dataset_split','evaluation']
                       and pd.api.types.is_numeric_dtype(policy[c])]
    tables=dict(final_event_catalog=catalog,final_event_metrics=raw,final_event_seed_metrics=seed_raw,
        final_event_summary=event_audit.seed_summary(seed_raw,[c for c in seed_raw if c not in ['model','seed','delay_conditioning','delay_na_reason']]),
        final_alert_metrics=policy,
        final_alert_summary=baseline.summarize(policy,['model','evaluation'],stability_metrics),
        final_policy_event_metrics=pd.concat(policy_events,ignore_index=True),
        final_false_alert_trajectory_metrics=pd.concat(false,ignore_index=True),
        final_notification_trace=pd.concat(traces,ignore_index=True))
    reduction=[]
    for model,g in policy.groupby('model',sort=True):
        a=g.loc[g.evaluation.eq('RAW')].false_notification_count.to_numpy(float)
        b=g.loc[g.evaluation.eq('LOCKED')].false_notification_count.to_numpy(float)
        if len(a)!=len(b):
            raise ValueError('Raw/locked run mismatch.')
        seed_ratio=np.where(a>0,100*(1-b/a),np.nan)
        reduction.append(dict(model=model,raw_mean_notifications=float(a.mean()),locked_mean_notifications=float(b.mean()),
            reduction_ratio_family_means_pct=float(100*(1-b.mean()/a.mean())) if a.mean()>0 else None,
            seed_reduction_mean_pct=float(np.nanmean(seed_ratio)) if np.isfinite(seed_ratio).any() else None,
            seed_reduction_sample_std=float(np.nanstd(seed_ratio,ddof=1)) if len(a)>1 else None,
            na_reason='raw_notification_count_zero' if a.mean()==0 else ''))
    tables['final_notification_reduction']=pd.DataFrame(reduction)
    return tables

def stage5_comparison(point,root):
    old=pd.read_csv(Path(root)/'outputs/metrics/stage5'/single.DEFAULT_DATA_DIR.name/'baseline_comparison/baseline_summary.csv')
    selected=['f1_score','recall','false_positive_rate','roc_auc','average_precision']
    result=point.loc[point.metric.isin(selected),['model','metric','mean','sample_std']].merge(
        old.loc[old.metric.isin(selected),['model','metric','mean','sample_std']],
        on=['model','metric'],suffixes=('_stage7','_stage5'),validate='one_to_one')
    result['delta_stage7_minus_stage5']=result.mean_stage7-result.mean_stage5
    result['interpretation']='Cohort difference; same frozen detector, no model-quality change implied'
    return result

def check_tables(tables):
    undefined=[]
    for name,frame in tables.items():
        for col in frame.select_dtypes(include=np.number):
            values=frame[col].to_numpy(float)
            if np.isinf(values).any():
                raise ValueError('Infinite final metric: '+name+'/'+col)
            n=int(np.isnan(values).sum())
            if n:
                undefined.append(dict(table=name,column=col,undefined_cells=n,
                    reason='Undefined statistic/missed-event delay/not-applicable identity; explicit support and per-row reasons retained'))
    return pd.DataFrame(undefined,columns=['table','column','undefined_cells','reason'])

def verify_results(root=contract.ROOT):
    root=Path(root)
    p=contract.read(root/contract.PROTOCOL)
    receipt=contract.read(root/contract.METRICS/'final_protocol_manifest.json')
    result=contract.read(root/contract.METRICS/'final_results_manifest.json')
    if contract.sha(root/contract.PROTOCOL)!=result['protocol_sha256'] or result['protocol_sha256']!=receipt['protocol_sha256']:
        raise ValueError('Result protocol changed.')
    contract.assert_hashes(root,result['result_file_hashes'])
    contract.assert_hashes(root,result['dataset_hashes'])
    contract.assert_hashes(root,result['model_hashes'])
    contract.assert_hashes(root,result['metric_producing_source_hashes'])
    if contract.sha(root/contract.DATA/'final_manifest.json')!=result['cohort_manifest_sha256']:
        raise ValueError('Cohort manifest changed.')
    return result

def run(root=contract.ROOT):
    root=Path(root)
    protocol,receipt=contract.verify_freeze(root)
    out=root/contract.METRICS
    if (out/'final_results_manifest.json').exists() or (out/'final_execution_attempt.json').exists():
        raise FileExistsError('Final evaluation attempted already; preserve existing results, no silent retry.')
    cohort=contract.read(root/contract.DATA/'final_manifest.json')
    contract.assert_hashes(root/contract.DATA,cohort['file_hashes'])
    contract.assert_hashes(root,contract.model_hashes(protocol['model_manifest']))
    torch.set_num_threads(1)
    detectors=[load_frozen_detector(m,root) for m in protocol['model_manifest']['models']]
    contract.save(out/'final_execution_attempt.json',dict(started_at=contract.now(),protocol_sha256=receipt['protocol_sha256'],
        git_sha=contract.git('rev-parse','HEAD',root=root),status='started',no_final_tuning=True))
    started=time.perf_counter()
    try:
        normal=read_final_csv(root/contract.DATA/'final_normal.csv')
        synthetic=read_final_csv(root/contract.DATA/'final_synthetic.csv')
        manifest=read_final_csv(root/contract.DATA/'synthetic_anomaly_manifest.csv')
        inputs=read_final_csv(root/contract.DATA/'input_manifest.csv')
        prior=read_final_csv(root/single_data_path()/'input_manifest.csv')
        validate_cohort(normal,synthetic,inputs,prior)
        evaluation=read_final_csv(root/contract.DATA/'final_evaluation.csv')
        from src.prepare_multiuser_dataset import evaluation_rows
        expected=evaluation_rows(pd.concat([normal,synthetic],ignore_index=True))
        pd.testing.assert_frame_equal(evaluation,expected,check_dtype=False)
        user_manifest=pd.DataFrame([dict(user_id=u,dataset_split='test',
            eligible_trajectory_count=int((inputs.user_id.eq(u)&inputs.eligible_for_dataset).sum())) for u in contract.USERS])
        predictions=[]
        for detector in detectors:
            m=detector['record']
            scored=infer(detector,evaluation)
            name=m['detector']+('' if m['seed'] is None else '_seed_'+str(m['seed']))
            scored.to_csv(out/('predictions_'+name+'.csv'),index=False)
            predictions.append((detector,scored))
            print('Frozen inference complete:',name,'rows',len(scored),flush=True)
        # Metrics are not displayed until all hashes and the result manifest are frozen.
        tables=metric_tables(predictions,user_manifest)
        tables.update(event_alert_tables(predictions,normal,synthetic,manifest))
        tables['stage5_vs_stage7']=stage5_comparison(tables['final_point_metrics'],root)
        tables['final_metric_na_report']=check_tables(tables)
        for name,frame in tables.items():
            path=out/(name+'.csv')
            if path.exists():
                raise FileExistsError('Refuse result overwrite: '+name)
            frame.to_csv(path,index=False)
        contract.verify_freeze(root)
        files=list(out.glob('predictions_*.csv'))+[out/(n+'.csv') for n in tables]
        result=dict(artifact_type='frozen_confirmatory_final_results',confirmatory=True,
            protocol_sha256=receipt['protocol_sha256'],protocol_freeze_commit_sha=receipt['protocol_freeze_commit_sha'],
            cohort_manifest_sha256=contract.sha(root/contract.DATA/'final_manifest.json'),
            dataset_hashes={(contract.DATA/name).as_posix():h for name,h in cohort['file_hashes'].items()},
            model_hashes=contract.model_hashes(protocol['model_manifest']),
            result_file_hashes={path.relative_to(root).as_posix():contract.sha(path) for path in sorted(files)},
            metric_producing_source_hashes=protocol['evaluator_source_hashes'],
            execution_started_at=contract.read(out/'final_execution_attempt.json')['started_at'],
            execution_completed_at=contract.now(),elapsed_seconds=time.perf_counter()-started,
            git_sha=contract.git('rev-parse','HEAD',root=root),training=False,scaler_fit=False,
            threshold_recalculation=False,policy_reselection=False,result_overwrite=False,
            prediction_row_count=len(evaluation),detector_runs=11,policy_comparators=['RAW G0_C0','LOCKED G0_C60'],
            cohort_users=list(contract.USERS),std_ddof=1)
        contract.save(out/'final_results_manifest.json',result)
        verify_results(root)
        print('Final results frozen:',contract.sha(out/'final_results_manifest.json'),'seconds',result['elapsed_seconds'])
        return result
    except Exception as exc:
        contract.save(out/'final_execution_failure.json',
            dict(error=type(exc).__name__,message=str(exc),timestamp=contract.now(),
                 original_outputs_preserved=True,confirmatory_completed=False,requires_explicit_bug_record_before_correction=True))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-existing',action='store_true')
    if ap.parse_args().verify_existing: verify_results()
    else: run()
