"""Independent final verifier: saved CSVs + scalar alert replay, no scorer imports."""
from __future__ import annotations
import hashlib, json, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/metrics/stage7/final'
DATA=ROOT/'data/processed/stage7/final_unseen_u020_039_first5'
USERS=[f'{i:03}' for i in range(20,40)]
FEATURES=['time_diff_sec','distance_m','speed_mps','acceleration_mps2','direction_change_deg','stop_duration_sec','bearing_sin','bearing_cos']
POINT=['accuracy','precision','recall','f1_score','false_positive_rate','roc_auc','average_precision']
IDENTITIES=['user_id','trajectory_id','original_trajectory_id','source_trajectory_id','sample_id','anomaly_segment_id']
KEY=['sample_id','source_trajectory_id','source_point_index']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def csv(path):
    f=pd.read_csv(path,dtype={k:str for k in IDENTITIES},float_precision='round_trip')
    if 'timestamp' in f:f['timestamp']=pd.to_datetime(f.timestamp,format='mixed')
    return f

def equal(actual,expected,name):
    if expected is None or (isinstance(expected,(float,np.floating)) and np.isnan(expected)):
        if not pd.isna(actual):raise AssertionError(name+': expected undefined')
    elif isinstance(expected,(int,float,np.number)):
        if not np.isclose(float(actual),float(expected),rtol=1e-9,atol=1e-10):
            raise AssertionError(f'{name}: {actual} != {expected}')
    elif actual!=expected:raise AssertionError(name)

def point(frame):
    y=frame.anomaly_label.to_numpy(int);decision=frame.predicted_anomaly.to_numpy(int)
    score=frame.anomaly_score.to_numpy(float)
    tn=int(((y==0)&(decision==0)).sum());fp=int(((y==0)&(decision==1)).sum())
    fn=int(((y==1)&(decision==0)).sum());tp=int(((y==1)&(decision==1)).sum())
    div=lambda a,b:a/b if b else None
    return dict(normal_row_count=tn+fp,anomaly_row_count=tp+fn,tn=tn,fp=fp,fn=fn,tp=tp,
        accuracy=div(tp+tn,len(y)),precision=div(tp,tp+fp),recall=div(tp,tp+fn),
        f1_score=div(2*tp,2*tp+fp+fn) if tp+fn else None,false_positive_rate=div(fp,fp+tn),
        roc_auc=float(roc_auc_score(y,score)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,score)) if len(set(y))==2 else None)

def replay(frame,cooldown):
    """Scalar RAW gate/run-start notifications; no use of production apply_policy."""
    g=frame.sort_values('source_point_index')
    prior_index=None;prior_time=None;prior_positive=False;last_emit=None;last_emit_index=None
    candidates=[];emissions=[];suppressors=[]
    for row in g.itertuples():
        index=int(row.source_point_index);time=pd.Timestamp(row.timestamp).value
        linked=prior_index is not None and index==prior_index+1 and 0<time-prior_time<=300*10**9
        positive=bool(row.predicted_anomaly)
        candidate=positive and not(prior_positive and linked)
        emit=candidate and (last_emit is None or time-last_emit>=cooldown*10**9)
        suppressors.append(last_emit_index if candidate and not emit else None)
        if emit:last_emit=time;last_emit_index=index
        candidates.append(candidate);emissions.append(emit)
        prior_index=index;prior_time=time;prior_positive=positive
    g=g.copy();g['independent_candidate']=candidates;g['independent_emit']=emissions
    g['independent_suppressor']=suppressors
    return g

def run_length(frame):
    g=frame.sort_values('source_point_index');longest=current=runs=0;previous=None
    for row in g.itertuples():
        connected=previous is not None and row.source_point_index==previous.source_point_index+1 and 0<(row.timestamp-previous.timestamp).total_seconds()<=300
        if row.predicted_anomaly:
            current=current+1 if connected and previous.predicted_anomaly else 1
            if current==1:runs+=1
            longest=max(longest,current)
        else:current=0
        previous=row
    return longest,runs

def fingerprint(frame):
    g=frame.sort_values('timestamp',kind='stable')
    elapsed=(g.timestamp-g.timestamp.iloc[0]).to_numpy(dtype='timedelta64[ns]').astype(np.int64)
    content='\n'.join(f'{a:.7f},{b:.7f},{int(t)}' for a,b,t in zip(g.latitude,g.longitude,elapsed))
    return hashlib.sha256(content.encode()).hexdigest()

def verify(root=ROOT):
    global ROOT,OUT,DATA
    ROOT=Path(root);OUT=ROOT/'outputs/metrics/stage7/final';DATA=ROOT/'data/processed/stage7/final_unseen_u020_039_first5'
    protocol=read(ROOT/'configs/stage7_final_protocol.json')
    receipt=read(OUT/'final_protocol_manifest.json');results=read(OUT/'final_results_manifest.json')
    integrity=read(OUT/'final_cohort_integrity.json');cohort=read(DATA/'final_manifest.json')
    assert protocol['final_users']==USERS and integrity['requested_users']==USERS
    assert protocol['feature_list']==FEATURES and protocol['alert_policy']=='G0_C60'
    assert protocol['model_seeds']==[7,21,42,100,2026] and protocol['synthetic_seed']==42
    assert digest(ROOT/'configs/stage7_final_protocol.json')==receipt['protocol_sha256']==results['protocol_sha256']
    committed=subprocess.check_output(['git','show',receipt['protocol_freeze_commit_sha']+':configs/stage7_final_protocol.json'],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==receipt['protocol_sha256']
    timeline=[receipt['timestamp'],cohort['cohort']['created_at'],results['execution_started_at'],results['execution_completed_at']]
    assert [datetime.fromisoformat(t) for t in timeline]==sorted(datetime.fromisoformat(t) for t in timeline)
    assert not(read(OUT/'prior_user_audit.json')['user_overlap'])
    assert integrity['lineage_errors']==0 and integrity['prior_user_overlap']==0
    for name,h in results['result_file_hashes'].items():assert digest(ROOT/name)==h,name
    for name,h in cohort['file_hashes'].items():assert digest(DATA/name)==h,name
    assert digest(DATA/'final_manifest.json')==results['cohort_manifest_sha256']
    for name,h in results['model_hashes'].items():assert digest(ROOT/name)==h,name
    for name,h in results['metric_producing_source_hashes'].items():assert digest(ROOT/name)==h,name
    snapshot=read(ROOT/'outputs/metrics/stage7_protected_snapshot.json')
    for name,h in snapshot.items():assert digest(ROOT/name)==h,name
    manifest=csv(DATA/'input_manifest.csv');exclusions=csv(DATA/'excluded_duplicates.csv')
    assert len(manifest)==integrity['trajectory_count'] and int(manifest.eligible_for_dataset.sum())==integrity['eligible_trajectory_count']
    assert len(exclusions)==integrity['excluded_duplicate_count']
    retained=manifest.loc[manifest.eligible_for_dataset]
    assert not retained.content_fingerprint.duplicated().any()
    prior=csv(ROOT/'data/processed/stage5/geolife_u000_019_first5_seed42_dedup/input_manifest.csv')
    assert not retained.content_fingerprint.isin(prior.content_fingerprint.dropna()).any()
    normal=csv(DATA/'final_normal.csv');synthetic=csv(DATA/'final_synthetic.csv')
    evaluation=csv(DATA/'final_evaluation.csv')
    assert set(normal.user_id).issubset(USERS) and not(set(normal.user_id)&set(prior.user_id))
    assert normal.source_point_index.ge(0).all()
    assert not normal.duplicated(['source_trajectory_id','source_point_index']).any()
    assert not synthetic.duplicated(KEY).any()
    lookup=normal.set_index(['source_trajectory_id','source_point_index'])
    selected=lookup.loc[list(synthetic[['source_trajectory_id','source_point_index']].itertuples(index=False,name=None))]
    assert np.array_equal(selected.user_id.to_numpy(),synthetic.user_id.to_numpy())
    for c in ['source_quality_valid','is_training_eligible','is_low_quality']:
        assert np.array_equal(selected[c].to_numpy(),synthetic[c].to_numpy()),c
    for source,g in normal.groupby('source_trajectory_id'):
        assert fingerprint(g)==manifest.loc[manifest.trajectory_id.eq(source),'content_fingerprint'].iloc[0]
    for _,g in synthetic.groupby('source_trajectory_id'):
        assert g.sample_id.nunique()==4
        assert set(g.anomaly_type)==set(protocol['anomaly_types'])
    expected=pd.concat([normal.loc[normal.source_quality_valid&normal.synthetic_value_valid],
        synthetic.loc[synthetic.source_quality_valid&synthetic.synthetic_value_valid&synthetic.anomaly_label.eq(1)]],ignore_index=True)
    pd.testing.assert_frame_equal(evaluation[KEY+FEATURES],expected[KEY+FEATURES],check_dtype=False)
    assert len(evaluation.loc[~evaluation.is_synthetic])==integrity['normal_rows']
    assert int(evaluation.anomaly_label.sum())==integrity['anomaly_labelled_rows']
    seed=csv(OUT/'final_seed_metrics.csv');user_seed=csv(OUT/'final_user_seed_metrics.csv')
    type_seed=csv(OUT/'final_anomaly_type_seed_metrics.csv');summary=csv(OUT/'final_point_metrics.csv')
    macro=csv(OUT/'final_user_macro_metrics.csv')
    events=csv(OUT/'final_event_metrics.csv');policy_events=csv(OUT/'final_policy_event_metrics.csv')
    false=csv(OUT/'final_false_alert_trajectory_metrics.csv');alert=csv(OUT/'final_alert_metrics.csv')
    catalog=csv(OUT/'final_event_catalog.csv')
    for c in ['event_start_timestamp','event_end_timestamp']:catalog[c]=pd.to_datetime(catalog[c],format='mixed')
    checked_points=checked_users=checked_events=checked_alerts=0
    for model in protocol['model_manifest']['models']:
        family=model['detector'];s=model['seed']
        matches=lambda f: f.model.eq(family)&(f.seed.isna() if s is None else f.seed.eq(s))
        name=family+('' if s is None else '_seed_'+str(s))
        pred=csv(OUT/('predictions_'+name+'.csv'))
        pd.testing.assert_frame_equal(pred[KEY+FEATURES],evaluation[KEY+FEATURES],check_dtype=False)
        assert np.isfinite(pred.anomaly_score).all()
        assert np.array_equal(pred.predicted_anomaly,(pred.anomaly_score>model['threshold']).astype(int))
        cfg=read(ROOT/model['config_path']);assert cfg['threshold']==model['threshold']
        measured=point(pred);stored=seed.loc[matches(seed)].iloc[0]
        for k,v in measured.items():equal(stored[k],v,name+'/'+k)
        user_values={metric:[] for metric in POINT}
        for u in USERS:
            part=pred.loc[pred.user_id.eq(u)];m=point(part)
            row=user_seed.loc[matches(user_seed)&user_seed.user_id.eq(u)].iloc[0]
            for k,v in m.items():equal(row[k],v,name+'/'+u+'/'+k)
            for metric in POINT:
                if m[metric] is not None:user_values[metric].append(m[metric])
            checked_users+=1
        for metric in ['precision','recall','f1_score','false_positive_rate','roc_auc','average_precision']:
            vals=user_values[metric];mrow=macro.loc[matches(macro)&macro.metric.eq(metric)].iloc[0]
            equal(mrow['mean'],np.mean(vals) if vals else None,'macro/'+name+'/'+metric)
            equal(mrow.user_sample_std,np.std(vals,ddof=1) if len(vals)>1 else None,'user std/'+name+'/'+metric)
            equal(mrow.defined_users,len(vals),'macro support')
        normals=pred.loc[~pred.is_synthetic]
        for kind in protocol['anomaly_types']:
            anomalies=pred.loc[pred.is_synthetic&pred.anomaly_type.eq(kind)]
            m=point(pd.concat([normals,anomalies],ignore_index=True))
            row=type_seed.loc[matches(type_seed)&type_seed.anomaly_type.eq(kind)].iloc[0]
            for metric in ['recall','roc_auc','average_precision']:equal(row[metric],m[metric],'type/'+name+'/'+kind+'/'+metric)
            equal(row.anomaly_row_count,len(anomalies),'type count')
        for source,g in normals.groupby('source_trajectory_id'):
            for policy,cooldown in [('RAW',0),('LOCKED',60)]:
                replayed=replay(g,cooldown);longest,runs=run_length(g)
                row=false.loc[matches(false)&false.source_trajectory_id.eq(source)&false.evaluation.eq(policy)].iloc[0]
                for col,value in [('raw_fp_points',int(g.predicted_anomaly.sum())),
                    ('false_gate_run_count',runs),('longest_false_gate_run_points',longest),
                    ('false_notification_count',int(replayed.independent_emit.sum())),('normal_point_count',len(g))]:
                    equal(row[col],value,'normal/'+name+'/'+source+'/'+policy+'/'+col)
                checked_alerts+=1
        for e in catalog.to_dict('records'):
            inside=pred.loc[pred.sample_id.eq(e['sample_id'])]
            g=synthetic.loc[synthetic.sample_id.eq(e['sample_id'])].sort_values('source_point_index')
            labels=g.loc[g.anomaly_label.eq(1)]
            assert len(labels)==e['event_point_count'] and labels.source_point_index.iloc[0]==e['event_start_index']
            assert labels.source_point_index.iloc[-1]==e['event_end_index']
            prefix=g.loc[g.source_point_index.lt(e['event_start_index'])]
            paired=lookup.loc[list(prefix[['source_trajectory_id','source_point_index']].itertuples(index=False,name=None))]
            for c in ['timestamp','latitude','longitude',*FEATURES]:assert np.array_equal(prefix[c].to_numpy(),paired[c].to_numpy()),c
            pre=normals.loc[normals.source_trajectory_id.eq(e['source_trajectory_id'])&
                             normals.source_point_index.lt(e['event_start_index'])].copy()
            pre['sample_id']=e['sample_id']
            context=pd.concat([pre,inside],ignore_index=True).sort_values('source_point_index')
            positives=inside.loc[inside.predicted_anomaly.eq(1)];hit=len(positives)>0
            offset=int(positives.source_point_index.iloc[0])-e['event_start_index'] if hit else None
            delay=(positives.timestamp.iloc[0]-e['event_start_timestamp']).total_seconds() if hit else None
            relative=offset/max(e['event_point_count']-1,1) if hit else None
            longest,runs=run_length(inside)
            row=events.loc[matches(events)&events.event_id.eq(e['event_id'])].iloc[0]
            for col,value in [('event_detected',hit),('delay_points',offset),('delay_seconds',delay),
                ('event_point_coverage',len(positives)/e['event_point_count']),
                ('longest_positive_run_points',longest),('positive_run_count',runs)]:
                equal(row[col],value,'event/'+name+'/'+col)
            for cutoff in [10,25,50]:equal(row['early_detection_'+str(cutoff)],bool(hit and relative<=cutoff/100),'event early')
            for policy,cooldown in [('RAW',0),('LOCKED',60)]:
                r=replay(context,cooldown);in_event=r.loc[r.source_point_index.between(e['event_start_index'],e['event_end_index'])]
                emitted=in_event.loc[in_event.independent_emit];notify=len(emitted)>0
                offset=int(emitted.source_point_index.iloc[0])-e['event_start_index'] if notify else None
                delay=(emitted.timestamp.iloc[0]-e['event_start_timestamp']).total_seconds() if notify else None
                relative=offset/max(e['event_point_count']-1,1) if notify else None
                record=policy_events.loc[matches(policy_events)&policy_events.event_id.eq(e['event_id'])&policy_events.evaluation.eq(policy)].iloc[0]
                for col,value in [('notification_detected',notify),('notification_delay_points',offset),
                                  ('notification_delay_seconds',delay),('notification_positive_count',len(emitted))]:
                    equal(record[col],value,'notification/'+name+'/'+policy+'/'+col)
                for cutoff in [10,25,50]:equal(record['notification_early'+str(cutoff)],bool(notify and relative<=cutoff/100),'notification early')
                checked_events+=1
        for policy in ['RAW','LOCKED']:
            f=false.loc[matches(false)&false.evaluation.eq(policy)]
            ev=policy_events.loc[matches(policy_events)&policy_events.evaluation.eq(policy)]
            row=alert.loc[matches(alert)&alert.evaluation.eq(policy)].iloc[0]
            equal(row.false_notification_count,f.false_notification_count.sum(),'pooled notification count')
            equal(row.false_notifications_per_1000_normal_points,1000*f.false_notification_count.sum()/len(normals),'notification burden')
            equal(row.any_notification_trajectory_percent,100*f.any_notification.mean(),'any notification')
            equal(row.notification_event_detection_rate,ev.notification_detected.mean(),'notification EDR')
            equal(row.notification_early25_rate,ev.notification_early25.mean(),'early25')
        checked_points+=len(pred)
    for row in summary.itertuples():
        vals=pd.to_numeric(seed.loc[seed.model.eq(row.model),row.metric],errors='coerce').dropna()
        for col,value in [('mean',vals.mean()),('sample_std',vals.std(ddof=1)),('min',vals.min()),('max',vals.max())]:
            equal(getattr(row,col),value,'seed aggregate/'+row.model+'/'+row.metric+'/'+col)
    for filename,groups,source in [('final_user_metrics.csv',['model','user_id'],user_seed),
                                  ('final_anomaly_type_metrics.csv',['model','anomaly_type'],type_seed)]:
        table=csv(OUT/filename)
        for row in table.to_dict('records'):
            mask=np.ones(len(source),dtype=bool)
            for key in groups:mask &= source[key].eq(row[key]).to_numpy()
            vals=pd.to_numeric(source.loc[mask,row['metric']],errors='coerce').dropna()
            for col,value in [('mean',vals.mean()),('sample_std',vals.std(ddof=1)),('min',vals.min()),('max',vals.max())]:
                equal(row[col],value,'group aggregate/'+filename+'/'+row['metric']+'/'+col)
    delta=csv(OUT/'stage5_vs_stage7.csv')
    for row in delta.itertuples():equal(row.delta_stage7_minus_stage5,row.mean_stage7-row.mean_stage5,'delta')
    report=dict(passed=True,independent_of_final_scorer=True,final_users=USERS,prior_user_overlap=0,
        trajectory_count=len(manifest),eligible_trajectories=len(retained),duplicate_exclusions=len(exclusions),
        lineage_errors=0,verified_prediction_rows=checked_points,per_user_checks=checked_users,
        scalar_policy_event_checks=checked_events,scalar_normal_alert_checks=checked_alerts,
        model_hashes_verified=True,protocol_commit_and_hash_verified=True,thresholds_unchanged=True,
        frozen_G0_C60_verified=True,seed_aggregation_ddof=1,protected_file_count=len(snapshot),
        protected_changed_files=0,result_manifest_sha256=digest(OUT/'final_results_manifest.json'),
        chronology=timeline,metric_recomputation='confusion counts, ROC-AUC/AP, user macro/std, anomaly-type, seed/group mean/std/min/max; scalar event/notification replay')
    path=OUT/'independent_verification.json'
    if path.exists():
        if read(path)!=report:raise FileExistsError('Independent receipt differs; do not overwrite.')
    else:
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':verify()
