"""Stage 6.5 frozen synthetic route label observability, not label editing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import audit_route_representation as previous
from src import audit_route_reference_coverage as spatial
from src import compare_stage5_baselines as baseline
from src import train_multiuser_autoencoder as single
from src.train_autoencoder import FEATURE_COLUMNS, add_bearing_features
from src.load_multiuser_geolife import read_dataset_csv, ID_COLUMNS
from src.feature_engineering import haversine_distance
from src.prepare_multiuser_dataset import validate_saved_dataset

KEY = previous.KEY
SCORE_KEY = previous.SCORE_KEY
FEATURE_ATOL, FEATURE_RTOL = 1e-6, 1e-8
COORD_ATOL_M = 1e-6
CONTEXT_ATOL_M, CONTEXT_RTOL = 1e-5, 1e-8
CONTEXT_SIGNALS = ('stage63_prefix', 'stage63_combined', 'w10g0_prefix', 'w10g0_combined')
TIERS = ('tier1', 'tier2', 'tier3', 'tier4')
DISPLACEMENT_THRESHOLDS_M = (1, 5, 10, 25, 50, 100, 200)
BIN_ORDER = ('zero', '(0,10)', '[10,25)', '[25,50)', '[50,100)', '[100,200)', '[200,inf)')
FEATURE_CATEGORY_ORDER = ('0', '1', '2-3', '4+')
LOW_SUPPORT_N = 10


def assert_protected(root, snapshot):
    root = Path(root).resolve()
    for relative, expected in snapshot.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Invalid/missing protected file: '+relative)
        if single.sha256_file(path) != expected: raise ValueError('Protected file changed: '+relative)


def finite(values):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all(): raise ValueError('Non-finite audit observation/score.')
    return result


def exact_join(left, right, keys, validate='one_to_one'):
    if left.duplicated(keys).any() and validate == 'one_to_one': raise ValueError('Duplicate left lineage.')
    if right.duplicated(keys).any(): raise ValueError('Duplicate right lineage.')
    result = left.merge(right, on=keys, how='left', validate=validate, indicator=True, sort=False)
    if result['_merge'].ne('both').any(): raise ValueError('Missing source/prediction/context lineage.')
    return result.drop(columns='_merge')


def displacement_bin(values):
    v = finite(values)
    if (v < 0).any(): raise ValueError('Negative displacement.')
    return np.select([v<=COORD_ATOL_M,v<10,v<25,v<50,v<100,v<200],BIN_ORDER[:-1],default=BIN_ORDER[-1])


def feature_category(counts):
    counts = np.asarray(counts)
    if not np.isin(counts, np.arange(9)).all(): raise ValueError('Invalid feature-change count.')
    return np.select([counts==0,counts==1,counts<=3],FEATURE_CATEGORY_ORDER[:3],default=FEATURE_CATEGORY_ORDER[-1])


def point_deltas(synthetic, originals):
    """Audit counterfactual pair, never a detector feature or input."""
    required = ['user_id','trajectory_id','dataset_split','timestamp','latitude','longitude']+FEATURE_COLUMNS
    original = add_bearing_features(originals)
    syn = add_bearing_features(synthetic)
    if original.duplicated(KEY).any() or syn.duplicated(SCORE_KEY).any(): raise ValueError('Duplicate lineage.')
    names = {c:('paired_trajectory_id' if c=='trajectory_id' else 'original_'+c) for c in required}
    source = original[KEY+required].rename(columns=names)
    result = exact_join(syn,source,KEY,'many_to_one')
    for column in ('user_id','trajectory_id','dataset_split'):
        if not result[column].eq(result['paired_trajectory_id' if column=='trajectory_id' else 'original_'+column]).all(): raise ValueError('Source identity mismatch: '+column)
    if result.timestamp.isna().any() or result.original_timestamp.isna().any(): raise ValueError('Missing timestamp.')
    coordinates = finite(result[['latitude','longitude','original_latitude','original_longitude']])
    if (np.abs(coordinates[:,[0,2]])>90).any() or (np.abs(coordinates[:,[1,3]])>180).any(): raise ValueError('Invalid GPS coordinate.')
    result['latitude_delta_deg'] = result.latitude-result.original_latitude
    result['longitude_delta_deg'] = result.longitude-result.original_longitude
    result['actual_displacement_m'] = haversine_distance(result.original_latitude,result.original_longitude,result.latitude,result.longitude)
    result['timestamp_delta_sec'] = (result.timestamp-result.original_timestamp).dt.total_seconds()
    result['coordinate_intervention_strict'] = result.latitude.ne(result.original_latitude)|result.longitude.ne(result.original_longitude)
    result['directly_displaced'] = result.actual_displacement_m.gt(COORD_ATOL_M)
    result['zero_displacement'] = ~result.directly_displaced
    result['timestamp_intervention'] = result.timestamp_delta_sec.ne(0)
    a,b=finite(result[FEATURE_COLUMNS]),finite(result[['original_'+c for c in FEATURE_COLUMNS]])
    changed = ~np.isclose(a,b,atol=FEATURE_ATOL,rtol=FEATURE_RTOL)
    for index,column in enumerate(FEATURE_COLUMNS):
        result[column+'_delta'] = a[:,index]-b[:,index]
        result[column+'_changed'] = changed[:,index]
    result['feature_changed_count'] = changed.sum(axis=1)
    result['feature_change_category'] = feature_category(result.feature_changed_count)
    result['feature_near_identical'] = result.feature_changed_count.eq(0)
    result['displacement_bin'] = displacement_bin(result.actual_displacement_m)
    return result


def segment_structure(full):
    """Full copied rows define consecutive runs and causal propagation, not filtered evaluation rows."""
    parts=[];runs=[]
    for sample,group in full.groupby('sample_id',sort=True):
        g=group.sort_values('source_point_index',kind='stable').reset_index(drop=True).copy()
        if g.user_id.nunique()!=1 or g.source_trajectory_id.nunique()!=1 or not g.source_point_index.diff().dropna().eq(1).all():
            raise ValueError('Ambiguous/noncontiguous sample stream.')
        if not g.timestamp.diff().dropna().gt(pd.Timedelta(0)).all(): raise ValueError('Non-increasing sample timestamps.')
        if not g.anomaly_label.isin([0,1]).all(): raise ValueError('Nonbinary label.')
        modified=g.directly_displaced
        g['previous_source_modified']=modified.shift(1,fill_value=False)
        g['next_source_modified']=modified.shift(-1,fill_value=False)
        g['past_modified_point_count']=modified.cumsum()-modified.astype(int)
        g['causal_raw_history_changed']=g.past_modified_point_count.gt(0)
        g['feature_propagation_only']=~modified & g.feature_changed_count.gt(0)
        # Next-point flag is descriptive; existing features use only current/past GPS.
        labels=g.anomaly_label.eq(1).to_numpy();starts=np.flatnonzero(labels & ~np.r_[False,labels[:-1]])
        ends=np.flatnonzero(labels & ~np.r_[labels[1:],False])
        g['label_run_id']='';g['segment_position']=-1;g['segment_length']=0;g['segment_relative_position']=np.nan
        g['segment_boundary']='outside';g['progression_bucket']='outside'
        for number,(start,end) in enumerate(zip(starts,ends),1):
            length=int(end-start+1);run=f'{sample}__run_{number:02d}';idx=np.arange(start,end+1);pos=np.arange(length)
            relative=pos/max(1,length-1)
            g.loc[idx,'label_run_id']=run;g.loc[idx,'segment_position']=pos;g.loc[idx,'segment_length']=length
            g.loc[idx,'segment_relative_position']=relative
            boundary=np.where(pos==0,'start',np.where(pos==length-1,'end','interior')).astype(object)
            if length==1:boundary[:]='singleton'
            g.loc[idx,'segment_boundary']=boundary
            g.loc[idx,'progression_bucket']=np.where(relative<1/3,'early',np.where(relative<2/3,'middle','late'))
            selected=g.loc[idx]
            runs.append({'sample_id':sample,'label_run_id':run,'user_id':str(g.user_id.iloc[0]),
                'source_trajectory_id':str(g.source_trajectory_id.iloc[0]),'run_start_index':int(g.source_point_index.iloc[start]),
                'run_end_index':int(g.source_point_index.iloc[end]),'run_start_timestamp':g.timestamp.iloc[start],
                'run_end_timestamp':g.timestamp.iloc[end],'full_labelled_length':length,
                'full_directly_displaced_length':int(selected.directly_displaced.sum()),
                'full_feature_observable_length':int(selected.feature_changed_count.gt(0).sum())})
        if not len(starts): raise ValueError('Route sample has no labelled run.')
        parts.append(g)
    return pd.concat(parts,ignore_index=True),pd.DataFrame(runs)


def classify_tiers(points):
    result=points.copy();changed=[];available=[]
    for signal in CONTEXT_SIGNALS:
        a=pd.to_numeric(result[signal+'_synthetic_score_m'],errors='raise').to_numpy(dtype=float)
        b=pd.to_numeric(result[signal+'_original_score_m'],errors='raise').to_numpy(dtype=float)
        if np.isinf(a).any() or np.isinf(b).any(): raise ValueError('Infinite context score.')
        valid=np.isfinite(a)&np.isfinite(b)
        result[signal+'_available']=valid
        result[signal+'_delta_m']=a-b
        mask=valid & ~np.isclose(a,b,atol=CONTEXT_ATOL_M,rtol=CONTEXT_RTOL)
        result[signal+'_changed']=mask;changed.append(mask);available.append(valid)
    result['context_changed_count']=np.asarray(changed).sum(axis=0)
    result['context_available_count']=np.asarray(available).sum(axis=0)
    result['context_comparison_complete']=result.context_available_count.eq(len(CONTEXT_SIGNALS))
    direct=result.directly_displaced.to_numpy();derived=result.feature_changed_count.gt(0).to_numpy()
    context=result.context_changed_count.gt(0).to_numpy()
    # Missing representations are not equality. Fail closed before declaring unidentifiability.
    undecidable=~direct & ~derived & ~context & ~result.context_comparison_complete.to_numpy()
    if undecidable.any(): raise ValueError('Incomplete context comparisons cannot establish Tier 4.')
    result['identifiability_tier']=np.select([direct,derived,context],TIERS[:3],default=TIERS[-1])
    result['audit_observable']=result.identifiability_tier.ne('tier4')
    if not result.identifiability_tier.isin(TIERS).all(): raise ValueError('Invalid tier partition.')
    return result


def attach_context(points,stage63_pairs,stage64_pairs,stage62_pairs):
    p=points.copy()
    for old,signal in [('causal_prefix','stage63_prefix'),('combined_personal','stage63_combined')]:
        source=stage63_pairs[SCORE_KEY+[old+'_distance_m','original_'+old+'_distance_m']].rename(columns={
            old+'_distance_m':signal+'_synthetic_score_m','original_'+old+'_distance_m':signal+'_original_score_m'})
        p=exact_join(p,source,SCORE_KEY)
    for ref,signal in [('prefix','w10g0_prefix'),('combined','w10g0_combined')]:
        g=stage64_pairs.loc[stage64_pairs.window_size.eq(10)&stage64_pairs.gap.eq(0)&stage64_pairs.reference_type.eq(ref)]
        source=g[SCORE_KEY+['synthetic_window_score_m','original_window_score_m']].rename(columns={
            'synthetic_window_score_m':signal+'_synthetic_score_m','original_window_score_m':signal+'_original_score_m'})
        p=exact_join(p,source,SCORE_KEY)
    return exact_join(p,stage62_pairs[SCORE_KEY+['synthetic_reference_distance_m','original_reference_distance_m']],SCORE_KEY)


def prediction_join(points, route_predictions, normal_predictions):
    columns=SCORE_KEY+['user_id','anomaly_score','predicted_anomaly']
    route=exact_join(points[SCORE_KEY+['user_id']],route_predictions[columns].rename(columns={'user_id':'prediction_user_id'}),SCORE_KEY)
    if not route.user_id.eq(route.prediction_user_id).all(): raise ValueError('Prediction user mismatch.')
    if not route.predicted_anomaly.isin([0,1]).all(): raise ValueError('Nonbinary saved prediction.')
    finite(route.anomaly_score)
    original=normal_predictions[KEY+['user_id','anomaly_score','predicted_anomaly']].rename(columns={
        'user_id':'normal_user_id','anomaly_score':'original_anomaly_score','predicted_anomaly':'original_predicted_anomaly'})
    result=exact_join(route.drop(columns='prediction_user_id'),original,KEY,'many_to_one')
    if not result.user_id.eq(result.normal_user_id).all(): raise ValueError('Normal prediction user mismatch.')
    if not result.original_predicted_anomaly.isin([0,1]).all(): raise ValueError('Nonbinary normal prediction.')
    finite(result.original_anomaly_score)
    return result.drop(columns='normal_user_id')


def load_predictions(root,dataset_id,test,points):
    records=[];input_checksums={}
    for model,seeds in [('statistical_rule',[None]),('isolation_forest',baseline.MODEL_SEEDS),('autoencoder',baseline.MODEL_SEEDS)]:
        for seed in seeds:
            if model=='autoencoder':
                directories=single.output_directories(root,dataset_id,seed)
                path=directories['metrics']/'test_predictions.csv';config_path=directories['model']/'training_config.json'
                frame=read_dataset_csv(path).rename(columns={'reconstruction_error':'anomaly_score'})
            else:
                suffix=Path(model) if seed is None else Path(model)/f'seed_{seed}'
                path=root/'outputs/metrics/stage5'/dataset_id/'baseline_comparison'/suffix/'test_predictions.csv'
                config_path=root/'models/stage5'/dataset_id/'baselines'/suffix/'run_config.json'
                frame=baseline.read_baseline_predictions(path)
            baseline.assert_same_rows(frame,test)
            threshold=float(json.loads(config_path.read_text(encoding='utf-8'))['threshold'])
            finite(frame.anomaly_score)
            if not np.array_equal(frame.predicted_anomaly,(frame.anomaly_score>threshold).astype(int)):
                raise ValueError('Stored prediction differs from stored score/threshold.')
            copied=frame.loc[frame.is_synthetic&frame.anomaly_label.eq(1)&frame.anomaly_type.eq('route_deviation')]
            normal=frame.loc[~frame.is_synthetic&frame.anomaly_label.eq(0)]
            if set(copied[SCORE_KEY].itertuples(index=False,name=None))!=set(points[SCORE_KEY].itertuples(index=False,name=None)):
                raise ValueError('Saved route prediction universe mismatch.')
            joined=prediction_join(points,copied,normal)
            joined.insert(0,'model',model);joined.insert(1,'seed',seed);joined['frozen_threshold']=threshold
            records.append(joined)
            input_checksums[path.relative_to(root).as_posix()]=single.sha256_file(path)
            input_checksums[config_path.relative_to(root).as_posix()]=single.sha256_file(config_path)
    return pd.concat(records,ignore_index=True),input_checksums


def recall_rows(points,predictions,group_column,categories):
    flags=points[SCORE_KEY+[group_column]]
    joined=exact_join(predictions,flags,SCORE_KEY,'many_to_one');records=[]
    for (model,seed),run in joined.groupby(['model','seed'],dropna=False,sort=True):
        for category in categories:
            g=run.loc[run[group_column].eq(category)];n=len(g);detected=int(g.predicted_anomaly.sum())
            records.append({'model':model,'seed':seed,group_column:category,'anomaly_row_count':n,'detected_count':detected,
                'recall':detected/n if n else np.nan,'low_support':0<n<LOW_SUPPORT_N,'na_reason':'' if n else 'no_rows'})
    return pd.DataFrame(records)


def recall_summary(seed_rows,group_column):
    records=[]
    for (model,category),g in seed_rows.groupby(['model',group_column],sort=True,dropna=False):
        values=g.recall.dropna()
        records.append({'model':model,group_column:category,'seed_count':len(g),'defined_seed_count':len(values),
            'anomaly_row_count':int(g.anomaly_row_count.iloc[0]),'detected_count_mean':float(g.detected_count.mean()),
            'recall_mean':float(values.mean()) if len(values) else np.nan,
            'recall_sample_std':float(values.std(ddof=1)) if len(values)>1 else np.nan,
            'low_support':bool(g.low_support.any()),'na_reason':'' if len(values) else 'no_rows'})
    return pd.DataFrame(records)


def counterfactual_metrics(points,predictions):
    joined=exact_join(predictions,points[SCORE_KEY+['audit_observable']],SCORE_KEY,'many_to_one');records=[]
    for (model,seed),g in joined.groupby(['model','seed'],dropna=False,sort=True):
        visible=g.loc[g.audit_observable];hidden=g.loc[~g.audit_observable]
        official=float(g.predicted_anomaly.mean()) if len(g) else np.nan
        counter=float(visible.predicted_anomaly.mean()) if len(visible) else np.nan
        records.append({'model':model,'seed':seed,'metric_scope':'counterfactual observable-subset recall',
            'official_route_rows':len(g),'official_detected':int(g.predicted_anomaly.sum()),'official_frozen_recall':official,
            'observable_rows':len(visible),'observable_detected':int(visible.predicted_anomaly.sum()),
            'counterfactual_observable_subset_recall':counter,'excluded_tier4_rows':len(hidden),
            'tier4_detected':int(hidden.predicted_anomaly.sum()),'tier4_recall':float(hidden.predicted_anomaly.mean()) if len(hidden) else np.nan,
            'audit_delta_percentage_points':100*(counter-official),'official_metric_changed':False})
    return pd.DataFrame(records)


def event_metrics(points,predictions,runs):
    fields=SCORE_KEY+['label_run_id','segment_position','timestamp']
    joined=exact_join(predictions,points[fields],SCORE_KEY,'many_to_one');records=[]
    for (model,seed),g in joined.groupby(['model','seed'],dropna=False,sort=True):
        for run in runs.itertuples(index=False):
            rows=g.loc[g.label_run_id.eq(run.label_run_id)].sort_values('source_point_index')
            hit=rows.loc[rows.predicted_anomaly.eq(1)];first=hit.iloc[0] if len(hit) else None
            records.append({'model':model,'seed':seed,'sample_id':run.sample_id,'label_run_id':run.label_run_id,
                'user_id':run.user_id,'source_trajectory_id':run.source_trajectory_id,'full_labelled_length':run.full_labelled_length,
                'evaluated_labelled_length':len(rows),'event_detected':bool(len(hit)),'detected_point_count':len(hit),
                'first_detection_source_index':int(first.source_point_index) if first is not None else np.nan,
                'first_detection_position':int(first.segment_position) if first is not None else np.nan,
                'detection_delay_points':int(first.source_point_index-run.run_start_index) if first is not None else np.nan,
                'detection_delay_seconds':float((first.timestamp-run.run_start_timestamp).total_seconds()) if first is not None else np.nan,
                'event_status':'detected' if len(hit) else ('not_detected' if len(rows) else 'unsupported_no_evaluation_rows'),
                'metric_scope':'audit-only at least one stored positive within the evaluated labelled event'})
    return pd.DataFrame(records)


def event_summary(events):
    records=[]
    for (model,seed),g in events.groupby(['model','seed'],dropna=False,sort=True):
        for user,subset in [('ALL',g)]+list(g.groupby('user_id',sort=True)):
            supported=subset.loc[subset.evaluated_labelled_length.gt(0)];detected=supported.loc[supported.event_detected]
            records.append({'model':model,'seed':seed,'user_id':user,'event_count':len(subset),'supported_event_count':len(supported),
                'detected_event_count':len(detected),'event_detection_rate':len(detected)/len(supported) if len(supported) else np.nan,
                'first_detection_delay_points_median':float(detected.detection_delay_points.median()) if len(detected) else np.nan,
                'first_detection_delay_seconds_median':float(detected.detection_delay_seconds.median()) if len(detected) else np.nan,
                'na_reason':'' if len(supported) else 'no_supported_events'})
    return pd.DataFrame(records)


def tier_summary(points):
    records=[]
    for tier in TIERS:
        g=points.loc[points.identifiability_tier.eq(tier)]
        records.append({'identifiability_tier':tier,'row_count':len(g),'percentage':100*len(g)/len(points) if len(points) else np.nan,
            'coordinate_displacement_median':float(g.actual_displacement_m.median()) if len(g) else np.nan,
            'feature_changed_count_median':float(g.feature_changed_count.median()) if len(g) else np.nan})
    return pd.DataFrame(records)


def feature_summary(points):
    records=[]
    for feature in FEATURE_COLUMNS:
        values=finite(points[feature+'_delta']);changed=points[feature+'_changed']
        records.append({'feature':feature,'row_count':len(points),'changed_count':int(changed.sum()),
            'changed_percentage':float(changed.mean()*100),'absolute_delta_median':float(np.median(np.abs(values))),
            'absolute_delta_p90':float(np.percentile(np.abs(values),90)),'absolute_delta_p95':float(np.percentile(np.abs(values),95))})
    return pd.DataFrame(records)


def score_by_tier(points):
    records=[]
    columns=['synthetic_reference_distance_m']+[s+'_synthetic_score_m' for s in CONTEXT_SIGNALS]
    for tier in TIERS+('observable_subset','unidentifiable_subset'):
        if tier=='observable_subset':g=points.loc[points.audit_observable]
        elif tier=='unidentifiable_subset':g=points.loc[~points.audit_observable]
        else:g=points.loc[points.identifiability_tier.eq(tier)]
        for column in columns:
            values=pd.to_numeric(g[column],errors='coerce').dropna()
            records.append({'identifiability_tier':tier,'score':column,**spatial.distance_summary(values)})
    return pd.DataFrame(records)


def position_summary(points,column):
    records=[]
    for position,g in points.groupby(column,sort=True):
        records.append({column:position,'labelled_rows':len(g),'displacement_median':float(g.actual_displacement_m.median()),
            'displacement_p90':float(g.actual_displacement_m.quantile(.9)),'feature_changed_count_median':float(g.feature_changed_count.median()),
            **{tier+'_count':int(g.identifiability_tier.eq(tier).sum()) for tier in TIERS},
            **{tier+'_pct':float(g.identifiability_tier.eq(tier).mean()*100) for tier in TIERS}})
    return pd.DataFrame(records)


def sample_summary(points,full,runs):
    records=[]
    for sample,g in points.groupby('sample_id',sort=True):
        copied=full.loc[full.sample_id.eq(sample)];labelled=copied.loc[copied.anomaly_label.eq(1)]
        records.append({'sample_id':sample,'user_id':str(g.user_id.iloc[0]),'source_trajectory_id':str(g.source_trajectory_id.iloc[0]),
            'full_copy_rows':len(copied),'full_labelled_rows':len(labelled),'evaluated_labelled_rows':len(g),
            'excluded_labelled_rows':len(labelled)-len(g),'directly_displaced_rows':int(g.directly_displaced.sum()),
            'displacement_median':float(g.actual_displacement_m.median()),'displacement_max':float(g.actual_displacement_m.max()),
            'observable_ratio':float(g.audit_observable.mean()),**{tier+'_count':int(g.identifiability_tier.eq(tier).sum()) for tier in TIERS}})
    return pd.DataFrame(records)


def user_summary(points,predictions,events):
    records=[]
    for user,g in points.groupby('user_id',sort=True):
        record={'user_id':user,'route_label_count':len(g),'observable_count':int(g.audit_observable.sum()),
            'observable_ratio':float(g.audit_observable.mean()),'tier4_proportion':float(g.identifiability_tier.eq('tier4').mean()),
            **{tier+'_count':int(g.identifiability_tier.eq(tier).sum()) for tier in TIERS},
            **{tier+'_pct':float(g.identifiability_tier.eq(tier).mean()*100) for tier in TIERS}}
        for model,group in predictions.loc[predictions.user_id.eq(user)].groupby('model',sort=True):
            joined=exact_join(group,g[SCORE_KEY+['audit_observable']],SCORE_KEY,'many_to_one')
            seed_rates=[]
            for _,run in joined.groupby('seed',dropna=False):seed_rates.append(run.loc[run.audit_observable,'predicted_anomaly'].mean())
            record[model+'_observable_recall_mean']=float(np.mean(seed_rates))
            per_event=events.loc[events.user_id.eq(user)&events.model.eq(model)]
            record[model+'_event_detection_rate_mean']=float(per_event.groupby('seed',dropna=False).event_detected.mean().mean())
        records.append(record)
    return pd.DataFrame(records)


def policy_simulation(points):
    direct=points.identifiability_tier.eq('tier1');observable=points.audit_observable;boundary=points.segment_boundary.isin(['start','end','singleton'])
    return pd.DataFrame([
        {'policy':'A direct displacement only','positive':int(direct.sum()),'boundary_ignore':0,'negative_candidate':int((~direct).sum()),'count_unit':'points'},
        {'policy':'B direct plus observable boundary','positive':int((direct|(observable&boundary)).sum()),'boundary_ignore':int((~observable&boundary).sum()),
            'negative_candidate':int((~(direct|(observable&boundary))&~(~observable&boundary)).sum()),'count_unit':'points'},
        {'policy':'C segment/event target','positive':int(points.label_run_id.nunique()),'boundary_ignore':0,'negative_candidate':0,'count_unit':'events'},
        {'policy':'D transition/window target (requires target design)','positive':np.nan,'boundary_ignore':np.nan,'negative_candidate':np.nan,'count_unit':'not_defined'},
        {'policy':'E separate unidentifiable boundary','positive':int(observable.sum()),'boundary_ignore':int((~observable).sum()),'negative_candidate':0,'count_unit':'points'}])


def build_tables(points,full,runs,predictions):
    tiers=tier_summary(points);tier_seeds=recall_rows(points,predictions,'identifiability_tier',TIERS)
    tier_recall=recall_summary(tier_seeds,'identifiability_tier')
    disp_seeds=recall_rows(points,predictions,'displacement_bin',BIN_ORDER);disp=recall_summary(disp_seeds,'displacement_bin')
    feature_seeds=recall_rows(points,predictions,'feature_change_category',FEATURE_CATEGORY_ORDER)
    feature_recall=recall_summary(feature_seeds,'feature_change_category')
    counter=counterfactual_metrics(points,predictions);events=event_metrics(points,predictions,runs);event_seeds=event_summary(events)
    run_records=[]
    for run in runs.itertuples(index=False):
        g=points.loc[points.label_run_id.eq(run.label_run_id)]
        run_records.append({**run._asdict(),'evaluated_labelled_length':len(g),'context_not_audited_length':run.full_labelled_length-len(g),
            'evaluated_directly_displaced_length':int(g.directly_displaced.sum()),
            'evaluated_feature_observable_length':int(g.feature_changed_count.gt(0).sum()),
            'evaluated_context_observable_length':int(g.context_changed_count.gt(0).sum()),
            'evaluated_unidentifiable_length':int(g.identifiability_tier.eq('tier4').sum()),
            **{tier+'_count':int(g.identifiability_tier.eq(tier).sum()) for tier in TIERS}})
    confusion=full.groupby(['anomaly_label','coordinate_intervention_strict'],dropna=False).size().rename('row_count').reset_index()
    # Include zero cells; full copied negatives exist, so no negative data is invented.
    grid=pd.MultiIndex.from_product([[0,1],[False,True]],names=['anomaly_label','coordinate_intervention_strict'])
    confusion=confusion.set_index(['anomaly_label','coordinate_intervention_strict']).reindex(grid,fill_value=0).reset_index()
    threshold_rows=[{'threshold_m':0,'comparison':'>','count':int(points.actual_displacement_m.gt(0).sum()),
        'percentage':float(points.actual_displacement_m.gt(0).mean()*100)}]
    threshold_rows += [{'threshold_m':value,'comparison':'>=','count':int(points.actual_displacement_m.ge(value).sum()),
        'percentage':float(points.actual_displacement_m.ge(value).mean()*100)} for value in DISPLACEMENT_THRESHOLDS_M]
    boundaries=position_summary(points,'segment_boundary');progression=position_summary(points,'progression_bucket')
    auxiliary=[]
    for scope,mask in [('all',np.ones(len(full),dtype=bool)),('labelled',full.anomaly_label.eq(1)),('not_labelled',full.anomaly_label.eq(0))]:
        g=full.loc[mask]
        auxiliary.append({'scope':scope,'copied_rows':len(g),'directly_modified_strict_count':int(g.coordinate_intervention_strict.sum()),
            'feature_observable_count':int(g.feature_changed_count.gt(0).sum()),
            'feature_propagation_only_count':int(g.feature_propagation_only.sum()),
            'propagation_previous_modified_count':int((g.feature_propagation_only&g.previous_source_modified).sum()),
            'propagation_next_modified_count':int((g.feature_propagation_only&g.next_source_modified).sum()),
            'propagation_carried_history_count':int((g.feature_propagation_only&~g.previous_source_modified).sum())})
    return {'route_label_point_audit':points,'route_label_full_copy_audit':full,'route_label_tier_summary':tiers,
        'route_label_feature_delta':feature_summary(points),'route_label_displacement_thresholds':pd.DataFrame(threshold_rows),
        'route_label_displacement_bins':disp,'detector_recall_by_displacement_seed':disp_seeds,
        'detector_recall_by_feature_count':feature_recall,'detector_recall_by_feature_count_seed':feature_seeds,
        'route_label_per_user':user_summary(points,predictions,events),'route_label_per_sample':sample_summary(points,full,runs),
        'route_label_segment_boundary':boundaries,'route_label_progression':progression,'route_label_consecutive_runs':pd.DataFrame(run_records),
        'label_intervention_confusion':confusion,'boundary_propagation_summary':pd.DataFrame(auxiliary),
        'detector_recall_by_identifiability_tier':tier_recall,'detector_recall_by_identifiability_tier_seed':tier_seeds,
        'route_label_detector_predictions':predictions,'counterfactual_observable_subset_metrics':counter,
        'event_level_detection_metrics':events,'event_level_detection_summary':event_seeds,
        'label_policy_simulation':policy_simulation(points),'spatial_window_score_by_tier':score_by_tier(points)}


def create_figures(tables,figure_dir):
    paths=[];colours=['#3073a5','#e89537','#62a269','#b7545b']
    def save(name):
        plt.tight_layout();p=figure_dir/(name+'.png');plt.savefig(p,dpi=160);plt.close();paths.append(p)
    tiers=tables['route_label_tier_summary'];plt.figure(figsize=(7,4));plt.bar(tiers.identifiability_tier,tiers.row_count,color=colours)
    for index,row in tiers.iterrows():plt.text(index,row.row_count,f'{int(row.row_count)} ({row.percentage:.2f}%)',ha='center',va='bottom')
    plt.ylim(0,max(1,tiers.row_count.max()*1.15));plt.ylabel('Evaluated labelled points');save('identifiability_tier_distribution')
    points=tables['route_label_point_audit'];plt.figure(figsize=(8,4));plt.hist(points.actual_displacement_m,bins=[0,1,5,10,25,50,100,200,300,400],color=colours[0])
    plt.xlabel('Stored coordinate displacement (m)');plt.ylabel('Labelled point count');save('coordinate_displacement_distribution')
    plt.figure(figsize=(7,4));count=points.feature_changed_count.value_counts().reindex(range(9),fill_value=0);plt.bar(count.index,count)
    plt.xticks(range(9));plt.xlabel('Changed features out of 8');plt.ylabel('Labelled point count');save('feature_change_count_distribution')
    plt.figure(figsize=(8,4));x=np.arange(4);recall=tables['detector_recall_by_identifiability_tier']
    for i,(model,g) in enumerate(recall.groupby('model',sort=True)):
        g=g.set_index('identifiability_tier').reindex(TIERS);plt.bar(x+(i-1)*.24,g.recall_mean,width=.24,label=model)
    plt.xticks(x,TIERS);plt.ylim(0,1);plt.ylabel('Stored prediction recall (seed mean)');
    if int(tiers.loc[tiers.identifiability_tier.eq('tier3'),'row_count'].sum())==0:plt.title('Tier 3 has no rows: NA, not zero recall')
    plt.legend();save('detector_recall_by_tier')
    plt.figure(figsize=(8,4));u=tables['route_label_per_user'];bottom=np.zeros(len(u))
    for tier,colour in zip(TIERS,colours):plt.bar(u.user_id,u[tier+'_pct'],bottom=bottom,label=tier,color=colour);bottom+=u[tier+'_pct'].to_numpy()
    plt.ylabel('Percentage of evaluated route labels');plt.ylim(0,100);plt.legend();save('tier_distribution_by_user')
    plt.figure(figsize=(8,4));s=tables['route_label_segment_boundary'].set_index('segment_boundary').reindex(['start','interior','end']);bottom=np.zeros(len(s))
    for tier,colour in zip(TIERS,colours):plt.bar(s.index,s[tier+'_count'],bottom=bottom,label=tier,color=colour);bottom+=s[tier+'_count'].to_numpy()
    plt.ylabel('Evaluated labelled points');plt.legend();save('tier_by_segment_position')
    plt.figure(figsize=(9,4));events=tables['event_level_detection_summary'];counter=tables['counterfactual_observable_subset_metrics']
    models=sorted(counter.model.unique());x=np.arange(len(models));official=counter.groupby('model').official_frozen_recall.mean().reindex(models)
    observable=counter.groupby('model').counterfactual_observable_subset_recall.mean().reindex(models)
    event=events.loc[events.user_id.eq('ALL')].groupby('model').event_detection_rate.mean().reindex(models)
    for offset,values,label in [(-.25,official,'Frozen point recall'),(0,observable,'Audit observable-subset recall'),(.25,event,'Audit event detection')]:plt.bar(x+offset,values,width=.25,label=label)
    plt.xticks(x,models);plt.ylim(0,1.05);plt.ylabel('Stored predictions, seed mean');plt.legend(fontsize=8);save('event_detection_summary')
    return paths


def read_frozen(path):
    return pd.read_csv(path,dtype={c:'string' for c in ID_COLUMNS},low_memory=False)


def json_safe(value):
    if value is None or value is pd.NA or value is pd.NaT:return None
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(float,np.floating)):
        if np.isnan(value):return None
        if not np.isfinite(value):raise ValueError('Infinite report metric.')
        return float(value)
    if isinstance(value,pd.Timestamp):return value.isoformat()
    return value


def run_audit(data_dir=single.DEFAULT_DATA_DIR,output_root=single.ROOT,protected_snapshot=None):
    started=time.perf_counter();root=Path(output_root).resolve();data_dir=Path(data_dir).resolve()
    metrics=root/'outputs/metrics/stage6/route_label_audit';figures=root/'outputs/figures/stage6/route_label_audit'
    if metrics.exists() or figures.exists():raise FileExistsError('Label audit overwrite refused.')
    snapshot=json.loads(Path(protected_snapshot or root/'outputs/metrics/stage65_protected_snapshot.json').read_text())
    required=[data_dir,'models/stage5','outputs/metrics/stage5','outputs/metrics/stage6/context_reference',
        'outputs/metrics/stage6/window_similarity','outputs/figures/stage6/window_similarity','src/synthetic_anomalies.py']
    for required_path in required:
        relative=Path(required_path).relative_to(root).as_posix() if isinstance(required_path,Path) else required_path
        if not any(k.replace('\\','/')==relative or k.replace('\\','/').startswith(relative+'/') for k in snapshot):
            raise ValueError('Incomplete protected snapshot: '+relative)
    assert_protected(root,snapshot);validate_saved_dataset(data_dir)
    summary,users=single.read_stage5_metadata(data_dir);test=single.load_stage5_split(data_dir,'test',summary,users)
    original=read_dataset_csv(data_dir/'source_normal.csv');original=original.loc[original.dataset_split.eq('test')]
    full_test=read_dataset_csv(data_dir/'test.csv');copies=full_test.loc[full_test.is_synthetic&full_test.anomaly_type.eq('route_deviation')]
    manifest=read_dataset_csv(data_dir/'synthetic_anomaly_manifest.csv');manifest=manifest.loc[manifest.dataset_split.eq('test')&manifest.anomaly_type.eq('route_deviation')]
    allowed=set(users.loc[users.dataset_split.eq('test'),'user_id'])
    if not set(copies.user_id).issubset(allowed) or manifest.sample_id.duplicated().any() or set(copies.sample_id)!=set(manifest.sample_id):
        raise ValueError('Unauthorized/ambiguous route sample catalog.')
    full,runs=segment_structure(point_deltas(copies,original))
    if full.timestamp_intervention.any():raise ValueError('Unexpected timestamp intervention for frozen route generator.')
    for row in manifest.itertuples(index=False):
        g=full.loc[full.sample_id.eq(row.sample_id)];labelled=g.loc[g.anomaly_label.eq(1)]
        if len(labelled)!=row.anomaly_point_count or len(g)!=row.generated_point_count or labelled.timestamp.min()!=row.anomaly_start_time or labelled.timestamp.max()!=row.anomaly_end_time:
            raise ValueError('Stored manifest segment mismatch.')
    evaluation=test.loc[test.is_synthetic&test.anomaly_type.eq('route_deviation')]
    points=exact_join(evaluation[SCORE_KEY],full,SCORE_KEY)
    stage63=read_frozen(root/'outputs/metrics/stage6/context_reference/source_matched_context_delta.csv')
    stage64=read_frozen(root/'outputs/metrics/stage6/window_similarity/source_matched_window_delta.csv')
    stage62=read_frozen(root/'outputs/metrics/stage6/route_reference_coverage/source_matched_distance_delta.csv')
    points=classify_tiers(attach_context(points,stage63,stage64,stage62))
    # Historical 38/19 regression is based on recomputed coordinates/features, not copied audit flags.
    old_samples=read_dataset_csv(root/'outputs/metrics/stage6'/data_dir.name/'route_audit/route_deviation_samples.csv')
    old_samples=old_samples.loc[old_samples.dataset_split.eq('test')&old_samples.evaluation_included]
    check=exact_join(points[SCORE_KEY+['zero_displacement','feature_near_identical']],old_samples[SCORE_KEY+['zero_displacement','feature_near_identical']].rename(columns={
        'zero_displacement':'old_zero','feature_near_identical':'old_identical'}),SCORE_KEY)
    for new,old in [('zero_displacement','old_zero'),('feature_near_identical','old_identical')]:
        if not np.array_equal(check[new],check[old].astype(str).str.lower().eq('true')):raise ValueError('Frozen boundary regression mismatch.')
    predictions,prediction_inputs=load_predictions(root,data_dir.name,test,points)
    tables=build_tables(points,full,runs,predictions)
    # Confirm unchanged official route recall against Stage 5.4 per-seed counts.
    official=read_frozen(root/'outputs/metrics/stage5'/data_dir.name/'baseline_comparison/per_anomaly_type_baseline_metrics.csv')
    official=official.loc[official.anomaly_type.eq('route_deviation')]
    for row in tables['counterfactual_observable_subset_metrics'].itertuples(index=False):
        match=official.loc[official.model.eq(row.model)&(official.seed.isna() if pd.isna(row.seed) else official.seed.eq(row.seed))]
        if len(match)!=1 or int(match.anomaly_row_count.iloc[0])!=row.official_route_rows or int(match.detected_anomaly_row_count.iloc[0])!=row.official_detected:
            raise ValueError('Official frozen detector count mismatch.')
        if not np.isclose(match.recall.iloc[0],row.official_frozen_recall,atol=1e-12):raise ValueError('Official recall mismatch.')
    assert_protected(root,snapshot);metrics.mkdir(parents=True);figures.mkdir(parents=True)
    for name,frame in tables.items():frame.to_csv(metrics/(name+'.csv'),index=False,na_rep='NA')
    saved_figures=create_figures(tables,figures)
    tier4=exact_join(predictions,points[SCORE_KEY+['identifiability_tier']],SCORE_KEY,'many_to_one');tier4=tier4.loc[tier4.identifiability_tier.eq('tier4')]
    report={'stage':'6.5','name':'Synthetic Route Label Identifiability & Boundary Audit','dataset_id':data_dir.name,'analysis_only':True,
        'counts':{'evaluation_route_labels':len(points),'full_copy_rows':len(full),'full_copy_labels':int(full.anomaly_label.sum()),
            'excluded_labelled_rows':int(full.anomaly_label.sum())-len(points),'samples':full.sample_id.nunique(),'label_runs':len(runs),
            'zero_displacement':int(points.zero_displacement.sum()),'identical_features':int(points.feature_near_identical.sum()),
            'observable_subset':int(points.audit_observable.sum()),'unidentifiable_subset':int((~points.audit_observable).sum()),
            'unlabelled_feature_observable_rows':int((full.anomaly_label.eq(0)&full.feature_changed_count.gt(0)).sum())},
        'tier_summary':tables['route_label_tier_summary'].to_dict(orient='records'),
        'feature_change_count_distribution':{str(k):int(v) for k,v in points.feature_changed_count.value_counts().sort_index().items()},
        'tolerances':{'coordinate_atol_m':COORD_ATOL_M,'feature_atol':FEATURE_ATOL,'feature_rtol':FEATURE_RTOL,
            'context_atol_m':CONTEXT_ATOL_M,'context_rtol':CONTEXT_RTOL},
        'context_signals':list(CONTEXT_SIGNALS),'context_scope':'W10/gap0 is predefined shortest boundary audit, not selected detector configuration.',
        'all_context_comparisons_complete':bool(points.context_comparison_complete.all()),
        'displacement_thresholds_m':['>0']+list(DISPLACEMENT_THRESHOLDS_M),'low_support_minimum_rows':LOW_SUPPORT_N,
        'generator_review':{'source_file':'src/synthetic_anomalies.py','generation_function':'generate_anomaly_sample',
            'selection':'Independent full source copy, contiguous uniform integer length 10..30, start>=1, end exclusive <=n-1.',
            'displacement':'Uniform configured amplitude 100..300m; sin(linspace(0,pi,L)) weights and heading-derived north/east offsets.',
            'boundary':'Labels include sine-zero first/last segment points; raw stored end coordinates may round to unchanged.',
            'recomputation':'Full copy features recalculated; route timestamps preserved; past transitions/carried features can affect unlabelled following rows.',
            'stage5_seed':'SHA256 first 8 bytes of JSON [dataset_seed,source_id,anomaly_type,1]; no RNG/generation invoked by this audit.',
            'source_point_index':'Stage 5 copies ordered original source index after verifying unchanged point count.',
            'metadata_present':list(manifest.columns),'actual_modified_mask_stored':False,'configured_amplitude_stored':False,
            'intervention_reconstruction':'Existing stored raw coordinate/timestamp counterfactual pair, no synthetic regeneration.'},
        'detector_policy':'Stored Stage 5 Rule + five IF + five AE predictions and thresholds only; no scoring/inference/fitting.',
        'prediction_input_checksums':prediction_inputs,'tier4_paired_prediction_equal':bool(tier4.predicted_anomaly.eq(tier4.original_predicted_anomaly).all()),
        'tier4_paired_score_equal':bool(np.isclose(tier4.anomaly_score,tier4.original_anomaly_score,atol=1e-12,rtol=1e-12).all()),
        'tier4_warning':'Unidentifiable for the audited representations is not automatically an incorrect label; positive predictions may be paired-normal false positives.',
        'point_label_changed':False,'synthetic_generated':False,'model_inference_or_training':False,'official_metrics_changed':False,
        'source_original_used_for_detector_input':False,'test_threshold_tuning':False,'user_split_changed':False,'stage66_implemented':False,
        'protected_file_count':len(snapshot),'protected_changed_files':0,'audit_only_separation_violations':0,'cross_split_leakage':0,
        'execution_seconds':time.perf_counter()-started,'source_sha256':single.sha256_file(Path(__file__)),
        'limitations':['Tier tests observable difference, not sufficient anomaly information or diagnostic accuracy.',
            'Four frozen context representations do not cover all possible observable history functions.',
            'Feature tolerance uses native units; context equality is metre tolerance. Positive-bin cutoffs are descriptive, not tuned labels.',
            'Full copy contains 379 labels, evaluation has 378; omitted quality-invalid row is not assigned a tier or fabricated prediction.',
            'Event rates are conditional on existing synthetic events, with no negative-event false-alarm analysis or new operating point.',
            'Seed means and point proportions are descriptive; points in one segment are dependent. No causal user explanation.']}
    assert_protected(root,snapshot);single.save_json(metrics/'route_label_audit.json',json_safe(report))
    verification={'passed':True,'protected_file_count':len(snapshot),'protected_changed_files':0,
        'source_sha256':report['source_sha256'],'output_checksums':{p.name:single.sha256_file(p) for p in metrics.glob('*')},
        'figure_checksums':{p.name:single.sha256_file(p) for p in saved_figures}}
    single.save_json(metrics/'stage6_5_verification.json',verification)
    print(json.dumps(json_safe({'counts':report['counts'],'tiers':report['tier_summary'],'execution_seconds':report['execution_seconds']}),indent=2),flush=True)
    return {'report':json_safe(report),'tables':tables,'metrics_dir':metrics,'figure_dir':figures,'verification':verification}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root',type=Path,default=single.ROOT);parser.add_argument('--protected-snapshot',type=Path)
    args=parser.parse_args();run_audit(args.data_dir,args.output_root,args.protected_snapshot)


if __name__=='__main__':main()
