"""Stage 6.3 causal, label-blind GPS context replay; analysis only."""
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
from sklearn.neighbors import BallTree
from src import audit_route_reference_coverage as spatial
from src import audit_route_representation as route_audit
from src import train_multiuser_autoencoder as single
from src.load_multiuser_geolife import read_dataset_csv
from src.feature_engineering import EARTH_RADIUS_M, haversine_distance
from src.prepare_multiuser_dataset import validate_saved_dataset

OBS = ['user_id', 'trajectory_id', 'source_trajectory_id', 'source_point_index', 'timestamp', 'latitude', 'longitude']
REFERENCES = ('global_train_causal', 'prior_personal', 'causal_prefix', 'combined_personal', 'global_personal_min')
MIN_CONTEXT = (10, 25, 50, 100)
RADII = (10, 25, 50, 100, 200, 500)
DURATION_SEC = (300, 900, 1800)
KEY = spatial.KEY
SCORE_KEY = spatial.SCORE_KEY
BLOCK_SIZE = 512


def observable(frame):
    """Only observed GPS/identity/time cross the scoring boundary."""
    result = frame[OBS].copy()
    spatial.coordinates(result)
    if result.timestamp.isna().any(): raise ValueError('Missing observed timestamp.')
    result['timestamp'] = pd.to_datetime(result.timestamp, errors='raise')
    index = result.source_point_index.to_numpy(dtype=float)
    if not np.isfinite(index).all() or not np.equal(index, np.floor(index)).all() or (index < 0).any():
        raise ValueError('Invalid observed source point index.')
    if not result.trajectory_id.eq(result.source_trajectory_id).all(): raise ValueError('Observed source identity differs.')
    return result


def validate_stream(stream):
    if stream.empty or stream.user_id.nunique() != 1 or stream.source_trajectory_id.nunique() != 1:
        raise ValueError('A single nonempty user trajectory stream required.')
    if not stream.timestamp.diff().dropna().gt(pd.Timedelta(0)).all():
        raise ValueError('Stream timestamps must be strictly increasing.')
    if not stream.source_point_index.diff().dropna().gt(0).all():
        raise ValueError('Stream source indices must be strictly increasing.')


def trajectory_catalog(originals, allowed_users):
    frame = observable(originals)
    if not set(frame.user_id).issubset(set(allowed_users)) or frame.duplicated(KEY).any():
        raise ValueError('Unauthorized personal user or duplicate observed lineage.')
    records = []
    for source, group in frame.groupby('source_trajectory_id', sort=True):
        validate_stream(group)
        records.append({'user_id': str(group.user_id.iloc[0]), 'source_trajectory_id': str(source),
            'start_timestamp': group.timestamp.iloc[0], 'end_timestamp': group.timestamp.iloc[-1], 'point_count': len(group)})
    catalog = pd.DataFrame(records).sort_values(['user_id', 'start_timestamp', 'source_trajectory_id'], kind='stable')
    catalog['trajectory_order'] = catalog.groupby('user_id').cumcount()
    # Equal start times cannot establish a strict prior trajectory order.
    if catalog.duplicated(['user_id', 'start_timestamp']).any(): raise ValueError('Ambiguous same-user trajectory start order.')
    return catalog.reset_index(drop=True)


class PastIndex:
    """Exact nearest past using a frozen BallTree plus a recent point buffer."""
    def __init__(self, points):
        self.rows = observable(points).sort_values(['timestamp', 'user_id', 'source_trajectory_id', 'source_point_index'], kind='stable').reset_index(drop=True)
        self.times = self.rows.timestamp.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        self.lat = self.rows.latitude.to_numpy(dtype=float); self.lon = self.rows.longitude.to_numpy(dtype=float)
        self.radians = np.radians(np.column_stack([self.lat, self.lon]))
        self.tree = None; self.base_end = 0; self.last_t = None

    def nearest(self, timestamp, latitude, longitude):
        t = int(pd.Timestamp(timestamp).value)
        if self.last_t is not None and t < self.last_t: raise ValueError('Query time moved backwards.')
        self.last_t = t
        end = int(np.searchsorted(self.times, t, side='left'))
        if end and not self.times[end - 1] < t: raise ValueError('Future/self point in reference.')
        if end - self.base_end >= BLOCK_SIZE:
            self.tree = BallTree(self.radians[:end], metric='haversine', leaf_size=40); self.base_end = end
        best = np.inf; nearest = -1
        if self.tree is not None:
            distance, index = self.tree.query(np.radians([[latitude, longitude]]), k=1)
            best = float(distance[0, 0] * EARTH_RADIUS_M); nearest = int(index[0, 0])
        if end > self.base_end:
            d = haversine_distance(latitude, longitude, self.lat[self.base_end:end], self.lon[self.base_end:end])
            j = int(np.argmin(d))
            if float(d[j]) < best:
                best = float(d[j]); nearest = self.base_end + j
        duration = float((self.times[end - 1] - self.times[0]) / 1e9) if end else 0.
        if nearest < 0:
            return np.nan, end, duration, '', '', -1, pd.NaT
        row = self.rows.iloc[nearest]
        if row.timestamp >= pd.Timestamp(timestamp): raise ValueError('Nearest reference is not past.')
        return best, end, duration, str(row.user_id), str(row.source_trajectory_id), int(row.source_point_index), row.timestamp


def prior_pool(originals, catalog, user, source):
    current = catalog.loc[catalog.user_id.eq(user) & catalog.source_trajectory_id.eq(source)]
    if len(current) != 1: raise ValueError('Current trajectory absent/ambiguous in personal catalog.')
    row = current.iloc[0]
    prior = catalog.loc[catalog.user_id.eq(user) & catalog.trajectory_order.lt(row.trajectory_order)]
    if not prior.start_timestamp.lt(row.start_timestamp).all(): raise ValueError('Future trajectory in prior history.')
    points = observable(originals.loc[originals.user_id.eq(user) & originals.source_trajectory_id.isin(prior.source_trajectory_id)])
    return points, row, len(prior)


def score_stream(stream, history, permitted_prior_sources, allowed_user):
    """No labels, onset, paired originals or global references enter this replay."""
    stream = observable(stream); history = observable(history); validate_stream(stream)
    source = str(stream.source_trajectory_id.iloc[0]); user = str(stream.user_id.iloc[0])
    if user != allowed_user or not history.user_id.eq(user).all(): raise ValueError('Cross-user context contamination.')
    if source in set(history.source_trajectory_id) or not set(history.source_trajectory_id).issubset(set(permitted_prior_sources)):
        raise ValueError('Current/future/unauthorized trajectory in prior reference.')
    if history.duplicated(KEY).any(): raise ValueError('Duplicate personal history point.')
    if not history.empty and not history.groupby('source_trajectory_id').timestamp.min().lt(stream.timestamp.iloc[0]).all():
        raise ValueError('Future trajectory start in personal history.')
    for _, group in history.groupby('source_trajectory_id',sort=False): validate_stream(group)
    prior_index = PastIndex(history); prefix_index = PastIndex(stream)
    results = []
    for q in stream.itertuples(index=False):
        p = prior_index.nearest(q.timestamp, q.latitude, q.longitude)
        c = prefix_index.nearest(q.timestamp, q.latitude, q.longitude)
        if c[1] and not int(c[5]) < q.source_point_index: raise ValueError('Prefix current/future source index.')
        combined_distance = float(np.fmin(p[0], c[0])) if p[1] + c[1] else np.nan
        chosen = p if p[1] and (not c[1] or p[0] <= c[0]) else c
        first_time = min([index.times[0] for index, result in ((prior_index, p), (prefix_index, c)) if result[1]], default=None)
        # Span of all observed context points, excluding the current point.
        last_observed = max([index.times[result[1]-1] for index, result in ((prior_index,p),(prefix_index,c)) if result[1]], default=None)
        combined_duration = (last_observed-first_time)/1e9 if first_time is not None else 0.
        record = {k: getattr(q, k) for k in OBS}
        for name, result in [('prior_personal',p),('causal_prefix',c),
                ('combined_personal',(combined_distance,p[1]+c[1],combined_duration,*chosen[3:]))]:
            record[name+'_distance_m'] = result[0]; record[name+'_context_points'] = result[1]
            record[name+'_duration_sec'] = result[2]
            components = [(prior_index,p)] if name=='prior_personal' else [(prefix_index,c)] if name=='causal_prefix' else [(prior_index,p),(prefix_index,c)]
            firsts = [index.times[0] for index,value in components if value[1]]
            lasts = [index.times[value[1]-1] for index,value in components if value[1]]
            record[name+'_first_context_timestamp'] = pd.Timestamp(min(firsts)) if firsts else pd.NaT
            record[name+'_last_context_timestamp'] = pd.Timestamp(max(lasts)) if lasts else pd.NaT
            record[name+'_nearest_user_id'] = result[3]; record[name+'_nearest_source_id'] = result[4]
            record[name+'_nearest_point_index'] = result[5]; record[name+'_nearest_timestamp'] = result[6]
            record[name+'_status'] = 'available' if result[1] else ('no_personal_history' if name=='prior_personal' else 'no_context')
        results.append(record)
    return pd.DataFrame(results)


def score_global(train, queries, allowed_users):
    points = observable(train)
    if not set(points.user_id).issubset(set(allowed_users)): raise ValueError('Unauthorized global reference user.')
    index = PastIndex(points); ordered = queries.sort_values('timestamp', kind='stable')
    result = []
    for q in ordered.itertuples():
        d,count,duration,user,source,point,stamp = index.nearest(q.timestamp,q.latitude,q.longitude)
        result.append({'row_key':q.Index,'global_train_causal_distance_m':d,
            'global_train_causal_context_points':count,'global_train_causal_duration_sec':duration,
            'global_train_causal_nearest_user_id':user,'global_train_causal_nearest_source_id':source,
            'global_train_causal_nearest_point_index':point,'global_train_causal_nearest_timestamp':stamp,
            'global_train_causal_status':'available' if count else 'no_train_past',
            'global_train_causal_first_context_timestamp':pd.Timestamp(index.times[0]) if count else pd.NaT,
            'global_train_causal_last_context_timestamp':pd.Timestamp(index.times[count-1]) if count else pd.NaT})
    return pd.DataFrame(result).set_index('row_key').reindex(queries.index).reset_index(drop=True)

def summarize(normal, route):
    availability, coverage, metrics, support = [], [], [], []
    for user in ['ALL'] + sorted(normal.user_id.unique().tolist()):
        n = normal if user=='ALL' else normal.loc[normal.user_id.eq(user)]
        a = route if user=='ALL' else route.loc[route.user_id.eq(user)]
        for ref in REFERENCES:
            distance = ref+'_distance_m'; count = ref+'_context_points'; duration = ref+'_duration_sec'
            nv = n[distance].dropna(); av = a[distance].dropna()
            availability.append({'user_id':user,'reference':ref,'normal_total':len(n),'anomaly_total':len(a),
                'normal_available_count':len(nv),'anomaly_available_count':len(av),
                'normal_available_pct':100*len(nv)/len(n),'anomaly_available_pct':100*len(av)/len(a) if len(a) else None,
                'normal_no_context_count':int(n[count].eq(0).sum()),'anomaly_no_context_count':int(a[count].eq(0).sum()),
                **{f'normal_history_ge_{s}sec_pct':100*float(n[duration].ge(s).mean()) for s in DURATION_SEC}})
            stats=spatial.distance_summary(nv)
            metrics.append({'user_id':user,'reference':ref,'normal_available_pct':100*len(nv)/len(n),
                'anomaly_available_pct':100*len(av)/len(a) if len(a) else None,
                'normal_median':stats['median'],'normal_p90':stats['p90'],'normal_p95':stats['p95'],
                'route_median':float(av.median()) if len(av) else None,**spatial.separation(nv,av)})
            for radius in RADII:
                covered=int(n[distance].le(radius).sum())
                coverage.append({'user_id':user,'reference':ref,'radius_m':radius,'normal_total':len(n),
                    'normal_available_count':len(nv),'unsupported_count':len(n)-len(nv),'covered_count':covered,
                    'coverage_all_pct':100*covered/len(n),'coverage_available_pct':100*covered/len(nv) if len(nv) else None})
            for minimum in MIN_CONTEXT:
                nn=n.loc[n[count].ge(minimum),distance].dropna(); aa=a.loc[a[count].ge(minimum),distance].dropna()
                support.append({'user_id':user,'reference':ref,'minimum_context_points':minimum,
                    'normal_total':len(n),'anomaly_total':len(a),'normal_supported_count':len(nn),'anomaly_supported_count':len(aa),
                    'normal_evaluable_pct':100*len(nn)/len(n),'anomaly_evaluable_pct':100*len(aa)/len(a) if len(a) else None,
                    'normal_insufficient_context_count':int((n[count].gt(0)&n[count].lt(minimum)).sum()),
                    'anomaly_insufficient_context_count':int((a[count].gt(0)&a[count].lt(minimum)).sum()),
                    'normal_median':float(nn.median()) if len(nn) else None,'anomaly_median':float(aa.median()) if len(aa) else None,
                    **spatial.separation(nn,aa)})
    return tuple(pd.DataFrame(rows) for rows in (availability,coverage,metrics,support))


def paired_audit(normal, route):
    cols=KEY+[ref+'_distance_m' for ref in REFERENCES]
    original=normal[cols].rename(columns={c:'original_'+c for c in cols if c not in KEY})
    pairs=route.merge(original,on=KEY,how='left',validate='one_to_one',indicator=True)
    if pairs['_merge'].ne('both').any(): raise ValueError('Missing audit-only original source.')
    pairs=pairs.drop(columns='_merge')
    for ref in REFERENCES: pairs[ref+'_delta_m']=pairs[ref+'_distance_m']-pairs['original_'+ref+'_distance_m']
    return pairs


def progression_audit(pairs):
    records=[]
    for bucket,group in pairs.groupby('progression_bucket',sort=False):
        for ref in REFERENCES:
            d=group[ref+'_distance_m'].dropna(); delta=group[ref+'_delta_m'].dropna()
            records.append({'progression_bucket':bucket,'reference':ref,'point_count':len(group),
                'available_count':len(d),'distance_median':float(d.median()) if len(d) else None,
                'distance_p90':float(d.quantile(.9)) if len(d) else None,
                'source_matched_delta_median':float(delta.median()) if len(delta) else None,
                'past_labelled_context_median':float(group.past_labelled_prefix_count.median())})
    return pd.DataFrame(records)


def create_figures(normal,route,availability,coverage,metrics,support,progression,folder):
    paths=[]
    def save(name):
        p=folder/name;plt.tight_layout();plt.savefig(p,dpi=150);plt.close();paths.append(p)
    plt.figure(figsize=(9,5))
    for ref,group in coverage.loc[coverage.user_id.eq('ALL')].groupby('reference',sort=False):
        plt.plot(group.radius_m,group.coverage_all_pct,'o-',label=ref)
    plt.xscale('log');plt.xticks(RADII,[str(r) for r in RADII]);plt.gca().minorticks_off()
    plt.ylim(0,100);plt.xlabel('Radius (m)');plt.ylabel('Covered / all normal (%)');plt.legend(fontsize=8)
    save('coverage_by_reference_type.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,ref in zip(axes,['prior_personal','causal_prefix','combined_personal']):
        for f,label in [(normal,'Normal'),(route,'Route')]:
            x=np.sort(f[ref+'_distance_m'].dropna());ax.plot(x,np.arange(1,len(x)+1)/max(1,len(x)),label=label)
        ax.set(xscale='symlog',xlabel='Past-context distance (m)',ylabel='ECDF',title=ref);ax.legend()
    save('normal_vs_route_context_distance.png')
    a=availability.loc[availability.user_id.ne('ALL')].pivot(index='user_id',columns='reference',values='normal_available_pct')
    a.plot.bar(figsize=(10,5));plt.ylabel('Available / all normal (%)');plt.ylim(0,105);plt.legend(fontsize=8)
    save('context_availability_by_user.png')
    m=metrics.loc[metrics.user_id.eq('ALL')]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(m.reference,m.roc_auc);axes[0].axhline(.5,color='gray',linestyle='--');axes[0].set_ylabel('ROC-AUC')
    axes[1].bar(m.reference,m.average_precision);axes[1].plot(np.arange(len(m)),m.prevalence,'ro',label='AP prevalence baseline');axes[1].set_ylabel('AP');axes[1].legend()
    for ax in axes: ax.tick_params(axis='x',rotation=30);ax.set_ylim(0,1)
    save('roc_auc_by_reference_type.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ref,g in support.loc[support.user_id.eq('ALL')].groupby('reference',sort=False):
        axes[0].plot(g.minimum_context_points,g.normal_evaluable_pct,'o-',label=ref)
        axes[1].plot(g.minimum_context_points,g.roc_auc,'o-',label=ref)
    axes[0].set(ylabel='Normal evaluable (%)',xlabel='Minimum past points');axes[1].set(ylabel='Supported ROC-AUC',xlabel='Minimum past points')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);save('support_vs_separation.png')
    plt.figure(figsize=(9,5))
    order=['early','middle','late']
    for ref in ['prior_personal','causal_prefix','combined_personal']:
        g=progression.loc[progression.reference.eq(ref)].set_index('progression_bucket').reindex(order)
        plt.plot(order,g.distance_median,'o-',label=ref)
    plt.ylabel('Route distance median (m)');plt.legend();save('anomaly_progression_distance.png')
    u=coverage.loc[coverage.user_id.ne('ALL')&coverage.radius_m.eq(50)].pivot(index='user_id',columns='reference',values='coverage_all_pct')
    u.plot.bar(figsize=(10,5));plt.ylabel('Normal coverage@50m / all (%)');plt.ylim(0,100);plt.legend(fontsize=8)
    save('per_user_context_coverage.png')
    return paths

def select_evaluation(scored,evaluation):
    if scored.duplicated(KEY).any() or evaluation.duplicated(KEY).any(): raise ValueError('Ambiguous evaluation lineage.')
    check=evaluation[OBS].merge(scored[OBS],on=KEY,how='left',validate='one_to_one',suffixes=('','_observed'),indicator=True)
    if check['_merge'].ne('both').any(): raise ValueError('Evaluation point was not replayed.')
    for c in OBS:
        if c not in KEY and not check[c].eq(check[c+'_observed']).all(): raise ValueError('Replay/evaluation identity mismatch: '+c)
    extra=[c for c in scored if c not in OBS]
    return evaluation.merge(scored[KEY+extra],on=KEY,how='left',validate='one_to_one',sort=False)


def run_audit(data_dir=single.DEFAULT_DATA_DIR,output_root=single.ROOT,protected_snapshot=None):
    started=time.perf_counter();root=Path(output_root).resolve();data_dir=Path(data_dir).resolve()
    metrics_dir=root/'outputs/metrics/stage6/context_reference';fig_dir=root/'outputs/figures/stage6/context_reference'
    if metrics_dir.exists() or fig_dir.exists(): raise FileExistsError('Context audit overwrite refused.')
    protected_snapshot=Path(protected_snapshot or root/'outputs/metrics/stage63_protected_snapshot.json')
    expected=json.loads(protected_snapshot.read_text());dataset_id=data_dir.name
    required=[data_dir.relative_to(root),Path('models/stage5')/dataset_id,Path('outputs/metrics/stage5')/dataset_id,
        Path('outputs/figures/stage5')/dataset_id,Path('outputs/metrics/stage6')/dataset_id/'route_audit',
        Path('outputs/figures/stage6')/dataset_id/'route_audit',Path('outputs/metrics/stage6/route_reference_coverage'),
        Path('outputs/figures/stage6/route_reference_coverage')]
    if {k.replace('\\','/') for k in expected}!={p.as_posix() for p in required}: raise ValueError('Incomplete Stage 5/6.1/6.2 protected snapshot.')
    route_audit.assert_protected(root,expected);leakage=validate_saved_dataset(data_dir)
    summary,users=single.read_stage5_metadata(data_dir)
    train=single.load_stage5_split(data_dir,'train',summary,users)
    test=single.load_stage5_split(data_dir,'test',summary,users)
    spatial.validate_reference(train,users,read_dataset_csv(data_dir/'trajectory_quality_summary.csv'))
    allowed_test=set(users.loc[users.dataset_split.eq('test'),'user_id'])
    allowed_train=set(users.loc[users.dataset_split.eq('train'),'user_id'])
    all_originals=read_dataset_csv(data_dir/'source_normal.csv')
    original=observable(all_originals.loc[all_originals.dataset_split.eq('test')])
    catalog=trajectory_catalog(original,allowed_test)
    full_test=read_dataset_csv(data_dir/'test.csv')
    copies=full_test.loc[full_test.is_synthetic&full_test.anomaly_type.eq('route_deviation')]
    if not set(copies.user_id).issubset(allowed_test): raise ValueError('Unauthorized route stream user.')
    normal_eval=test.loc[~test.is_synthetic];route_eval=test.loc[test.is_synthetic&test.anomaly_type.eq('route_deviation')]
    history_records=[];normal_records=[];route_records=[]
    build_started=time.perf_counter()
    for row in catalog.itertuples(index=False):
        history,entry,previous_count=prior_pool(original,catalog,row.user_id,row.source_trajectory_id)
        permitted=set(history.source_trajectory_id)
        stream=original.loc[original.source_trajectory_id.eq(row.source_trajectory_id)]
        evaluated=normal_eval.loc[normal_eval.source_trajectory_id.eq(row.source_trajectory_id)]
        scored=score_stream(stream,history,permitted,row.user_id)
        normal_records.append(select_evaluation(scored,evaluated))
        history_records.append({'user_id':row.user_id,'source_trajectory_id':row.source_trajectory_id,
            'trajectory_order':row.trajectory_order,'start_timestamp':row.start_timestamp,'end_timestamp':row.end_timestamp,
            'previous_trajectory_count':previous_count,'previous_history_logged_points':len(history),
            'past_history_points_at_start':int(scored.prior_personal_context_points.iloc[0]),
            'past_history_duration_at_start_sec':float(scored.prior_personal_duration_sec.iloc[0]),
            'history_status_at_start':scored.prior_personal_status.iloc[0]})
        scenario=copies.loc[copies.source_trajectory_id.eq(row.source_trajectory_id)]
        if scenario.sample_id.nunique()!=1: raise ValueError('Expected one independent route scenario per source.')
        scenario=scenario.sort_values('source_point_index')
        synthetic_score=score_stream(scenario,history,permitted,row.user_id)
        labelled=scenario.loc[scenario.anomaly_label.eq(1)]
        positions=dict(zip(labelled.source_point_index,np.arange(len(labelled))))
        counts=dict(zip(scenario.source_point_index,(scenario.anomaly_label.eq(1).cumsum()-scenario.anomaly_label.eq(1).astype(int)).astype(int)))
        selected=select_evaluation(synthetic_score,route_eval.loc[route_eval.source_trajectory_id.eq(row.source_trajectory_id)])
        selected['segment_position']=selected.source_point_index.map(positions)
        selected['segment_length']=len(labelled)
        selected['segment_relative_position']=selected.segment_position/max(1,len(labelled)-1)
        selected['progression_bucket']=np.where(selected.segment_relative_position.lt(1/3),'early',np.where(selected.segment_relative_position.lt(2/3),'middle','late'))
        selected['past_labelled_prefix_count']=selected.source_point_index.map(counts)
        selected['audit_is_segment_first']=selected.segment_position.eq(0)
        route_records.append(selected)
    replay_seconds=time.perf_counter()-build_started
    normal=pd.concat(normal_records,ignore_index=True);route=pd.concat(route_records,ignore_index=True)
    global_start=time.perf_counter();all_queries=pd.concat([normal[OBS],route[OBS]],ignore_index=True)
    global_scores=score_global(train,all_queries,allowed_train)
    normal=pd.concat([normal.reset_index(drop=True),global_scores.iloc[:len(normal)].reset_index(drop=True)],axis=1)
    route=pd.concat([route.reset_index(drop=True),global_scores.iloc[len(normal):].reset_index(drop=True)],axis=1)
    global_seconds=time.perf_counter()-global_start
    for frame in (normal,route):
        frame['global_personal_min_distance_m']=np.fmin(frame.global_train_causal_distance_m,frame.combined_personal_distance_m)
        frame['global_personal_min_context_points']=frame.global_train_causal_context_points+frame.combined_personal_context_points
        first_context=frame[['global_train_causal_first_context_timestamp','combined_personal_first_context_timestamp']].min(axis=1)
        last_context=frame[['global_train_causal_last_context_timestamp','combined_personal_last_context_timestamp']].max(axis=1)
        frame['global_personal_min_duration_sec']=(last_context-first_context).dt.total_seconds().fillna(0.)
        frame['global_personal_min_status']=np.where(frame.global_personal_min_context_points.gt(0),'available','no_context')
        for ref in REFERENCES[:-1]:
            valid=frame[ref+'_context_points'].gt(0)
            if frame.loc[valid,ref+'_distance_m'].isna().any() or frame.loc[valid,ref+'_nearest_timestamp'].ge(frame.loc[valid,'timestamp']).any():
                raise ValueError('Non-past or missing available context.')
    # Frozen static Stage 6.2 baseline is kept separate from causal deployable scores.
    old_dir=root/'outputs/metrics/stage6/route_reference_coverage'
    for frame,name in [(normal,'normal_reference_distances'),(route,'route_deviation_reference_distances')]:
        baseline=read_dataset_csv(old_dir/(name+'.csv'))
        merged=frame.merge(baseline[SCORE_KEY+[spatial.DISTANCE]],on=SCORE_KEY,how='left',validate='one_to_one',indicator=True)
        if merged['_merge'].ne('both').any(): raise ValueError('Missing frozen Stage 6.2 baseline.')
        frame['global_train_static_baseline_distance_m']=merged[spatial.DISTANCE].to_numpy()
    availability,coverage,per_user,support=summarize(normal,route)
    pairs=paired_audit(normal,route)
    # Stage 6.2 boundary flags are audit metadata; never reference/gate inputs.
    old_pairs=read_dataset_csv(old_dir/'source_matched_distance_delta.csv')
    flags=SCORE_KEY+['zero_displacement','feature_near_identical','actual_displacement_m']
    pairs=pairs.merge(old_pairs[flags],on=SCORE_KEY,how='left',validate='one_to_one',indicator=True)
    if pairs['_merge'].ne('both').any(): raise ValueError('Missing boundary audit pair.')
    pairs=pairs.drop(columns='_merge')
    for flag in ('zero_displacement','feature_near_identical'):
        pairs[flag]=pairs[flag].astype(str).str.lower().eq('true')
    boundary=pairs.loc[pairs.zero_displacement|pairs.feature_near_identical].copy()
    progression=progression_audit(pairs)
    first=pairs.loc[pairs.audit_is_segment_first].copy()
    paired_summary=pd.DataFrame([{'reference':ref,**spatial.distance_summary(pairs[ref+'_delta_m'].dropna())} for ref in REFERENCES])
    static_report=json.loads((old_dir/'route_reference_audit.json').read_text())
    static_metrics={'reference':'global_train_static_baseline','deployable_causal':False,
        **spatial.separation(normal.global_train_static_baseline_distance_m,route.global_train_static_baseline_distance_m)}
    for field in ('roc_auc','average_precision'):
        np.testing.assert_allclose(static_metrics[field],static_report['full_test_separation'][field],atol=1e-12)
    tables={'normal_context_distances':normal,'route_deviation_context_distances':route,
        'context_availability':availability,'coverage_by_reference':coverage,'per_user_context_metrics':per_user,
        'source_matched_context_delta':pairs,'anomaly_progression':progression,'support_analysis':support,
        'trajectory_history':pd.DataFrame(history_records),'label_boundary_context_audit':boundary,
        'segment_first_context_audit':first,'paired_context_delta_summary':paired_summary}
    route_audit.assert_protected(root,expected)
    metrics_dir.mkdir(parents=True);fig_dir.mkdir(parents=True)
    for name,frame in tables.items():frame.to_csv(metrics_dir/(name+'.csv'),index=False,na_rep='NA')
    figures=create_figures(normal,route,availability,coverage,per_user,support,progression,fig_dir)
    boundary_report={}
    for flag in ('zero_displacement','feature_near_identical'):
        b=pairs.loc[pairs[flag]]
        boundary_report[flag]={'count':len(b),'by_reference':{ref:spatial.distance_summary(b[ref+'_delta_m'].dropna()) for ref in REFERENCES}}
    report={'stage':'6.3','name':'Inference-observable Context Reference & Abstention Audit','dataset_id':dataset_id,
        'analysis_only':True,'counts':{'normal':len(normal),'route':len(route),'source_pairs':len(pairs),'test_trajectories':len(catalog)},
        'references':list(REFERENCES),'static_baseline':static_metrics,'reference_context_construction_seconds':replay_seconds,
        'causal_global_query_seconds':global_seconds,'execution_seconds':time.perf_counter()-started,
        'minimum_context_candidates':list(MIN_CONTEXT),'duration_candidates_sec':list(DURATION_SEC),'radii_m':list(RADII),
        'earth_radius_m':EARTH_RADIUS_M,'computation_block_size':BLOCK_SIZE,'boundary':boundary_report,
        'causal_reference_timestamp_policy':'All query reference points have timestamp < query timestamp; prefix also requires index < current index.',
        'scenario_policy':'Each route copy independently replays observed synthetic prefix; earlier logged original trajectories are unchanged in this fixture. No label filtering.',
        'history_quality_policy':'All finite observed coordinates in retained eligible trajectories; source/model quality flags do not oracle-filter replay memory.',
        'cross_split_leakage':0,'causality_violations':0,'label_based_reference_filtering':False,
        'source_matching_used_for_gate':False,'distance_cutoff_support':False,'test_parameter_tuning':False,
        'model_trained_or_inferred':False,'threshold_or_scaler_changed':False,'protected_file_count':sum(map(len,expected.values())),
        'source_sha256':single.sha256_file(Path(__file__)),'limitations':['Static global Stage 6.2 reuse is not historical-causal baseline.',
            'Previously logged original personal trajectories are assumed available; other past anomaly scenarios are not modeled.',
            'Short-term prefix nearest distance can track local step length rather than habitual route departure.',
            'Progression labels and paired originals are audit-only; buckets do not identify causal contamination effect.',
            'Personal history changes unseen-user information conditions from Train-only cold start; no retrospective oracle cleaning.']}
    route_audit.assert_protected(root,expected);single.save_json(metrics_dir/'context_reference_audit.json',report)
    verification={'passed':True,'protected_files_unchanged':True,'protected_file_count':report['protected_file_count'],
        'causality_violations':0,'source_sha256':report['source_sha256'],
        'output_checksums':{p.name:single.sha256_file(p) for p in metrics_dir.glob('*')},
        'figure_checksums':{p.name:single.sha256_file(p) for p in figures}}
    single.save_json(metrics_dir/'stage6_3_verification.json',verification)
    print(json.dumps({'counts':report['counts'],'execution_seconds':report['execution_seconds'],
        'reference_context_construction_seconds':replay_seconds,'causal_global_query_seconds':global_seconds},indent=2),flush=True)
    return {'report':report,'tables':tables,'metrics_dir':metrics_dir,'figure_dir':fig_dir,'verification':verification}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root',type=Path,default=single.ROOT)
    parser.add_argument('--protected-snapshot',type=Path)
    args=parser.parse_args();run_audit(args.data_dir,args.output_root,args.protected_snapshot)


if __name__=='__main__':main()
