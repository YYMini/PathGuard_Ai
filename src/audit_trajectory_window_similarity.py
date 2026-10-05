"""Stage 6.4: causal absolute-space window identifiability; no model fitting."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import audit_context_reference as context
from src import audit_route_reference_coverage as spatial
from src import audit_route_representation as previous
from src import evaluate_multi_seed_stability as multi
from src import train_multiuser_autoencoder as single
from src.load_multiuser_geolife import read_dataset_csv
from src.feature_engineering import EARTH_RADIUS_M
from src.prepare_multiuser_dataset import validate_saved_dataset

WINDOWS = (10, 25, 50)
GAPS = (0, 10, 25, 50)
REFERENCES = ('prior', 'prefix', 'combined')
STRIDE = 10
TREE_BLOCK = 128
KEY = spatial.KEY
SCORE_KEY = spatial.SCORE_KEY
SCORE = 'window_similarity_score_m'
ATOL, RTOL = 1e-5, 1e-8


def metric_coordinates(frame, anchor):
    """Common Train-fixed spherical ECEF -> ENU, in metres, without window centering.

    All three components are retained: a rigid transform of spherical ECEF.
    Euclidean distance is spherical chord distance, not surface arc distance.
    """
    radians = spatial.coordinates(frame)
    lat, lon = radians.T
    xyz = EARTH_RADIUS_M * np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    a, b = np.radians(np.asarray(anchor, dtype=float))
    origin = EARTH_RADIUS_M*np.array([np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)])
    rotation = np.array([[-np.sin(b), np.cos(b), 0], [-np.sin(a)*np.cos(b), -np.sin(a)*np.sin(b), np.cos(a)],
                         [np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)]])
    result = np.ascontiguousarray((xyz-origin) @ rotation.T)
    if not np.isfinite(result).all(): raise ValueError('Non-finite projected coordinates.')
    return result


@njit(cache=True, fastmath=False)
def _frechet(a, b, cutoff=np.inf):
    """Exact discrete Frechet DP; row lower bound permits safe abandonment."""
    n, m = len(a), len(b)
    old = np.full(m, np.inf)
    for i in range(n):
        row = np.empty(m)
        for j in range(m):
            squared = 0.
            for k in range(a.shape[1]):
                diff = a[i,k]-b[j,k]
                squared += diff*diff
            d = np.sqrt(squared)
            if i == 0 and j == 0: row[j] = d
            elif i == 0: row[j] = max(d, row[j-1])
            elif j == 0: row[j] = max(d, old[j])
            else: row[j] = max(d, min(old[j], row[j-1], old[j-1]))
        if np.min(row) > cutoff: return np.inf
        old = row
    return old[-1]


def frechet_distance(a, b):
    a, b = np.ascontiguousarray(a, dtype=float), np.ascontiguousarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        raise ValueError('Nonempty matching-dimensional windows required.')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Non-finite window.')
    return float(_frechet(a, b))


@njit(cache=True, fastmath=False)
def _search_windows(current, windows, indices, lower, seed, initial):
    best, winner, computed = initial, seed, 1
    for p in range(len(indices)):
        idx = indices[p]
        if idx == seed: continue
        if lower[p] > best+max(1e-7, abs(best)*1e-12): break
        value = _frechet(current, windows[idx], best)
        computed += 1
        if value < best or (value == best and idx < winner): best, winner = value, idx
    return best, winner, computed


@dataclass(frozen=True)
class Match:
    score: float = np.nan
    winner: int = -1
    aligned_mean: float = np.nan
    aligned_max: float = np.nan
    candidates: int = 0
    computed: int = 0


class WindowBank:
    """Exact minimum over a fixed stride grid, endpoint pruning without missed minima.

    The mutable index contains only the eligible prefix. A frozen tree plus a
    recent buffer avoids indexing future windows. Ties retain first bank index.
    """
    def __init__(self, streams, anchor, width, stride=STRIDE):
        windows, records = [], []
        for stream in streams:
            observed = context.observable(stream).reset_index(drop=True)
            context.validate_stream(observed)
            xyz = metric_coordinates(observed, anchor)
            for start in range(0, len(observed)-width+1, stride):
                end = start+width-1
                windows.append(xyz[start:end+1])
                records.append((str(observed.user_id.iloc[0]), str(observed.source_trajectory_id.iloc[0]),
                    int(observed.source_point_index.iloc[start]), int(observed.source_point_index.iloc[end]),
                    observed.timestamp.iloc[start].value, observed.timestamp.iloc[end].value))
        self.windows = np.ascontiguousarray(np.asarray(windows, dtype=float).reshape(-1, width, 3))
        self.meta = records
        self.endpoint = np.concatenate([self.windows[:, 0], self.windows[:, -1]], axis=1)
        self.end_indices = np.array([r[3] for r in records], dtype=np.int64)
        self.end_times = np.array([r[5] for r in records], dtype=np.int64)
        self.tree = None
        self.base_end = self.last_limit = 0

    def nearest(self, current, limit=None):
        limit = len(self.meta) if limit is None else int(limit)
        if limit < self.last_limit or limit < 0 or limit > len(self.meta): raise ValueError('Invalid backward candidate limit.')
        self.last_limit = limit
        if not limit: return Match()
        if limit-self.base_end >= TREE_BLOCK:
            self.tree = cKDTree(self.endpoint[:limit]); self.base_end = limit
        endpoint = np.r_[current[0], current[-1]]
        seed, seed_distance = -1, np.inf
        if self.tree is not None:
            seed_distance, seed = self.tree.query(endpoint, k=1)
            seed = int(seed)
        if limit > self.base_end:
            squared = np.sum((self.endpoint[self.base_end:limit]-endpoint)**2, axis=1)
            idx = int(np.argmin(squared))
            if squared[idx] <= seed_distance**2: seed = self.base_end+idx
        initial = float(_frechet(current, self.windows[seed]))
        # Frechet >= max(start distance, end distance), hence endpoint L2 <= sqrt(2)*Frechet.
        radius = np.sqrt(2)*initial+max(1e-7, abs(initial)*1e-12)
        candidates = list(self.tree.query_ball_point(endpoint, radius, eps=0)) if self.tree is not None else []
        if limit > self.base_end:
            squared = np.sum((self.endpoint[self.base_end:limit]-endpoint)**2, axis=1)
            candidates.extend((self.base_end+np.flatnonzero(squared <= radius**2)).tolist())
        candidates = np.asarray(candidates, dtype=np.int64)
        lower = np.maximum(np.linalg.norm(self.windows[candidates, 0]-current[0], axis=1),
                           np.linalg.norm(self.windows[candidates, -1]-current[-1], axis=1))
        order = np.lexsort((candidates, lower))
        best, winner, computed = _search_windows(current, self.windows, candidates[order], lower[order], seed, initial)
        aligned = np.linalg.norm(current-self.windows[winner], axis=1)
        return Match(float(best), int(winner), float(aligned.mean()), float(aligned.max()), limit, int(computed))


def completed_prior(originals, catalog, user, source):
    current = catalog.loc[catalog.source_trajectory_id.eq(source)]
    if len(current) != 1 or str(current.user_id.iloc[0]) != str(user): raise ValueError('Unknown current trajectory/user.')
    row = current.iloc[0]
    eligible = catalog.loc[catalog.user_id.eq(user) & catalog.end_timestamp.lt(row.start_timestamp)]
    if source in set(eligible.source_trajectory_id): raise ValueError('Self trajectory in prior history.')
    return [originals.loc[originals.source_trajectory_id.eq(s)] for s in eligible.source_trajectory_id]


def assert_reference(meta, current, width, gap, kind, current_trajectory_start):
    user, source, start, end, t0, t1 = meta
    if user != str(current.user_id.iloc[0]): raise ValueError('Cross-user reference.')
    if t1 >= current.timestamp.iloc[0].value: raise ValueError('Temporal leakage: reference not strictly before window.')
    if kind == 'prefix':
        if source != str(current.source_trajectory_id.iloc[0]): raise ValueError('Invalid prefix identity.')
        if end > int(current.source_point_index.iloc[0])-gap-1: raise ValueError('Window overlap/gap violation.')
        if set(range(start, end+1)).intersection(set(current.source_point_index)): raise ValueError('Shared source indices.')
    else:
        if source == str(current.source_trajectory_id.iloc[0]) or t1 >= pd.Timestamp(current_trajectory_start).value:
            raise ValueError('Self/future prior trajectory.')


def score_stream(stream, evaluation, prior_streams, anchor, width, gap, prior_bank=None, prior_matches=None):
    """No labels, source-pair values, distance features or audit flags enter matching."""
    observed = context.observable(stream).reset_index(drop=True); context.validate_stream(observed)
    if not observed.source_point_index.diff().dropna().eq(1).all(): raise ValueError('Contiguous point indices required for point-unit windows/gaps.')
    for history in prior_streams:
        logged = context.observable(history); context.validate_stream(logged)
        if (str(logged.user_id.iloc[0]) != str(observed.user_id.iloc[0]) or
            str(logged.source_trajectory_id.iloc[0]) == str(observed.source_trajectory_id.iloc[0]) or
            logged.timestamp.iloc[-1] >= observed.timestamp.iloc[0]):
            raise ValueError('Prior trajectory must be same-user, distinct and completed before current start.')
    positions = dict(zip(observed.source_point_index, range(len(observed))))
    ends = [positions[int(x)] for x in evaluation.source_point_index]
    if ends != sorted(ends): raise ValueError('Evaluation query order must be increasing.')
    xyz = metric_coordinates(observed, anchor)
    prefix_bank = WindowBank([observed], anchor, width)
    prior_bank = prior_bank or WindowBank(prior_streams, anchor, width)
    if prior_matches is None: prior_matches = {}
    output = []
    for end in ends:
        start = end-width+1
        common = {'source_point_index': int(observed.source_point_index.iloc[end]), 'window_size':width, 'gap':gap,
            'current_window_start_index': int(observed.source_point_index.iloc[start]) if start >= 0 else -1,
            'current_window_end_index':int(observed.source_point_index.iloc[end]),
            'current_window_start_timestamp': observed.timestamp.iloc[start] if start >= 0 else pd.NaT,
            'current_window_end_timestamp':observed.timestamp.iloc[end]}
        prior, prefix = Match(), Match()
        if start >= 0:
            current = observed.iloc[start:end+1]
            coordinates = xyz[start:end+1]
            if end not in prior_matches: prior_matches[end] = prior_bank.nearest(coordinates)
            prior = prior_matches[end]
            bound = int(current.source_point_index.iloc[0])-gap-1
            limit = int(np.searchsorted(prefix_bank.end_indices, bound, side='right'))
            time_limit = int(np.searchsorted(prefix_bank.end_times, current.timestamp.iloc[0].value, side='left'))
            prefix = prefix_bank.nearest(coordinates, min(limit, time_limit))
        chosen = prior if prior.winner >= 0 and (prefix.winner < 0 or prior.score <= prefix.score) else prefix
        for kind, match in [('prior',prior), ('prefix',prefix), ('combined',chosen)]:
            selected_kind = ('prior' if chosen is prior else 'prefix') if kind == 'combined' else kind
            bank = prior_bank if selected_kind == 'prior' else prefix_bank
            count = prior.candidates+prefix.candidates if kind == 'combined' else match.candidates
            status = 'available' if match.winner >= 0 else ('insufficient_current_window' if start < 0 else
                ('no_history' if kind == 'prior' and not prior_streams else 'insufficient_history'))
            record = {**common,'reference_type':kind, SCORE:match.score,'aligned_mean_score_m':match.aligned_mean,
                'aligned_max_score_m':match.aligned_max,'available_reference_count':count,'status':status,
                'selected_reference_type':selected_kind if match.winner >= 0 else '', 'reference_user_id':'',
                'reference_trajectory_id':'','reference_start_index':-1,'reference_end_index':-1,
                'reference_start_timestamp':pd.NaT,'reference_end_timestamp':pd.NaT,'overlap_count':0,
                'frechet_computed_count':(prior.computed+prefix.computed if kind == 'combined' else match.computed)}
            if match.winner >= 0:
                meta = bank.meta[match.winner]
                assert_reference(meta, current, width, gap, selected_kind, observed.timestamp.iloc[0])
                record.update(dict(zip(['reference_user_id','reference_trajectory_id','reference_start_index',
                    'reference_end_index','reference_start_timestamp','reference_end_timestamp'],
                    (*meta[:4],pd.Timestamp(meta[4]),pd.Timestamp(meta[5])))))
            output.append(record)
    return pd.DataFrame(output), prior_matches


def correlation(scores, baseline):
    x, y = np.asarray(scores,dtype=float), np.asarray(baseline,dtype=float)
    valid = np.isfinite(x) & np.isfinite(y); x, y = x[valid], y[valid]
    result = {'count':len(x),'pearson':None,'spearman':None,'equal_fraction':None,'na_reason':''}
    if not len(x): result['na_reason']='no_paired_finite_rows'; return result
    result['equal_fraction']=float(np.isclose(x,y,atol=ATOL,rtol=RTOL).mean())
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0: result['na_reason']='constant_or_insufficient_rows'; return result
    result['pearson']=float(np.corrcoef(x,y)[0,1])
    result['spearman']=float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    return result


def availability(frame):
    valid=frame.status.eq('available')
    if not np.isfinite(frame.loc[valid,SCORE]).all() or frame.loc[~valid,SCORE].notna().any():
        raise ValueError('Availability/finite-score mismatch.')
    return {'total_count':len(frame),'evaluable_count':int(valid.sum()),'unsupported_count':int((~valid).sum()),
        'availability_pct':float(valid.mean()*100) if len(frame) else None,
        **{reason+'_count':int(frame.status.eq(reason).sum()) for reason in
            ('no_history','insufficient_history','insufficient_current_window')}}


def paired_audit(normal, route):
    keep=KEY+[SCORE,'status']
    original=normal[keep].rename(columns={SCORE:'original_window_score_m','status':'original_status'})
    result=route.merge(original,on=KEY,how='left',validate='one_to_one',indicator=True)
    if result['_merge'].ne('both').any(): raise ValueError('Missing source-matched normal evaluation endpoint.')
    result=result.drop(columns='_merge')
    result['synthetic_window_score_m']=result[SCORE]
    result['window_delta_m']=result[SCORE]-result.original_window_score_m
    return result


def delta_summary(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    return {**spatial.distance_summary(values), 'positive_pct':float((values>0).mean()*100) if len(values) else None,
        **{f'ge_{threshold}m_pct':float((values>=threshold).mean()*100) if len(values) else None for threshold in (10,25,50,100)}}


def hard_subset(frame):
    if 'all_8_inside_train_p01_p99' not in frame: raise ValueError('Frozen hard-subset flag required.')
    return frame.loc[frame.all_8_inside_train_p01_p99].copy()


def compare_config(normal, route, config, subset='all', cut=None):
    n=normal.loc[normal.status.eq('available')];a=route.loc[route.status.eq('available')]
    result={**config,'subset':subset,'train_distance_cutoff_m':cut,
        **{'normal_'+k:v for k,v in availability(normal).items()},
        **{'route_'+k:v for k,v in availability(route).items()},
        **{'normal_score_'+k:v for k,v in spatial.distance_summary(n[SCORE]).items()},
        **{'route_score_'+k:v for k,v in spatial.distance_summary(a[SCORE]).items()},
        **spatial.separation(n[SCORE],a[SCORE])}
    for scope,frame in [('normal',n),('route',a),('all',pd.concat([n,a],ignore_index=True))]:
        for baseline in ('distance_m','causal_prefix_distance_m'):
            result.update({scope+'_'+baseline+'_'+k:v for k,v in correlation(frame[SCORE],frame[baseline]).items()})
    # Paired-support baseline comparisons: unavailable endpoints excluded from all methods equally.
    for baseline in ('distance_m','causal_prefix_distance_m'):
        result.update({baseline+'_'+k:v for k,v in spatial.separation(n[baseline],a[baseline]).items()})
    return result


def progression_audit(pairs, config):
    return pd.DataFrame([{**config,'progression_bucket':bucket,
        **{'score_'+k:v for k,v in spatial.distance_summary(g[SCORE].dropna()).items()},
        **{'delta_'+k:v for k,v in delta_summary(g.window_delta_m).items()},
        **availability(g)} for bucket,g in pairs.groupby('progression_bucket',sort=True)])


def user_metrics(normal,route,config):
    records=[]
    for user in sorted(set(normal.user_id)|set(route.user_id)):
        n=normal.loc[normal.user_id.eq(user)];a=route.loc[route.user_id.eq(user)]
        record={**compare_config(n,a,config),'user_id':user}
        hard=compare_config(n,hard_subset(a),config,'hard')
        record.update({'hard_'+k:hard[k] for k in ('roc_auc','average_precision','route_evaluable_count','route_total_count','route_availability_pct','route_score_median')})
        records.append(record)
    return pd.DataFrame(records)


def peak_memory_mb():
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        value=Counters();value.cb=ctypes.sizeof(value)
        kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=wintypes.HANDLE
        api=ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];api.restype=wintypes.BOOL
        if not api(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb): return None
        return float(value.PeakWorkingSetSize/(1024**2))
    except (AttributeError,OSError): return None


def load_frozen_metadata(root,data_dir,normal,route):
    old=root/'outputs/metrics/stage6/context_reference'
    cols=SCORE_KEY+['causal_prefix_distance_m']
    for frame,name in [(normal,'normal_context_distances'),(route,'route_deviation_context_distances')]:
        baseline=read_dataset_csv(old/(name+'.csv'))
        linked=frame.merge(baseline[cols],on=SCORE_KEY,how='left',validate='one_to_one',indicator=True)
        if linked['_merge'].ne('both').any(): raise ValueError('Missing Stage 6.3 endpoint baseline.')
        frame['causal_prefix_distance_m']=linked.causal_prefix_distance_m.to_numpy()
    flags=read_dataset_csv(root/'outputs/metrics/stage6'/data_dir.name/'route_audit/route_deviation_samples.csv')
    flags=flags.loc[flags.dataset_split.eq('test') & flags.evaluation_included]
    select=SCORE_KEY+['all_8_inside_train_p01_p99','zero_displacement','feature_near_identical']
    route=route.merge(flags[select],on=SCORE_KEY,how='left',validate='one_to_one',indicator=True)
    if route['_merge'].ne('both').any(): raise ValueError('Missing Stage 6.1 audit row.')
    route=route.drop(columns='_merge')
    for flag in select[3:]:
        values=route[flag].astype(str).str.lower()
        if not values.isin(['true','false']).all(): raise ValueError('Invalid frozen boolean audit flag.')
        route[flag]=values.eq('true')
    progression=read_dataset_csv(old/'route_deviation_context_distances.csv')
    route=route.merge(progression[SCORE_KEY+['progression_bucket','segment_position','segment_length','past_labelled_prefix_count']],
        on=SCORE_KEY,how='left',validate='one_to_one',indicator=True)
    if route['_merge'].ne('both').any(): raise ValueError('Missing Stage 6.3 progression metadata.')
    return normal,route.drop(columns='_merge')


def attach_evaluation(scores, evaluation):
    keep=SCORE_KEY+['user_id','trajectory_id','distance_m','causal_prefix_distance_m']
    keep += [x for x in ('all_8_inside_train_p01_p99','zero_displacement','feature_near_identical',
                         'progression_bucket','segment_position','segment_length','past_labelled_prefix_count') if x in evaluation]
    result=scores.merge(evaluation[keep],on=['source_point_index'],how='left',validate='many_to_one',indicator=True)
    if result['_merge'].ne('both').any(): raise ValueError('Missing evaluation identity.')
    return result.drop(columns='_merge')


def create_figures(configs,hard,users,progress,sample,fig_dir):
    """All configurations remain visible; scatter rendering alone is thinned deterministically."""
    def save(name):
        plt.tight_layout();p=fig_dir/(name+'.png');plt.savefig(p,dpi=160);plt.close();return p
    figures=[];table=pd.DataFrame(configs);label=table.apply(lambda r:f"W{r.window_size}/g{r.gap}/{r.reference_type}",axis=1)
    x=np.arange(len(table))
    plt.figure(figsize=(15,5));plt.plot(x,table.normal_score_median,'o-',label='normal');plt.plot(x,table.route_score_median,'o-',label='route')
    plt.xticks(x,label,rotation=90);plt.ylabel('Median Frechet score (m)');plt.legend();figures.append(save('normal_vs_route_window_score'))
    for metric,name in [('roc_auc','auc_by_window_size'),('average_precision','ap_by_window_size')]:
        plt.figure(figsize=(9,5))
        for (ref,gap),g in table.groupby(['reference_type','gap']):plt.plot(g.window_size,g[metric],'o-',label=f'{ref}, gap {gap}')
        plt.xticks(WINDOWS);plt.ylabel(metric);plt.xlabel('Window size (points)');plt.legend(fontsize=7,ncol=3);figures.append(save(name))
    plt.figure(figsize=(9,5))
    for group,g in sample.groupby('endpoint_class'):plt.scatter(g.distance_m,g[SCORE],s=4,alpha=.3,label=group)
    plt.xscale('symlog',linthresh=1);plt.yscale('symlog',linthresh=1);plt.xlabel('distance_m');plt.ylabel('W10 gap0 combined Frechet (m), display sample');plt.legend();figures.append(save('window_score_vs_distance_m'))
    h=pd.DataFrame(hard);plt.figure(figsize=(15,5));plt.plot(x,h.normal_score_median,'o-',label='all normal');plt.plot(x,h.route_score_median,'o-',label='hard route')
    plt.xticks(x,label,rotation=90);plt.ylabel('Median Frechet score (m)');plt.legend();figures.append(save('hard_subset_window_score'))
    u=pd.concat(users,ignore_index=True);plt.figure(figsize=(13,5))
    for user,g in u.groupby('user_id'):plt.plot(x,g.roc_auc.to_numpy(),'o-',label=user)
    plt.xticks(x,label,rotation=90);plt.ylabel('Per-user ROC-AUC');plt.legend();figures.append(save('per_user_window_auc'))
    p=pd.concat(progress,ignore_index=True);plt.figure(figsize=(13,6))
    for (width,gap,ref),g in p.groupby(['window_size','gap','reference_type']):
        g=g.set_index('progression_bucket').reindex(['early','middle','late']);plt.plot(range(3),g.score_median,alpha=.7,label=f'{width}/{gap}/{ref}')
    plt.xticks(range(3),['early','middle','late']);plt.yscale('symlog',linthresh=1);plt.ylabel('Median Frechet score (m)');plt.legend(fontsize=6,ncol=6);figures.append(save('anomaly_progression_window_score'))
    return figures


def run_audit(data_dir=single.DEFAULT_DATA_DIR,output_root=single.ROOT,protected_snapshot=None):
    started=time.perf_counter();root=Path(output_root).resolve();data_dir=Path(data_dir).resolve()
    metrics=root/'outputs/metrics/stage6/window_similarity';figures=root/'outputs/figures/stage6/window_similarity'
    if metrics.exists() or figures.exists(): raise FileExistsError('Window audit overwrite refused.')
    expected=json.loads(Path(protected_snapshot or root/'outputs/metrics/stage64_protected_snapshot.json').read_text())
    required=[data_dir.relative_to(root),Path('models/stage5')/data_dir.name,Path('outputs/metrics/stage5')/data_dir.name,
        Path('outputs/figures/stage5')/data_dir.name,Path('outputs/metrics/stage6')/data_dir.name/'route_audit',
        Path('outputs/figures/stage6')/data_dir.name/'route_audit',Path('outputs/metrics/stage6/route_reference_coverage'),
        Path('outputs/figures/stage6/route_reference_coverage'),Path('outputs/metrics/stage6/context_reference'),Path('outputs/figures/stage6/context_reference')]
    if {k.replace('\\','/') for k in expected}!={p.as_posix() for p in required}: raise ValueError('Incomplete Stage 5 through 6.3 protected snapshot.')
    previous.assert_protected(root,expected);validate_saved_dataset(data_dir)
    summary,users=single.read_stage5_metadata(data_dir)
    train=single.load_stage5_split(data_dir,'train',summary,users)
    test=single.load_stage5_split(data_dir,'test',summary,users)
    spatial.validate_reference(train,users,read_dataset_csv(data_dir/'trajectory_quality_summary.csv'))
    anchor=(float(train.latitude.median()),float(train.longitude.median()))
    cutoffs={'train_median':float(train.distance_m.median()),'train_p75':float(train.distance_m.quantile(.75))}
    normal=test.loc[~test.is_synthetic].copy();route=test.loc[test.is_synthetic&test.anomaly_type.eq('route_deviation')].copy()
    normal,route=load_frozen_metadata(root,data_dir,normal,route)
    originals=read_dataset_csv(data_dir/'source_normal.csv');originals=context.observable(originals.loc[originals.dataset_split.eq('test')])
    catalog=context.trajectory_catalog(originals,set(users.loc[users.dataset_split.eq('test'),'user_id']))
    full=read_dataset_csv(data_dir/'test.csv');copies=full.loc[full.is_synthetic & full.anomaly_type.eq('route_deviation')]
    observed_original={s:g.reset_index(drop=True) for s,g in originals.groupby('source_trajectory_id',sort=True)}
    observed_route={s:context.observable(g.sort_values('source_point_index')).reset_index(drop=True) for s,g in copies.groupby('source_trajectory_id',sort=True)}
    n_eval={s:g.sort_values('source_point_index').reset_index(drop=True) for s,g in normal.groupby('source_trajectory_id')}
    a_eval={s:g.sort_values('source_point_index').reset_index(drop=True) for s,g in route.groupby('source_trajectory_id')}
    if set(observed_route)!=set(observed_original) or any(g.sample_id.nunique()!=1 for _,g in copies.groupby('source_trajectory_id')):
        raise ValueError('Independent route scenario per eligible Test trajectory required.')
    if not set(route.user_id).issubset(set(users.loc[users.dataset_split.eq('test'),'user_id'])): raise ValueError('Unauthorized Test scenario.')
    metrics.mkdir(parents=True);figures.mkdir(parents=True)
    configurations=[];hard_records=[];low_records=[];statistics=[];user_records=[];paired_records=[];progression=[];boundary=[]
    score_path=metrics/'window_similarity_scores.csv';candidate_counts=[];display=pd.DataFrame();elapsed=[]
    for width in WINDOWS:
        caches={}
        for row in catalog.itertuples(index=False):
            prior=completed_prior(originals,catalog,row.user_id,row.source_trajectory_id)
            caches[row.source_trajectory_id]=(prior,WindowBank(prior,anchor,width),{}, {})
        for gap in GAPS:
            block_start=time.perf_counter();n_parts=[];a_parts=[];last_notice=block_start
            for row in catalog.itertuples(index=False):
                source=row.source_trajectory_id;prior,bank,n_cache,a_cache=caches[source]
                for observed,evaluated,cache,parts,kind in [(observed_original[source],n_eval[source],n_cache,n_parts,'normal'),
                    (observed_route[source],a_eval[source],a_cache,a_parts,'route')]:
                    scores,_=score_stream(observed,evaluated,prior,anchor,width,gap,bank,cache)
                    result=attach_evaluation(scores,evaluated);result['endpoint_class']=kind
                    result['prior_trajectory_count']=len(prior)
                    parts.append(result)
                if time.perf_counter()-last_notice>25:
                    print(f'W={width} gap={gap}: {source}, elapsed {time.perf_counter()-started:.1f}s',flush=True);last_notice=time.perf_counter()
            n=pd.concat(n_parts,ignore_index=True);a=pd.concat(a_parts,ignore_index=True)
            # A single header/schema for streaming all 1.68M rows. All route-specific flags are audit-only.
            extra=['all_8_inside_train_p01_p99','zero_displacement','feature_near_identical','progression_bucket','segment_position','segment_length','past_labelled_prefix_count']
            for column in extra:n[column]=pd.NA
            ordered=list(n.columns);a=a.reindex(columns=ordered)
            for ref in REFERENCES:
                config={'window_size':width,'gap':gap,'reference_type':ref,'reference_stride':STRIDE}
                nr=n.loc[n.reference_type.eq(ref)].copy();ar=a.loc[a.reference_type.eq(ref)].copy()
                if len(nr)!=len(normal) or len(ar)!=len(route): raise ValueError('Evaluation endpoints were dropped.')
                configurations.append(compare_config(nr,ar,config))
                hard_records.append(compare_config(nr,hard_subset(ar),config,'hard_8_marginal_train_p01_p99'))
                for name,cut in cutoffs.items():
                    low_records.append(compare_config(nr.loc[nr.distance_m.le(cut)],ar.loc[ar.distance_m.le(cut)],config,name,cut))
                user_records.append(user_metrics(nr,ar,config))
                pairs=paired_audit(nr,ar);paired_records.append(pairs)
                progression.append(progression_audit(pairs,config))
                boundary.append(pairs.loc[pairs.zero_displacement|pairs.feature_near_identical].copy())
                for kind,frame in [('normal',nr),('route',ar)]:
                    statistics.append({**config,'endpoint_class':kind,**availability(frame),**spatial.distance_summary(frame[SCORE].dropna())})
                candidate_counts.append({**config,'normal_candidate_comparisons':int(nr.available_reference_count.sum()),
                    'route_candidate_comparisons':int(ar.available_reference_count.sum()),
                    'normal_frechet_computed':int(nr.frechet_computed_count.sum()),'route_frechet_computed':int(ar.frechet_computed_count.sum()),
                    'prior_bank_window_entries':sum(len(b.meta) for _,b,_,_ in caches.values())})
                if width==10 and gap==0 and ref=='combined':
                    display=pd.concat([nr.iloc[::max(1,len(nr)//5000)],ar],ignore_index=True)
            pd.concat([n,a],ignore_index=True).to_csv(score_path,index=False,mode='a' if score_path.exists() else 'w',header=not score_path.exists(),na_rep='NA')
            elapsed.append({'window_size':width,'gap':gap,'seconds':time.perf_counter()-block_start})
            print(f'W={width} gap={gap} complete: {len(n)//3} normal / {len(a)//3} route endpoints; {elapsed[-1]["seconds"]:.1f}s',flush=True)
        del caches
    pairs=pd.concat(paired_records,ignore_index=True);b=pd.concat(boundary,ignore_index=True)
    pair_summary=[];boundary_summary=[]
    for keys,g in pairs.groupby(['window_size','gap','reference_type'],sort=False):
        config=dict(zip(['window_size','gap','reference_type'],keys))
        pair_summary.append({**config,'total_pair_count':len(g),**delta_summary(g.window_delta_m)})
        for flag in ('zero_displacement','feature_near_identical'):
            subset=g.loc[g[flag]]
            boundary_summary.append({**config,'boundary_type':flag,'total_pair_count':len(subset),
                **delta_summary(subset.window_delta_m), 'zero_delta_count':int(np.isclose(subset.window_delta_m,0,atol=ATOL,rtol=RTOL).sum())})
    tables={'window_similarity_summary':pd.DataFrame(statistics),'window_config_comparison':pd.DataFrame(configurations),
        'per_user_window_metrics':pd.concat(user_records,ignore_index=True),'hard_subset_metrics':pd.DataFrame(hard_records),
        'low_distance_challenge_metrics':pd.DataFrame(low_records),'source_matched_window_delta':pairs,
        'source_matched_window_delta_summary':pd.DataFrame(pair_summary),'window_progression':pd.concat(progression,ignore_index=True),
        'boundary_window_audit':b,'boundary_window_summary':pd.DataFrame(boundary_summary),'candidate_window_counts':pd.DataFrame(candidate_counts)}
    for name,frame in tables.items():frame.to_csv(metrics/(name+'.csv'),index=False,na_rep='NA')
    saved_figures=create_figures(configurations,hard_records,user_records,progression,display,figures)
    previous.assert_protected(root,expected)
    report={'stage':'6.4','name':'Causal Trajectory / Window Similarity Identifiability Audit','dataset_id':data_dir.name,
        'analysis_only':True,'counts':{'normal_endpoints':len(normal),'route_endpoints':len(route),'test_trajectories':len(catalog),
            'hard_route_endpoints':len(hard_subset(route)),'zero_displacement_endpoints':int(route.zero_displacement.sum()),
            'identical_feature_endpoints':int(route.feature_near_identical.sum()),'configurations':len(configurations),
            'score_rows':(len(normal)+len(route))*len(configurations),'normal_full_observed_rows':len(originals)},
        'window_sizes':list(WINDOWS),'gaps':list(GAPS),'reference_types':list(REFERENCES),'reference_stride':STRIDE,
        'primary_metric':'Minimum discrete Frechet distance over all eligible stride-10 reference windows; high = anomaly.',
        'auxiliary_metric':'Mean/max aligned point distances for the selected Frechet-minimizing window.',
        'projection':{'name':'Spherical ECEF rigidly transformed to common Train-fixed 3D ENU','units':'metres',
            'anchor_lat_lon':list(anchor),'earth_radius_m':EARTH_RADIUS_M,'per_window_translation_normalization':False,
            'distance':'spherical chord, not surface arc','chord_arc_ratio_at_100km':float(2*EARTH_RADIUS_M*np.sin(100000/(2*EARTH_RADIUS_M))/100000)},
        'candidate_search':'Exact over stride grid: endpoint lower bounds and exact cKDTree radius, no approximate nearest search.',
        'grid_limitation':'Stride 10 restricts reference starts to 0,10,...; may miss minima over stride-1 windows. Query endpoints are not downsampled.',
        'prior_policy':'Same Test user, complete original logged trajectory end < current trajectory start; no current trajectory.',
        'prefix_policy':'Reference end index <= current window start - gap - 1 AND reference max timestamp < current window min timestamp.',
        'gap_scope':'Point gap applies to same-trajectory prefix only; prior results invariant across gap.',
        'history_policy':'All finite observed GPS in retained eligible trajectories; no quality/label oracle filtering. Synthetic current history replayed as observed.',
        'train_distance_cutoffs_m':cutoffs,'equality_tolerance':{'atol':ATOL,'rtol':RTOL},
        'hard_subset_policy':'Frozen Stage 6.1 all-eight marginal Train p01-p99 flag; all evaluable Test normals are the comparison class.',
        'causality_violations':0,'overlap_violations':0,'cross_split_leakage':0,'protected_files_unchanged':True,
        'protected_file_count':sum(map(len,expected.values())),'model_trained_or_inferred':False,'threshold_or_scaler_changed':False,
        'test_parameter_tuning':False,'labels_or_pairs_used_for_scoring':False,'synthetic_generated':False,
        'candidate_counts':candidate_counts,'block_times':elapsed,'execution_seconds':time.perf_counter()-started,
        'peak_process_working_set_mb':peak_memory_mb(),'source_sha256':single.sha256_file(Path(__file__)),
        'limitations':['Correlation and marginal-hard separation do not prove conditional information or detector improvement.',
            'Per-point window metrics have repeated dependent trajectories/segments; no independence or confidence claims.',
            'Previously logged original routes are assumed observed unchanged; unlogged history and prior anomaly scenarios not simulated.',
            'Prefix history is short-term context, not proof of learned habitual route memory.',
            'Progression mixes changing displacement/window composition; cannot identify causal memory contamination from bucket medians.']}
    single.save_json(metrics/'window_similarity_audit.json',report)
    verification={'passed':True,'protected_file_count':report['protected_file_count'],'protected_files_unchanged':True,
        'source_sha256':report['source_sha256'],'output_checksums':{p.name:single.sha256_file(p) for p in metrics.glob('*')},
        'figure_checksums':{p.name:single.sha256_file(p) for p in saved_figures}}
    single.save_json(metrics/'stage6_4_verification.json',verification)
    print(json.dumps({'counts':report['counts'],'execution_seconds':report['execution_seconds'],'peak_memory_mb':report['peak_process_working_set_mb']},indent=2),flush=True)
    return {'report':report,'tables':tables,'metrics_dir':metrics,'figure_dir':figures,'verification':verification}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root',type=Path,default=single.ROOT)
    parser.add_argument('--protected-snapshot',type=Path)
    args=parser.parse_args();run_audit(args.data_dir,args.output_root,args.protected_snapshot)


if __name__=='__main__':main()
