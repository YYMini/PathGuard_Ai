"""Supplementary Stage 6.6 audit using frozen Stage 5 decisions only.

Event geometry comes from full labelled copies. Unknown/excluded decisions are
never imputed; runs break at missing source indices and existing long gaps.
"""
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

from src import audit_synthetic_route_labels as labels
from src import compare_stage5_baselines as baseline
from src import train_multiuser_autoencoder as single
from src.data_quality import LONG_GAP_THRESHOLD_SEC
from src.load_multiuser_geolife import read_dataset_csv

KEY = labels.KEY
SCORE_KEY = labels.SCORE_KEY
EARLY_CUTOFFS = (0.10, 0.25, 0.50)
MODEL_NAMES = ('statistical_rule', 'isolation_forest', 'autoencoder')
RUN_COLUMNS = ['start_index', 'end_index', 'start_timestamp', 'end_timestamp',
               'length_points', 'duration_seconds']


def validate_stream(frame):
    """One source stream, unique integral indices, strictly increasing time."""
    g = frame.sort_values('source_point_index', kind='stable').copy()
    if g.empty:
        raise ValueError('Empty stream.')
    if g.source_trajectory_id.nunique() != 1 or g.user_id.nunique() != 1:
        raise ValueError('Ambiguous trajectory/user stream.')
    if 'sample_id' in g and g.sample_id.nunique() != 1:
        raise ValueError('Ambiguous sample stream.')
    indices = pd.to_numeric(g.source_point_index, errors='raise').to_numpy(float)
    if not np.isfinite(indices).all() or (indices < 0).any() or (indices != np.floor(indices)).any():
        raise ValueError('Invalid source index.')
    if g.source_point_index.duplicated().any():
        raise ValueError('Duplicate source index.')
    g['timestamp'] = pd.to_datetime(g.timestamp, errors='raise')
    if g.timestamp.isna().any() or not g.timestamp.diff().dropna().gt(pd.Timedelta(0)).all():
        raise ValueError('Missing/non-increasing timestamp.')
    return g.reset_index(drop=True)


def observation_links(frame):
    """Both observed endpoints, adjacent raw indices, positive interval <= 300 s."""
    g = validate_stream(frame)
    seconds = g.timestamp.diff().dt.total_seconds()
    links = g.source_point_index.diff().eq(1) & seconds.gt(0) & seconds.le(LONG_GAP_THRESHOLD_SEC)
    return g, links, seconds


def positive_runs(frame):
    g, links, seconds = observation_links(frame)
    if not g.predicted_anomaly.isin([0, 1]).all():
        raise ValueError('Nonbinary/unknown stored prediction.')
    positive = g.predicted_anomaly.eq(1).to_numpy()
    starts = positive & ~(np.r_[False, positive[:-1]] & links.to_numpy())
    groups = np.cumsum(starts)
    records = []
    for number in np.unique(groups[positive]):
        selected = g.loc[positive & (groups == number)]
        records.append({'start_index': int(selected.source_point_index.iloc[0]),
                        'end_index': int(selected.source_point_index.iloc[-1]),
                        'start_timestamp': selected.timestamp.iloc[0],
                        'end_timestamp': selected.timestamp.iloc[-1],
                        'length_points': len(selected),
                        'duration_seconds': (selected.timestamp.iloc[-1] - selected.timestamp.iloc[0]).total_seconds()})
    return pd.DataFrame(records, columns=RUN_COLUMNS)


def normal_exposure(frame):
    g, links, seconds = observation_links(frame)
    # Existing feature time_diff_sec must agree on intervals we count. Never bridge holes.
    if 'time_diff_sec' in g and not np.allclose(g.loc[links, 'time_diff_sec'], seconds.loc[links], rtol=0, atol=1e-9):
        raise ValueError('Stored interval differs from observed timestamp difference.')
    exposure = float(seconds.loc[links].sum())
    return {'observation_exposure_sec': exposure,
            'valid_observation_intervals': int(links.sum()),
            'excluded_observation_links': max(0, len(g) - 1) - int(links.sum()),
            'time_exposure_na_reason': '' if exposure > 0 else 'no_valid_adjacent_observation_intervals'}


def build_events(full, manifest, tiers):
    """Keep every labelled run, including Tier 4; validate against frozen manifest."""
    if manifest.sample_id.duplicated().any() or set(full.sample_id) != set(manifest.sample_id):
        raise ValueError('Sample catalog mismatch.')
    if tiers.duplicated(SCORE_KEY).any():
        raise ValueError('Duplicate tier lineage.')
    records = []
    expected_tier_keys = set()
    for sample, group in full.groupby('sample_id', sort=True):
        g = validate_stream(group)
        if not g.source_point_index.diff().dropna().eq(1).all():
            raise ValueError('Full sample has source holes.')
        if not g.anomaly_label.isin([0, 1]).all():
            raise ValueError('Nonbinary full labels.')
        positive = g.anomaly_label.eq(1).to_numpy()
        starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
        ends = np.flatnonzero(positive & ~np.r_[positive[1:], False])
        metadata = manifest.loc[manifest.sample_id.eq(sample)].iloc[0]
        labelled = g.loc[positive]
        if len(labelled) == 0 or len(labelled) != int(metadata.anomaly_point_count):
            raise ValueError('Manifest label count mismatch.')
        if (labelled.timestamp.iloc[0] != pd.Timestamp(metadata.anomaly_start_time)
                or labelled.timestamp.iloc[-1] != pd.Timestamp(metadata.anomaly_end_time)
                or str(metadata.user_id) != str(g.user_id.iloc[0])
                or str(metadata.source_trajectory_id) != str(g.source_trajectory_id.iloc[0])):
            raise ValueError('Manifest boundary/identity mismatch.')
        if 'generated_point_count' in manifest and len(g) != int(metadata.generated_point_count):
            raise ValueError('Full copy length mismatch.')
        for number, (start, end) in enumerate(zip(starts, ends), 1):
            segment = g.iloc[start:end + 1]
            a, b = int(segment.source_point_index.iloc[0]), int(segment.source_point_index.iloc[-1])
            t = tiers.loc[tiers.sample_id.eq(sample) & tiers.source_point_index.between(a, b)]
            if not t.source_trajectory_id.eq(g.source_trajectory_id.iloc[0]).all() or not t.user_id.eq(g.user_id.iloc[0]).all():
                raise ValueError('Tier/event identity mismatch.')
            if not t.identifiability_tier.isin(labels.TIERS).all():
                raise ValueError('Invalid tier.')
            expected_tier_keys.update(t[SCORE_KEY].itertuples(index=False, name=None))
            seconds = segment.timestamp.diff().dt.total_seconds().dropna()
            valid_duration = seconds.gt(0).all() and seconds.le(LONG_GAP_THRESHOLD_SEC).all()
            raw_duration = (segment.timestamp.iloc[-1] - segment.timestamp.iloc[0]).total_seconds()
            row = {'event_id': f'{sample}__run_{number:02d}', 'sample_id': sample,
                   'synthetic_sample_id': sample, 'user_id': str(g.user_id.iloc[0]),
                   'source_trajectory_id': str(g.source_trajectory_id.iloc[0]),
                   'anomaly_segment_id': str(segment.anomaly_segment_id.iloc[0]),
                   'event_start_index': a, 'event_end_index': b,
                   'event_start_timestamp': segment.timestamp.iloc[0],
                   'event_end_timestamp': segment.timestamp.iloc[-1], 'event_point_count': len(segment),
                   'evaluated_point_count': len(t), 'excluded_quality_point_count': len(segment) - len(t),
                   'event_duration_sec': raw_duration if valid_duration else np.nan,
                   'raw_event_span_sec': raw_duration,
                   'duration_na_reason': '' if valid_duration else 'long_gap_inside_labelled_event'}
            for tier in labels.TIERS:
                row[tier + '_count'] = int(t.identifiability_tier.eq(tier).sum())
            row['context_not_audited_count'] = len(segment) - len(t)
            row['observable_ratio'] = (len(t) - row['tier4_count']) / len(t) if len(t) else np.nan
            row['contains_tier4'] = row['tier4_count'] > 0
            records.append(row)
    if expected_tier_keys != set(tiers[SCORE_KEY].itertuples(index=False, name=None)):
        raise ValueError('Tier points outside event catalog.')
    events = pd.DataFrame(records)
    for column, name in [('event_point_count', 'length_group'), ('event_duration_sec', 'duration_group')]:
        lower, upper = events[column].quantile([.33, .66])
        events[name] = np.select([events[column].isna(), events[column].le(lower), events[column].le(upper)],
                                 ['not_available', 'short', 'medium'], default='long')
        events[name + '_p33'] = lower
        events[name + '_p66'] = upper
    return events


def evaluate_event(event, frame):
    g = validate_stream(frame)
    if not g.source_point_index.between(event['event_start_index'], event['event_end_index']).all():
        raise ValueError('Prediction outside event.')
    if len(g) != event['evaluated_point_count']:
        raise ValueError('Missing/extra evaluated event predictions.')
    if str(g.sample_id.iloc[0]) != str(event['sample_id']) or str(g.user_id.iloc[0]) != str(event['user_id']) or str(g.source_trajectory_id.iloc[0]) != str(event['source_trajectory_id']):
        raise ValueError('Event prediction identity mismatch.')
    runs = positive_runs(g)
    positives = g.loc[g.predicted_anomaly.eq(1)]
    hit = len(positives) > 0
    offset = int(positives.source_point_index.iloc[0]) - event['event_start_index'] if hit else np.nan
    delay = (positives.timestamp.iloc[0] - event['event_start_timestamp']).total_seconds() if hit else np.nan
    if hit and delay < 0:
        raise ValueError('Negative detection delay.')
    relative = offset / max(event['event_point_count'] - 1, 1) if hit else np.nan
    longest = int(runs.length_points.max()) if len(runs) else 0
    row = dict(event)
    row.update({'event_detected': hit, 'event_status': 'detected' if hit else 'missed_event',
                'detected_point_count': len(positives),
                'first_detection_source_index': int(positives.source_point_index.iloc[0]) if hit else np.nan,
                'first_detection_timestamp': positives.timestamp.iloc[0] if hit else pd.NaT,
                'delay_points': offset, 'first_detection_point_offset': offset,
                'relative_detection_position': relative, 'delay_seconds': delay,
                'delay_na_reason': '' if hit else 'missed_event',
                'delay_seconds_crosses_long_gap': bool(hit and delay > 0 and g.loc[g.source_point_index.le(positives.source_point_index.iloc[0]), 'timestamp'].diff().dt.total_seconds().gt(LONG_GAP_THRESHOLD_SEC).any()),
                # Requested full-label coverage. Excluded point has no saved decision.
                'event_point_coverage': len(positives) / event['event_point_count'],
                'evaluated_point_coverage': len(positives) / event['evaluated_point_count'],
                'coverage_available_ratio': event['evaluated_point_count'] / event['event_point_count'],
                'longest_positive_run_points': longest,
                'longest_positive_run_ratio': longest / event['event_point_count'],
                'positive_run_count': len(runs),
                'average_run_length': len(positives) / len(runs) if len(runs) else np.nan})
    for cutoff in EARLY_CUTOFFS:
        row[f'early_detection_{int(cutoff * 100)}'] = bool(hit and relative <= cutoff)
    return row


def validate_fixed_decisions(frame, threshold):
    scores = pd.to_numeric(frame.anomaly_score, errors='raise').to_numpy(float)
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Non-finite stored score/threshold.')
    if not frame.predicted_anomaly.isin([0, 1]).all() or not np.array_equal(frame.predicted_anomaly, (scores > threshold).astype(int)):
        raise ValueError('Stored decisions differ from frozen threshold.')


def prediction_specs(root, dataset_id):
    for model in MODEL_NAMES:
        for seed in ([None] if model == 'statistical_rule' else baseline.MODEL_SEEDS):
            if model == 'autoencoder':
                directory = single.output_directories(root, dataset_id, seed)
                yield model, seed, directory['metrics'] / 'test_predictions.csv', directory['model'] / 'training_config.json'
            else:
                suffix = Path(model) if seed is None else Path(model) / f'seed_{seed}'
                yield model, seed, root / 'outputs/metrics/stage5' / dataset_id / 'baseline_comparison' / suffix / 'test_predictions.csv', root / 'models/stage5' / dataset_id / 'baselines' / suffix / 'run_config.json'


def load_decisions(root, dataset_id, test, tiers):
    route_parts, normal_parts, checksums, configs = [], [], {}, []
    expected_normal = test.loc[~test.is_synthetic & test.anomaly_label.eq(0)]
    route_columns = SCORE_KEY + ['user_id', 'timestamp', 'predicted_anomaly']
    normal_columns = KEY + ['user_id', 'sample_id', 'timestamp', 'time_diff_sec', 'predicted_anomaly']
    for model, seed, path, config_path in prediction_specs(root, dataset_id):
        frame = (read_dataset_csv(path).rename(columns={'reconstruction_error': 'anomaly_score'})
                 if model == 'autoencoder' else baseline.read_baseline_predictions(path))
        baseline.assert_same_rows(frame, test)
        threshold = float(json.loads(config_path.read_text(encoding='utf-8'))['threshold'])
        validate_fixed_decisions(frame, threshold)
        copied = frame.loc[frame.is_synthetic & frame.anomaly_label.eq(1) & frame.anomaly_type.eq('route_deviation')]
        normals = frame.loc[~frame.is_synthetic & frame.anomaly_label.eq(0)]
        if set(copied[SCORE_KEY].itertuples(index=False, name=None)) != set(tiers[SCORE_KEY].itertuples(index=False, name=None)):
            raise ValueError('Route prediction universe mismatch.')
        if set(normals[KEY].itertuples(index=False, name=None)) != set(expected_normal[KEY].itertuples(index=False, name=None)):
            raise ValueError('Normal prediction universe mismatch.')
        matched = labels.exact_join(tiers[SCORE_KEY + ['user_id', 'timestamp']], copied[route_columns], SCORE_KEY)
        if not matched.user_id_x.eq(matched.user_id_y).all() or not pd.to_datetime(matched.timestamp_x).eq(matched.timestamp_y).all():
            raise ValueError('Prediction/tier metadata mismatch.')
        for destination, selected, columns in [(route_parts, copied, route_columns), (normal_parts, normals, normal_columns)]:
            part = selected[columns].copy()
            part.insert(0, 'model', model)
            part.insert(1, 'seed', seed)
            destination.append(part)
        for source in (path, config_path):
            checksums[source.relative_to(root).as_posix()] = single.sha256_file(source)
        official_path = path.parent / 'test_metrics.json'
        official = json.loads(official_path.read_text(encoding='utf-8'))
        if int(normals.predicted_anomaly.sum()) != official['fp'] or len(normals) != official['normal_row_count']:
            raise ValueError('Official Stage 5 normal metric mismatch.')
        checksums[official_path.relative_to(root).as_posix()] = single.sha256_file(official_path)
        configs.append({'model': model, 'seed': seed, 'frozen_threshold': threshold,
                        'prediction_path': path.relative_to(root).as_posix(),
                        'official_fp': official['fp'], 'official_fpr': official['false_positive_rate']})
    return pd.concat(route_parts, ignore_index=True), pd.concat(normal_parts, ignore_index=True), checksums, configs


def evaluate_normal_predictions(normals):
    run_parts, trajectory_rows = [], []
    for (model, seed, user, source), frame in normals.groupby(['model', 'seed', 'user_id', 'source_trajectory_id'], dropna=False, sort=True):
        runs = positive_runs(frame)
        identity = {'model': model, 'seed': seed, 'user_id': str(user), 'source_trajectory_id': str(source)}
        exposure = normal_exposure(frame)
        for number, row in enumerate(runs.to_dict('records'), 1):
            run_parts.append(dict(identity, false_alert_run_id=f'{source}__alert_{number:04d}', **row))
        trajectory_rows.append(dict(identity, normal_point_count=len(frame), fp_point_count=int(frame.predicted_anomaly.sum()),
                                    fpr=float(frame.predicted_anomaly.mean()), false_alert_run_count=len(runs),
                                    longest_false_alert_run_points=int(runs.length_points.max()) if len(runs) else 0,
                                    any_false_alert=bool(len(runs)), **exposure))
    columns = ['model', 'seed', 'user_id', 'source_trajectory_id', 'false_alert_run_id'] + RUN_COLUMNS
    return pd.DataFrame(run_parts, columns=columns), pd.DataFrame(trajectory_rows)


def normal_summary(trajectories):
    points = int(trajectories.normal_point_count.sum())
    fp = int(trajectories.fp_point_count.sum())
    runs = int(trajectories.false_alert_run_count.sum())
    exposure = float(trajectories.observation_exposure_sec.sum())
    return {'normal_point_count': points, 'normal_trajectory_count': len(trajectories),
            'fp_point_count': fp, 'fpr': fp / points if points else np.nan,
            'false_alert_run_count': runs, 'false_alert_runs_per_trajectory': runs / len(trajectories),
            'normal_trajectories_with_alert_percent': 100 * float(trajectories.any_false_alert.mean()),
            'fp_per_1000_normal_points': 1000 * fp / points if points else np.nan,
            'normal_observation_exposure_sec': exposure,
            'valid_observation_intervals': int(trajectories.valid_observation_intervals.sum()),
            'false_alert_runs_per_hour': runs / (exposure / 3600) if exposure > 0 else np.nan,
            'exposure_na_reason': '' if exposure > 0 else 'no_valid_adjacent_observation_intervals'}


def quantiles(values, prefix, probabilities):
    values = pd.Series(values, dtype=float).dropna()
    result = {prefix + '_mean': float(values.mean()) if len(values) else np.nan}
    for name, p in probabilities.items():
        result[prefix + '_' + name] = float(values.quantile(p)) if len(values) else np.nan
    return result


def event_summary(events):
    detected = events.loc[events.event_detected]
    result = {'event_count': len(events), 'detected_event_count': int(events.event_detected.sum()),
              'missed_event_count': int((~events.event_detected).sum()),
              'event_detection_rate': float(events.event_detected.mean()),
              'point_recall': events.detected_point_count.sum() / events.evaluated_point_count.sum(),
              'full_label_observed_positive_fraction': events.detected_point_count.sum() / events.event_point_count.sum(),
              'median_longest_positive_run_points': float(events.longest_positive_run_points.median()),
              'median_longest_positive_run_ratio': float(events.longest_positive_run_ratio.median()),
              'mean_positive_run_count': float(events.positive_run_count.mean()),
              'total_positive_run_count': int(events.positive_run_count.sum()),
              'delay_conditioning': 'detected_events_only',
              'delay_na_reason': '' if len(detected) else 'all_events_missed'}
    for cutoff in EARLY_CUTOFFS:
        column = f'early_detection_{int(cutoff * 100)}'
        result[column + '_rate'] = float(events[column].mean())
        result[column + '_detected_conditional_rate'] = float(detected[column].mean()) if len(detected) else np.nan
    for column in ('delay_points', 'delay_seconds', 'relative_detection_position'):
        result.update(quantiles(detected[column], column, {'median': .5, 'p75': .75, 'p90': .90, 'p95': .95}))
    for column in ('event_point_coverage', 'evaluated_point_coverage'):
        result.update(quantiles(events[column], column, {'median': .5, 'p25': .25, 'p75': .75, 'p90': .90}))
    return result


def grouped_summary(frame, keys, function):
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        rows.append(dict(zip(keys, group_key), **function(group)))
    return pd.DataFrame(rows)


def seed_summary(seed_rows, metrics):
    """One row per model/metric; Rule has an explicitly NA sample std."""
    records = []
    for model, group in seed_rows.groupby('model', sort=False):
        expected = 1 if model == 'statistical_rule' else len(baseline.MODEL_SEEDS)
        if len(group) != expected or (model != 'statistical_rule' and set(group.seed.astype(int)) != set(baseline.MODEL_SEEDS)):
            raise ValueError('Incomplete/repeated detector seeds.')
        if model == 'statistical_rule' and not group.seed.isna().all():
            raise ValueError('Rule must have no invented seed.')
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors='raise')
            if np.isinf(values).any():
                raise ValueError('Infinite aggregate.')
            valid = values.dropna()
            records.append({'model': model, 'metric': metric, 'seed_count': len(group),
                            'available_seed_count': len(valid), 'mean': valid.mean(),
                            'sample_std': valid.std(ddof=1) if model != 'statistical_rule' and len(valid) >= 2 else np.nan,
                            'min': valid.min(), 'max': valid.max(),
                            'std_na_reason': 'deterministic_single_result' if model == 'statistical_rule' else ('insufficient_available_seeds' if len(valid) < 2 else '')})
    return pd.DataFrame(records)


def build_tables(events, route_predictions, normal_predictions):
    rows = []
    for (model, seed), frame in route_predictions.groupby(['model', 'seed'], dropna=False, sort=True):
        for event in events.to_dict('records'):
            selected = frame.loc[frame.sample_id.eq(event['sample_id']) & frame.source_point_index.between(event['event_start_index'], event['event_end_index'])]
            rows.append(dict(model=model, seed=seed, **evaluate_event(event, selected)))
    positive = pd.DataFrame(rows)
    runs, trajectories = evaluate_normal_predictions(normal_predictions)
    detection = grouped_summary(positive, ['model', 'seed'], event_summary)
    normal = grouped_summary(trajectories, ['model', 'seed'], normal_summary)
    comparison = detection.merge(normal, on=['model', 'seed'], validate='one_to_one')
    user_events = grouped_summary(positive, ['model', 'seed', 'user_id'], event_summary)
    user_normal = grouped_summary(trajectories, ['model', 'seed', 'user_id'], normal_summary)
    users = user_events.merge(user_normal, on=['model', 'seed', 'user_id'], validate='one_to_one')
    metrics = [c for c in comparison if pd.api.types.is_numeric_dtype(comparison[c]) and c != 'seed']
    stability = seed_summary(comparison, metrics)
    models = stability.pivot(index='model', columns='metric', values='mean').reset_index()
    stds = stability.pivot(index='model', columns='metric', values='sample_std').add_suffix('_sample_std').reset_index()
    models = models.merge(stds, on='model', validate='one_to_one')
    macro_metrics = ['event_detection_rate', 'early_detection_25_rate', 'event_point_coverage_median',
                     'delay_points_median', 'fp_per_1000_normal_points', 'normal_trajectories_with_alert_percent']
    macro = users.groupby(['model', 'seed'], dropna=False)[macro_metrics].mean().add_prefix('macro_user_').reset_index()
    tier_rows = []
    for (model, seed), group in positive.groupby(['model', 'seed'], dropna=False, sort=True):
        for contains in (True, False):
            selected = group.loc[group.contains_tier4.eq(contains)]
            row = {'model': model, 'seed': seed, 'contains_tier4': contains, 'event_count': len(selected),
                   'na_reason': '' if len(selected) else 'no_events_without_tier4_boundary'}
            if len(selected):
                row.update(event_summary(selected))
            else:
                row.update({c: np.nan for c in ('event_detection_rate', 'delay_points_median', 'event_point_coverage_median')})
            tier_rows.append(row)
    return {'positive_event_metrics': positive, 'event_detection_by_seed': detection,
            'event_detection_by_model': models, 'event_delay_metrics': positive[['model', 'seed', 'event_id', 'user_id', 'event_status', 'first_detection_source_index', 'first_detection_timestamp', 'first_detection_point_offset', 'relative_detection_position', 'delay_points', 'delay_seconds', 'delay_na_reason']],
            'event_coverage_metrics': positive[['model', 'seed', 'event_id', 'user_id', 'event_point_count', 'evaluated_point_count', 'excluded_quality_point_count', 'detected_point_count', 'event_point_coverage', 'evaluated_point_coverage', 'coverage_available_ratio']],
            'event_run_metrics': positive[['model', 'seed', 'event_id', 'user_id', 'longest_positive_run_points', 'longest_positive_run_ratio', 'positive_run_count', 'average_run_length']],
            'normal_false_alert_runs': runs, 'per_trajectory_false_alert_metrics': trajectories,
            'normal_false_alert_summary': normal, 'per_user_event_metrics': users,
            'event_length_analysis': grouped_summary(positive, ['model', 'seed', 'length_group'], event_summary),
            'event_duration_analysis': grouped_summary(positive, ['model', 'seed', 'duration_group'], event_summary),
            'event_tier_analysis': pd.DataFrame(tier_rows), 'point_vs_event_comparison': comparison,
            'seed_stability_summary': stability, 'macro_user_event_metrics': macro, 'event_catalog': events}


def make_figures(tables, destination):
    destination.mkdir(parents=True, exist_ok=False)
    colors = ['#4169a1', '#d08228', '#3b906b']
    pretty = {'statistical_rule': 'Rule', 'isolation_forest': 'IF', 'autoencoder': 'AE'}
    m = tables['event_detection_by_model'].set_index('model').loc[list(MODEL_NAMES)]
    x = np.arange(3)

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(destination / (name + '.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - .18, m.point_recall * 100, .36, label='Point recall (378 evaluated labels)', color='#647b99')
    ax.bar(x + .18, m.event_detection_rate * 100, .36, label='Event detection (19 full events)', color='#57a58a')
    ax.set(xticks=x, xticklabels=['Rule', 'IF', 'AE'], ylabel='Percent', ylim=(0, 105), title='Same frozen decisions, different evaluation units')
    ax.legend(loc='upper left', fontsize=8)
    finish(fig, 'point_recall_vs_event_detection')

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x, m.event_detection_rate * 100, color=colors, yerr=m.event_detection_rate_sample_std.fillna(0) * 100, capsize=5)
    ax.set(xticks=x, xticklabels=['Rule', 'IF', 'AE'], ylabel='Event detection (%)', ylim=(0, 105), title='Rule single result; IF/AE five-seed mean and sample std')
    finish(fig, 'event_detection_by_model')

    p = tables['positive_event_metrics']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for model, color in zip(MODEL_NAMES, colors):
        g = p.loc[p.model.eq(model) & p.event_detected]
        axes[0].scatter(g.relative_detection_position, np.full(len(g), MODEL_NAMES.index(model)), color=color, alpha=.35, label=pretty[model])
        data = np.sort(g.delay_seconds.dropna())
        if len(data):
            axes[1].step(data, np.arange(1, len(data) + 1) / len(data), where='post', color=color, label=pretty[model])
    axes[0].set(xlabel='First detection / full event position', yticks=x, yticklabels=['Rule', 'IF', 'AE'], xlim=(-.02, 1.02))
    axes[1].set(xlabel='Wall-clock delay (seconds)', ylabel='Detected-only cumulative fraction', ylim=(0, 1.05))
    axes[1].legend()
    fig.suptitle('Detected events only; IF/AE event-seed observations pooled')
    finish(fig, 'detection_delay_distribution')

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.boxplot([p.loc[p.model.eq(model), 'event_point_coverage'] * 100 for model in MODEL_NAMES], tick_labels=['Rule', 'IF', 'AE'], showmeans=True)
    ax.set(ylabel='Stored positive points / full labelled points (%)', title='Coverage: all events, including misses; IF/AE pooled seeds', ylim=(-2, 102))
    finish(fig, 'event_coverage_distribution')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, column, label in zip(axes, ['fp_per_1000_normal_points', 'false_alert_runs_per_trajectory'], ['FP / 1,000 original normal points', 'False alert runs / original trajectory']):
        ax.bar(x, m[column], color=colors, yerr=m[column + '_sample_std'].fillna(0), capsize=5)
        ax.set(xticks=x, xticklabels=['Rule', 'IF', 'AE'], ylabel=label)
    fig.suptitle('46,299 original normal points; no smoothing or gap bridging')
    finish(fig, 'false_alert_burden_by_model')

    fig, ax = plt.subplots(figsize=(8, 4.6))
    comparison = tables['point_vs_event_comparison']
    for model, color in zip(MODEL_NAMES, colors):
        g = comparison.loc[comparison.model.eq(model)]
        ax.scatter(g.fp_per_1000_normal_points, g.event_detection_rate * 100, label=pretty[model], color=color, s=65)
    ax.set(xlabel='FP / 1,000 original normal points', ylabel='Event detection (%)', ylim=(75, 103), title='Every frozen detector run; positive and negative scopes separate')
    ax.legend()
    finish(fig, 'event_detection_vs_false_alerts')

    fig, ax = plt.subplots(figsize=(9, 4.6))
    u = tables['per_user_event_metrics']
    users = sorted(u.user_id.unique())
    for index, (model, color) in enumerate(zip(MODEL_NAMES, colors)):
        g = u.loc[u.model.eq(model)].groupby('user_id').event_detection_rate
        ax.bar(np.arange(len(users)) + (index - 1) * .25, g.mean().reindex(users) * 100, .25, yerr=g.std(ddof=1).reindex(users).fillna(0) * 100, color=color, label=pretty[model], capsize=3)
    ax.set(xticks=np.arange(len(users)), xticklabels=users, ylabel='Event detection (%)', ylim=(0, 112), title='User event detection: equal seed means, sample std')
    ax.legend(loc='lower right')
    finish(fig, 'per_user_event_detection')

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for index, (model, color) in enumerate(zip(MODEL_NAMES, colors)):
        columns = [f'early_detection_{int(c * 100)}_rate' for c in EARLY_CUTOFFS]
        values = m.loc[model, columns].to_numpy(float) * 100
        errors = m.loc[model, [c + '_sample_std' for c in columns]].fillna(0).to_numpy(float) * 100
        ax.bar(x + (index - 1) * .25, values, .25, yerr=errors, capsize=3, label=pretty[model], color=color)
    ax.set(xticks=x, xticklabels=['Within first 10%', 'Within first 25%', 'Within first 50%'], ylabel='All-event early detection (%)', ylim=(0, 105), title='Fixed cutoffs; missed events remain in denominator')
    ax.legend(loc='lower right')
    finish(fig, 'early_detection_by_model')


def verify_previous_metrics(root, dataset_id, tables):
    official = labels.read_frozen(root / 'outputs/metrics/stage5' / dataset_id / 'baseline_comparison/per_anomaly_type_baseline_metrics.csv')
    official = official.loc[official.anomaly_type.eq('route_deviation')]
    old_events = labels.read_frozen(root / 'outputs/metrics/stage6/route_label_audit/event_level_detection_metrics.csv')
    current = tables['positive_event_metrics']
    for (model, seed), group in current.groupby(['model', 'seed'], dropna=False):
        matches = official.loc[official.model.eq(model) & (official.seed.isna() if pd.isna(seed) else official.seed.eq(seed))]
        if len(matches) != 1 or not np.isclose(group.detected_point_count.sum() / group.evaluated_point_count.sum(), float(matches.recall.iloc[0]), rtol=0, atol=1e-12):
            raise ValueError('Official point recall disagreement.')
        old = old_events.loc[old_events.model.eq(model) & (old_events.seed.isna() if pd.isna(seed) else old_events.seed.eq(seed))]
        joined = group.merge(old, on='sample_id', suffixes=('_new', '_old'), validate='one_to_one')
        if len(joined) != len(group) or not joined.event_detected_new.eq(joined.event_detected_old).all():
            raise ValueError('Stage 6.5 event detection disagreement.')
        for a, b in [('delay_points', 'detection_delay_points'), ('delay_seconds', 'detection_delay_seconds')]:
            if not np.allclose(joined[a], joined[b], rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError('Stage 6.5 event delay disagreement.')


def run_audit(data_dir=single.DEFAULT_DATA_DIR, output_root=single.ROOT, protected_snapshot=None):
    started = time.perf_counter()
    root, data_dir = Path(output_root).resolve(), Path(data_dir).resolve()
    metrics = root / 'outputs/metrics/stage6/event_evaluation'
    figures = root / 'outputs/figures/stage6/event_evaluation'
    if metrics.exists() or figures.exists():
        raise FileExistsError('Event audit overwrite refused.')
    snapshot = json.loads(Path(protected_snapshot or root / 'outputs/metrics/stage66_protected_snapshot.json').read_text(encoding='utf-8'))
    required = [data_dir.relative_to(root).as_posix(), 'models/stage5', 'outputs/metrics/stage5',
                'outputs/metrics/stage6/route_label_audit', 'outputs/figures/stage6/route_label_audit',
                'src/synthetic_anomalies.py', 'src/audit_synthetic_route_labels.py']
    for relative in required:
        if not any(k.replace('\\', '/') == relative or k.replace('\\', '/').startswith(relative + '/') for k in snapshot):
            raise ValueError('Incomplete protected snapshot: ' + relative)
    labels.assert_protected(root, snapshot)
    summary, users = single.read_stage5_metadata(data_dir)
    test = single.load_stage5_split(data_dir, 'test', summary, users)
    source = root / 'outputs/metrics/stage6/route_label_audit'
    full = labels.read_frozen(source / 'route_label_full_copy_audit.csv')
    tiers = labels.read_frozen(source / 'route_label_point_audit.csv')
    manifest = read_dataset_csv(data_dir / 'synthetic_anomaly_manifest.csv')
    manifest = manifest.loc[manifest.dataset_split.eq('test') & manifest.anomaly_type.eq('route_deviation')]
    events = build_events(full, manifest, tiers)
    route, normals, checksums, configs = load_decisions(root, data_dir.name, test, tiers)
    tables = build_tables(events, route, normals)
    verify_previous_metrics(root, data_dir.name, tables)
    metrics.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(metrics / (name + '.csv'), index=False)
    make_figures(tables, figures)
    report = {'stage': '6.6', 'supplementary_evaluation': True, 'dataset_id': data_dir.name,
              'positive_event_count': len(events), 'full_labelled_points': int(events.event_point_count.sum()),
              'evaluated_labelled_points': int(events.evaluated_point_count.sum()),
              'excluded_labelled_points_without_prediction': int(events.excluded_quality_point_count.sum()),
              'normal_points_per_run': int((~test.is_synthetic & test.anomaly_label.eq(0)).sum()),
              'detector_runs': len(configs), 'detector_event_rows': len(tables['positive_event_metrics']),
              'model_seeds': list(baseline.MODEL_SEEDS), 'frozen_detector_configs': configs,
              'event_length_description': events.event_point_count.describe(percentiles=[.33, .66]).to_dict(),
              'event_duration_description': events.event_duration_sec.describe(percentiles=[.33, .66]).to_dict(),
              'duration_na_events': int(events.event_duration_sec.isna().sum()),
              'contains_tier4_event_count': int(events.contains_tier4.sum()),
              'without_tier4_event_count': int((~events.contains_tier4).sum()),
              'tier_comparison_na_reason': 'all_events_contain_tier4; no between-group effect estimable' if events.contains_tier4.all() else '',
              'early_cutoffs': list(EARLY_CUTOFFS), 'early_denominator': 'all_positive_events; misses are false',
              'delay_scope': 'detected events only; misses NA, never imputed',
              'coverage_scope': 'event_point_coverage=stored TP/full label geometry (379); evaluated_point_coverage=TP/available labels (378); excluded decisions unknown, not negative',
              'point_recall_link': 'sum(TP)/sum(evaluated length) equals official recall; full-length weighted coverage uses 379 instead',
              'run_policy': 'positive runs within one source/sample; break at negative, missing source index, or timestamp interval > existing long-gap threshold',
              'long_gap_threshold_sec': LONG_GAP_THRESHOLD_SEC,
              'time_exposure_policy': 'sum adjacent evaluated original-normal intervals with both endpoints present and 0 < delta <= 300; validate existing time_diff_sec; not last-first span',
              'event_precision': None, 'event_precision_na_reason': 'independent original and synthetic copies; no fully scored continuous combined stream',
              'alarm_matching_performed': False, 'official_stage5_metrics_unchanged': True,
              'official_point_and_normal_metrics_verified': True, 'stage65_event_results_verified': True,
              'model_training_or_inference': False, 'threshold_modified': False, 'labels_regenerated': False,
              'prediction_smoothing': False, 'protected_file_count': len(snapshot), 'protected_changed_files': 0,
              'prediction_config_and_official_checksums': checksums,
              'source_sha256': single.sha256_file(Path(__file__)),
              'output_checksums': {p.name: single.sha256_file(p) for p in sorted(metrics.glob('*.csv'))},
              'figure_checksums': {p.name: single.sha256_file(p) for p in sorted(figures.glob('*.png'))},
              'runtime_seconds': time.perf_counter() - started,
              'recommended_protocol': 'B extended with coverage, fixed early rates, exposure burden, macro-user and seed summaries; supplementary until prospectively approved',
              'stage67_recommendation_only': 'A: prospectively fix event-aware protocol; B: if needed, Validation-only alert policy study; no implementation',
              'stage66_committed': False, 'stage66_pushed': False, 'pr_created': False, 'main_merged': False}
    labels.assert_protected(root, snapshot)
    single.save_json(metrics / 'event_protocol_summary.json', labels.json_safe(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=single.ROOT)
    parser.add_argument('--protected-snapshot', type=Path)
    args = parser.parse_args()
    report = run_audit(args.data_dir, args.output_root, args.protected_snapshot)
    print(json.dumps(labels.json_safe({k: report[k] for k in ('stage', 'positive_event_count', 'detector_event_rows', 'normal_points_per_run', 'runtime_seconds', 'protected_file_count', 'protected_changed_files')}), indent=2))


if __name__ == '__main__':
    main()
