"""Stage 6.7: frozen candidates -> Validation-only selection -> lock -> Test.

This cohort was observed in Stage 6.6. Test is a held-out policy application,
not an untouched final benchmark. Existing point predictions are immutable.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import audit_event_evaluation_protocol as event_audit
from src import audit_synthetic_route_labels as labels
from src import compare_stage5_baselines as baseline
from src import train_multiuser_autoencoder as single
from src.train_autoencoder import FEATURE_COLUMNS, add_bearing_features
from src.load_multiuser_geolife import read_dataset_csv
from src.data_quality import LONG_GAP_THRESHOLD_SEC
from src.alert_policy import AlertPolicy, LockedPolicies, apply_policy, GATE_DEFINITIONS, COMPLEXITY_ORDER, COOLDOWNS

METRICS = Path('outputs/metrics/stage6/alert_aggregation')
FIGURES = Path('outputs/figures/stage6/alert_aggregation')
CONFIG = Path('configs/stage67_alert_policy_candidates.json')
FAMILIES = event_audit.MODEL_NAMES
RAW = AlertPolicy('G0', 0)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    single.save_json(path, labels.json_safe(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def frozen_candidates(root):
    manifest = read_json(root / METRICS / 'candidate_policy_manifest.json')
    if single.sha256_file(root / CONFIG) != manifest['candidate_config_sha256']:
        raise ValueError('Candidate config hash changed after freeze.')
    config = read_json(root / CONFIG)
    policies = tuple(AlertPolicy.from_dict(p) for p in config['policies'])
    expected = {f'{gate}_C{cooldown}' for gate in GATE_DEFINITIONS for cooldown in COOLDOWNS}
    if len(policies) != 21 or {p.policy_id for p in policies} != expected:
        raise ValueError('Frozen candidate set must contain exactly 7 x 3 policies.')
    if config['selection']['split'] != 'validation':
        raise ValueError('Selection split changed.')
    return config, policies, manifest


def protect(root):
    snapshot = read_json(root / 'outputs/metrics/stage67_protected_snapshot.json')
    for prefix in ['models/stage5', 'outputs/metrics/stage5', 'outputs/metrics/stage6/event_evaluation',
                   'src/audit_event_evaluation_protocol.py', 'data/processed/stage5']:
        if not any(k.replace('\\', '/') == prefix or k.replace('\\', '/').startswith(prefix + '/') for k in snapshot):
            raise ValueError('Incomplete protected snapshot: ' + prefix)
    labels.assert_protected(root, snapshot)
    return snapshot


def build_catalog(full, manifest, evaluated):
    """No detector decisions or tier filtering determine event boundaries."""
    records = []
    if manifest.sample_id.duplicated().any() or set(full.sample_id) != set(manifest.sample_id):
        raise ValueError('Event sample/manifest catalog mismatch.')
    for sample, part in full.groupby('sample_id', sort=True):
        g = event_audit.validate_stream(part)
        if not g.source_point_index.diff().dropna().eq(1).all():
            raise ValueError('Full copy has missing source points.')
        flags = g.anomaly_label.eq(1).to_numpy()
        starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
        ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError('Expected frozen one-segment sample; unscored intervening copies cannot be filled.')
        segment = g.iloc[starts[0]:ends[0] + 1]
        m = manifest.loc[manifest.sample_id.eq(sample)].iloc[0]
        if len(segment) != m.anomaly_point_count or len(g) != m.generated_point_count:
            raise ValueError('Manifest count mismatch.')
        if segment.timestamp.iloc[0] != pd.Timestamp(m.anomaly_start_time) or segment.timestamp.iloc[-1] != pd.Timestamp(m.anomaly_end_time):
            raise ValueError('Manifest boundary mismatch.')
        if str(m.user_id) != str(g.user_id.iloc[0]) or str(m.source_trajectory_id) != str(g.source_trajectory_id.iloc[0]):
            raise ValueError('Manifest identity mismatch.')
        observed = evaluated.loc[evaluated.sample_id.eq(sample)]
        if not observed.source_point_index.isin(segment.source_point_index).all():
            raise ValueError('Evaluated predictions outside label segment.')
        seconds = segment.timestamp.diff().dt.total_seconds().dropna()
        valid_duration = seconds.gt(0).all() and seconds.le(LONG_GAP_THRESHOLD_SEC).all()
        records.append({'event_id': sample + '__run_01', 'sample_id': sample,
                        'user_id': str(g.user_id.iloc[0]), 'source_trajectory_id': str(g.source_trajectory_id.iloc[0]),
                        'anomaly_segment_id': str(segment.anomaly_segment_id.iloc[0]),
                        'event_start_index': int(segment.source_point_index.iloc[0]),
                        'event_end_index': int(segment.source_point_index.iloc[-1]),
                        'event_start_timestamp': segment.timestamp.iloc[0], 'event_end_timestamp': segment.timestamp.iloc[-1],
                        'event_point_count': len(segment), 'evaluated_point_count': len(observed),
                        'excluded_quality_point_count': len(segment) - len(observed),
                        'event_duration_sec': (segment.timestamp.iloc[-1] - segment.timestamp.iloc[0]).total_seconds() if valid_duration else np.nan})
    return pd.DataFrame(records)


def load_split_inputs(root, data_dir, split):
    if split not in ('validation', 'test'):
        raise ValueError('Unsupported split.')
    summary, users = single.read_stage5_metadata(data_dir)
    evaluated = single.load_stage5_split(data_dir, split, summary, users)
    originals = add_bearing_features(read_dataset_csv(data_dir / 'source_normal.csv'))
    originals = originals.loc[originals.dataset_split.eq(split)].copy()
    full = add_bearing_features(read_dataset_csv(data_dir / (split + '.csv')))
    full = full.loc[full.is_synthetic & full.anomaly_type.eq('route_deviation')].copy()
    manifest = read_dataset_csv(data_dir / 'synthetic_anomaly_manifest.csv')
    manifest = manifest.loc[manifest.dataset_split.eq(split) & manifest.anomaly_type.eq('route_deviation')]
    route = evaluated.loc[evaluated.is_synthetic & evaluated.anomaly_type.eq('route_deviation')]
    events = build_catalog(full, manifest, route)
    parts, artifact_hashes, detector_configs = [], {}, []
    for model, seed, test_path, config_path in event_audit.prediction_specs(root, data_dir.name):
        path = test_path.with_name(split + '_predictions.csv')
        if not path.is_file():
            raise FileNotFoundError('Frozen prediction artifact missing: ' + str(path))
        frame = (read_dataset_csv(path).rename(columns={'reconstruction_error': 'anomaly_score'})
                 if model == 'autoencoder' else baseline.read_baseline_predictions(path))
        baseline.assert_same_rows(frame, evaluated)
        threshold = float(read_json(config_path)['threshold'])
        event_audit.validate_fixed_decisions(frame, threshold)
        normal = frame.loc[~frame.is_synthetic & frame.anomaly_label.eq(0)].copy()
        copied = frame.loc[frame.is_synthetic & frame.anomaly_type.eq('route_deviation') & frame.anomaly_label.eq(1)].copy()
        if set(copied[labels.SCORE_KEY].itertuples(index=False, name=None)) != set(route[labels.SCORE_KEY].itertuples(index=False, name=None)):
            raise ValueError('Frozen route prediction universe mismatch.')
        columns = labels.SCORE_KEY + ['user_id', 'timestamp', 'time_diff_sec', 'predicted_anomaly']
        parts.append({'model': model, 'seed': seed, 'normal': normal[columns].copy(), 'route': copied[columns].copy()})
        for source in (path, config_path):
            artifact_hashes[source.relative_to(root).as_posix()] = single.sha256_file(source)
        detector_configs.append({'model': model, 'seed': seed, 'threshold': threshold, 'predictions_reused': True})
    return {'dataset_split': split, 'events': events, 'full': full, 'originals': originals,
            'detectors': parts, 'prediction_config_checksums': artifact_hashes, 'detector_configs': detector_configs,
            'normal_point_count': int((~evaluated.is_synthetic & evaluated.anomaly_label.eq(0)).sum()),
            'route_evaluated_count': len(route)}


def make_context_stream(event, full, originals, normal, copied):
    """Only existing causal copy prefix + saved labelled decisions through event end.

    These stateless point detectors can reuse a source-normal decision only
    after exact raw GPS/time and feature equality with the pre-event copy.
    No predictions for changed/unscored post-event copied rows are invented.
    """
    sample = full.loc[full.sample_id.eq(event['sample_id'])]
    prefix = sample.loc[sample.source_point_index.lt(event['event_start_index'])].copy()
    raw_source = originals.loc[originals.source_trajectory_id.eq(event['source_trajectory_id'])]
    columns = labels.KEY + ['user_id', 'timestamp', 'latitude', 'longitude'] + FEATURE_COLUMNS
    paired = labels.exact_join(prefix[columns], raw_source[columns], labels.KEY)
    for column in ['user_id', 'timestamp', 'latitude', 'longitude'] + FEATURE_COLUMNS:
        a, b = paired[column + '_x'], paired[column + '_y']
        if not a.eq(b).all():
            raise ValueError('Changed copied prefix cannot reuse a normal prediction: ' + column)
    pre = normal.loc[normal.source_trajectory_id.eq(event['source_trajectory_id']) & normal.source_point_index.lt(event['event_start_index'])].copy()
    expected = prefix.loc[prefix.source_quality_valid & ~prefix.is_low_quality & prefix.is_training_eligible & prefix.synthetic_value_valid]
    if set(pre.source_point_index) != set(expected.source_point_index):
        raise ValueError('Unscored/missing quality-valid prefix.')
    pre['sample_id'] = event['sample_id']
    inside = copied.loc[copied.sample_id.eq(event['sample_id'])].copy()
    if not inside.source_point_index.between(event['event_start_index'], event['event_end_index']).all() or len(inside) != event['evaluated_point_count']:
        raise ValueError('Event prediction extent mismatch.')
    return pd.concat([pre, inside], ignore_index=True).sort_values('source_point_index', kind='stable')


def event_metrics(event, replay):
    g = replay.loc[replay.source_point_index.between(event['event_start_index'], event['event_end_index'])]
    if len(g) != event['evaluated_point_count']:
        raise ValueError('Missing event replay decisions.')
    result = dict(event)
    for kind, column in [('raw', 'predicted_anomaly'), ('gate', 'gate_positive'), ('notification', 'notification_emitted')]:
        detected = g.loc[g[column].eq(1)]
        hit = len(detected) > 0
        first = detected.iloc[0] if hit else None
        offset = int(first.source_point_index) - event['event_start_index'] if hit else np.nan
        relative = offset / max(event['event_point_count'] - 1, 1) if hit else np.nan
        result[kind + '_detected'] = hit
        result[kind + '_positive_count'] = len(detected)
        result[kind + '_first_index'] = int(first.source_point_index) if hit else np.nan
        result[kind + '_delay_points'] = offset
        result[kind + '_delay_seconds'] = (first.timestamp - event['event_start_timestamp']).total_seconds() if hit else np.nan
        result[kind + '_relative_position'] = relative
        result[kind + '_coverage'] = len(detected) / event['event_point_count']
        result[kind + '_evaluated_coverage'] = len(detected) / len(g) if len(g) else np.nan
        for cutoff in (10, 25, 50):
            result[f'{kind}_early{cutoff}'] = bool(hit and relative <= cutoff / 100)
    gate_frame = g.copy()
    gate_frame['predicted_anomaly'] = gate_frame.gate_positive.astype(int)
    runs = event_audit.positive_runs(gate_frame)
    result['longest_gate_run_points'] = int(runs.length_points.max()) if len(runs) else 0
    result['longest_gate_run_ratio'] = result['longest_gate_run_points'] / event['event_point_count']
    result['gate_run_count'] = len(runs)
    result['average_gate_run_length'] = int(g.gate_positive.sum()) / len(runs) if len(runs) else np.nan
    result['event_status'] = 'detected' if result['notification_detected'] else 'missed_event'
    suppressed = g.loc[g.suppressed_by_cooldown]
    relevant = suppressed.loc[suppressed.source_point_index.lt(result['notification_first_index'])] if result['notification_detected'] else suppressed
    result['cooldown_suppressed_candidates'] = len(suppressed)
    result['suppressed_by_cooldown'] = len(relevant) > 0
    result['cooldown_induced_miss'] = bool(result['gate_detected'] and not result['notification_detected'] and len(suppressed))
    result['cooldown_induced_delay'] = bool(result['notification_detected'] and len(relevant))
    result['suppressed_by_pre_event_false_notification'] = bool((relevant.suppressing_notification_source_index < event['event_start_index']).any())
    result['gate_induced_miss'] = bool(result['raw_detected'] and not result['gate_detected'])
    result['suppressed_by_gate'] = bool(result['raw_detected'] and (not result['gate_detected'] or result['gate_first_index'] > result['raw_first_index']))
    result['gate_suppression_reasons'] = ';'.join(sorted(set(g.loc[g.predicted_anomaly.eq(1) & ~g.gate_positive, 'gate_suppression_reason']) - {''}))
    candidates = g.loc[g.notification_candidate]
    result['event_has_gate_run_start'] = bool(len(candidates))
    result['carried_gate_run_without_event_notification'] = bool(result['gate_detected'] and not result['notification_detected'] and not len(candidates))
    result['notification_delay_na_reason'] = '' if result['notification_detected'] else 'missed_event'
    result['raw_to_notification_delay_increase_points'] = result['notification_delay_points'] - result['raw_delay_points'] if result['raw_detected'] and result['notification_detected'] else np.nan
    return result


def summarize_event_rows(events):
    result = {'event_count': len(events)}
    for kind in ('raw', 'gate', 'notification'):
        detected = events.loc[events[kind + '_detected']]
        result[kind + '_detected_count'] = int(events[kind + '_detected'].sum())
        result[kind + '_event_detection_rate'] = float(events[kind + '_detected'].mean())
        for cutoff in (10, 25, 50):
            result[f'{kind}_early{cutoff}_rate'] = float(events[f'{kind}_early{cutoff}'].mean())
        for unit in ('points', 'seconds'):
            result.update(event_audit.quantiles(detected[f'{kind}_delay_{unit}'], f'{kind}_delay_{unit}', {'median': .5, 'p90': .9, 'p95': .95}))
        result[kind + '_coverage_median'] = float(events[kind + '_coverage'].median())
        result[kind + '_coverage_mean'] = float(events[kind + '_coverage'].mean())
    result['raw_point_recall'] = events.raw_positive_count.sum() / events.evaluated_point_count.sum()
    result['gate_longest_run_ratio_median'] = float(events.longest_gate_run_ratio.median())
    result['gate_run_count_mean'] = float(events.gate_run_count.mean())
    result['gate_induced_missed_events'] = int(events.gate_induced_miss.sum())
    result['cooldown_induced_missed_events'] = int(events.cooldown_induced_miss.sum())
    result['cooldown_delayed_events'] = int(events.cooldown_induced_delay.sum())
    result['pre_event_false_notification_suppressed_events'] = int(events.suppressed_by_pre_event_false_notification.sum())
    result['carried_gate_run_missed_events'] = int(events.carried_gate_run_without_event_notification.sum())
    return result


def summarize_false_rows(trajectories):
    count = int(trajectories.normal_point_count.sum())
    raw = int(trajectories.raw_fp_points.sum())
    gate_points = int(trajectories.aggregated_positive_points.sum())
    gate_runs = int(trajectories.false_gate_run_count.sum())
    notifications = int(trajectories.false_notification_count.sum())
    exposure = float(trajectories.observation_exposure_sec.sum())
    return {'normal_point_count': count, 'normal_trajectory_count': len(trajectories),
            'raw_fp_points': raw, 'raw_fpr': raw / count,
            'aggregated_positive_points': gate_points, 'aggregated_fpr': gate_points / count,
            'false_gate_run_count': gate_runs, 'false_notification_count': notifications,
            'false_gate_runs_per_1000_normal_points': 1000 * gate_runs / count,
            'false_notifications_per_1000_normal_points': 1000 * notifications / count,
            'false_notifications_per_trajectory': notifications / len(trajectories),
            'any_notification_trajectory_percent': float(trajectories.any_notification.mean()) * 100,
            'longest_false_gate_run_points': int(trajectories.longest_false_gate_run_points.max()),
            'observation_exposure_sec': exposure, 'false_notifications_per_hour': notifications / (exposure / 3600) if exposure else np.nan,
            'valid_observation_intervals': int(trajectories.valid_observation_intervals.sum())}


def prepare_contexts(inputs):
    for detector in inputs['detectors']:
        detector['normal_trajectories'] = [(source, part.copy()) for source, part in detector['normal'].groupby('source_trajectory_id', sort=True)]
        detector['event_contexts'] = [(event, make_context_stream(event, inputs['full'], inputs['originals'], detector['normal'], detector['route']))
                                      for event in inputs['events'].to_dict('records')]


def evaluate_detector_policy(detector, policy, split, capture=False):
    event_rows, false_rows, traces = [], [], []
    identity = {'model': detector['model'], 'seed': detector['seed'], 'policy_id': policy.policy_id, 'dataset_split': split}
    for source, frame in detector['normal_trajectories']:
        replay = apply_policy(frame, policy)
        gate_frame = replay.copy()
        gate_frame['predicted_anomaly'] = gate_frame.gate_positive.astype(int)
        runs = event_audit.positive_runs(gate_frame)
        false_rows.append(dict(identity, user_id=str(frame.user_id.iloc[0]), source_trajectory_id=str(source),
                               normal_point_count=len(replay), raw_fp_points=int(replay.predicted_anomaly.sum()),
                               aggregated_positive_points=int(replay.gate_positive.sum()), false_gate_run_count=len(runs),
                               false_notification_count=int(replay.notification_emitted.sum()),
                               any_notification=bool(replay.notification_emitted.any()),
                               longest_false_gate_run_points=int(runs.length_points.max()) if len(runs) else 0,
                               **event_audit.normal_exposure(frame)))
        if capture:
            selected = replay.loc[replay.notification_candidate].copy()
            for key, value in identity.items(): selected[key] = value
            selected['stream_scope'] = 'original_normal'
            traces.append(selected)
    for event, frame in detector['event_contexts']:
        replay = apply_policy(frame, policy)
        event_rows.append(dict(identity, **event_metrics(event, replay)))
        if capture:
            selected = replay.loc[replay.notification_candidate].copy()
            for key, value in identity.items(): selected[key] = value
            selected['stream_scope'] = 'existing_synthetic_prefix_through_event_end'
            selected['event_id'] = event['event_id']
            selected['inside_event'] = selected.source_point_index.between(event['event_start_index'], event['event_end_index'])
            traces.append(selected)
    events, false = pd.DataFrame(event_rows), pd.DataFrame(false_rows)
    metrics = dict(identity, **summarize_event_rows(events), **summarize_false_rows(false))
    return metrics, events, false, pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()


def select_validation_policies(grid, config):
    """Pure selector: only Validation metric rows; no paths/Test input."""
    if not isinstance(grid, pd.DataFrame) or grid.empty or 'dataset_split' not in grid or not grid.dataset_split.eq('validation').all():
        raise ValueError('Validation selector rejects Test or untagged data.')
    if set(grid.model) != set(FAMILIES):
        raise ValueError('Missing detector family.')
    rule = config['selection']
    if rule['split'] != 'validation':
        raise ValueError('Only Validation can select.')
    policies = {p.policy_id: p for p in (AlertPolicy.from_dict(p) for p in config['policies'])}
    family_grid, selected = [], []
    tolerance = rule['float_tolerance']
    for model, family in grid.groupby('model', sort=True):
        if set(family.policy_id) != set(policies):
            raise ValueError('Incomplete Validation candidate set.')
        raw = family.loc[family.policy_id.eq(RAW.policy_id)].copy()
        expected_seeds = 1 if model == 'statistical_rule' else len(baseline.MODEL_SEEDS)
        if len(raw) != expected_seeds:
            raise ValueError('Incomplete raw seeds.')
        baseline_edr = raw.notification_event_detection_rate.mean()
        baseline_early = raw.notification_early25_rate.mean()
        for policy_id, g in family.groupby('policy_id', sort=True):
            if len(g) != expected_seeds or g.seed.duplicated().any():
                raise ValueError('Incomplete/repeated candidate seeds.')
            if model == 'statistical_rule':
                if not g.seed.isna().all(): raise ValueError('Rule has no seed.')
            elif set(g.seed.astype(int)) != set(baseline.MODEL_SEEDS):
                raise ValueError('Wrong seed universe.')
            policy = policies[policy_id]
            numeric = [c for c in g if pd.api.types.is_numeric_dtype(g[c]) and c != 'seed']
            row = dict(model=model, dataset_split='validation', policy_id=policy_id,
                       seed_count=len(g), **{c: g[c].mean() for c in numeric})
            for metric in numeric:
                row[metric + '_sample_std'] = g[metric].std(ddof=1) if model != 'statistical_rule' else np.nan
            edr_ok = row['notification_event_detection_rate'] + tolerance >= rule['mean_edr_retention_min'] * baseline_edr
            early_ok = row['notification_early25_rate'] + tolerance >= rule['mean_early25_retention_min'] * baseline_early
            seed_guard = g[['seed', 'notification_event_detection_rate']].merge(raw[['seed', 'notification_event_detection_rate']], on='seed', suffixes=('_candidate', '_raw'), validate='one_to_one')
            pass_count = int((seed_guard.notification_event_detection_rate_candidate + tolerance >= rule['seed_edr_retention_min'] * seed_guard.notification_event_detection_rate_raw).sum())
            stable_ok = model == 'statistical_rule' or pass_count >= rule['minimum_guardrail_seeds']
            row.update(eligible=bool(edr_ok and early_ok and stable_ok), edr_guard_passed=bool(edr_ok),
                       early25_guard_passed=bool(early_ok), seed_guard_pass_count=pass_count,
                       seed_guard_passed=bool(stable_ok), raw_mean_notification_edr=baseline_edr,
                       raw_mean_early25=baseline_early, complexity_gate_rank=policy.complexity[0],
                       complexity_cooldown_rank=policy.complexity[1],
                       rejection_reason=';'.join(reason for good, reason in [(edr_ok, 'mean_EDR_retention'), (early_ok, 'mean_early25_retention'), (stable_ok, 'four_of_five_seed_EDR_guard')] if not good))
            family_grid.append(row)
        rows = [r for r in family_grid if r['model'] == model and r['eligible']]
        def key(r):
            delay = r['notification_delay_points_median']
            return (r['false_notifications_per_1000_normal_points'], r['false_notifications_per_trajectory'],
                    delay if pd.notna(delay) else float('inf'), -r['notification_event_detection_rate'],
                    r['complexity_gate_rank'], r['complexity_cooldown_rank'])
        if not rows or not any(r['policy_id'] == RAW.policy_id for r in rows):
            raise ValueError('Raw baseline must remain eligible.')
        best = min(rows, key=key)
        selected.append(dict(best, selected=True, selection_reason='minimum eligible lexicographic objective; no weighted utility'))
    return pd.DataFrame(family_grid), pd.DataFrame(selected)


def write_lock(root, selection, provenance, candidate_manifest):
    path = root / METRICS / 'locked_alert_policies.json'
    if path.exists(): raise FileExistsError('Policy lock overwrite refused.')
    config, candidates, unused = frozen_candidates(root)
    policy_by_id = {p.policy_id: p for p in candidates}
    if len(selection) != 3 or set(selection.model) != set(FAMILIES) or not selection.eligible.all() or not selection.dataset_split.eq('validation').all():
        raise ValueError('Lock requires exactly three eligible Validation selections.')
    record = {'artifact_type': 'locked_alert_policies', 'schema_version': 1, 'selection_split': 'validation',
              'candidate_config_sha256': candidate_manifest['candidate_config_sha256'],
              'candidate_manifest_sha256': single.sha256_file(root / METRICS / 'candidate_policy_manifest.json'),
              'validation_prediction_checksums': provenance['prediction_config_checksums'],
              'source_checksums': provenance['source_checksums'], 'locked_at_utc': now(),
              'git_commit_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'test_metrics_used': False, 'policy_family_post_hoc_to_stage66_test': True,
              'policies': {row.model: dict(policy_by_id[row.policy_id].to_dict(), selection_metrics=labels.json_safe(row._asdict())) for row in selection.itertuples(index=False)}}
    save(path, record)
    receipt = {'locked_policy_sha256': single.sha256_file(path), 'candidate_config_sha256': record['candidate_config_sha256'],
               'locked_at_utc': record['locked_at_utc'], 'test_started': False}
    save(root / METRICS / 'locked_policy_verification.json', receipt)
    return path


def load_locked_policies(root, path):
    if not isinstance(path, (str, Path)):
        raise TypeError('Test accepts a locked-policy file, not a candidate grid.')
    path = Path(path).resolve()
    canonical = (root / METRICS / 'locked_alert_policies.json').resolve()
    if path != canonical:
        raise ValueError('Test accepts only the canonical locked-policy artifact.')
    receipt = read_json(root / METRICS / 'locked_policy_verification.json')
    digest = single.sha256_file(path)
    if digest != receipt['locked_policy_sha256']:
        raise ValueError('Locked policy hash changed.')
    record = read_json(path)
    manifest = read_json(root / METRICS / 'candidate_policy_manifest.json')
    candidate_digest = single.sha256_file(root / CONFIG)
    if candidate_digest != record['candidate_config_sha256'] or candidate_digest != manifest['candidate_config_sha256'] or candidate_digest != receipt['candidate_config_sha256']:
        raise ValueError('Candidate config hash changed.')
    if single.sha256_file(root / METRICS / 'candidate_policy_manifest.json') != record['candidate_manifest_sha256']:
        raise ValueError('Candidate manifest changed.')
    if record.get('artifact_type') != 'locked_alert_policies' or record.get('selection_split') != 'validation' or record.get('test_metrics_used') is not False:
        raise ValueError('Test requires a Validation-only lock.')
    for name, expected in {**record['validation_prediction_checksums'], **record['source_checksums']}.items():
        if single.sha256_file(root / name) != expected:
            raise ValueError('Locked source/Validation artifact changed: ' + name)
    concrete = tuple((model, AlertPolicy.from_dict(policy)) for model, policy in record['policies'].items())
    return LockedPolicies(concrete, digest, candidate_digest)


def validate_test_raw_regression(root, seed_rows, event_rows):
    """Compare detector/gate units, not the newly defined notification unit."""
    old = labels.read_frozen(root / 'outputs/metrics/stage6/event_evaluation/point_vs_event_comparison.csv')
    old_events = labels.read_frozen(root / 'outputs/metrics/stage6/event_evaluation/positive_event_metrics.csv')
    raw = seed_rows.loc[seed_rows.evaluation.eq('raw')]
    mappings = {'raw_event_detection_rate': 'event_detection_rate', 'gate_event_detection_rate': 'event_detection_rate',
                'gate_early25_rate': 'early_detection_25_rate', 'gate_delay_points_median': 'delay_points_median',
                'gate_delay_seconds_median': 'delay_seconds_median', 'gate_coverage_median': 'event_point_coverage_median',
                'raw_point_recall': 'point_recall', 'raw_fp_points': 'fp_point_count', 'raw_fpr': 'fpr',
                'false_gate_run_count': 'false_alert_run_count', 'observation_exposure_sec': 'normal_observation_exposure_sec'}
    joined = raw.merge(old, on=['model', 'seed'], suffixes=('_new', '_old'), validate='one_to_one')
    if len(joined) != len(old): raise ValueError('Stage 6.6 raw detector runs missing.')
    for new, previous in mappings.items():
        a = joined[new + '_new'] if new in old else joined[new]
        b = joined[previous + '_old'] if previous in raw else joined[previous]
        if not np.allclose(a, b, rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError('Stage 6.6 raw regression mismatch: ' + new)
    current = event_rows.loc[event_rows.evaluation.eq('raw')]
    e = current.merge(old_events, on=['model', 'seed', 'event_id'], suffixes=('_new', '_old'), validate='one_to_one')
    if len(e) != len(old_events): raise ValueError('Stage 6.6 raw event count mismatch.')
    for new, previous in [('raw_detected', 'event_detected'), ('gate_detected', 'event_detected'),
                          ('raw_delay_points', 'delay_points'), ('gate_delay_seconds', 'delay_seconds'),
                          ('raw_coverage', 'event_point_coverage')]:
        if not np.allclose(e[new], e[previous], rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError('Stage 6.6 raw event regression mismatch: ' + new)
    return {'passed': True, 'detector_runs': len(raw), 'events_verified': len(current),
            'gate_edr_exactly_matches_stage66': True, 'raw_fp_and_gate_runs_exactly_match': True,
            'notification_scope_distinguished': True}


def evaluate_locked_test(inputs, locked):
    """No grid/config/selection/ranking argument or action in Test evaluation."""
    if not isinstance(locked, LockedPolicies):
        raise TypeError('Test evaluator requires LockedPolicies, never a candidate grid.')
    if inputs.get('dataset_split') != 'test' or 'candidate_grid' in inputs:
        raise ValueError('Test evaluator accepts Test inputs and lock only.')
    rows, events, false, traces = [], [], [], []
    for detector in inputs['detectors']:
        for evaluation, policy in [('raw', RAW), ('locked', locked.for_model(detector['model']))]:
            metrics, e, f, trace = evaluate_detector_policy(detector, policy, 'test', capture=True)
            metrics['evaluation'] = evaluation
            rows.append(metrics)
            for frame, target in [(e, events), (f, false), (trace, traces)]:
                frame['evaluation'] = evaluation
                target.append(frame)
    return pd.DataFrame(rows), pd.concat(events, ignore_index=True), pd.concat(false, ignore_index=True), pd.concat(traces, ignore_index=True)


def reductions(seed_rows):
    raw = seed_rows.loc[seed_rows.evaluation.eq('raw')]
    locked = seed_rows.loc[seed_rows.evaluation.eq('locked')]
    joined = raw.merge(locked, on=['model', 'seed'], suffixes=('_raw', '_locked'), validate='one_to_one')
    result = joined[['model', 'seed', 'policy_id_locked']].copy()
    for metric, name in [('false_notification_count', 'false_notification_reduction_pct'),
                         ('false_gate_run_count', 'false_gate_run_reduction_pct')]:
        a, b = joined[metric + '_raw'], joined[metric + '_locked']
        result[name] = np.where(a.gt(0), 100 * (1 - b / a), np.nan)
    for metric, name in [('notification_event_detection_rate', 'edr_retention_pct'),
                         ('notification_early25_rate', 'early25_retention_pct')]:
        a, b = joined[metric + '_raw'], joined[metric + '_locked']
        result[name] = np.where(a.gt(0), 100 * b / a, np.nan)
    for metric in ['notification_delay_points_median', 'notification_delay_points_p90',
                   'notification_delay_seconds_median', 'notification_delay_seconds_p90',
                   'gate_coverage_median', 'notification_coverage_median']:
        result[metric + '_change'] = joined[metric + '_locked'] - joined[metric + '_raw']
    return result


def seed_stability(rows, keys=('evaluation',)):
    frames = []
    for key, group in rows.groupby(list(keys), dropna=False, sort=True):
        if not isinstance(key, tuple): key = (key,)
        numeric = [c for c in group if pd.api.types.is_numeric_dtype(group[c]) and c != 'seed']
        result = event_audit.seed_summary(group, numeric)
        for column, value in zip(keys, key): result[column] = value
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def family_means(rows):
    return rows.groupby(['model', 'evaluation'], dropna=False, sort=True).mean(numeric_only=True).reset_index()


def make_validation_figures(grid, selection, figures):
    figures.mkdir(parents=True, exist_ok=True)
    colors = {'statistical_rule': '#4169a1', 'isolation_forest': '#d08228', 'autoencoder': '#3b906b'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in grid.groupby('model'):
        ax.scatter(group.false_notifications_per_1000_normal_points, group.notification_event_detection_rate * 100,
                   color=colors[model], alpha=.55, label=model)
    for row in selection.itertuples(index=False):
        ax.scatter(row.false_notifications_per_1000_normal_points, row.notification_event_detection_rate * 100,
                   marker='*', color=colors[row.model], edgecolor='black', s=180)
        ax.annotate(row.policy_id, (row.false_notifications_per_1000_normal_points, row.notification_event_detection_rate * 100), xytext=(4, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel='False notifications / 1,000 Validation normal points', ylabel='Notification EDR (%)', title='VALIDATION selection: 21 fixed policies / family; stars selected')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / 'validation_false_alert_vs_edr.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, model in zip(axes, FAMILIES):
        g = grid.loc[grid.model.eq(model)].sort_values('false_notifications_per_1000_normal_points')
        ax.scatter(g.notification_early25_rate * 100, g.false_notifications_per_1000_normal_points,
                   c=np.where(g.eligible, '#3b906b', '#bbbbbb'), s=55)
        chosen = selection.loc[selection.model.eq(model)].iloc[0]
        ax.scatter(chosen.notification_early25_rate * 100, chosen.false_notifications_per_1000_normal_points, marker='*', color='#d08228', s=160)
        ax.set(title=model, xlabel='Notification early@25 (%)')
    axes[0].set_ylabel('False notifications / 1,000 normal points')
    fig.suptitle('VALIDATION: eligibility and tradeoffs; green eligible, gray rejected')
    fig.tight_layout(); fig.savefig(figures / 'validation_policy_tradeoff.png', dpi=160); plt.close(fig)


def make_test_figures(rows, users, stability, figures):
    means = family_means(rows)
    colors = ['#647b99', '#57a58a']
    x = np.arange(3)
    specs = [('notification_event_detection_rate', 'Notification EDR (%)', 'test_raw_vs_locked_edr', 100),
             ('false_notifications_per_1000_normal_points', 'False notifications / 1,000 normal points', 'test_false_notification_reduction', 1),
             ('notification_early25_rate', 'Notification early@25 (%)', 'test_early_detection_raw_vs_locked', 100)]
    for column, ylabel, name, factor in specs:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for index, evaluation in enumerate(('raw', 'locked')):
            g = means.loc[means.evaluation.eq(evaluation)].set_index('model').loc[list(FAMILIES)]
            std = rows.loc[rows.evaluation.eq(evaluation)].groupby('model')[column].std(ddof=1).reindex(FAMILIES).fillna(0)
            ax.bar(x + (index-.5)*.35, g[column]*factor, .35, yerr=std*factor, capsize=4, label=evaluation.upper(), color=colors[index])
        ax.set(xticks=x, xticklabels=['Rule', 'IF', 'AE'], ylabel=ylabel, title='TEST: frozen raw vs Validation-locked; seed mean/sample std')
        ax.legend(); fig.tight_layout(); fig.savefig(figures / (name+'.png'), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, metric in zip(axes, ['notification_delay_points_median', 'notification_delay_points_p90']):
        for index, evaluation in enumerate(('raw', 'locked')):
            g = means.loc[means.evaluation.eq(evaluation)].set_index('model').loc[list(FAMILIES)]
            ax.bar(x+(index-.5)*.35, g[metric], .35, color=colors[index], label=evaluation.upper())
        ax.set(xticks=x, xticklabels=['Rule','IF','AE'], ylabel=metric.replace('_',' '))
    axes[0].legend(); fig.suptitle('TEST: detected-only delay; miss counts reported separately')
    fig.tight_layout(); fig.savefig(figures / 'test_delay_raw_vs_locked.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, model in zip(axes, FAMILIES):
        g = users.loc[users.model.eq(model)].groupby(['user_id','evaluation']).false_notifications_per_1000_normal_points.mean().unstack()
        positions = np.arange(len(g))
        for index, evaluation in enumerate(('raw','locked')):
            ax.bar(positions+(index-.5)*.35, g[evaluation], .35, color=colors[index], label=evaluation.upper())
        ax.set(xticks=positions, xticklabels=g.index, title=model, xlabel='Test user')
    axes[0].set_ylabel('False notifications / 1,000 normal points'); axes[0].legend()
    fig.suptitle('TEST: one family-wide lock; no user-specific selection')
    fig.tight_layout(); fig.savefig(figures / 'test_per_user_false_alerts.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5,4.6))
    for ax, model in zip(axes, ('isolation_forest','autoencoder')):
        for index, evaluation in enumerate(('raw','locked')):
            g = rows.loc[rows.model.eq(model)&rows.evaluation.eq(evaluation)].sort_values('seed')
            ax.scatter(g.false_notifications_per_1000_normal_points, g.notification_event_detection_rate*100, color=colors[index], label=evaluation.upper())
            for row in g.itertuples(index=False): ax.annotate(str(int(row.seed)),(row.false_notifications_per_1000_normal_points,row.notification_event_detection_rate*100),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(title=model,xlabel='False notifications / 1,000 normal points',ylabel='Notification EDR (%)'); ax.legend()
    fig.suptitle('TEST: locked policy seed stability; five fixed seeds')
    fig.tight_layout(); fig.savefig(figures / 'test_seed_stability.png',dpi=160); plt.close(fig)


def run_validation_selection(root=single.ROOT, data_dir=single.DEFAULT_DATA_DIR):
    started = time.perf_counter()
    root, data_dir = Path(root).resolve(), Path(data_dir).resolve()
    if (root / METRICS / 'validation_policy_grid.csv').exists() or (root / METRICS / 'locked_alert_policies.json').exists():
        raise FileExistsError('Validation selection overwrite refused.')
    snapshot = protect(root)
    config, policies, manifest = frozen_candidates(root)
    # Config manifest already exists before ANY Validation prediction/metric load.
    source_hashes = {name: single.sha256_file(root / name) for name in ['src/alert_policy.py', 'src/audit_alert_aggregation.py']}
    began = now()
    inputs = load_split_inputs(root, data_dir, 'validation')
    prepare_contexts(inputs)
    provenance = {'dataset_split': 'validation', 'started_at_utc': began, 'source_checksums': source_hashes,
                  'prediction_config_checksums': inputs['prediction_config_checksums'],
                  'detector_configs': inputs['detector_configs'], 'normal_point_count': inputs['normal_point_count'],
                  'route_event_count': len(inputs['events']), 'route_evaluated_count': inputs['route_evaluated_count'],
                  'model_training': False, 'frozen_inference': False, 'test_metrics_used': False}
    save(root / METRICS / 'validation_input_provenance.json', provenance)
    rows, event_frames, false_frames = [], [], []
    for detector in inputs['detectors']:
        for policy in policies:
            metrics, events, false, trace = evaluate_detector_policy(detector, policy, 'validation')
            rows.append(metrics); event_frames.append(events); false_frames.append(false)
    seed_grid = pd.DataFrame(rows)
    family_grid, selection = select_validation_policies(seed_grid, config)
    for name, frame in [('validation_policy_seed_grid',seed_grid), ('validation_policy_grid',family_grid),
                        ('validation_policy_selection',selection), ('validation_event_metrics',pd.concat(event_frames,ignore_index=True)),
                        ('validation_false_alert_metrics',pd.concat(false_frames,ignore_index=True)), ('validation_event_catalog',inputs['events'])]:
        frame.to_csv(root / METRICS / (name+'.csv'), index=False)
    lock_path = write_lock(root, selection, provenance, manifest)
    make_validation_figures(family_grid,selection,root/FIGURES)
    protect(root); frozen_candidates(root)
    result = {'stage':'6.7','phase':'validation-select','normal_rows':inputs['normal_point_count'],
              'route_events':len(inputs['events']),'candidate_count_per_family':len(policies),
              'locked_policies':dict(zip(selection.model,selection.policy_id)),
              'candidate_config_sha256':manifest['candidate_config_sha256'],'locked_policy_sha256':single.sha256_file(lock_path),
              'protected_file_count':len(snapshot),'protected_changed_files':0,'runtime_seconds':time.perf_counter()-started,
              'finished_at_utc':now(),'test_metrics_used':False}
    save(root/METRICS/'validation_phase_summary.json',result)
    return result


def run_test_evaluation(root=single.ROOT, data_dir=single.DEFAULT_DATA_DIR, locked_policy=None):
    started = time.perf_counter()
    root, data_dir = Path(root).resolve(), Path(data_dir).resolve()
    if (root/METRICS/'test_raw_vs_locked.csv').exists(): raise FileExistsError('Test application overwrite refused.')
    if locked_policy is None: raise ValueError('Test phase requires explicit locked-policy path.')
    locked = load_locked_policies(root, locked_policy)
    snapshot = protect(root)
    began = now()
    inputs = load_split_inputs(root, data_dir, 'test')
    prepare_contexts(inputs)
    rows, events, false, traces = evaluate_locked_test(inputs, locked)
    regression = validate_test_raw_regression(root, rows, events)
    user_events = event_audit.grouped_summary(events,['model','seed','evaluation','user_id'],summarize_event_rows)
    user_false = event_audit.grouped_summary(false,['model','seed','evaluation','user_id'],summarize_false_rows)
    users = user_events.merge(user_false,on=['model','seed','evaluation','user_id'],validate='one_to_one')
    reduction_rows = reductions(rows)
    stability = seed_stability(rows)
    reduction_stability = event_audit.seed_summary(reduction_rows.drop(columns='policy_id_locked'),[c for c in reduction_rows if c not in ['model','seed','policy_id_locked']])
    suppression_columns = ['model','seed','evaluation','policy_id','event_id','user_id','raw_detected','gate_detected','notification_detected',
                           'suppressed_by_gate','gate_induced_miss','gate_suppression_reasons','suppressed_by_cooldown','cooldown_induced_miss','cooldown_induced_delay',
                           'suppressed_by_pre_event_false_notification','cooldown_suppressed_candidates','carried_gate_run_without_event_notification',
                           'raw_delay_points','gate_delay_points','notification_delay_points','raw_to_notification_delay_increase_points']
    for name, frame in [('test_raw_vs_locked',rows),('test_event_metrics',events),('test_false_alert_metrics',false),
                        ('test_per_user_alert_metrics',users),('test_seed_stability',stability),('test_reduction_metrics',reduction_rows),
                        ('test_reduction_seed_stability',reduction_stability),('test_notification_trace',traces),
                        ('test_event_catalog',inputs['events']),('event_suppression_analysis',events[suppression_columns])]:
        frame.to_csv(root/METRICS/(name+'.csv'),index=False)
    make_test_figures(rows,users,stability,root/FIGURES)
    save(root/METRICS/'test_input_provenance.json',{'dataset_split':'test','started_at_utc':began,
         'prediction_config_checksums':inputs['prediction_config_checksums'],'detector_configs':inputs['detector_configs'],
         'model_training':False,'frozen_inference':False,'policies_locked_before_test_load':True})
    after = load_locked_policies(root,locked_policy)
    if after != locked: raise ValueError('Policy lock changed during Test application.')
    protect(root)
    report = {'stage':'6.7','validation':read_json(root/METRICS/'validation_phase_summary.json'),
              'test_started_at_utc':began,'test_finished_at_utc':now(),'test_normal_rows':inputs['normal_point_count'],
              'test_route_events':len(inputs['events']),'raw_stage66_regression':regression,
              'locked_policies':{m:p.policy_id for m,p in locked.family_policies},
              'candidate_config_sha256':locked.candidate_config_sha256,'locked_policy_sha256':locked.lock_sha256,
              'protected_file_count':len(snapshot),'protected_changed_files':0,
              'test_runtime_seconds':time.perf_counter()-started,'model_training':False,'frozen_inference':False,
              'threshold_changed':False,'model_changed':False,'labels_changed':False,'user_split_changed':False,
              'test_parameter_selection':False,'test_candidate_grid_evaluated':False,'oracle_test_best_computed':False,
              'selection_leakage_checks_passed':True,'policy_family_post_hoc_to_stage66_test':True,'untouched_final_test':False,
              'confirmatory_new_user_cohort_needed':True,'stage67_committed':False,'stage67_pushed':False,
              'pr_created':False,'main_merged':False,'stage68_implemented':False,
              'coverage_note':'notification occurrence coverage and gate-positive coverage distinct; full/evaluable denominators both retained',
              'causal_prefix_note':'only existing unmodified pre-event copy prefix reused after exact eight-feature/GPS/time validation; no combined original/synthetic invented stream, no changed post-event predictions',
              'unknown_prediction_policy':'quality-excluded points unknown; gate history/reset at source holes; cooldown retains trajectory time',
              'output_checksums':{p.name:single.sha256_file(p) for p in sorted((root/METRICS).glob('*')) if p.is_file()},
              'figure_checksums':{p.name:single.sha256_file(p) for p in sorted((root/FIGURES).glob('*.png'))}}
    save(root/METRICS/'alert_policy_audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['validation-select','test-evaluate'])
    parser.add_argument('--output-root',type=Path,default=single.ROOT)
    parser.add_argument('--data-dir',type=Path,default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--locked-policy',type=Path)
    args=parser.parse_args()
    if args.phase=='validation-select':
        if args.locked_policy is not None: parser.error('Validation phase creates, not consumes, a policy lock.')
        result=run_validation_selection(args.output_root,args.data_dir)
    else:
        if args.locked_policy is None: parser.error('Test phase requires --locked-policy.')
        result=run_test_evaluation(args.output_root,args.data_dir,args.locked_policy)
    print(json.dumps(labels.json_safe({k:result[k] for k in result if k in ['phase','normal_rows','route_events','locked_policies','candidate_config_sha256','locked_policy_sha256','runtime_seconds','test_runtime_seconds','protected_file_count','protected_changed_files']}),indent=2))


if __name__=='__main__':
    main()
