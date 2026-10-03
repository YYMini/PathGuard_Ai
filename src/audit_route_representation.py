"""Read-only Stage 6.1 audit: route labels, paired features and spatial identifiability."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from typing import Sequence
import numpy as np
import pandas as pd
from src import train_multiuser_autoencoder as single
from src import evaluate_multi_seed_stability as multi
from src import compare_stage5_baselines as baseline
from src.feature_engineering import EARTH_RADIUS_M, INPUT_COLUMNS, haversine_distance, generate_trajectory_features
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import validate_saved_dataset
from src.train_autoencoder import FEATURE_COLUMNS, add_bearing_features

KEY = ['source_trajectory_id', 'source_point_index']
SCORE_KEY = ['sample_id'] + KEY
QUANTILES = [1, 5, 25, 50, 75, 95, 99]
NUMERICAL_ATOL = 1e-6
NUMERICAL_RTOL = 1e-8


def describe(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Non-finite descriptive input.')
    if not len(values):
        return {'n': 0, **{name: None for name in ['mean', 'median', 'std', 'iqr', 'min', 'max'] + [f'p{q:02}' for q in QUANTILES]}, 'std_ddof': 1, 'na_reason': 'no_rows'}
    qs = np.percentile(values, QUANTILES, method='linear')
    return {'n': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'std': float(values.std(ddof=1)) if len(values) > 1 else None, 'iqr': float(qs[4] - qs[2]),
            'min': float(values.min()), 'max': float(values.max()), **{f'p{q:02}': float(v) for q, v in zip(QUANTILES, qs)},
            'std_ddof': 1, 'na_reason': '' if len(values) > 1 else 'std_requires_two_rows'}


def empirical_percentile(reference, values) -> np.ndarray:
    reference = np.asarray(reference, dtype=float); values = np.asarray(values, dtype=float)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError('Finite nonempty Train reference required.')
    ordered = np.sort(reference)
    low = np.searchsorted(ordered, values, side='left'); high = np.searchsorted(ordered, values, side='right')
    return 100. * (low + high) / (2 * len(ordered))


def robust_reference(train: pd.DataFrame) -> dict:
    baseline.require_train_normal(train)
    values = baseline.feature_values(train); center = np.median(values, axis=0)
    quartiles = np.percentile(values, [25, 75], axis=0); iqr = quartiles[1] - quartiles[0]; std = values.std(axis=0, ddof=0)
    scales = np.where(iqr > 0, iqr / 1.349, np.where(std > 0, std, 1e-9 * np.maximum(1., np.abs(center))))
    return {feature: {'median': float(center[j]), 'scale': float(scales[j]),
                     'p01': float(np.percentile(values[:, j], 1)), 'p99': float(np.percentile(values[:, j], 99)),
                     'fit_source': 'Train original normal only', 'fit_rows': len(train)} for j, feature in enumerate(FEATURE_COLUMNS)}


def align_source(synthetic: pd.DataFrame, originals: pd.DataFrame) -> pd.DataFrame:
    if originals.duplicated(KEY).any() or synthetic.duplicated(SCORE_KEY).any(): raise ValueError('Duplicate original/sample lineage.')
    cols = KEY + ['user_id', 'trajectory_id', 'dataset_split', 'timestamp', 'latitude', 'longitude', 'bearing_deg', 'source_quality_valid'] + FEATURE_COLUMNS
    reference = originals[cols].rename(columns={c: ('paired_trajectory_id' if c == 'trajectory_id' else 'original_' + c) for c in cols if c not in KEY})
    aligned = synthetic.merge(reference, on=KEY, how='left', validate='many_to_one', indicator=True, sort=False)
    if aligned['_merge'].ne('both').any(): raise ValueError('Route sample has missing original lineage.')
    for c in ('user_id', 'trajectory_id', 'dataset_split', 'timestamp', 'source_quality_valid'):
        if not aligned[c].eq(aligned['paired_trajectory_id' if c == 'trajectory_id' else 'original_' + c]).all(): raise ValueError(f'Route original/sample identity mismatch: {c}')
    return aligned.drop(columns='_merge')


def full_route_audit(full_routes: pd.DataFrame, originals: pd.DataFrame, manifest: pd.DataFrame,
                     evaluation_keys: set[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full copies supply neighbors and lag effects, never filtered adjacency."""
    aligned = align_source(full_routes, originals); records, segments = [], []
    metadata = manifest.loc[manifest.anomaly_type.eq('route_deviation')].set_index('sample_id')
    if metadata.index.duplicated().any() or set(metadata.index) != set(aligned.sample_id): raise ValueError('Route manifest/sample mismatch.')
    for sample_id, group in aligned.groupby('sample_id', sort=True):
        g = group.sort_values('source_point_index').reset_index(drop=True).copy()
        row = metadata.loc[sample_id]; labels = g.anomaly_label.eq(1); positions = np.flatnonzero(labels.to_numpy())
        if (not len(positions) or not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1))
                or len(g) != int(row.original_point_count) or labels.sum() != int(row.anomaly_point_count)):
            raise ValueError('Noncontiguous label segment or manifest counts differ.')
        if g.loc[labels, 'timestamp'].iloc[0] != row.anomaly_start_time or g.loc[labels, 'timestamp'].iloc[-1] != row.anomaly_end_time:
            raise ValueError('Route label timestamps differ from manifest.')
        if positions[0] == 0 or positions[-1] == len(g) - 1: raise ValueError('Route labels lack full previous/next neighbors.')
        lat = g.latitude.to_numpy(); lon = g.longitude.to_numpy(); olat = g.original_latitude.to_numpy(); olon = g.original_longitude.to_numpy()
        displacement = haversine_distance(olat, olon, lat, lon)
        prev = np.r_[0., haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])]
        following = np.r_[haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:]), np.nan]
        oprev = np.r_[0., haversine_distance(olat[:-1], olon[:-1], olat[1:], olon[1:])]
        onext = np.r_[haversine_distance(olat[:-1], olon[:-1], olat[1:], olon[1:]), np.nan]
        np.testing.assert_allclose(prev, g.distance_m, rtol=1e-8, atol=1e-5)
        recalculated, removed = generate_trajectory_features(g[INPUT_COLUMNS])
        if removed: raise ValueError('Route feature recheck removed points.')
        recalculated = add_bearing_features(recalculated)
        np.testing.assert_allclose(recalculated[FEATURE_COLUMNS], g[FEATURE_COLUMNS], rtol=1e-7, atol=1e-5)
        delta = g[FEATURE_COLUMNS].to_numpy() - g[['original_' + c for c in FEATURE_COLUMNS]].to_numpy()
        unchanged = np.isclose(g[FEATURE_COLUMNS].to_numpy(), g[['original_' + c for c in FEATURE_COLUMNS]].to_numpy(), rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL).all(axis=1)
        g['displacement_m'] = displacement; g['latitude_delta_deg'] = lat - olat; g['longitude_delta_deg'] = lon - olon
        g['previous_distance_m'] = prev; g['next_distance_m'] = following
        g['original_previous_distance_m'] = oprev; g['original_next_distance_m'] = onext
        g['bearing_change_deg'] = (g.bearing_deg - g.original_bearing_deg + 180.) % 360. - 180.
        g['bearing_change_abs_deg'] = g.bearing_change_deg.abs(); g['feature_near_identical'] = unchanged
        g['zero_displacement'] = displacement <= NUMERICAL_ATOL; g['trajectory_position'] = np.arange(len(g))
        g['trajectory_fraction'] = np.arange(len(g)) / max(1, len(g) - 1); g['segment_position'] = np.arange(len(g)) - positions[0]
        g['segment_point_count'] = len(positions)
        g['segment_duration_sec'] = (g.loc[labels, 'timestamp'].iloc[-1] - g.loc[labels, 'timestamp'].iloc[0]).total_seconds()
        for j, name in enumerate(FEATURE_COLUMNS): g[name + '_delta'] = delta[:, j]
        g['evaluation_included'] = [tuple(key) in evaluation_keys for key in g[SCORE_KEY].itertuples(index=False, name=None)]
        if int(g.loc[labels, 'source_quality_valid'].sum()) != int(row.source_anomaly_quality_valid_rows): raise ValueError('Manifest source quality counts differ.')
        if not g.loc[g.evaluation_included, 'anomaly_label'].eq(1).all(): raise ValueError('Evaluation contains synthetic normal route points.')
        selected = g.loc[labels].copy()
        if selected[['previous_distance_m', 'next_distance_m']].isna().any().any(): raise ValueError('Missing label neighbors.')
        records.append(selected)
        segments.append({'sample_id': sample_id, 'user_id': str(g.user_id.iloc[0]), 'dataset_split': str(g.dataset_split.iloc[0]),
            'source_trajectory_id': str(g.source_trajectory_id.iloc[0]), 'random_seed': int(row.random_seed), 'trajectory_point_count': len(g),
            'segment_start_position': int(positions[0]), 'segment_end_position': int(positions[-1]), 'label_point_count': len(positions),
            'evaluation_point_count': int(selected.evaluation_included.sum()), 'source_quality_excluded_label_count': int((~selected.evaluation_included).sum()),
            'segment_duration_sec': float(selected.segment_duration_sec.iloc[0]), 'zero_displacement_label_count': int(selected.zero_displacement.sum()),
            'near_identical_feature_label_count': int(selected.feature_near_identical.sum()),
            'unlabelled_coordinate_changed_count': int(((~labels) & (displacement > NUMERICAL_ATOL)).sum()),
            'unlabelled_feature_changed_count': int(((~labels) & ~unchanged).sum()), 'max_displacement_m': float(selected.displacement_m.max()),
            'endpoint_zero_displacement_count': int(g.loc[[positions[0], positions[-1]], 'zero_displacement'].sum()),
            'synthetic_speed_gt_50_count': int(selected.speed_mps.gt(50).sum()),
            'synthetic_jump_count': int(((selected.previous_distance_m > 1000) & selected.time_diff_sec.le(60)).sum())})
    return pd.concat(records, ignore_index=True), pd.DataFrame(segments)


def feature_analysis(train: pd.DataFrame, test: pd.DataFrame, samples: pd.DataFrame) -> tuple:
    reference = robust_reference(train)
    route = samples.loc[samples.dataset_split.eq('test') & samples.evaluation_included].copy()
    original = route[['original_' + c for c in FEATURE_COLUMNS]].rename(columns=lambda c: c.removeprefix('original_'))
    cohorts = {'train_normal': train, 'test_normal': test.loc[~test.is_synthetic], 'test_route_deviation': route,
               'paired_test_source_original': original,
               'validation_route_deviation': samples.loc[samples.dataset_split.eq('validation') & samples.evaluation_included]}
    for user, group in route.groupby('user_id', sort=True): cohorts['test_route_user_' + user] = group
    summaries = []
    for cohort, frame in cohorts.items():
        for name in FEATURE_COLUMNS:
            record = describe(frame[name].to_numpy())
            effect = abs(record['median'] - reference[name]['median']) / reference[name]['scale'] if record['n'] else None
            summaries.append({'cohort': cohort, 'feature': name, **record, 'train_robust_scale': reference[name]['scale'],
                              'standardized_median_effect': effect, 'reference_source': 'Train original normal only'})
    percentiles = []
    for name in FEATURE_COLUMNS:
        r = reference[name]; original_p = empirical_percentile(train[name], samples['original_' + name]); synthetic_p = empirical_percentile(train[name], samples[name])
        for j, (_, row) in enumerate(samples.iterrows()):
            percentiles.append({**{c: row[c] for c in SCORE_KEY + ['user_id', 'dataset_split', 'evaluation_included']},
                'feature': name, 'original_value': row['original_' + name], 'synthetic_value': row[name], 'feature_delta': row[name + '_delta'],
                'train_original_percentile': float(original_p[j]), 'train_synthetic_percentile': float(synthetic_p[j]),
                'inside_train_p01_p99': r['p01'] <= row[name] <= r['p99'],
                'paired_absolute_delta_train_scale': abs(row[name + '_delta']) / r['scale']})
    percentile_frame = pd.DataFrame(percentiles); sample_keys = pd.MultiIndex.from_frame(samples[SCORE_KEY])
    matrix = percentile_frame.pivot(index=SCORE_KEY, columns='feature', values='inside_train_p01_p99').reindex(sample_keys)
    samples = samples.copy(); samples['all_8_inside_train_p01_p99'] = matrix.all(axis=1).to_numpy()
    delta_matrix = percentile_frame.pivot(index=SCORE_KEY, columns='feature', values='paired_absolute_delta_train_scale').reindex(sample_keys)
    samples['max_paired_delta_train_scale'] = delta_matrix.max(axis=1).to_numpy()
    return reference, pd.DataFrame(summaries), percentile_frame, samples


def noise_proxy(originals: pd.DataFrame) -> dict:
    """Train-only interpolation residual; GPS ground-truth error is unobserved."""
    all_values, stationary_values = [], []
    for _, frame in originals.loc[originals.dataset_split.eq('train')].groupby('source_trajectory_id', sort=True):
        g = frame.sort_values('source_point_index').reset_index(drop=True)
        if len(g) < 3: continue
        times = g.timestamp.astype('int64').to_numpy() / 1e9
        alpha = (times[1:-1] - times[:-2]) / (times[2:] - times[:-2])
        lat = g.latitude.to_numpy(); lon = np.degrees(np.unwrap(np.radians(g.longitude.to_numpy())))
        predicted_lat = lat[:-2] + alpha * (lat[2:] - lat[:-2]); predicted_lon = lon[:-2] + alpha * (lon[2:] - lon[:-2])
        residual = haversine_distance(lat[1:-1], lon[1:-1], predicted_lat, predicted_lon)
        valid = g.source_quality_valid.to_numpy(); mask = valid[:-2] & valid[1:-1] & valid[2:]
        stationary = mask & (g.distance_m.to_numpy()[1:-1] <= 3) & (g.distance_m.to_numpy()[2:] <= 3)
        stationary &= (g.speed_mps.to_numpy()[1:-1] <= .5) & (g.speed_mps.to_numpy()[2:] <= .5)
        all_values.extend(residual[mask]); stationary_values.extend(residual[stationary])
    return {'train_normal_interpolation_residual_m': describe(all_values),
            'train_stationary_triplet_interpolation_residual_m': describe(stationary_values),
            'interpretation': 'Proxy only: interpolation includes motion, turns, sampling and GPS error; true GPS noise is unobserved.',
            'reference_source': 'Train original normal only'}


def longitude_rotation_probe(original: pd.DataFrame) -> dict:
    """Memory-only Train counterexample; 200m anchor offset is user-specified."""
    original = original.sort_values('timestamp').reset_index(drop=True); lat0 = float(original.latitude.iloc[0])
    angle = 2 * np.arcsin(np.sin(200. / (2 * EARTH_RADIUS_M)) / np.cos(np.radians(lat0)))
    shifted = original[INPUT_COLUMNS].copy(); shifted['longitude'] = (shifted.longitude + np.degrees(angle) + 180.) % 360. - 180.
    first, removed1 = generate_trajectory_features(original[INPUT_COLUMNS]); second, removed2 = generate_trajectory_features(shifted)
    if removed1 or removed2: raise ValueError('Rotation probe time ordering invalid.')
    first = add_bearing_features(first); second = add_bearing_features(second)
    displacement = haversine_distance(original.latitude, original.longitude, shifted.latitude, shifted.longitude)
    delta = np.abs(first[FEATURE_COLUMNS].to_numpy() - second[FEATURE_COLUMNS].to_numpy())
    return {'source_trajectory_id': str(original.source_trajectory_id.iloc[0]), 'source_split': str(original.dataset_split.iloc[0]),
            'row_count': len(original), 'longitude_rotation_deg': float(np.degrees(angle)), 'anchor_displacement_m': float(displacement[0]),
            'displacement_m': describe(displacement), 'max_absolute_feature_delta': {name: float(delta[:, j].max()) for j, name in enumerate(FEATURE_COLUMNS)},
            'all_8_equal_within_numerical_tolerance': bool(np.allclose(first[FEATURE_COLUMNS], second[FEATURE_COLUMNS], rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)),
            'absolute_location_or_route_reference_in_model_features': False,
            'interpretation': 'Longitude rotation is a spherical isometry. Same eight local movement features do not identify absolute location or departure from an external habitual route.',
            'models_executed': False, 'dataset_written': False}


def load_saved_route_scores(root: Path, dataset_id: str, test: pd.DataFrame, samples: pd.DataFrame) -> tuple:
    """Stored scores and decisions only; no fitting/inference/percentile selection."""
    eval_samples = samples.loc[samples.dataset_split.eq('test') & samples.evaluation_included]
    keys = set(eval_samples[SCORE_KEY].itertuples(index=False, name=None)); records = []
    for model, seeds in [('statistical_rule', [None]), ('isolation_forest', baseline.MODEL_SEEDS), ('autoencoder', baseline.MODEL_SEEDS)]:
        for seed in seeds:
            if model == 'autoencoder':
                dirs = single.output_directories(root, dataset_id, seed)
                frame = read_dataset_csv(dirs['metrics'] / 'test_predictions.csv').rename(columns={'reconstruction_error': 'anomaly_score'})
                config = json.loads((dirs['model'] / 'training_config.json').read_text(encoding='utf-8'))
            else:
                suffix = Path(model) if seed is None else Path(model) / f'seed_{seed}'
                frame = baseline.read_baseline_predictions(root / 'outputs/metrics/stage5' / dataset_id / 'baseline_comparison' / suffix / 'test_predictions.csv')
                config = json.loads((root / 'models/stage5' / dataset_id / 'baselines' / suffix / 'run_config.json').read_text(encoding='utf-8'))
            baseline.assert_same_rows(frame, test); threshold = float(config['threshold'])
            np.testing.assert_array_equal(frame.predicted_anomaly, (frame.anomaly_score > threshold).astype(int))
            route = frame.loc[frame.is_synthetic & frame.anomaly_label.eq(1) & frame.anomaly_type.eq('route_deviation')]
            if set(route[SCORE_KEY].itertuples(index=False, name=None)) != keys: raise ValueError('Stored score/route lineage mismatch.')
            scored = route[SCORE_KEY + ['user_id', 'anomaly_score', 'predicted_anomaly']].copy()
            scored.insert(0, 'model', model); scored.insert(1, 'seed', seed); scored['threshold'] = threshold; records.append(scored)
    scores = pd.concat(records, ignore_index=True)
    scores = scores.merge(eval_samples[SCORE_KEY + ['displacement_m', 'max_paired_delta_train_scale', 'feature_near_identical']], on=SCORE_KEY, how='left', validate='many_to_one')
    seed_rows = []
    for (model, seed), group in scores.groupby(['model', 'seed'], dropna=False, sort=True):
        for user, g in [('ALL', group)] + list(group.groupby('user_id', sort=True)):
            stats = describe(g.anomaly_score)
            seed_rows.append({'model': model, 'seed': seed, 'user_id': user, 'route_row_count': len(g),
                              'detected_count': int(g.predicted_anomaly.sum()), 'recall': float(g.predicted_anomaly.mean()),
                              **{'score_' + k: stats[k] for k in ('mean', 'median', 'std', 'p95', 'p99', 'min', 'max')}})
    seeds = pd.DataFrame(seed_rows)
    summary = multi.summarize_values(seeds, ['model', 'user_id'], ['recall', 'detected_count', 'score_mean', 'score_median', 'score_p95', 'score_p99'])
    old = pd.read_csv(root / 'outputs/metrics/stage5' / dataset_id / 'baseline_comparison/per_anomaly_type_baseline_metrics.csv')
    old = old.loc[old.anomaly_type.eq('route_deviation')]
    for row in seeds.loc[seeds.user_id.eq('ALL')].itertuples():
        matches = old.loc[old.model.eq(row.model) & (old.seed.isna() if pd.isna(row.seed) else old.seed.eq(row.seed))]
        if len(matches) != 1 or int(matches.iloc[0].anomaly_row_count) != row.route_row_count or int(matches.iloc[0].detected_anomaly_row_count) != row.detected_count: raise ValueError('Route detection differs from Stage 5.4.')
        np.testing.assert_allclose(matches.iloc[0].recall, row.recall, rtol=1e-12)
    return scores, seeds, summary


def user_summary(samples: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user, g in samples.loc[samples.dataset_split.eq('test') & samples.evaluation_included].groupby('user_id', sort=True):
        for model, runs in seeds.loc[seeds.user_id.eq(user)].groupby('model', sort=True):
            scores = runs.score_mean
            rows.append({'user_id': str(user), 'model': model, 'route_row_count': len(g), 'segment_count': g.sample_id.nunique(),
                'run_count': len(runs), 'recall_mean': float(runs.recall.mean()), 'recall_sample_std': float(runs.recall.std(ddof=1)) if len(runs) > 1 else None,
                'recall_min': float(runs.recall.min()), 'recall_max': float(runs.recall.max()), 'detected_count_mean': float(runs.detected_count.mean()),
                'score_mean': float(scores.mean()), 'score_mean_sample_std': float(scores.std(ddof=1)) if len(runs) > 1 else None,
                **{'displacement_' + k: v for k, v in describe(g.displacement_m).items() if k in ('mean', 'median', 'p95', 'p99', 'min', 'max')},
                'zero_displacement_rows': int(g.zero_displacement.sum()), 'near_identical_feature_rows': int(g.feature_near_identical.sum()),
                'all_8_inside_train_p01_p99_rows': int(g.all_8_inside_train_p01_p99.sum())})
    return pd.DataFrame(rows)


def create_figures(samples: pd.DataFrame, summaries: pd.DataFrame, scores: pd.DataFrame,
                   users: pd.DataFrame, train: pd.DataFrame, directory: Path) -> list[Path]:
    plt = single.plt; paths = []; route = samples.loc[samples.dataset_split.eq('test') & samples.evaluation_included]
    def save(fig, name):
        fig.tight_layout(); path = directory / name; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.hist(route.displacement_m, bins=30)
    ax.set(xlabel='Displacement (m)', ylabel='Test labelled points', title=f'Route displacement: {len(route)} evaluation rows'); save(fig, 'route_displacement_distribution.png')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for name, ax in zip(FEATURE_COLUMNS, axes.ravel()):
        ax.ecdf(train[name], label='Train normal'); ax.ecdf(route[name], label='Test route'); ax.ecdf(route['original_' + name], label='Paired original')
        if name not in ('bearing_sin', 'bearing_cos'): ax.set_xscale('symlog', linthresh=1)
        ax.set(xlabel=name, ylabel='ECDF'); ax.legend(fontsize=7)
    save(fig, 'normal_route_feature_distributions.png')
    frame = summaries.loc[summaries.cohort.eq('test_route_deviation')].set_index('feature').reindex(FEATURE_COLUMNS)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(FEATURE_COLUMNS, frame.standardized_median_effect)
    ax.tick_params(axis='x', rotation=30); ax.set(ylabel='Absolute median difference / Train robust scale', title='Descriptive effect; no classifier or threshold tuning'); save(fig, 'feature_effect_size_comparison.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for model, ax in zip(baseline.MODEL_NAMES, axes):
        for seed, g in scores.loc[scores.model.eq(model)].groupby('seed', dropna=False):
            ax.ecdf(g.anomaly_score, label='single' if pd.isna(seed) else str(int(seed))); ax.axvline(g.threshold.iloc[0], alpha=.3)
        if model != 'isolation_forest': ax.set_xscale('symlog', linthresh=.01)
        ax.set(xlabel='Stored anomaly score', ylabel='Route ECDF', title=model); ax.legend(fontsize=7)
    save(fig, 'route_score_distributions_by_model.png')
    ids = sorted(users.user_id.unique()); x = np.arange(len(ids)); width = .25; fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(baseline.MODEL_NAMES):
        g = users.loc[users.model.eq(model)].set_index('user_id').reindex(ids)
        ax.bar(x + (i - 1) * width, g.recall_mean, width, yerr=None if model == 'statistical_rule' else g.recall_sample_std, capsize=3, label=model)
    ax.set_xticks(x, ids); ax.set(ylabel='Route Recall', xlabel='Test user', ylim=(0, 1.05)); ax.legend(); save(fig, 'route_recall_by_user.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for model, ax in zip(baseline.MODEL_NAMES, axes):
        g = scores.loc[scores.model.eq(model)]; avg = g.groupby(SCORE_KEY, sort=False)[['displacement_m', 'anomaly_score']].mean()
        ax.scatter(avg.displacement_m, avg.anomaly_score, s=10, alpha=.5)
        if model != 'isolation_forest': ax.set_yscale('symlog', linthresh=.01)
        ax.set(xlabel='Displacement (m)', ylabel='Stored score (seed mean)', title=model)
    save(fig, 'displacement_vs_anomaly_score.png')
    return paths


def assert_protected(root: Path, expected: dict) -> None:
    for relative, hashes in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError('Unsafe protected root.')
        multi.assert_snapshot(path, hashes)


def run_audit(data_dir: Path = single.DEFAULT_DATA_DIR, output_root: Path = single.ROOT,
              protected_snapshot_path: Path | None = None) -> dict:
    started = time.perf_counter(); data_dir = Path(data_dir).resolve(); root = Path(output_root).resolve()
    summary, users = single.read_stage5_metadata(data_dir); dataset_id = summary['dataset_id']; single.output_directories(root, dataset_id, 42)
    metrics_dir = root / 'outputs/metrics/stage6' / dataset_id / 'route_audit'; figure_dir = root / 'outputs/figures/stage6' / dataset_id / 'route_audit'
    if metrics_dir.exists() or figure_dir.exists(): raise FileExistsError('Stage 6 audit outputs exist; no overwrite allowed.')
    snapshot_path = protected_snapshot_path or root / 'outputs/metrics/stage54_frozen_artifacts.json'
    expected = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
    required = [data_dir, root / 'models/stage5' / dataset_id, root / 'outputs/metrics/stage5' / dataset_id, root / 'outputs/figures/stage5' / dataset_id]
    if set(expected) != {str(p.relative_to(root)) for p in required}: raise ValueError('Protected root set differs.')
    assert_protected(root, expected); integrity = validate_saved_dataset(data_dir)
    if not integrity['ready_for_stage5_b']: raise ValueError('Dataset integrity failed.')
    frames = {k: single.load_stage5_split(data_dir, k, summary, users) for k in ('train', 'validation', 'test')}
    originals = add_bearing_features(read_dataset_csv(data_dir / 'source_normal.csv'))
    full_routes = pd.concat([read_dataset_csv(data_dir / (s + '.csv')).loc[lambda f: f.is_synthetic & f.anomaly_type.eq('route_deviation')] for s in ('validation', 'test')], ignore_index=True)
    full_routes = add_bearing_features(full_routes)
    evaluation = pd.concat([frames['validation'], frames['test']]); evaluation = evaluation.loc[evaluation.is_synthetic & evaluation.anomaly_type.eq('route_deviation') & evaluation.anomaly_label.eq(1)]
    evaluation_keys = set(evaluation[SCORE_KEY].itertuples(index=False, name=None))
    samples, segments = full_route_audit(full_routes, originals, read_dataset_csv(data_dir / 'synthetic_anomaly_manifest.csv'), evaluation_keys)
    if set(samples.loc[samples.evaluation_included, SCORE_KEY].itertuples(index=False, name=None)) != evaluation_keys: raise ValueError('Missing route evaluation sample.')
    reference, feature_summary, percentiles, samples = feature_analysis(frames['train'], frames['test'], samples)
    scores, seeds, seed_summary = load_saved_route_scores(root, dataset_id, frames['test'], samples); user_stats = user_summary(samples, seeds)
    source_id = sorted(originals.loc[originals.dataset_split.eq('train'), 'source_trajectory_id'].unique())[0]
    probe = longitude_rotation_probe(originals.loc[originals.source_trajectory_id.eq(source_id)]); noise = noise_proxy(originals)
    selected = samples.loc[samples.dataset_split.eq('test') & samples.evaluation_included]
    acceleration_p99 = float(np.percentile(frames['train'].acceleration_mps2.abs(), 99))
    metrics_dir.mkdir(parents=True); figure_dir.mkdir(parents=True)
    tables = {'route_deviation_samples': samples, 'route_segment_summary': segments, 'route_feature_summary': feature_summary,
              'route_feature_percentiles': percentiles, 'route_user_summary': user_stats, 'route_model_scores': scores,
              'route_seed_metrics': seeds, 'route_seed_summary': seed_summary}
    for name, table in tables.items(): table.to_csv(metrics_dir / (name + '.csv'), index=False, na_rep='NA')
    figures = create_figures(samples, feature_summary, scores, user_stats, frames['train'], figure_dir)
    report = {'stage': '6.1', 'name': 'Route Representation & Synthetic Label Audit', 'dataset_id': dataset_id,
        'analysis_only': True, 'feature_columns': FEATURE_COLUMNS, 'train_reference': reference,
        'source_code_sha256': {name: single.sha256_file(single.ROOT / 'src' / name) for name in ['synthetic_anomalies.py', 'feature_engineering.py', 'train_autoencoder.py', 'prepare_multiuser_dataset.py']},
        'generation_rule': {'point_selection': 'Contiguous random 10..30 points; start>=1, excludes trajectory endpoints.',
            'amplitude_m': [100, 300], 'shape': 'sin(linspace(0,pi,size)), indexed by point not elapsed time',
            'offset_direction': 'Perpendicular to start/end chord in local north/east coordinates',
            'latitude_delta': '-cos(heading)*amplitude*weight / 111320',
            'longitude_delta': 'sin(heading)*amplitude*weight / (111320*cos(updated latitude))',
            'timestamps_modified': False, 'feature_recalculation': 'Entire transformed copy; bearing sin/cos added at model input.',
            'label_scope': 'All selected points including zero-offset endpoints.',
            'source_quality_policy': 'Original quality flags inherited; synthetic physical plausibility not re-filtered.'},
        'counts': {'route_segments_all_splits': len(segments), 'labelled_points_all_splits': len(samples),
            'test_segments': int(segments.dataset_split.eq('test').sum()), 'test_labelled_points_before_quality': int(samples.dataset_split.eq('test').sum()),
            'test_evaluation_route_points': len(selected), 'test_source_quality_excluded_labels': int((samples.dataset_split.eq('test') & ~samples.evaluation_included).sum()),
            'test_zero_displacement_labels': int(selected.zero_displacement.sum()), 'test_near_identical_feature_labels': int(selected.feature_near_identical.sum()),
            'test_all_8_inside_train_p01_p99_rows': int(selected.all_8_inside_train_p01_p99.sum()),
            'test_unlabelled_feature_changed_rows_full_copies': int(segments.loc[segments.dataset_split.eq('test'), 'unlabelled_feature_changed_count'].sum())},
        'test_displacement_m': describe(selected.displacement_m), 'test_segment_duration_sec': describe(segments.loc[segments.dataset_split.eq('test'), 'segment_duration_sec']),
        'test_paired_max_scaled_feature_delta': describe(selected.max_paired_delta_train_scale),
        'physical_review': {'speed_gt_50_count': int(selected.speed_mps.gt(50).sum()), 'original_speed_gt_50_count': int(selected.original_speed_mps.gt(50).sum()),
            'jump_distance_gt_1000_dt_le_60_count': int(((selected.previous_distance_m > 1000) & selected.time_diff_sec.le(60)).sum()),
            'abs_acceleration_gt_train_p99_count': int(selected.acceleration_mps2.abs().gt(acceleration_p99).sum()),
            'train_abs_acceleration_p99': acceleration_p99, 'synthetic_speed_mps': describe(selected.speed_mps), 'original_speed_mps': describe(selected.original_speed_mps),
            'synthetic_abs_acceleration_mps2': describe(selected.acceleration_mps2.abs()),
            'interpretation': 'Existing quality speed/jump flags and Train acceleration extremes are review indicators, not behavior detectors or universal physical bounds.'},
        'normal_noise_proxy': noise, 'spatial_information_probe': probe,
        'model_reuse': {'training': False, 'inference': False, 'threshold_recalculated': False, 'scaler_created': False,
                       'model_seeds': list(baseline.MODEL_SEEDS), 'scores': 'Read existing Stage 5 CSVs only'},
        'fit_or_tuning_on_test': False, 'candidate_features_added_to_pipeline': False,
        'percentile_definition': 'Train empirical midrank; p01..p99 are descriptive marginal intervals, not joint normality or classifier thresholds.',
        'std_ddof': 1, 'figures': [str(p) for p in figures], 'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['True GPS noise and habitual normal routes of unseen Test users are unobserved.',
            'Synthetic source alignment is an audit oracle, unavailable as a production route reference.',
            'Marginal overlap and translation invariance do not prove all route anomalies have no usable signal.']}
    assert_protected(root, expected)
    verification = {'stage': '6.1', 'passed': True, 'dataset_immutable': True, 'all_stage5_artifacts_immutable': True,
        'protected_file_count': sum(len(x) for x in expected.values()), 'protected_root_checksums': expected,
        'lineage_aligned': True, 'route_manifest_label_counts_and_timestamps_verified': True,
        'route_features_recomputed_from_saved_coordinates': True, 'scores_and_route_recalls_match_stage54': True,
        'cross_split_leakage': 0, 'feature_schema_unchanged': True, 'no_model_train_infer_threshold_or_scaler': True,
        'no_test_driven_tuning': True, 'no_production_candidate_feature_added': True,
        'output_checksums': {name + '.csv': single.sha256_file(metrics_dir / (name + '.csv')) for name in tables},
        'figure_checksums': {p.name: single.sha256_file(p) for p in figures}}
    single.save_json(metrics_dir / 'synthetic_route_audit.json', report)
    verification['output_checksums']['synthetic_route_audit.json'] = single.sha256_file(metrics_dir / 'synthetic_route_audit.json')
    single.save_json(metrics_dir / 'stage6_1_verification.json', verification)
    print(json.dumps({'counts': report['counts'], 'physical_review': report['physical_review'], 'probe': probe,
                      'elapsed_seconds': report['elapsed_seconds']}, indent=2), flush=True)
    return {'report': report, 'verification': verification, 'tables': tables, 'metrics_dir': metrics_dir, 'figure_dir': figure_dir}


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Read-only route representation and synthetic label audit.')
    parser.add_argument('--data-dir', type=Path, default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=single.ROOT)
    parser.add_argument('--protected-snapshot', type=Path)
    options = parser.parse_args(args); run_audit(options.data_dir, options.output_root, options.protected_snapshot)


if __name__ == '__main__': main()
