"""Stage 6.2: frozen Train-only spatial coverage; no model fitting or selection."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import BallTree
from src import audit_route_representation as previous
from src import train_multiuser_autoencoder as single
from src.feature_engineering import EARTH_RADIUS_M, haversine_distance
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import validate_saved_dataset

COVERAGE_RADII_M = (10, 25, 50, 100, 200, 500, 1000)
ABSTENTION_RADII_M = (25, 50, 100, 200, 500)
KEY = previous.KEY
SCORE_KEY = previous.SCORE_KEY
DISTANCE = 'nearest_train_reference_distance_m'
META = ['user_id', 'trajectory_id', 'source_point_index']


def coordinates(frame: pd.DataFrame) -> np.ndarray:
    values = frame[['latitude', 'longitude']].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or np.any(np.abs(values[:, 0]) > 90)
            or np.any(np.abs(values[:, 1]) > 180)):
        raise ValueError('Finite valid latitude/longitude required.')
    return np.radians(values)


def finite(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite distance or score.')
    return result


def distance_summary(values) -> dict:
    values = finite(values)
    if not len(values):
        return {'count': 0, **{k: None for k in ('mean', 'median', 'min', 'max',
                'p25', 'p50', 'p75', 'p90', 'p95', 'p99')}, 'na_reason': 'no_rows'}
    return {'count': len(values), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'min': float(values.min()),
            'max': float(values.max()), **{f'p{q}': float(np.percentile(values, q))
            for q in (25, 50, 75, 90, 95, 99)}, 'na_reason': ''}


def separation(normal, anomaly) -> dict:
    normal, anomaly = finite(normal), finite(anomaly)
    if not len(normal) or not len(anomaly):
        return {'roc_auc': None, 'average_precision': None,
                'na_reason': 'both_classes_required', 'normal_count': len(normal),
                'anomaly_count': len(anomaly), 'prevalence': None}
    labels = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]
    scores = np.r_[normal, anomaly]
    return {'roc_auc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores)),
            'na_reason': '', 'normal_count': len(normal), 'anomaly_count': len(anomaly),
            'prevalence': float(labels.mean())}


def validate_reference(train: pd.DataFrame, users: pd.DataFrame,
                       trajectories: pd.DataFrame) -> dict:
    """Fail closed on forbidden rows, manifest identity, or duplicate sources."""
    if train.empty or users.user_id.duplicated().any() or trajectories.trajectory_id.duplicated().any():
        raise ValueError('Empty reference or ambiguous manifest.')
    for column in ('is_synthetic', 'is_low_quality', 'source_quality_valid',
                   'is_training_eligible', 'synthetic_value_valid'):
        if not pd.api.types.is_bool_dtype(train[column]):
            raise ValueError('Reference quality flags must be boolean.')
    if (not train.dataset_split.eq('train').all() or train.is_synthetic.any()
            or not train.anomaly_label.eq(0).all()):
        raise ValueError('Reference requires Train original normal only.')
    if (train.is_low_quality.any() or not train.source_quality_valid.all()
            or not train.is_training_eligible.all() or not train.synthetic_value_valid.all()):
        raise ValueError('Low-quality reference row.')
    if train.duplicated(KEY).any() or not train.trajectory_id.eq(train.source_trajectory_id).all():
        raise ValueError('Duplicate or invalid reference lineage.')
    allowed_users = set(users.loc[users.dataset_split.eq('train'), 'user_id'])
    if not set(train.user_id).issubset(allowed_users):
        raise ValueError('Validation/Test/unknown user in Train reference.')
    eligible = trajectories.loc[trajectories.dataset_split.eq('train') &
                                trajectories.eligible_for_dataset].copy()
    if eligible.is_exact_duplicate.any() or not eligible.quality_eligible_for_dataset.all():
        raise ValueError('Duplicate or low-quality trajectory marked eligible.')
    if eligible.content_fingerprint.eq('').any() or eligible.content_fingerprint.duplicated().any():
        raise ValueError('Exact duplicate trajectory fingerprint in reference.')
    if set(train.source_trajectory_id) != set(eligible.trajectory_id):
        raise ValueError('Excluded duplicate/trajectory or missing eligible Train source.')
    identity = train[['source_trajectory_id', 'user_id']].drop_duplicates()
    expected = eligible.set_index('trajectory_id').user_id
    if not identity.user_id.eq(identity.source_trajectory_id.map(expected)).all():
        raise ValueError('Reference source/user identity mismatch.')
    coordinates(train)
    return {'row_count': len(train), 'user_count': train.user_id.nunique(),
            'trajectory_count': train.source_trajectory_id.nunique(),
            'user_ids': sorted(train.user_id.unique().tolist()),
            'low_quality_rows': 0, 'synthetic_rows': 0, 'duplicate_trajectories': 0,
            'validation_or_test_users': 0, 'reference_leakage': 0,
            'repeated_coordinate_rows': int(train.duplicated(['latitude', 'longitude']).sum())}


@dataclass
class SpatialReference:
    rows: pd.DataFrame
    tree: BallTree
    canonical: np.ndarray
    audit: dict

    @classmethod
    def build(cls, train, users, trajectories):
        audit = validate_reference(train, users, trajectories)
        rows = train.sort_values(['user_id'] + KEY, kind='stable').reset_index(drop=True).copy()
        # Keep every row; select lexical metadata for identical coordinates.
        keys = list(zip(rows.latitude, rows.longitude))
        first = {}
        canonical = np.array([first.setdefault(key, i) for i, key in enumerate(keys)])
        return cls(rows, BallTree(coordinates(rows), metric='haversine', leaf_size=40), canonical, audit)

    def query(self, frame):
        result = frame.reset_index(drop=True).copy()
        if frame.empty:
            for name in META: result['nearest_train_' + name] = pd.Series(dtype='object')
            result[DISTANCE] = pd.Series(dtype=float)
            return result
        angles, indices = self.tree.query(coordinates(frame), k=1)
        indices = self.canonical[indices[:, 0]]
        for name in META:
            result['nearest_train_' + name] = self.rows.iloc[indices][name].to_numpy()
        result[DISTANCE] = angles[:, 0] * EARTH_RADIUS_M
        finite(result[DISTANCE])
        return result

def source_matched(route: pd.DataFrame, normal: pd.DataFrame,
                   reference: SpatialReference) -> pd.DataFrame:
    if normal.duplicated(KEY).any() or route.duplicated(SCORE_KEY).any():
        raise ValueError('Ambiguous source pairing.')
    if (not route.is_synthetic.all() or not route.anomaly_label.eq(1).all()
            or not route.anomaly_type.eq('route_deviation').all()):
        raise ValueError('Route labels required.')
    cols = KEY + ['user_id', 'trajectory_id', 'dataset_split', 'timestamp',
                  'latitude', 'longitude', DISTANCE] + single.FEATURE_COLUMNS
    originals = normal[cols].rename(columns={c: ('paired_trajectory_id' if c == 'trajectory_id' else 'original_' + c) for c in cols if c not in KEY})
    pairs = reference.query(route).merge(originals, on=KEY, how='left',
                                        validate='many_to_one', indicator=True, sort=False)
    if pairs['_merge'].ne('both').any():
        raise ValueError('Missing normal source lineage.')
    for name in ('user_id', 'trajectory_id', 'dataset_split', 'timestamp'):
        if not pairs[name].eq(pairs['paired_trajectory_id' if name == 'trajectory_id' else 'original_' + name]).all():
            raise ValueError('Source identity mismatch: ' + name)
    pairs = pairs.drop(columns='_merge').rename(columns={
        DISTANCE: 'synthetic_reference_distance_m',
        'original_' + DISTANCE: 'original_reference_distance_m'})
    pairs['delta_reference_distance_m'] = pairs.synthetic_reference_distance_m - pairs.original_reference_distance_m
    pairs['actual_displacement_m'] = haversine_distance(pairs.original_latitude,
                        pairs.original_longitude, pairs.latitude, pairs.longitude)
    pairs['feature_near_identical'] = np.isclose(pairs[single.FEATURE_COLUMNS].to_numpy(dtype=float),
        pairs[['original_' + c for c in single.FEATURE_COLUMNS]].to_numpy(dtype=float),
        atol=previous.NUMERICAL_ATOL, rtol=previous.NUMERICAL_RTOL).all(axis=1)
    pairs['zero_displacement'] = pairs.actual_displacement_m.le(previous.NUMERICAL_ATOL)
    # Distance to a set is 1-Lipschitz under this spherical metric.
    if (pairs.delta_reference_distance_m.abs() > pairs.actual_displacement_m + 1e-5).any():
        raise ValueError('Reference delta exceeds actual displacement.')
    pairs['configured_amplitude_m'] = np.nan
    pairs['amplitude_na_reason'] = 'Not stored in saved manifest; no regeneration performed.'
    return pairs


def coverage(normal, radii=COVERAGE_RADII_M) -> pd.DataFrame:
    distances = finite(normal[DISTANCE])
    return pd.DataFrame([{'radius_m': r, 'normal_count': len(normal),
        'normal_covered_count': int((distances <= r).sum()),
        'normal_coverage_pct': float(100 * (distances <= r).mean()) if len(normal) else None}
        for r in radii])


def abstention(normal, pairs) -> pd.DataFrame:
    records = []
    for radius in ABSTENTION_RADII_M:
        n = normal.loc[normal[DISTANCE].le(radius), DISTANCE]
        a = pairs.loc[pairs.original_reference_distance_m.le(radius), 'synthetic_reference_distance_m']
        records.append({'radius_m': radius, 'normal_total': len(normal), 'route_total': len(pairs),
            'normal_covered_count': len(n), 'route_source_covered_count': len(a),
            'normal_coverage_pct': 100 * len(n) / len(normal) if len(normal) else None,
            'route_source_coverage_pct': 100 * len(a) / len(pairs) if len(pairs) else None,
            'normal_abstained_count': len(normal) - len(n),
            'route_source_abstained_count': len(pairs) - len(a), **separation(n, a)})
    return pd.DataFrame(records)


def delta_summary(pairs) -> dict:
    values = finite(pairs.delta_reference_distance_m)
    return {**distance_summary(values), **{name: float(mask.mean()) if len(values) else None
        for name, mask in [('delta_gt_0_fraction', values > 0)] +
        [(f'delta_ge_{r}m_fraction', values >= r) for r in (10, 25, 50, 100, 200)]}}


def user_analysis(normal, pairs) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, conditions = [], []
    for user, n in normal.groupby('user_id', sort=True):
        p = pairs.loc[pairs.user_id.eq(user)]
        row = {'user_id': str(user), **{'normal_' + k: v for k, v in distance_summary(n[DISTANCE]).items()},
               'route_count': len(p), 'route_distance_median': float(p.synthetic_reference_distance_m.median()) if len(p) else None,
               'source_matched_delta_median': float(p.delta_reference_distance_m.median()) if len(p) else None,
               **{'full_' + k: v for k, v in separation(n[DISTANCE], p.synthetic_reference_distance_m).items()}}
        for radius in COVERAGE_RADII_M:
            row[f'coverage_{radius}m_pct'] = 100 * float(n[DISTANCE].le(radius).mean())
        records.append(row)
        group = abstention(n, p); group.insert(0, 'user_id', str(user)); conditions.append(group)
    return pd.DataFrame(records), pd.concat(conditions, ignore_index=True)


def nearest_user_frequencies(normal) -> pd.DataFrame:
    counts = normal.groupby(['user_id', 'nearest_train_user_id'], sort=True).size().rename('point_count').reset_index()
    totals = normal.groupby('user_id').size()
    counts['fraction'] = counts.point_count / counts.user_id.map(totals)
    return counts


def verify_stage61_boundary(pairs, path):
    """Cross-check frozen audit results by keys rather than positional alignment."""
    saved = read_dataset_csv(path)
    saved = saved.loc[saved.dataset_split.eq('test') & saved.evaluation_included.astype(str).str.lower().eq('true')]
    if saved.duplicated(SCORE_KEY).any(): raise ValueError('Duplicate Stage 6.1 sample.')
    columns = SCORE_KEY + ['displacement_m', 'zero_displacement', 'feature_near_identical']
    checked = pairs.merge(saved[columns], on=SCORE_KEY, how='outer', validate='one_to_one',
                          suffixes=('', '_stage61'), indicator=True)
    if checked['_merge'].ne('both').any(): raise ValueError('Stage 6.1 route sample mismatch.')
    np.testing.assert_allclose(checked.actual_displacement_m, checked.displacement_m, atol=1e-6, rtol=1e-8)
    for name in ('zero_displacement', 'feature_near_identical'):
        flags = checked[name + '_stage61'].astype(str).str.lower().eq('true')
        if not checked[name].eq(flags).all(): raise ValueError('Stage 6.1 boundary flags differ.')


def create_figures(normal, pairs, cover, per_user, abstain, directory):
    paths = []
    def save(name):
        path = directory / name
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(); paths.append(path)
    def ecdf(values, label):
        x = np.sort(finite(values)); plt.plot(x, np.arange(1, len(x) + 1) / len(x), label=label)
    plt.figure(figsize=(8, 5))
    ecdf(normal[DISTANCE], 'Test original normal'); ecdf(pairs.synthetic_reference_distance_m, 'Test route deviation')
    plt.xscale('symlog', linthresh=1); plt.xlabel('Nearest Train point distance (m; symlog)'); plt.ylabel('ECDF')
    plt.legend(); plt.title('Frozen Test spatial distance distributions; no rows removed')
    save('normal_vs_route_distance_distribution.png')
    plt.figure(figsize=(8, 5)); plt.hist(pairs.delta_reference_distance_m, bins=45)
    plt.axvline(0, color='black', linestyle='--'); plt.xlabel('Synthetic minus source distance (m)'); plt.ylabel('Route points')
    plt.title('Source-matched reference distance delta'); save('source_matched_delta_distribution.png')
    plt.figure(figsize=(8, 5)); plt.plot(cover.radius_m, cover.normal_coverage_pct, 'o-')
    plt.xscale('log'); plt.xlabel('Coverage radius (m)'); plt.ylabel('Test normal covered (%)'); plt.ylim(0, 100)
    plt.title('Train-only point reference coverage'); save('normal_coverage_by_radius.png')
    plt.figure(figsize=(8, 5))
    for _, row in per_user.iterrows():
        plt.plot(COVERAGE_RADII_M, [row[f'coverage_{r}m_pct'] for r in COVERAGE_RADII_M], 'o-', label=str(row.user_id))
    plt.xscale('log'); plt.ylim(0, 100); plt.xlabel('Coverage radius (m)'); plt.ylabel('Test normal covered (%)')
    plt.legend(title='Test user'); plt.title('User-level geographic coverage'); save('coverage_by_user.png')
    plt.figure(figsize=(7, 5)); labels = np.r_[np.zeros(len(normal)), np.ones(len(pairs))]
    fpr, tpr, _ = roc_curve(labels, np.r_[normal[DISTANCE], pairs.synthetic_reference_distance_m])
    plt.plot(fpr, tpr, label='Full Test: spatial distance only')
    plt.plot([0, 1], [0, 1], '--', color='gray'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.legend(); plt.title('Descriptive ROC; no decision threshold selected'); save('spatial_distance_roc.png')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(abstain.radius_m, abstain.normal_coverage_pct, 'o-', label='Normal coverage')
    ax[0].plot(abstain.radius_m, abstain.route_source_coverage_pct, 's-', label='Anomaly source coverage')
    ax[0].set(xscale='log', xlabel='Source coverage radius (m)', ylabel='Covered (%)', ylim=(0, 100)); ax[0].legend()
    ax[1].plot(abstain.normal_coverage_pct, abstain.roc_auc, 'o-', label='Covered ROC-AUC')
    ax[1].plot(abstain.normal_coverage_pct, abstain.average_precision, 's-', label='Covered AP')
    ax[1].plot(abstain.normal_coverage_pct, abstain.prevalence, '--', label='Covered prevalence (AP baseline)')
    for _, row in abstain.iterrows():
        if pd.notna(row.roc_auc):
            ax[1].annotate(f'{int(row.radius_m)}m', (row.normal_coverage_pct, row.roc_auc))
    ax[1].set(xlabel='Normal covered (%)', ylabel='Metric', ylim=(0, 1)); ax[1].legend()
    fig.suptitle('Source-oracle conditional audit; not a deployable abstention policy')
    save('abstention_coverage_curve.png')
    return paths

def run_audit(data_dir=single.DEFAULT_DATA_DIR, output_root=single.ROOT,
              protected_snapshot=None) -> dict:
    started = time.perf_counter()
    root = Path(output_root).resolve(); data_dir = Path(data_dir).resolve()
    metrics = root / 'outputs/metrics/stage6/route_reference_coverage'
    figures = root / 'outputs/figures/stage6/route_reference_coverage'
    if metrics.exists() or figures.exists():
        raise FileExistsError('Stage 6.2 output exists; overwrite refused.')
    protected_snapshot = Path(protected_snapshot or root / 'outputs/metrics/stage62_protected_snapshot.json')
    expected = json.loads(protected_snapshot.read_text(encoding='utf-8'))
    dataset_id = data_dir.name
    stage61_dir = root / 'outputs/metrics/stage6' / dataset_id / 'route_audit'
    required = {str(data_dir.relative_to(root)),
        str(Path('models/stage5') / dataset_id), str(Path('outputs/metrics/stage5') / dataset_id),
        str(Path('outputs/figures/stage5') / dataset_id),
        str(stage61_dir.relative_to(root)), str(Path('outputs/figures/stage6') / dataset_id / 'route_audit')}
    if {k.replace('\\', '/') for k in expected} != {k.replace('\\', '/') for k in required}:
        raise ValueError('Snapshot must cover the dataset, all Stage 5 and Stage 6.1 output roots.')
    previous.assert_protected(root, expected)
    leakage = validate_saved_dataset(data_dir)
    summary, users = single.read_stage5_metadata(data_dir)
    train = single.load_stage5_split(data_dir, 'train', summary, users)
    test = single.load_stage5_split(data_dir, 'test', summary, users)
    validation = single.load_stage5_split(data_dir, 'validation', summary, users)
    trajectories = read_dataset_csv(data_dir / 'trajectory_quality_summary.csv')
    reference = SpatialReference.build(train, users, trajectories)
    normal = reference.query(test.loc[~test.is_synthetic])
    val_normal = reference.query(validation.loc[~validation.is_synthetic])
    route = test.loc[test.is_synthetic & test.anomaly_label.eq(1) & test.anomaly_type.eq('route_deviation')]
    pairs = source_matched(route, normal, reference)
    verify_stage61_boundary(pairs, stage61_dir / 'route_deviation_samples.csv')
    normal_cover = coverage(normal); val_cover = coverage(val_normal)
    per_user, per_user_abstain = user_analysis(normal, pairs)
    abstain = abstention(normal, pairs)
    boundary = pairs.loc[pairs.zero_displacement | pairs.feature_near_identical].copy()
    frequencies = nearest_user_frequencies(normal)
    route_distances = pairs[single.LINEAGE_COLUMNS + ['latitude', 'longitude'] +
        ['nearest_train_' + c for c in META] + ['synthetic_reference_distance_m']].rename(
            columns={'synthetic_reference_distance_m': DISTANCE})
    normal_columns = single.LINEAGE_COLUMNS + ['latitude', 'longitude'] + ['nearest_train_' + c for c in META] + [DISTANCE]
    tables = {'normal_reference_distances': normal[normal_columns],
              'route_deviation_reference_distances': route_distances,
              'source_matched_distance_delta': pairs, 'coverage_summary': normal_cover,
              'per_user_coverage': per_user, 'abstention_analysis': abstain,
              'nearest_reference_user_summary': frequencies,
              'label_boundary_spatial_audit': boundary,
              'per_user_abstention_analysis': per_user_abstain,
              'validation_normal_reference_distances': val_normal[normal_columns],
              'validation_coverage_summary': val_cover}
    manifest = read_dataset_csv(data_dir / 'synthetic_anomaly_manifest.csv')
    if 'configured_amplitude_m' in manifest or 'amplitude_m' in manifest:
        raise ValueError('New amplitude metadata requires an explicit audit adapter.')
    previous.assert_protected(root, expected)
    metrics.mkdir(parents=True); figures.mkdir(parents=True)
    for name, frame in tables.items(): frame.to_csv(metrics / (name + '.csv'), index=False, na_rep='NA')
    figure_paths = create_figures(normal, pairs, normal_cover, per_user, abstain, figures)
    boundary_stats = {}
    for flag in ('zero_displacement', 'feature_near_identical'):
        b = pairs.loc[pairs[flag]]
        boundary_stats[flag] = {'count': len(b), 'actual_displacement_m': distance_summary(b.actual_displacement_m),
            'original_reference_distance_m': distance_summary(b.original_reference_distance_m),
            'synthetic_reference_distance_m': distance_summary(b.synthetic_reference_distance_m),
            'delta_reference_distance_m': distance_summary(b.delta_reference_distance_m),
            'spatially_unchanged_count': int(b.delta_reference_distance_m.abs().le(1e-6).sum())}
    corr = pairs.actual_displacement_m.corr(pairs.delta_reference_distance_m, method='spearman')
    report = {'stage': '6.2', 'name': 'Train-only Route Reference Coverage & Abstention Audit',
        'dataset_id': dataset_id, 'analysis_only': True, 'earth_radius_m': EARTH_RADIUS_M,
        'reference': reference.audit, 'reference_method': 'BallTree haversine, latitude/longitude radians, leaf_size=40',
        'tie_policy': 'Sorted lexical lineage; identical coordinate metadata uses first lexical row. Other equidistant coordinates follow deterministic BallTree ordering in the recorded version.',
        'test_normal': distance_summary(normal[DISTANCE]),
        'validation_normal': distance_summary(val_normal[DISTANCE]),
        'test_route_deviation': distance_summary(pairs.synthetic_reference_distance_m),
        'full_test_separation': separation(normal[DISTANCE], pairs.synthetic_reference_distance_m),
        'paired_original_reference_distance_m': distance_summary(pairs.original_reference_distance_m),
        'source_matched_delta': delta_summary(pairs),
        'actual_displacement_m': distance_summary(pairs.actual_displacement_m),
        'actual_displacement_vs_delta_spearman': float(corr) if np.isfinite(corr) else None,
        'amplitude': {'configured_range_m_from_unchanged_generator': [100, 300],
            'per_sample_amplitude_available': False, 'reason': 'Not stored in frozen manifest; seed replay and anomaly regeneration forbidden.'},
        'coverage_radii_m': list(COVERAGE_RADII_M), 'abstention_radii_m': list(ABSTENTION_RADII_M),
        'boundary': boundary_stats, 'cross_split_leakage': 0,
        'leakage_checks': {k: leakage[k] for k in ('user_overlap', 'source_trajectory_overlap',
                        'sample_overlap', 'cross_split_duplicate_fingerprint_count')},
        'source_coverage_is_audit_oracle': True, 'decision_threshold_selected': False,
        'model_trained_or_inferred': False, 'scaler_or_threshold_recomputed': False,
        'features_or_synthetic_data_changed': False, 'test_parameter_tuning': False,
        'protected_file_count': sum(len(v) for v in expected.values()),
        'source_sha256': single.sha256_file(Path(__file__)),
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                     'sklearn': __import__('sklearn').__version__},
        'limitations': ['Nearest Train point is not a continuous route or personal normal route.',
            'Source coverage is available only in this paired synthetic audit, not at inference.',
            'ROC/AP are descriptive representation metrics; AP depends on subset prevalence.',
            'Temporal/spatial correlation limits row-level inference; no causal attribution.',
            'Case judgement is qualitative; no Test-based radius or production policy selection.'],
        'elapsed_seconds': time.perf_counter() - started}
    previous.assert_protected(root, expected)
    single.save_json(metrics / 'route_reference_audit.json', report)
    verification = {'passed': True, 'protected_file_count': report['protected_file_count'],
        'protected_artifacts_unchanged': True, 'reference_train_only': True,
        'lineage_and_stage61_boundary_verified': True, 'cross_split_leakage': 0,
        'no_model_fit_inference_threshold_scaler_or_test_tuning': True,
        'source_sha256': report['source_sha256'],
        'protected_snapshot_sha256': single.sha256_file(protected_snapshot),
        'output_checksums': {p.name: single.sha256_file(p) for p in metrics.glob('*') if p.is_file()},
        'figure_checksums': {p.name: single.sha256_file(p) for p in figure_paths}}
    single.save_json(metrics / 'stage6_2_verification.json', verification)
    print(json.dumps({'reference': report['reference'], 'test_normal': report['test_normal'],
        'test_route_deviation': report['test_route_deviation'], 'source_matched_delta': report['source_matched_delta'],
        'separation': report['full_test_separation'], 'elapsed_seconds': report['elapsed_seconds']}, indent=2))
    return {'report': report, 'tables': tables, 'verification': verification,
            'metrics_dir': metrics, 'figure_dir': figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=single.ROOT)
    parser.add_argument('--protected-snapshot', type=Path)
    args = parser.parse_args()
    run_audit(args.data_dir, args.output_root, args.protected_snapshot)


if __name__ == '__main__': main()
