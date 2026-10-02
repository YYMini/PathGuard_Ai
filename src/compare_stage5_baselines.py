"""Stage 5.4: predeclared baselines on frozen rows; read-only Stage 5.3 AE reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from src import train_multiuser_autoencoder as single
from src import evaluate_multi_seed_stability as multi
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import validate_saved_dataset
from src.train_autoencoder import FEATURE_COLUMNS, fit_train_scaler, transform_with_scaler

# Fixed before baseline Test scores were computed; see docs/stage5_baseline_comparison.md.
MODEL_SEEDS = multi.MODEL_SEEDS
MODEL_NAMES = ('statistical_rule', 'isolation_forest', 'autoencoder')
IF_PARAMETERS = dict(n_estimators=200, max_samples='auto', contamination='auto',
                     max_features=1.0, bootstrap=False, n_jobs=None, verbose=0, warm_start=False)
RULE_DEFINITION = {
    'center': 'Train original normal median', 'primary_scale': 'IQR / 1.349',
    'zero_iqr_fallback': 'Train population std (ddof=0)',
    'constant_fallback': '1e-9 * max(1, abs(median))',
    'score': 'max(abs(x_j - median_j) / scale_j), all eight features',
    'quantile_method': 'linear', 'feature_columns': FEATURE_COLUMNS,
    'bearing_handling': 'Both sin/cos coordinates included; no angular seam; not rotation invariant.',
}
GLOBAL_METRICS = single.METRIC_NAMES + ['tn', 'fp', 'fn', 'tp']
USER_METRICS = multi.MACRO_METRICS
TYPE_METRICS = ['anomaly_row_count', 'detected_anomaly_row_count', 'recall', 'roc_auc',
                'average_precision', 'anomaly_score_mean', 'anomaly_score_median',
                'anomaly_score_p95', 'anomaly_score_p99']


def feature_values(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty or any(name not in frame for name in FEATURE_COLUMNS):
        raise ValueError('Nonempty fixed eight-feature input required.')
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite model features.')
    return values


def require_train_normal(train: pd.DataFrame) -> None:
    feature_values(train)
    if (not train.dataset_split.eq('train').all() or train.is_synthetic.any()
            or not train.anomaly_label.eq(0).all()):
        raise ValueError('Fit accepts Train original normal only.')


class StatisticalRule:
    """Eight-feature maximum robust deviation, including both bearing coordinates."""
    def fit(self, train: pd.DataFrame) -> StatisticalRule:
        require_train_normal(train)
        values = feature_values(train)
        self.center_ = np.median(values, axis=0)
        quartiles = np.percentile(values, [25, 75], axis=0, method='linear')
        iqr = quartiles[1] - quartiles[0]
        std = values.std(axis=0, ddof=0)
        constant = 1e-9 * np.maximum(1., np.abs(self.center_))
        self.scale_ = np.where(iqr > 0, iqr / 1.349, np.where(std > 0, std, constant))
        self.scale_sources_ = np.where(iqr > 0, 'iqr', np.where(std > 0, 'population_std', 'constant_floor')).tolist()
        self.fit_row_count_ = len(train)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        scores = np.max(np.abs(feature_values(frame) - self.center_) / self.scale_, axis=1)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite Rule score.')
        return scores

    def parameters(self) -> dict:
        return {'definition': RULE_DEFINITION, 'center': self.center_.tolist(),
                'scale': self.scale_.tolist(), 'scale_sources': self.scale_sources_,
                'fit_row_count': self.fit_row_count_}


def fit_isolation_forest(train: pd.DataFrame, seed: int) -> dict:
    require_train_normal(train)
    if seed not in MODEL_SEEDS:
        raise ValueError('Only the five fixed model seeds are allowed.')
    scaler, values = fit_train_scaler(train)
    model = IsolationForest(random_state=seed, **IF_PARAMETERS).fit(values)
    return {'model': model, 'scaler': scaler, 'fit_row_count': len(train)}


def anomaly_scores(selection: dict, frame: pd.DataFrame) -> np.ndarray:
    feature_values(frame)
    if selection['model_name'] == 'statistical_rule':
        scores = selection['model'].score(frame)
    elif selection['model_name'] == 'isolation_forest':
        scores = -selection['model'].score_samples(transform_with_scaler(selection['scaler'], frame))
    else:
        raise ValueError('AE is read-only; new inference is not supported.')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite anomaly scores.')
    return np.asarray(scores, dtype=float)


def select_baseline(train: pd.DataFrame, validation: pd.DataFrame,
                    model_name: str, seed: int | None = None) -> dict:
    """No Test argument. Fit and threshold receive Train/Val original normals only."""
    require_train_normal(train)
    if not validation.dataset_split.eq('validation').all():
        raise ValueError('Threshold input must be the Validation split.')
    normal = single.validation_normal_rows(validation)
    feature_values(normal)
    if model_name == 'statistical_rule':
        if seed is not None:
            raise ValueError('Deterministic Rule has no model seed.')
        selection = {'model': StatisticalRule().fit(train), 'fit_row_count': len(train)}
    elif model_name == 'isolation_forest':
        selection = fit_isolation_forest(train, seed)
    else:
        raise ValueError('Unknown baseline.')
    selection.update(model_name=model_name, seed=seed)
    scores = anomaly_scores(selection, normal)
    selection.update(threshold=float(np.percentile(scores, 95, method='linear')),
                     validation_normal_row_count=len(normal))
    return selection


def predictions(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    if len(scores) != len(frame) or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Finite aligned scores and threshold required.')
    result = frame.copy()
    result['anomaly_score'] = scores
    result['predicted_anomaly'] = (scores > threshold).astype(int)
    return result



def read_baseline_predictions(path: Path) -> pd.DataFrame:
    """Preserve ID/boolean semantics and exact score bits, including threshold ties."""
    frame = read_dataset_csv(path)
    scores = pd.read_csv(path, usecols=['anomaly_score'], float_precision='round_trip')
    frame['anomaly_score'] = scores.anomaly_score.to_numpy()
    return frame


def metrics_adapter(frame: pd.DataFrame) -> pd.DataFrame:
    # Reuse the audited Stage 5 evaluation logic; the alias is internal only.
    return frame.assign(reconstruction_error=frame.anomaly_score)


def evaluate(frame: pd.DataFrame, threshold: float, users: pd.DataFrame) -> dict:
    adapted = metrics_adapter(frame)
    overall = single.calculate_metrics(adapted, threshold)
    per_user, macro = single.per_user_metrics(adapted, threshold, users)
    per_type = single.per_anomaly_type_metrics(adapted, threshold).rename(
        columns={f'reconstruction_error_{suffix}': f'anomaly_score_{suffix}'
                 for suffix in ('mean', 'median', 'p95', 'p99')})
    return {'metrics': overall, 'per_user': per_user, 'macro': macro, 'per_type': per_type}


def assert_same_rows(saved: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Require row order, explicit lineage and all eight input features to match."""
    pd.testing.assert_frame_equal(saved[single.LINEAGE_COLUMNS], frozen[single.LINEAGE_COLUMNS], check_dtype=False)
    np.testing.assert_allclose(feature_values(saved), feature_values(frozen), rtol=1e-10, atol=1e-12)


def row_digest(frame: pd.DataFrame) -> str:
    subset = frame[single.LINEAGE_COLUMNS + FEATURE_COLUMNS]
    return hashlib.sha256(pd.util.hash_pandas_object(subset, index=False).values.tobytes()).hexdigest()


def load_existing_autoencoders(data_dir: Path, output_root: Path, summary: dict,
                               users: pd.DataFrame, frames: dict) -> tuple[dict, dict]:
    """Read Stage 5.3 results and verify provenance; never infer/train/reselect AE."""
    source_dir = output_root / 'outputs/metrics/stage5' / summary['dataset_id'] / 'multi_seed'
    source_report = json.loads((source_dir / 'multi_seed_summary.json').read_text(encoding='utf-8'))
    if (source_report['model_seeds'] != list(MODEL_SEEDS)
            or source_report['dataset_id'] != summary['dataset_id']
            or source_report['feature_columns'] != FEATURE_COLUMNS
            or source_report['dataset_checksums'] != multi.file_snapshot(data_dir)):
        raise ValueError('Stage 5.3 dataset/schema/seeds differ from this comparison.')
    results, provenance = {}, {}
    for seed in MODEL_SEEDS:
        paths = single.output_directories(output_root, summary['dataset_id'], seed)
        hashes = source_report['seed_artifact_verification'][str(seed)]['artifact_checksums']
        for name, path in paths.items():
            multi.assert_snapshot(path, hashes[name])
        config = json.loads((paths['model'] / 'training_config.json').read_text(encoding='utf-8'))
        multi.validate_config(config, summary, data_dir, seed)
        for split in ('validation', 'test'):
            saved = read_dataset_csv(paths['metrics'] / f'{split}_predictions.csv')
            assert_same_rows(saved, frames[split])
            scores = saved.reconstruction_error.to_numpy(dtype=float)
            if not np.isfinite(scores).all() or (scores < 0).any():
                raise ValueError('Invalid saved AE scores.')
            if not np.array_equal(saved.predicted_anomaly, (scores > config['threshold']).astype(int)):
                raise ValueError('Saved AE decisions differ from saved threshold.')
            stored = json.loads((paths['metrics'] / f'{split}_metrics.json').read_text(encoding='utf-8'))
            multi.assert_metric_dict(stored, single.calculate_metrics(saved, config['threshold']))
        # Test saved was read last. Threshold is the stored value, never recomputed.
        comparison_frame = saved.rename(columns={'reconstruction_error': 'anomaly_score'})
        result = evaluate(comparison_frame, config['threshold'], users)
        user_stored = pd.read_csv(paths['metrics'] / 'per_user_metrics.csv', dtype={'user_id': str})
        pd.testing.assert_frame_equal(user_stored, result['per_user'], check_dtype=False, rtol=1e-7, atol=1e-9)
        type_stored = pd.read_csv(paths['metrics'] / 'per_anomaly_type_metrics.csv').rename(
            columns={f'reconstruction_error_{s}': f'anomaly_score_{s}' for s in ('mean', 'median', 'p95', 'p99')})
        pd.testing.assert_frame_equal(type_stored, result['per_type'], check_dtype=False, rtol=1e-7, atol=1e-9)
        macro_stored = json.loads((paths['metrics'] / 'user_macro_metrics.json').read_text(encoding='utf-8'))
        if result['macro'] != macro_stored:
            raise ValueError('Saved AE macro metrics differ.')
        result.update(threshold=config['threshold'], elapsed_seconds=0., reused_existing=True)
        results[seed] = result
        provenance[str(seed)] = {'threshold': config['threshold'], 'artifact_checksums': hashes,
                                'config_sha256': single.sha256_file(paths['model'] / 'training_config.json'),
                                'fit_train_row_count': config['scaler_fit_row_count'],
                                'threshold_normal_row_count': config['validation_normal_row_count']}
    return results, {'source_report_sha256': single.sha256_file(source_dir / 'multi_seed_summary.json'),
                     'seeds': provenance, 'threshold_recalculated': False,
                     'model_inference_executed': False, 'model_training_executed': False}


def summarize(frame: pd.DataFrame, groups: list[str], names: list[str]) -> pd.DataFrame:
    result = multi.summarize_values(frame, groups, names).rename(
        columns={'seed_count': 'run_count', 'defined_seed_count': 'defined_run_count'})
    result.loc[result.model.eq('statistical_rule') & result.defined_run_count.eq(1), 'na_reason'] = 'deterministic_single_run_std_undefined'
    return result


def collect_tables(records: list[tuple[str, int | None, dict]]) -> dict[str, pd.DataFrame]:
    global_rows, user_rows, type_rows = [], [], []
    for name, seed, result in records:
        identity = {'model': name, 'seed': seed, 'deterministic': name == 'statistical_rule',
                    'reused_existing': name == 'autoencoder'}
        row = {**identity, **result['metrics'], 'elapsed_seconds': result['elapsed_seconds']}
        row['metric_na_reasons'] = json.dumps(row['metric_na_reasons'], sort_keys=True)
        for metric in USER_METRICS:
            row[f'user_macro_{metric}'] = result['macro']['metric_means'][metric]
            row[f'user_macro_{metric}_defined_users'] = result['macro']['defined_user_counts'][metric]
        global_rows.append(row)
        for destination, table in [(user_rows, result['per_user']), (type_rows, result['per_type'])]:
            destination.extend({**identity, **record} for record in table.to_dict('records'))
    overall, users, types = map(pd.DataFrame, (global_rows, user_rows, type_rows))
    return {'baseline_results': overall,
            'baseline_summary': summarize(overall, ['model'], GLOBAL_METRICS + [f'user_macro_{m}' for m in USER_METRICS]),
            'per_user_baseline_metrics': users,
            'per_user_baseline_summary': summarize(users, ['model', 'user_id'], USER_METRICS),
            'per_anomaly_type_baseline_metrics': types,
            'per_anomaly_type_baseline_summary': summarize(types, ['model', 'anomaly_type'], TYPE_METRICS)}


def create_figures(tables: dict, figure_dir: Path) -> list[Path]:
    plt = single.plt
    paths = []
    for metric in ('f1_score', 'roc_auc', 'average_precision', 'false_positive_rate'):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, name in enumerate(MODEL_NAMES):
            row = tables['baseline_summary'].loc[lambda f: f.model.eq(name) & f.metric.eq(metric)].iloc[0]
            if pd.isna(row['mean']):
                ax.text(i, .01, 'NA', ha='center')
            else:
                ax.bar(i, row['mean'], yerr=None if pd.isna(row.sample_std) else row.sample_std, capsize=4)
        ax.set_xticks(range(3), ['Statistical Rule (single)', 'Isolation Forest (5)', 'Autoencoder (5)'])
        ax.set(ylabel=metric, ylim=(0, 1.05), title='Single value or mean +/- sample std (ddof=1)')
        fig.tight_layout()
        filename = 'model_fpr_comparison.png' if metric == 'false_positive_rate' else f'model_{"f1" if metric == "f1_score" else metric}_comparison.png'
        path = figure_dir / filename; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    for key, group, metric, filename in [
        ('per_anomaly_type_baseline_summary', 'anomaly_type', 'recall', 'anomaly_recall_by_model.png'),
        ('per_user_baseline_summary', 'user_id', 'f1_score', 'per_user_f1_by_model.png')]:
        table = tables[key].loc[lambda f: f.metric.eq(metric)]
        labels = sorted(table[group].unique()); x = np.arange(len(labels)); width = .25
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(MODEL_NAMES):
            rows = table.loc[table.model.eq(name)].set_index(group).reindex(labels)
            positions = x + (i - 1) * width
            valid = rows['mean'].notna().to_numpy()
            ax.bar(positions[valid], rows['mean'].to_numpy()[valid], width,
                   yerr=None if name == 'statistical_rule' else rows.sample_std.fillna(0).to_numpy()[valid],
                   capsize=3, label=name + (' (single)' if name == 'statistical_rule' else ' (5 seeds)'))
            for pos in positions[~valid]: ax.text(pos, .01, 'NA', ha='center')
        ax.set_xticks(x, labels); ax.set(ylabel=metric, ylim=(0, 1.08), title='Single value or mean +/- sample std (ddof=1)')
        ax.legend(); fig.tight_layout()
        path = figure_dir / filename; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return paths


def run_comparison(data_dir: Path = single.DEFAULT_DATA_DIR, output_root: Path = single.ROOT) -> dict:
    started = time.perf_counter(); data_dir = Path(data_dir).resolve(); output_root = Path(output_root).resolve()
    summary, users = single.read_stage5_metadata(data_dir)
    # output_directories also validates the dataset ID for safe filesystem paths.
    single.output_directories(output_root, summary['dataset_id'], 42)
    metrics_dir = output_root / 'outputs/metrics/stage5' / summary['dataset_id'] / 'baseline_comparison'
    model_dir = output_root / 'models/stage5' / summary['dataset_id'] / 'baselines'
    figure_dir = output_root / 'outputs/figures/stage5' / summary['dataset_id'] / 'baseline_comparison'
    if any(p.exists() for p in (metrics_dir, model_dir, figure_dir)):
        raise FileExistsError('Baseline outputs exist; no overwrite allowed.')
    integrity = validate_saved_dataset(data_dir)
    if not integrity['ready_for_stage5_b']:
        raise ValueError('Invalid frozen dataset.')
    dataset_before = multi.file_snapshot(data_dir)
    frames = {s: single.load_stage5_split(data_dir, s, summary, users) for s in ('train', 'validation', 'test')}
    ae, reuse = load_existing_autoencoders(data_dir, output_root, summary, users, frames)
    multi_dir = output_root / 'outputs/metrics/stage5' / summary['dataset_id'] / 'multi_seed'
    multi_before = multi.file_snapshot(multi_dir)
    normal_validation = single.validation_normal_rows(frames['validation'])
    validation = {
        'stage': '5.4', 'dataset_id': summary['dataset_id'], 'passed': True,
        'dataset_checksums': dataset_before, 'dataset_summary_sha256': single.sha256_file(data_dir / 'dataset_summary.json'),
        'row_counts': {s: {'total': len(f), 'normal': int((~f.is_synthetic & f.anomaly_label.eq(0)).sum()),
                          'anomaly': int((f.is_synthetic & f.anomaly_label.eq(1)).sum())} for s, f in frames.items()},
        'train_normal_row_count': len(frames['train']), 'validation_normal_row_count': len(normal_validation),
        'feature_columns': FEATURE_COLUMNS, 'test_users': sorted(frames['test'].user_id.unique()),
        'user_splits': {s: summary['splits'][s]['users'] for s in frames},
        'row_identity_feature_lineage_sha256': {s: row_digest(f) for s, f in frames.items()},
        'validation_normal_identity_feature_lineage_sha256': row_digest(normal_validation),
        'synthetic_manifest_sha256': dataset_before['synthetic_anomaly_manifest.csv'],
        'dedup_report_sha256': dataset_before['duplicate_report.json'],
        'leakage_report_sha256': dataset_before['leakage_report.json'],
        'checks': {k: True for k in ['same_train_rows', 'same_validation_normal_rows', 'same_test_rows',
            'same_eight_feature_schema', 'same_test_users', 'same_synthetic_anomalies', 'same_source_lineage',
            'same_95th_percentile_rule', 'train_normal_fit_only', 'validation_normal_threshold_only', 'no_test_tuning']},
        'threshold_rule': {'percentile': 95, 'method': 'linear', 'source': 'Validation original normal only', 'comparison': 'score > threshold'},
        'autoencoder_reuse': reuse, 'condition_deviations': [],
        'method_differences': ['Rule robust scaling vs Train-only StandardScaler for IF/AE.',
            'Score units differ; compare metrics, never raw scores/threshold magnitudes.',
            'Rule single deterministic run vs IF/AE five model seeds.',
            'AE thresholds are reused with audited Stage 5.3 provenance, not recalculated.'],
    }
    for directory in (metrics_dir, model_dir, figure_dir): directory.mkdir(parents=True)
    design = {'stage': '5.4', 'rule': RULE_DEFINITION, 'isolation_forest': IF_PARAMETERS,
              'model_seeds': list(MODEL_SEEDS), 'feature_columns': FEATURE_COLUMNS,
              'threshold_rule': validation['threshold_rule'],
              'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                              'numpy': np.__version__, 'pandas': pd.__version__}, 'test_tuning': False}
    single.save_json(metrics_dir / 'experiment_design.json', design)
    single.save_json(metrics_dir / 'comparison_validation.json', validation)
    records = []
    for name, seeds in [('statistical_rule', [None]), ('isolation_forest', MODEL_SEEDS)]:
        for seed in seeds:
            run_started = time.perf_counter()
            suffix = Path(name) if seed is None else Path(name) / f'seed_{seed}'
            run_metrics = metrics_dir / suffix; run_model = model_dir / suffix
            run_metrics.mkdir(parents=True); run_model.mkdir(parents=True)
            selection = select_baseline(frames['train'], frames['validation'], name, seed)
            config = {**design, 'model': name, 'seed': seed, 'fit_source': 'Train original normal only',
                      'fit_row_count': selection['fit_row_count'], 'threshold': selection['threshold'],
                      'validation_normal_row_count': selection['validation_normal_row_count'],
                      'test_used_for_fit_or_threshold': False, 'validation_anomaly_used_for_fit_or_threshold': False,
                      'input_row_hashes': validation['row_identity_feature_lineage_sha256']}
            if name == 'statistical_rule':
                single.save_json(run_model / 'rule_parameters.json', selection['model'].parameters())
            else:
                joblib.dump(selection['model'], run_model / 'model.joblib')
                joblib.dump(selection['scaler'], run_model / 'scaler.joblib')
                config['scaler_fit_row_count'] = int(selection['scaler'].n_samples_seen_)
                config['effective_max_samples'] = int(selection['model'].max_samples_)
            # Persist fixed model and selected threshold before any baseline Test inference.
            single.save_json(run_model / 'run_config.json', config)
            for split in ('validation', 'test'):
                scored = predictions(frames[split], anomaly_scores(selection, frames[split]), selection['threshold'])
                assert_same_rows(scored, frames[split])
                scored.to_csv(run_metrics / f'{split}_predictions.csv', index=False)
                saved = read_baseline_predictions(run_metrics / f'{split}_predictions.csv')
                assert_same_rows(saved, frames[split])
                np.testing.assert_array_equal(saved.predicted_anomaly, (saved.anomaly_score > selection['threshold']).astype(int))
                split_metrics = single.calculate_metrics(metrics_adapter(saved), selection['threshold'])
                single.save_json(run_metrics / f'{split}_metrics.json', split_metrics)
            result = evaluate(saved, selection['threshold'], users)
            result.update(elapsed_seconds=time.perf_counter() - run_started, threshold=selection['threshold'])
            result['per_user'].to_csv(run_metrics / 'per_user_metrics.csv', index=False, na_rep='NA')
            result['per_type'].to_csv(run_metrics / 'per_anomaly_type_metrics.csv', index=False, na_rep='NA')
            single.save_json(run_metrics / 'user_macro_metrics.json', result['macro'])
            records.append((name, seed, result))
            print(f'{name} seed={seed}: F1={result["metrics"]["f1_score"]}; elapsed={result["elapsed_seconds"]:.2f}s', flush=True)
    records.extend(('autoencoder', seed, ae[seed]) for seed in MODEL_SEEDS)
    tables = collect_tables(records)
    for name, table in tables.items(): table.to_csv(metrics_dir / f'{name}.csv', index=False, na_rep='NA')
    figures = create_figures(tables, figure_dir)
    multi.assert_snapshot(data_dir, dataset_before); multi.assert_snapshot(multi_dir, multi_before)
    for seed in MODEL_SEEDS:
        for name, path in single.output_directories(output_root, summary['dataset_id'], seed).items():
            multi.assert_snapshot(path, reuse['seeds'][str(seed)]['artifact_checksums'][name])
    validation.update(dataset_unchanged=True, all_autoencoder_artifacts_unchanged=True, stage53_summary_unchanged=True)
    single.save_json(metrics_dir / 'comparison_validation.json', validation)
    report = {**design, 'dataset_id': summary['dataset_id'], 'comparison_validation': validation,
              'aggregation': {'std_ddof': 1, 'rule_run_count': 1, 'isolation_forest_seed_count': 5,
                              'autoencoder_seed_count': 5, 'na_policy': 'Undefined metrics excluded with defined run/user counts; Rule std remains NA.'},
              'elapsed_seconds': time.perf_counter() - started,
              'baseline_run_elapsed_seconds': sum(r['elapsed_seconds'] for n, _, r in records if n != 'autoencoder'),
              'ae_historical_training_time_included': False,
              'figures': [str(p) for p in figures],
              'summary': json.loads(tables['baseline_summary'].to_json(orient='records')),
              'limitations': ['One frozen user split and synthetic anomaly generator; five seeds are not independent datasets.',
                  'No causal attribution to feature representation, architecture or objective from this comparison alone.',
                  'Coordinatewise bearing statistics are not rotation invariant; raw score units are incomparable.']}
    single.save_json(metrics_dir / 'baseline_comparison.json', report)
    print(f'Stage 5.4 complete: {report["elapsed_seconds"]:.2f}s; dataset and AE unchanged.', flush=True)
    return {'report': report, 'tables': tables, 'metrics_dir': metrics_dir, 'figure_dir': figure_dir}


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Fixed Stage 5.4 baseline comparison; Stage 5.3 AE results reused read-only.')
    parser.add_argument('--data-dir', type=Path, default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=single.ROOT)
    parsed = parser.parse_args(args)
    run_comparison(parsed.data_dir, parsed.output_root)


if __name__ == '__main__':
    main()