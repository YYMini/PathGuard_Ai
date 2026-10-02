"""Stage 5.3: fixed-data, five-model-seed stability with audited seed-42 reuse."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.autoencoder import Autoencoder
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import validate_saved_dataset
from src.synthetic_anomalies import ANOMALY_TYPES
from src.train_autoencoder import (
    FEATURE_COLUMNS, MODEL_ARCHITECTURE, TrainingOptions,
    reconstruction_errors, transform_with_scaler,
)
from src import train_multiuser_autoencoder as single

MODEL_SEEDS = (7, 21, 42, 100, 2026)
USER_METRICS = ['f1_score', 'recall', 'false_positive_rate', 'roc_auc', 'average_precision']
MACRO_METRICS = ['precision', 'recall', 'f1_score', 'false_positive_rate', 'roc_auc', 'average_precision']
SEED_METRICS = ['best_epoch', 'best_validation_normal_loss', 'threshold'] + single.METRIC_NAMES + ['tn', 'fp', 'fn', 'tp']


def file_snapshot(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): single.sha256_file(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def assert_snapshot(directory: Path, snapshot: dict[str, str]) -> None:
    if file_snapshot(directory) != snapshot:
        raise ValueError(f'Protected files changed: {directory}')


def validate_config(config: dict, summary: dict, data_dir: Path, seed: int) -> None:
    """Match all fixed experiment settings; historical stage labels are metadata."""
    expected = {
        **asdict(TrainingOptions(seed=seed)), 'dataset_id': summary['dataset_id'],
        'dataset_path': str(data_dir.resolve()),
        'dataset_summary_sha256': single.sha256_file(data_dir / 'dataset_summary.json'),
        'dataset_file_checksums': summary['file_checksums'],
        'feature_columns': FEATURE_COLUMNS, 'model_architecture': MODEL_ARCHITECTURE,
        'scaler_fit_row_count': summary['splits']['train']['normal_original_rows'],
        'validation_normal_row_count': summary['splits']['validation']['normal_original_rows'],
        'train_row_count': summary['splits']['train']['evaluation_rows'],
        'validation_evaluation_row_count': summary['splits']['validation']['evaluation_rows'],
        'test_evaluation_row_count': summary['splits']['test']['evaluation_rows'],
        'user_splits': {s: summary['splits'][s]['users'] for s in ('train', 'validation', 'test')},
        'scaler_fit_source': 'Train original normal only',
        'threshold_source': 'best checkpoint, Validation original normal only',
        'checkpoint_selection_source': 'Validation original normal MSE only',
        'threshold_comparison': 'reconstruction_error > threshold',
        'test_used_for_selection_or_threshold': False,
        'validation_anomaly_used_for_selection_or_threshold': False,
    }
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError(f'Incompatible seed {seed} config: {mismatches}')
    if config.get('stage') not in ('5-B', '5.2', '5.3'):
        raise ValueError('Unexpected seed artifact stage.')
    current_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.get('device') != current_device or config.get('torch_num_threads') != torch.get_num_threads():
        raise ValueError('Seed artifact device/thread configuration differs from this experiment.')
    if config.get('environment') != {
        'python': single.platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': single.sklearn.__version__}:
        raise ValueError('Seed artifact software environment differs from this experiment.')


def assert_metric_dict(actual: dict, expected: dict) -> None:
    if actual.keys() != expected.keys():
        raise ValueError('Saved metric schema differs from recomputed metrics.')
    for key, value in expected.items():
        saved = actual[key]
        if isinstance(value, float):
            if saved is None or not np.isclose(saved, value, rtol=1e-7, atol=1e-9):
                raise ValueError(f'Saved metric differs from recomputed value: {key}')
        elif saved != value:
            raise ValueError(f'Saved metric differs from recomputed value: {key}')


def verify_seed_artifacts(data_dir: Path, output_root: Path, summary: dict,
                          users: pd.DataFrame, seed: int) -> dict:
    """Verify config, lineage, checkpoint scores, scaler and all aggregation inputs."""
    for name, digest in summary['file_checksums'].items():
        if single.sha256_file(data_dir / name) != digest:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    paths = single.output_directories(output_root, summary['dataset_id'], seed)
    model_dir, metrics_dir = paths['model'], paths['metrics']
    config = json.loads((model_dir / 'training_config.json').read_text(encoding='utf-8'))
    validate_config(config, summary, data_dir, seed)
    run_report = json.loads((metrics_dir / 'run_report.json').read_text(encoding='utf-8'))
    if run_report['config'] != config or not run_report.get('dataset_unchanged'):
        raise ValueError('Incomplete or inconsistent saved run report.')
    features = json.loads((model_dir / 'feature_columns.json').read_text(encoding='utf-8'))
    if features != {'feature_columns': FEATURE_COLUMNS}:
        raise ValueError('Saved feature allowlist changed.')
    history = pd.read_csv(metrics_dir / 'training_history.csv', float_precision='round_trip')
    best_row = history.loc[history.validation_normal_loss.idxmin()]
    if (len(history) != config['actual_epochs'] or int(best_row.epoch) != config['best_epoch']
            or not np.isclose(best_row.validation_normal_loss, config['best_validation_normal_loss'], rtol=1e-10)):
        raise ValueError('Checkpoint selection differs from normal-only history.')
    frames = {s: single.load_stage5_split(data_dir, s, summary, users)
              for s in ('train', 'validation', 'test')}
    scaler = joblib.load(model_dir / 'scaler.joblib')
    fresh_scaler, _ = single.fit_train_scaler(frames['train'])
    if int(scaler.n_samples_seen_) != len(frames['train']):
        raise ValueError('Saved scaler fit count differs from Train rows.')
    for name in ('mean_', 'var_', 'scale_'):
        np.testing.assert_array_equal(getattr(scaler, name), getattr(fresh_scaler, name))
    device = torch.device(config['device'])
    model = Autoencoder(8).to(device)
    model.load_state_dict(torch.load(model_dir / 'model.pt', map_location=device, weights_only=True))
    predictions = {}
    for split in ('validation', 'test'):
        saved = read_dataset_csv(metrics_dir / f'{split}_predictions.csv')
        pd.testing.assert_frame_equal(saved[single.LINEAGE_COLUMNS], frames[split][single.LINEAGE_COLUMNS], check_dtype=False)
        errors = reconstruction_errors(model, transform_with_scaler(scaler, frames[split]), device)
        np.testing.assert_allclose(saved.reconstruction_error, errors, rtol=1e-7, atol=1e-8)
        if not np.array_equal(saved.predicted_anomaly, (errors > config['threshold']).astype(int)):
            raise ValueError('Saved predictions differ from restored checkpoint/threshold.')
        recomputed = single.calculate_metrics(saved, config['threshold'])
        stored = json.loads((metrics_dir / f'{split}_metrics.json').read_text(encoding='utf-8'))
        assert_metric_dict(stored, recomputed)
        assert_metric_dict(run_report[f'{split}_metrics'], recomputed)
        predictions[split] = saved
    normal = predictions['validation'].loc[lambda f: ~f.is_synthetic & f.anomaly_label.eq(0)]
    if not np.isclose(np.percentile(normal.reconstruction_error, 95), config['threshold'], rtol=1e-7, atol=1e-9):
        raise ValueError('Saved threshold is not Validation normal 95th percentile.')
    per_user, macro = single.per_user_metrics(predictions['test'], config['threshold'], users)
    stored_user = pd.read_csv(metrics_dir / 'per_user_metrics.csv', dtype={'user_id': str})
    pd.testing.assert_frame_equal(stored_user, per_user, check_dtype=False, rtol=1e-7, atol=1e-9)
    stored_macro = json.loads((metrics_dir / 'user_macro_metrics.json').read_text(encoding='utf-8'))
    if stored_macro != macro or run_report['user_macro_metrics'] != macro:
        raise ValueError('Saved user macro metrics differ from recomputed metrics.')
    per_type = single.per_anomaly_type_metrics(predictions['test'], config['threshold'])
    stored_type = pd.read_csv(metrics_dir / 'per_anomaly_type_metrics.csv')
    pd.testing.assert_frame_equal(stored_type, per_type, check_dtype=False, rtol=1e-7, atol=1e-9)
    required_figures = ['training_loss.png', 'reconstruction_error_distribution.png',
                        'reconstruction_error_distribution_zoom.png', 'confusion_matrix.png',
                        'roc_curve.png', 'precision_recall_curve.png', 'anomaly_score_by_type.png',
                        'per_user_f1.png', 'per_user_fpr.png']
    if any(not (paths['figures'] / name).is_file() for name in required_figures):
        raise ValueError('Incomplete seed figures.')
    return {'config': config, 'test_metrics': single.calculate_metrics(predictions['test'], config['threshold']),
            'macro': stored_macro, 'per_user': per_user, 'per_type': per_type,
            'artifact_checksums': {name: file_snapshot(path) for name, path in paths.items()},
            'verification': 'Fixed config, dataset checksum, original lineage, restored scores/predictions, Train-only scaler, normal-only threshold/history, user/type metrics verified.'}


def summarize_values(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    """Tidy summaries with sample std (ddof=1) and defined-seed support."""
    groups = frame.groupby(group_columns, sort=True) if group_columns else [((), frame)]
    records = []
    for key, rows in groups:
        key = key if isinstance(key, tuple) else (key,)
        group = dict(zip(group_columns, key))
        for name in metrics:
            values = pd.to_numeric(rows[name], errors='coerce').dropna()
            records.append({**group, 'metric': name, 'seed_count': len(rows), 'defined_seed_count': len(values),
                            'mean': float(values.mean()) if len(values) else None,
                            'sample_std': float(values.std(ddof=1)) if len(values) >= 2 else None,
                            'min': float(values.min()) if len(values) else None,
                            'max': float(values.max()) if len(values) else None,
                            'na_reason': 'no_defined_seeds' if not len(values) else ('sample_std_requires_two_seeds' if len(values) == 1 else '')})
    return pd.DataFrame(records)


def collect_results(verified: dict[int, dict], reused: set[int]) -> dict[str, pd.DataFrame]:
    seed_rows, user_rows, type_rows = [], [], []
    for seed in MODEL_SEEDS:
        result = verified[seed]
        config, metrics, macro = result['config'], result['test_metrics'], result['macro']
        row = {'seed': seed, 'reused_existing': seed in reused,
               'actual_epochs': config['actual_epochs'], 'elapsed_seconds': config['elapsed_seconds'],
               'dataset_summary_sha256': config['dataset_summary_sha256'],
               **{key: config[key] for key in ('best_epoch', 'best_validation_normal_loss', 'threshold')},
               **{key: metrics[key] for key in single.METRIC_NAMES + ['tn', 'fp', 'fn', 'tp']}}
        for name in MACRO_METRICS:
            row[f'user_macro_{name}'] = macro['metric_means'][name]
            row[f'user_macro_{name}_defined_users'] = macro['defined_user_counts'][name]
        seed_rows.append(row)
        user_frame = result['per_user'][['user_id', 'eligible_trajectory_count', 'normal_row_count',
                                        'anomaly_row_count'] + USER_METRICS + ['na_reasons']].copy()
        user_frame.insert(0, 'seed', seed)
        user_rows.extend(user_frame.to_dict('records'))
        type_frame = result['per_type'][['anomaly_type', 'anomaly_row_count', 'detected_anomaly_row_count',
                                        'recall', 'na_reasons']].copy()
        type_frame.insert(0, 'seed', seed)
        type_rows.extend(type_frame.to_dict('records'))
    seeds, users, types = map(pd.DataFrame, (seed_rows, user_rows, type_rows))
    return {'seed_results': seeds,
            'seed_summary': summarize_values(seeds, [], SEED_METRICS + [f'user_macro_{name}' for name in MACRO_METRICS]),
            'per_user_seed_results': users,
            'per_user_seed_summary': summarize_values(users, ['user_id'], USER_METRICS),
            'per_anomaly_type_seed_results': types,
            'per_anomaly_type_seed_summary': summarize_values(types, ['anomaly_type'], ['recall'])}


def create_figures(tables: dict[str, pd.DataFrame], figure_dir: Path) -> list[Path]:
    paths = []
    seeds = tables['seed_results']
    for name, ylabel in [('f1_score', 'F1'), ('roc_auc', 'ROC-AUC'),
                         ('average_precision', 'Average Precision (AP)'), ('threshold', 'Validation normal threshold')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = seeds[name].to_numpy(dtype=float)
        ax.plot(range(len(seeds)), values, marker='o')
        ax.set_xticks(range(len(seeds)), seeds.seed.astype(str))
        ax.set(xlabel='Model seed', ylabel=ylabel)
        if name != 'threshold':
            ax.set_ylim(0, 1)
        fig.tight_layout()
        filename = {'f1_score': 'f1_by_seed.png', 'roc_auc': 'roc_auc_by_seed.png',
                    'average_precision': 'average_precision_by_seed.png', 'threshold': 'threshold_by_seed.png'}[name]
        path = figure_dir / filename
        fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    for frame, group, metric, filename in [
        (tables['per_anomaly_type_seed_results'], 'anomaly_type', 'recall', 'anomaly_recall_by_seed.png'),
        (tables['per_user_seed_results'], 'user_id', 'f1_score', 'per_user_f1_by_seed.png')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, rows in frame.groupby(group, sort=True):
            rows = rows.set_index('seed').reindex(MODEL_SEEDS)
            ax.plot(range(len(MODEL_SEEDS)), rows[metric], marker='o', label=str(label))
        ax.set_xticks(range(len(MODEL_SEEDS)), [str(seed) for seed in MODEL_SEEDS])
        ax.set(xlabel='Model seed', ylabel='Recall' if metric == 'recall' else 'F1', ylim=(0, 1))
        ax.legend(); fig.tight_layout()
        path = figure_dir / filename
        fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return paths


def run_multi_seed(data_dir: Path = single.DEFAULT_DATA_DIR, output_root: Path = single.ROOT) -> dict:
    started = time.perf_counter()
    data_dir, output_root = Path(data_dir).resolve(), Path(output_root).resolve()
    summary, users = single.read_stage5_metadata(data_dir)
    integrity = validate_saved_dataset(data_dir)
    if not integrity['ready_for_stage5_b']:
        raise ValueError('Dataset is not eligible for the stability experiment.')
    dataset_snapshot = file_snapshot(data_dir)
    directories = {seed: single.output_directories(output_root, summary['dataset_id'], seed) for seed in MODEL_SEEDS}
    metrics_dir = output_root / 'outputs' / 'metrics' / 'stage5' / summary['dataset_id'] / 'multi_seed'
    figure_dir = output_root / 'outputs' / 'figures' / 'stage5' / summary['dataset_id'] / 'multi_seed'
    if metrics_dir.exists() or figure_dir.exists():
        raise FileExistsError('Multi-seed outputs already exist; no overwrite allowed.')
    for seed, paths in directories.items():
        if seed != 42 and any(path.exists() for path in paths.values()):
            raise FileExistsError(f'Seed {seed} output already exists; no overwrite allowed.')
    verified = {}
    reused = set()
    protected_seed42 = None
    if any(path.exists() for path in directories[42].values()):
        verified[42] = verify_seed_artifacts(data_dir, output_root, summary, users, 42)
        reused.add(42)
        protected_seed42 = verified[42]['artifact_checksums']
        print('Seed 42: existing Stage 5.2 result verified and reused; all artifacts retained.', flush=True)
    metrics_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)
    logs = metrics_dir / 'logs'; logs.mkdir()
    for seed in MODEL_SEEDS:
        if seed in reused:
            continue
        assert_snapshot(data_dir, dataset_snapshot)
        print(f'Seed {seed}: training starts with frozen dataset/config; log={logs / f"seed_{seed}.log"}', flush=True)
        with (logs / f'seed_{seed}.log').open('w', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
            single.run_pipeline(data_dir, output_root, TrainingOptions(seed=seed), experiment_stage='5.3')
        verified[seed] = verify_seed_artifacts(data_dir, output_root, summary, users, seed)
        print(f'Seed {seed}: best_epoch={verified[seed]["config"]["best_epoch"]}; '
              f'F1={verified[seed]["test_metrics"]["f1_score"]:.6f}; verified.', flush=True)
    assert_snapshot(data_dir, dataset_snapshot)
    if protected_seed42 is not None:
        for name, hashes in protected_seed42.items():
            assert_snapshot(directories[42][name], hashes)
    tables = collect_results(verified, reused)
    for name, table in tables.items():
        table.to_csv(metrics_dir / f'{name}.csv', index=False, na_rep='NA')
    figures = create_figures(tables, figure_dir)
    results = tables['seed_results']
    metric_summary = {row.metric: {name: None if pd.isna(getattr(row, name)) else getattr(row, name) for name in ['mean', 'sample_std', 'min', 'max', 'defined_seed_count']}
                      for row in tables['seed_summary'].itertuples()}
    seed42_positions = {}
    for metric in single.METRIC_NAMES:
        values = pd.to_numeric(results[metric], errors='coerce')
        raw_score = values.loc[results.seed.eq(42)].iloc[0]
        score = None if pd.isna(raw_score) else float(raw_score)
        record = metric_summary[metric]
        ranks = values.rank(method='min')
        seed42_positions[metric] = {'value': score, 'mean': record['mean'],
                                   'difference_from_mean': score - record['mean'] if score is not None and record['mean'] is not None else None,
                                   'z_from_mean': (score - record['mean']) / record['sample_std'] if score is not None and record['sample_std'] else None,
                                   'ascending_rank': int(ranks.loc[results.seed.eq(42)].iloc[0]) if score is not None else None,
                                   'rank_interpretation': 'Ascending numeric rank; lower FPR and higher other metrics are preferable.'}
    report = {
        'stage': '5.3', 'dataset_id': summary['dataset_id'], 'model_seeds': list(MODEL_SEEDS),
        'fixed_training_options': {k: v for k, v in asdict(TrainingOptions()).items() if k != 'seed'},
        'feature_columns': FEATURE_COLUMNS, 'model_architecture': MODEL_ARCHITECTURE,
        'dataset_summary_sha256': single.sha256_file(data_dir / 'dataset_summary.json'),
        'dataset_checksums': dataset_snapshot,
        'user_split_manifest_sha256': dataset_snapshot['user_split_manifest.csv'],
        'synthetic_manifest_sha256': dataset_snapshot['synthetic_anomaly_manifest.csv'],
        'dataset_unchanged': True, 'seed42_artifacts_unchanged': protected_seed42 is not None,
        'reused_seeds': sorted(reused), 'newly_trained_seeds': [s for s in MODEL_SEEDS if s not in reused],
        'seed_artifact_verification': {str(seed): {'reused_existing': seed in reused,
                'verification': verified[seed]['verification'], 'artifact_checksums': verified[seed]['artifact_checksums']}
                for seed in MODEL_SEEDS},
        'aggregation': {'std_ddof': 1, 'seed_count': len(MODEL_SEEDS),
                        'na_policy': 'Exclude undefined metric seeds; record defined_seed_count; std is NA for fewer than two defined seeds.'},
        'metric_summary': metric_summary, 'seed42_positions': seed42_positions,
        'largest_std_test_metric': max(single.METRIC_NAMES, key=lambda key: metric_summary[key]['sample_std'] if metric_summary[key]['sample_std'] is not None else -1),
        'new_training_elapsed_seconds': sum(verified[s]['config']['elapsed_seconds'] for s in MODEL_SEEDS if s not in reused),
        'elapsed_seconds': time.perf_counter() - started,
        'figures': [str(p) for p in figures],
        'limitations': ['Five model seeds estimate initialization/shuffling variability on one fixed dataset.',
                       'Original normal labels are research assumptions and synthetic anomalies are experimental labels.',
                       'Neither stable low performance nor seed variance identifies a unique causal architectural limitation.'],
    }
    single.save_json(metrics_dir / 'multi_seed_summary.json', report)
    print(json.dumps({k: report[k] for k in ['model_seeds', 'reused_seeds', 'metric_summary', 'largest_std_test_metric', 'elapsed_seconds']}, indent=2), flush=True)
    return {'report': report, 'tables': tables, 'metrics_dir': metrics_dir, 'figure_dir': figure_dir}


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Stage 5.3 Multi-seed Stability on the immutable Stage 5.1 dataset.')
    parser.add_argument('--data-dir', type=Path, default=single.DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=single.ROOT)
    parsed = parser.parse_args(args)
    run_multi_seed(parsed.data_dir, parsed.output_root)


if __name__ == '__main__':
    main()