"""Stage 5-B: a fixed Stage 4 autoencoder evaluated on unseen users.

Selection accepts only Train and Validation; Test inference happens after the
checkpoint and normal-only percentile threshold have been saved. Stage 5-A is
read-only, with explicit point lineage rather than Stage 4 positional matching.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from typing import Sequence

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch

from src.autoencoder import Autoencoder
from src.load_multiuser_geolife import read_dataset_csv
from src.prepare_multiuser_dataset import DEFAULT_DATASET_ID, validate_saved_dataset
from src.synthetic_anomalies import ANOMALY_TYPES
from src.train_autoencoder import (
    FEATURE_COLUMNS, MODEL_ARCHITECTURE, TrainingOptions, add_bearing_features,
    calculate_threshold, create_figures, fit_train_scaler, reconstruction_errors,
    set_seed, train_model, transform_with_scaler, validate_feature_values,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / 'data' / 'processed' / 'stage5' / DEFAULT_DATASET_ID
METRIC_NAMES = ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate',
                'roc_auc', 'average_precision']
LINEAGE_COLUMNS = ['user_id', 'trajectory_id', 'original_trajectory_id',
                   'source_trajectory_id', 'source_point_index', 'sample_id',
                   'dataset_split', 'timestamp', 'is_synthetic', 'anomaly_label',
                   'anomaly_type', 'anomaly_segment_id']
COMPARISON_NOTE = (
    'Stage 4 evaluates other trajectories of user 000; Stage 5 evaluates unseen users. '
    'Dataset sizes, anomaly prevalence and source trajectories differ. '
    'Deltas describe the changed evaluation condition, not model superiority.'
)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_stage5_metadata(data_dir: Path) -> tuple[dict, pd.DataFrame]:
    summary = json.loads((data_dir / 'dataset_summary.json').read_text(encoding='utf-8'))
    if summary.get('stage') != '5-A' or 'deduplication' not in summary:
        raise ValueError('A deduplicated Stage 5-A dataset is required.')
    if not summary.get('ready_for_stage5_b'):
        raise ValueError('Stage 5-A dataset is not ready for Stage 5-B.')
    if summary['model_feature_columns_for_stage5_b'] != FEATURE_COLUMNS:
        raise ValueError('Stage 5-A feature schema differs from Stage 4.')
    return summary, read_dataset_csv(data_dir / 'user_split_manifest.csv')


def load_stage5_split(data_dir: Path, split: str, summary: dict,
                      user_manifest: pd.DataFrame) -> pd.DataFrame:
    """Load the frozen model/evaluation CSV and reject silent row filtering."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown dataset split.')
    name = 'train.csv' if split == 'train' else f'{split}_evaluation.csv'
    frame = read_dataset_csv(data_dir / name)
    missing = set(LINEAGE_COLUMNS + ['source_quality_valid', 'is_low_quality',
                                    'is_training_eligible', 'synthetic_value_valid']) - set(frame.columns)
    if missing:
        raise ValueError(f'{split} missing lineage/quality columns: {sorted(missing)}')
    normal = ~frame.is_synthetic & frame.anomaly_label.eq(0)
    anomaly = frame.is_synthetic & frame.anomaly_label.eq(1)
    if frame.empty or not (normal | anomaly).all():
        raise ValueError(f'{split} contains synthetic normal or invalid labels.')
    if split == 'train' and not normal.all():
        raise ValueError('Train must contain original normal rows only, with zero synthetic rows.')
    if not (frame.source_quality_valid & ~frame.is_low_quality &
            frame.is_training_eligible & frame.synthetic_value_valid).all():
        raise ValueError(f'{split} contains invalid or low-quality source rows.')
    expected_users = set(user_manifest.loc[user_manifest.dataset_split.eq(split), 'user_id'])
    if not set(frame.user_id).issubset(expected_users) or not frame.dataset_split.eq(split).all():
        raise ValueError(f'{split} contains users from another split.')
    trajectories = read_dataset_csv(data_dir / 'trajectory_quality_summary.csv')
    expected_sources = set(trajectories.loc[trajectories.eligible_for_dataset &
                           trajectories.dataset_split.eq(split), 'trajectory_id'])
    if set(frame.source_trajectory_id) != expected_sources:
        raise ValueError(f'{split} includes excluded duplicates or misses eligible trajectories.')
    if frame.duplicated(['sample_id', 'source_trajectory_id', 'source_point_index']).any():
        raise ValueError(f'{split} contains duplicate sample lineage.')
    if not frame.loc[anomaly, 'anomaly_type'].isin(ANOMALY_TYPES).all():
        raise ValueError(f'{split} contains unknown anomaly types.')
    stats = summary['splits'][split]
    if (len(frame) != stats['evaluation_rows'] or int(normal.sum()) != stats['normal_original_rows']
            or int(anomaly.sum()) != stats['synthetic_anomaly_rows']):
        raise ValueError(f'{split} row counts differ from Stage 5-A summary.')
    result = add_bearing_features(frame)
    validate_feature_values(result, split)
    return result


def validation_normal_rows(validation: pd.DataFrame) -> pd.DataFrame:
    result = validation.loc[~validation.is_synthetic & validation.anomaly_label.eq(0)].copy()
    if result.empty or not result.source_quality_valid.all():
        raise ValueError('No valid Validation original normal rows for model selection.')
    return result


def fit_and_select(train: pd.DataFrame, validation: pd.DataFrame, options: TrainingOptions,
                   device: torch.device, model_path: Path) -> dict:
    """No Test argument; synthetic Validation rows never reach the trainer/scaler."""
    if train.is_synthetic.any() or not train.anomaly_label.eq(0).all():
        raise ValueError('Selection requires normal-only Train.')
    normal = validation_normal_rows(validation)
    scaler, train_values = fit_train_scaler(train)
    normal_values = transform_with_scaler(scaler, normal)
    model = Autoencoder(len(FEATURE_COLUMNS)).to(device)
    history, best_epoch = train_model(model, train_values, normal_values, options, device, model_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    errors = reconstruction_errors(model, normal_values, device)
    if not np.isfinite(errors).all():
        raise ValueError('Validation normal errors are not finite.')
    threshold = calculate_threshold(errors, normal.anomaly_label, options.threshold_percentile)
    return {'model': model, 'scaler': scaler, 'history': history, 'best_epoch': best_epoch,
            'best_validation_normal_loss': float(history.loc[history.epoch.eq(best_epoch),
                                                            'validation_normal_loss'].iloc[0]),
            'threshold': threshold, 'validation_normal_row_count': len(normal),
            'scaler_fit_row_count': int(scaler.n_samples_seen_)}


def score_split(frame: pd.DataFrame, selection: dict, device: torch.device) -> pd.DataFrame:
    values = transform_with_scaler(selection['scaler'], frame)
    errors = reconstruction_errors(selection['model'], values, device)
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('Non-finite or negative reconstruction errors.')
    # Preserve all input columns, including explicit lineage and quality audit.
    result = frame.copy()
    result['reconstruction_error'] = errors
    result['predicted_anomaly'] = (errors > selection['threshold']).astype(int)
    return result


def calculate_metrics(predictions: pd.DataFrame, threshold: float) -> dict:
    """Undefined metrics are null with explicit reasons, never coerced to zero."""
    labels = predictions.anomaly_label.to_numpy(dtype=int)
    scores = predictions.reconstruction_error.to_numpy(dtype=float)
    predicted = predictions.predicted_anomaly.to_numpy(dtype=int)
    if not np.isin(labels, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError('Metrics require binary labels and predictions.')
    if not np.isfinite(scores).all():
        raise ValueError('Metrics require finite scores.')
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    reasons = {}
    def ratio(name: str, numerator: int, denominator: int, reason: str):
        if denominator == 0:
            reasons[name] = reason
            return None
        return float(numerator / denominator)
    metrics = {
        'row_count': len(labels), 'normal_row_count': tn + fp, 'anomaly_row_count': fn + tp,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'threshold': float(threshold),
        'accuracy': ratio('accuracy', tn + tp, len(labels), 'no_rows'),
        'precision': ratio('precision', tp, tp + fp, 'no_predicted_anomaly_rows'),
        'recall': ratio('recall', tp, tp + fn, 'no_actual_anomaly_rows'),
        'false_positive_rate': ratio('false_positive_rate', fp, fp + tn, 'no_actual_normal_rows'),
    }
    if tp + fn == 0:
        metrics['f1_score'] = None
        reasons['f1_score'] = 'no_actual_anomaly_rows'
    else:
        metrics['f1_score'] = float(2 * tp / (2 * tp + fp + fn))
    if len(set(labels)) == 2:
        metrics['roc_auc'] = float(roc_auc_score(labels, scores))
        metrics['average_precision'] = float(average_precision_score(labels, scores))
    else:
        metrics['roc_auc'] = metrics['average_precision'] = None
        reasons['roc_auc'] = reasons['average_precision'] = 'requires_normal_and_anomaly_classes'
    metrics['metric_na_reasons'] = reasons
    return metrics


def per_user_metrics(predictions: pd.DataFrame, threshold: float,
                     user_manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    records = []
    for user in user_manifest.loc[user_manifest.dataset_split.eq('test')].itertuples():
        frame = predictions.loc[predictions.user_id.eq(user.user_id)]
        record = calculate_metrics(frame, threshold)
        reasons = record.pop('metric_na_reasons')
        records.append({'user_id': str(user.user_id),
                        'eligible_trajectory_count': int(user.eligible_trajectory_count),
                        **record, 'na_reasons': json.dumps(reasons, sort_keys=True)})
    result = pd.DataFrame(records)
    macro = {'user_count': len(result), 'metric_means': {}, 'defined_user_counts': {},
             'undefined_user_counts': {},
             'aggregation_policy': 'Unweighted mean across Test users with a defined metric; NA excluded and support recorded.'}
    for name in METRIC_NAMES:
        available = pd.to_numeric(result[name], errors='coerce').dropna()
        macro['metric_means'][name] = float(available.mean()) if len(available) else None
        macro['defined_user_counts'][name] = len(available)
        macro['undefined_user_counts'][name] = len(result) - len(available)
    return result, macro


def per_anomaly_type_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    normal = predictions.loc[~predictions.is_synthetic & predictions.anomaly_label.eq(0)]
    records = []
    for anomaly_type in ANOMALY_TYPES:
        anomaly = predictions.loc[predictions.is_synthetic & predictions.anomaly_label.eq(1) &
                                  predictions.anomaly_type.eq(anomaly_type)]
        # Other anomaly types are excluded; only original normal rows are negatives.
        comparison = calculate_metrics(pd.concat([normal, anomaly], ignore_index=True), threshold)
        scores = anomaly.reconstruction_error.to_numpy(dtype=float)
        reasons = {k: v for k, v in comparison['metric_na_reasons'].items()
                   if k in ('recall', 'roc_auc', 'average_precision')}
        if not len(scores):
            reasons['error_statistics'] = 'no_rows_for_anomaly_type'
        records.append({'anomaly_type': anomaly_type, 'anomaly_row_count': len(anomaly),
                        'detected_anomaly_row_count': int(anomaly.predicted_anomaly.sum()),
                        'recall': comparison['recall'], 'comparison_normal_row_count': len(normal),
                        'roc_auc': comparison['roc_auc'], 'average_precision': comparison['average_precision'],
                        'reconstruction_error_mean': float(np.mean(scores)) if len(scores) else None,
                        'reconstruction_error_median': float(np.median(scores)) if len(scores) else None,
                        'reconstruction_error_p95': float(np.percentile(scores, 95)) if len(scores) else None,
                        'reconstruction_error_p99': float(np.percentile(scores, 99)) if len(scores) else None,
                        'na_reasons': json.dumps(reasons, sort_keys=True)})
    return pd.DataFrame(records)


def compare_stage4_stage5(test_metrics: dict, stage4_path: Path) -> tuple[pd.DataFrame, dict]:
    if stage4_path.is_file():
        baseline = json.loads(stage4_path.read_text(encoding='utf-8'))
        baseline_source = str(stage4_path.resolve())
        baseline_hash = sha256_file(stage4_path)
    else:
        baseline = {'accuracy': .9189, 'precision': .5310, 'recall': .6897, 'f1_score': .6000,
                    'roc_auc': .8183, 'average_precision': .5148}
        baseline_source = 'User-provided rounded Stage 4 final metrics'
        baseline_hash = None
    if 'false_positive_rate' not in baseline and 'false_positive' in baseline:
        baseline['false_positive_rate'] = baseline['false_positive'] / (baseline['false_positive'] + baseline['true_negative'])
    records = []
    for name in METRIC_NAMES:
        before, after = baseline.get(name), test_metrics[name]
        records.append({'metric': name, 'stage4': before, 'stage5': after,
                        'delta_stage5_minus_stage4': after - before if before is not None and after is not None else None})
    report = {'stage4_source': baseline_source, 'stage4_source_sha256': baseline_hash,
              'interpretation': COMPARISON_NOTE, 'stage4_evaluation': 'same user, other trajectories',
              'stage5_evaluation': 'unseen Test users', 'metrics': records,
              'stage4_test_rows': baseline.get('row_count', 987),
              'stage5_test_rows': test_metrics['row_count'],
              'stage4_normal_rows': 900, 'stage4_anomaly_rows': 87,
              'stage5_normal_rows': test_metrics['normal_row_count'],
              'stage5_anomaly_rows': test_metrics['anomaly_row_count']}
    return pd.DataFrame(records), report


def create_stage5_figures(history: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                          per_user: pd.DataFrame, best_epoch: int, threshold: float,
                          figure_dir: Path) -> list[Path]:
    paths = create_figures(history, validation, test, best_epoch, threshold, figure_dir)
    for metric, filename, ylabel in [('f1_score', 'per_user_f1.png', 'F1'),
                                     ('false_positive_rate', 'per_user_fpr.png', 'False positive rate')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = pd.to_numeric(per_user[metric], errors='coerce')
        ax.bar(per_user.user_id, values)
        for i, value in enumerate(values):
            ax.text(i, 0 if pd.isna(value) else value, 'NA' if pd.isna(value) else f'{value:.3f}',
                    ha='center', va='bottom')
        ax.set(xlabel='Unseen Test user', ylabel=ylabel, ylim=(0, 1.08))
        fig.tight_layout()
        path = figure_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def output_directories(output_root: Path, dataset_id: str, seed: int) -> dict[str, Path]:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', dataset_id):
        raise ValueError('Unsafe dataset ID.')
    suffix = Path(dataset_id) / f'seed_{seed}'
    return {'model': output_root / 'models' / 'stage5' / suffix,
            'metrics': output_root / 'outputs' / 'metrics' / 'stage5' / suffix,
            'figures': output_root / 'outputs' / 'figures' / 'stage5' / suffix}


def run_pipeline(data_dir: Path = DEFAULT_DATA_DIR, output_root: Path = ROOT,
                 options: TrainingOptions = TrainingOptions(),
                 stage4_metrics_path: Path | None = None,
                 experiment_stage: str = "5.2") -> dict:
    """Single run. Existing outputs are refused rather than overwritten."""
    started = time.perf_counter()
    data_dir, output_root = Path(data_dir).resolve(), Path(output_root).resolve()
    if experiment_stage not in ('5.2', '5.3'):
        raise ValueError('Unknown Stage 5 experiment.')
    if experiment_stage == '5.3' and (options.seed not in (7, 21, 42, 100, 2026)
            or options != TrainingOptions(seed=options.seed)):
        raise ValueError('Stage 5.3 requires the exact five model seeds and fixed hyperparameters.')
    if options.threshold_percentile != 95 or (experiment_stage == '5.2' and options.seed != 42):
        raise ValueError('Stage 5.2 uses model seed 42; all Stage 5 runs use the fixed 95th percentile.')
    summary, users = read_stage5_metadata(data_dir)
    directories = output_directories(output_root, summary['dataset_id'], options.seed)
    if any(path.exists() for path in directories.values()):
        raise FileExistsError('Stage 5-B output directory already exists; this run will not overwrite it.')
    # Structural Test inspection is allowed here, but there is no Test scoring,
    # scaler fitting, epoch selection, or threshold adjustment in this preflight.
    integrity = validate_saved_dataset(data_dir)
    if not integrity['ready_for_stage5_b']:
        raise ValueError('Dataset leakage preflight failed.')
    dataset_hash_before = sha256_file(data_dir / 'dataset_summary.json')
    train = load_stage5_split(data_dir, 'train', summary, users)
    validation = load_stage5_split(data_dir, 'validation', summary, users)
    for path in directories.values():
        path.mkdir(parents=True)
    model_dir, metrics_dir, figure_dir = (directories[k] for k in ('model', 'metrics', 'figures'))
    set_seed(options.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Stage {experiment_stage}: device={device}; seed={options.seed}; Train rows={len(train)}; '
          f'Validation normal rows={len(validation_normal_rows(validation))}', flush=True)
    selection = fit_and_select(train, validation, options, device, model_dir / 'model.pt')
    joblib.dump(selection['scaler'], model_dir / 'scaler.joblib')
    save_json(model_dir / 'feature_columns.json', {'feature_columns': FEATURE_COLUMNS})
    selection['history'].to_csv(metrics_dir / 'training_history.csv', index=False)
    config = {
        'stage': experiment_stage, 'dataset_id': summary['dataset_id'], 'dataset_path': str(data_dir),
        'dataset_summary_sha256': dataset_hash_before, 'dataset_file_checksums': summary['file_checksums'],
        'feature_columns': FEATURE_COLUMNS, 'model_architecture': MODEL_ARCHITECTURE, **asdict(options),
        'actual_epochs': len(selection['history']), 'best_epoch': selection['best_epoch'],
        'best_validation_normal_loss': selection['best_validation_normal_loss'],
        'threshold': selection['threshold'], 'threshold_comparison': 'reconstruction_error > threshold',
        'threshold_source': 'best checkpoint, Validation original normal only',
        'checkpoint_selection_source': 'Validation original normal MSE only',
        'scaler_fit_source': 'Train original normal only',
        'scaler_fit_row_count': selection['scaler_fit_row_count'],
        'validation_normal_row_count': selection['validation_normal_row_count'],
        'train_row_count': len(train), 'validation_evaluation_row_count': len(validation),
        'device': str(device), 'torch_num_threads': torch.get_num_threads(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'user_splits': {s: sorted(users.loc[users.dataset_split.eq(s), 'user_id'].tolist())
                        for s in ('train', 'validation', 'test')},
        'test_used_for_selection_or_threshold': False, 'validation_anomaly_used_for_selection_or_threshold': False,
    }
    # Selection is frozen on disk before Test is loaded for inference.
    save_json(model_dir / 'training_config.json', config)
    print(f'Frozen checkpoint: epoch={selection["best_epoch"]}; '
          f'Validation normal loss={selection["best_validation_normal_loss"]:.8f}; '
          f'threshold={selection["threshold"]:.8f}. Test evaluation starts now.', flush=True)
    test = load_stage5_split(data_dir, 'test', summary, users)
    validation_predictions = score_split(validation, selection, device)
    test_predictions = score_split(test, selection, device)
    validation_metrics = calculate_metrics(validation_predictions, selection['threshold'])
    test_metrics = calculate_metrics(test_predictions, selection['threshold'])
    per_user, macro = per_user_metrics(test_predictions, selection['threshold'], users)
    per_type = per_anomaly_type_metrics(test_predictions, selection['threshold'])
    if any(int(per_user[key].sum()) != test_metrics[key] for key in ('tn', 'fp', 'fn', 'tp')):
        raise ValueError('Per-user confusion counts do not sum to overall counts.')
    if int(per_type.anomaly_row_count.sum()) != test_metrics['anomaly_row_count']:
        raise ValueError('Per-type counts do not match overall anomaly rows.')
    comparison, comparison_report = compare_stage4_stage5(
        test_metrics, Path(stage4_metrics_path) if stage4_metrics_path else ROOT / 'outputs' / 'metrics' / 'stage4_test_metrics.json')
    for name, frame in [('validation_predictions', validation_predictions), ('test_predictions', test_predictions),
                        ('per_user_metrics', per_user), ('per_anomaly_type_metrics', per_type),
                        ('stage4_stage5_comparison', comparison)]:
        frame.to_csv(metrics_dir / f'{name}.csv', index=False, na_rep='NA')
        if name.endswith('_predictions'):
            restored = read_dataset_csv(metrics_dir / f'{name}.csv')
            pd.testing.assert_frame_equal(frame[LINEAGE_COLUMNS], restored[LINEAGE_COLUMNS], check_dtype=False)
    for name, payload in [('validation_metrics', validation_metrics), ('test_metrics', test_metrics),
                          ('user_macro_metrics', macro), ('stage4_stage5_comparison', comparison_report),
                          ('dataset_validation', integrity)]:
        save_json(metrics_dir / f'{name}.json', payload)
    figure_paths = create_stage5_figures(selection['history'], validation_predictions, test_predictions,
                                        per_user, selection['best_epoch'], selection['threshold'], figure_dir)
    config.update(test_evaluation_row_count=len(test), elapsed_seconds=time.perf_counter() - started)
    save_json(model_dir / 'training_config.json', config)
    # Verify the immutable input dataset after all training/plotting work.
    if sha256_file(data_dir / 'dataset_summary.json') != dataset_hash_before:
        raise ValueError('Stage 5-A summary changed during training.')
    if any(sha256_file(data_dir / name) != digest for name, digest in summary['file_checksums'].items()):
        raise ValueError('Stage 5-A dataset changed during training.')
    report = {'config': config, 'validation_metrics': validation_metrics, 'test_metrics': test_metrics,
              'user_macro_metrics': macro, 'figures': [str(p) for p in figure_paths],
              'output_directories': {k: str(v) for k, v in directories.items()},
              'dataset_unchanged': True, 'interpretation': COMPARISON_NOTE}
    save_json(metrics_dir / 'run_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return {**report, 'selection': selection, 'per_user': per_user, 'per_type': per_type,
            'test_predictions': test_predictions, 'validation_predictions': validation_predictions}


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Stage 5-B fixed single-seed Autoencoder generalization experiment.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parsed = parser.parse_args(args)
    run_pipeline(parsed.data_dir, parsed.output_root, TrainingOptions())


if __name__ == '__main__':
    main()