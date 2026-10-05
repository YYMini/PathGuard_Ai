"""Pre-scoring contracts for confirmatory Stage 7. No final GPS is read here."""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib, json, platform, subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import compare_stage5_baselines as baseline
from src import train_multiuser_autoencoder as single
from src import train_autoencoder as training
from src.audit_alert_aggregation import load_locked_policies, METRICS as ALERT_METRICS
from src.synthetic_anomalies import ANOMALY_TYPES
from src.data_quality import MIN_TRAJECTORY_POINTS, MAX_LOW_QUALITY_RATIO

ROOT = Path(__file__).resolve().parents[1]
USERS = tuple(f'{i:03}' for i in range(20, 40))
SEEDS = (7, 21, 42, 100, 2026)
PROTOCOL = Path('configs/stage7_final_protocol.json')
METRICS = Path('outputs/metrics/stage7/final')
DATA = Path('data/processed/stage7/final_unseen_u020_039_first5')
SOURCE_FILES = ('src/stage7_protocol.py', 'src/prepare_final_cohort.py',
                'src/evaluate_final_validation.py')
METRIC_NAMES = tuple(single.METRIC_NAMES)
LOCK = ALERT_METRICS / 'locked_alert_policies.json'

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def git(*args, root=ROOT):
    return subprocess.check_output(['git', *args], cwd=root).decode().strip()

def assert_hashes(root, hashes):
    for path, expected in hashes.items():
        if sha(Path(root) / path) != expected:
            raise ValueError('Frozen artifact changed: ' + path)

def validate_protocol(p):
    expected = dict(stage=7, purpose='confirmatory_final_validation',
                    final_users=list(USERS), max_trajectories_per_user=5,
                    anomaly_types=list(ANOMALY_TYPES), model_seeds=list(SEEDS),
                    feature_list=training.FEATURE_COLUMNS, alert_policy='G0_C60',
                    no_tuning_after_evaluation=True, synthetic_seed=42,
                    min_trajectory_points=MIN_TRAJECTORY_POINTS,
                    max_low_quality_ratio=MAX_LOW_QUALITY_RATIO,
                    threshold_comparison='score > threshold',
                    evaluation_semantics='quality-valid original normal once + synthetic label=1 only',
                    std_ddof=1, missing_user_policy='report_and_fail_no_replacement')
    for k, v in expected.items():
        if p.get(k) != v:
            raise ValueError('Frozen protocol contract mismatch: ' + k)
    return p

def protected_snapshot(root):
    root = Path(root)
    old = read(root / 'outputs/metrics/stage67_protected_snapshot.json')
    assert_hashes(root, old)
    # Progress logs/publication receipts are not research outputs.
    for base in ['models/stage4', 'models/stage5', 'data/processed/stage5',
                 'outputs/metrics/stage5', 'outputs/metrics/stage6',
                 'outputs/figures/stage6']:
        for path in (root / base).rglob('*'):
            if path.is_file():
                old[path.relative_to(root).as_posix()] = sha(path)
    for path in git('ls-files', root=root).splitlines():
        if path != 'README.md' and not path.startswith(('configs/stage7', 'src/stage7', 'src/prepare_final', 'src/evaluate_final', 'tests/test_stage7')):
            old[path] = sha(root / path)
    return dict(sorted(old.items()))

def prior_user_audit(root):
    """Inspect user columns in all previous data/metrics; never open final raw GPS."""
    root = Path(root)
    records, prior = [], set()
    paths = list((root/'data/processed').glob('*.csv'))
    for base in ['data/processed/stage5', 'outputs/metrics/stage5', 'outputs/metrics/stage6']:
        paths += list((root/base).rglob('*.csv'))
    for path in sorted(paths):
        header = pd.read_csv(path, nrows=0)
        if 'user_id' not in header:
            continue
        observed = set(pd.read_csv(path, usecols=['user_id'], dtype={'user_id':str})['user_id'].dropna())
        ids = {u for u in observed if len(u)==3 and u.isascii() and u.isdigit()}
        prior |= ids
        records.append({'path':path.relative_to(root).as_posix(), 'sha256':sha(path), 'users':sorted(ids)})
    model_records = []
    for path in sorted((root/'models').rglob('*config.json')):
        obj = read(path)
        splits = obj.get('user_splits', {})
        ids = {u for group in splits.values() for u in group}
        prior |= ids
        model_records.append({'path':path.relative_to(root).as_posix(), 'sha256':sha(path), 'fitting_users':sorted(ids)})
    overlap = sorted(prior & set(USERS))
    if overlap:
        raise ValueError('Final users previously used: ' + ','.join(overlap))
    if not set(f'{i:03}' for i in range(20)).issubset(prior):
        raise ValueError('Incomplete prior-user audit.')
    return {'prior_users':sorted(prior), 'final_users':list(USERS), 'user_overlap':overlap,
            'checked_csvs':records, 'model_configs':model_records,
            'scope':'Local Stage 1-6 datasets, prediction/metric user columns, model fitting configs and frozen source hashes; undocumented external/manual access cannot be inferred.'}

def build_model_manifest(root):
    root = Path(root)
    summary, users = single.read_stage5_metadata(root / single.DEFAULT_DATA_DIR.relative_to(ROOT))
    train_users = sorted(users.loc[users.dataset_split.eq('train'), 'user_id'].astype(str))
    base = Path('models/stage5') / summary['dataset_id']
    records = []
    for family in baseline.MODEL_NAMES:
        for seed in ([None] if family=='statistical_rule' else SEEDS):
            directory = base / (f'seed_{seed}' if family=='autoencoder' else
                                'baselines/statistical_rule' if family=='statistical_rule' else
                                f'baselines/isolation_forest/seed_{seed}')
            config_path = directory / ('training_config.json' if family=='autoencoder' else 'run_config.json')
            config = read(root/config_path)
            if config['feature_columns'] != training.FEATURE_COLUMNS:
                raise ValueError('Frozen model features mismatch.')
            count = config.get('scaler_fit_row_count', config.get('fit_row_count'))
            if count != 86067:
                raise ValueError('Unexpected frozen Train row count.')
            if family=='autoencoder' and config['seed']!=seed:
                raise ValueError('AE seed mismatch.')
            if family=='isolation_forest' and (config['seed']!=seed or config['isolation_forest']!=baseline.IF_PARAMETERS):
                raise ValueError('IF frozen config mismatch.')
            model_path = directory / ('model.pt' if family=='autoencoder' else
                                      'rule_parameters.json' if family=='statistical_rule' else 'model.joblib')
            scaler_path = directory / 'scaler.joblib' if family!='statistical_rule' else None
            # All eleven original artifacts exist. No reconstruction or fitting needed.
            record = dict(detector=family, seed=seed, model_path=model_path.as_posix(),
                          model_sha256=sha(root/model_path),
                          scaler_path=scaler_path.as_posix() if scaler_path else None,
                          scaler_sha256=sha(root/scaler_path) if scaler_path else None,
                          threshold=float(config['threshold']), feature_list=training.FEATURE_COLUMNS,
                          training_users=train_users, training_row_count=count,
                          config_path=config_path.as_posix(), config_sha256=sha(root/config_path),
                          inference_device='cpu', reconstruction_performed=False)
            if not np.isfinite(record['threshold']):
                raise ValueError('Invalid threshold.')
            records.append(record)
    return dict(artifact_type='stage7_frozen_models', models=records,
                model_training=False, model_reconstruction=False, threshold_recalculation=False,
                feature_list=training.FEATURE_COLUMNS)

@contextmanager
def no_fit_guard():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    def deny(*args, **kwargs):
        raise RuntimeError('Final inference forbids fitting/training/threshold recomputation.')
    with ExitStack() as stack:
        for cls, name in [(StandardScaler,'fit'),(StandardScaler,'partial_fit'),
                          (IsolationForest,'fit'),(baseline.StatisticalRule,'fit')]:
            stack.enter_context(patch.object(cls, name, deny))
        for module, names in [(training, ['fit_train_scaler','train_model','calculate_threshold']),
                              (single, ['fit_and_select','fit_train_scaler','train_model','calculate_threshold']), (baseline, ['select_baseline','fit_isolation_forest'])]:
            for name in names:
                stack.enter_context(patch.object(module, name, deny))
        yield

def model_hashes(manifest):
    result = {}
    for m in manifest['models']:
        result[m['model_path']] = m['model_sha256']
        result[m['config_path']] = m['config_sha256']
        if m['scaler_path']:
            result[m['scaler_path']] = m['scaler_sha256']
    return result

def freeze(root=ROOT):
    root = Path(root)
    if (root/PROTOCOL).exists() or (root/METRICS/'final_protocol_manifest.json').exists():
        raise FileExistsError('Protocol already exists; refuse overwrite.')
    (root/METRICS).mkdir(parents=True, exist_ok=True)
    snapshot = protected_snapshot(root)
    audit = prior_user_audit(root)
    models = build_model_manifest(root)
    lock = load_locked_policies(root, root/LOCK)
    if any(lock.for_model(m).policy_id!='G0_C60' for m in baseline.MODEL_NAMES):
        raise ValueError('Existing lock is not G0_C60.')
    generators = ['src/synthetic_anomalies.py','src/prepare_multiuser_dataset.py',
                  'src/feature_engineering.py','src/data_quality.py','src/load_multiuser_geolife.py',
                  'src/load_geolife.py','src/prepare_dataset.py']
    sources = {p:sha(root/p) for p in SOURCE_FILES}
    protocol = dict(stage=7,purpose='confirmatory_final_validation',final_users=list(USERS),
        max_trajectories_per_user=5,trajectory_selection='lexical filename first5; no replacement',
        anomaly_types=list(ANOMALY_TYPES),samples_per_type=1,synthetic_seed=42,
        synthetic_seed_policy='Stage5 synthetic_seed_for(source,type,42); SHA256 [seed,source,type,1]',
        generator_configuration={'segment_points':[10,30],'route_metres':[100,300],
            'speed_multiplier':[3,5],'speed_clip_mps':[10,45],'long_stop_min_sec':120,
            'long_stop_jitter_metres':[-1,1],'direction_metres':[15,40]},
        generator_source_hashes={p:sha(root/p) for p in generators},
        evaluator_source_hashes=sources,model_seeds=list(SEEDS),feature_list=training.FEATURE_COLUMNS,
        min_trajectory_points=100,max_low_quality_ratio=.05,
        missing_user_policy='report_and_fail_no_replacement',
        quality_zero_eligible_user_policy='retain_requested_user_in_report; undefined metrics with reason',
        fingerprint_definition='Stage5 SHA256 rounded lat/lon7 + elapsed nanoseconds, sorted timestamp',
        duplicate_policy='exclude cross-stage copies first; within-final retain first original filename/global ID; no replacement',
        alert_policy='G0_C60',alert_lock_path=LOCK.as_posix(),alert_lock_sha256=sha(root/LOCK),
        point_metrics=list(METRIC_NAMES),event_metrics=['EDR','early10','early25','early50',
            'median/p90_delay_points','median/p90_delay_seconds','coverage','longest_run','fragmentation'],
        evaluation_semantics='quality-valid original normal once + synthetic label=1 only',
        threshold_comparison='score > threshold',std_ddof=1,no_tuning_after_evaluation=True,
        model_manifest=models, prior_user_audit_sha256=None,
        raw_notification_comparator='G0_C0 reference only; no candidate search',
        execution={'device':'cpu','torch_num_threads':1,'batch_size':1024},
        undefined_metric_policy='null/blank + explicit NA reason/support; no NaN/inf in defined metrics')
    save(root/METRICS/'final_model_manifest.json', models)
    save(root/METRICS/'prior_user_audit.json', audit)
    save(root/'outputs/metrics/stage7_protected_snapshot.json', snapshot)
    protocol['prior_user_audit_sha256'] = sha(root/METRICS/'prior_user_audit.json')
    validate_protocol(protocol)
    save(root/PROTOCOL, protocol)
    print('Pre-cohort freeze prepared:', len(snapshot), 'protected;', len(models['models']), 'frozen models; prior overlap 0')
    return protocol

def seal(root=ROOT):
    """Run only after the freeze commit and its tests, before cohort creation."""
    root = Path(root); p = validate_protocol(read(root/PROTOCOL))
    assert_hashes(root, p['generator_source_hashes'])
    assert_hashes(root, p['evaluator_source_hashes'])
    assert_hashes(root, model_hashes(p['model_manifest']))
    commit = git('rev-parse','HEAD',root=root)
    if git('log','-1','--format=%s',root=root)!='stage7: freeze final validation protocol':
        raise ValueError('Protocol must be committed before final data.')
    if git('status','--porcelain',root=root):
        raise ValueError('Freeze commit must have a clean worktree.')
    obj = dict(protocol_sha256=sha(root/PROTOCOL), protocol_freeze_commit_sha=commit,
        generator_hashes=p['generator_source_hashes'],model_hashes=model_hashes(p['model_manifest']),
        locked_alert_policy_sha256=p['alert_lock_sha256'],cohort_users=p['final_users'],
        evaluator_source_hashes=p['evaluator_source_hashes'], timestamp=now(),
        protocol_committed_before_cohort=True,model_manifest_sha256=sha(root/METRICS/'final_model_manifest.json'))
    save(root/METRICS/'final_protocol_manifest.json',obj)
    print('Protocol sealed:', commit, obj['protocol_sha256'])
    return obj

def verify_freeze(root=ROOT):
    root=Path(root); p=validate_protocol(read(root/PROTOCOL))
    receipt=read(root/METRICS/'final_protocol_manifest.json')
    if sha(root/PROTOCOL)!=receipt['protocol_sha256']:
        raise ValueError('Protocol modified after freeze.')
    freeze_sha = receipt['protocol_freeze_commit_sha']
    committed = subprocess.check_output(['git','show',freeze_sha+':'+PROTOCOL.as_posix()],cwd=root)
    if hashlib.sha256(committed).hexdigest()!=receipt['protocol_sha256']:
        raise ValueError('Protocol does not match its pre-scoring commit.')
    if git('log','-1','--format=%s',freeze_sha,root=root)!='stage7: freeze final validation protocol':
        raise ValueError('Invalid protocol freeze commit evidence.')
    assert_hashes(root,p['generator_source_hashes'])
    assert_hashes(root,p['evaluator_source_hashes'])
    assert_hashes(root,model_hashes(p['model_manifest']))
    if read(root/METRICS/'final_model_manifest.json')!=p['model_manifest']:
        raise ValueError('Model manifest modified.')
    if sha(root/LOCK)!=p['alert_lock_sha256']:
        raise ValueError('Alert lock modified.')
    if sha(root/METRICS/'prior_user_audit.json')!=p['prior_user_audit_sha256']:
        raise ValueError('Prior-user audit changed.')
    assert_hashes(root,read(root/'outputs/metrics/stage7_protected_snapshot.json'))
    return p, receipt

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['freeze','seal','verify'],required=True)
    phase=ap.parse_args().phase
    {'freeze':freeze,'seal':seal,'verify':verify_freeze}[phase]()
