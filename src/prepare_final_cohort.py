"""Create users 020-039 only AFTER committed protocol sealing; no model scoring."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src import stage7_protocol as contract
from src.load_geolife import DEFAULT_DATA_ROOT
from src.load_multiuser_geolife import load_multiuser_features, read_dataset_csv, trajectory_fingerprint
from src.data_quality import add_quality_flags, summarize_trajectories
from src.prepare_dataset import add_normal_metadata
from src.prepare_multiuser_dataset import (generate_evaluation_synthetic, evaluation_rows,
    validate_numeric_values, attach_source_quality, QUALITY_FIELDS)
from src.train_autoencoder import add_bearing_features, FEATURE_COLUMNS

KEY = ['source_trajectory_id','source_point_index']
EXCLUSION_COLUMNS = ['duplicate_fingerprint','user_id','trajectory_id','retained_trajectory',
                     'excluded_trajectory','split','prior_split','excluded_rows','exclusion_reason']

def select_first5(data_root, users=contract.USERS):
    """Only directory names are examined, not GPS content or performance."""
    records=[]
    for user in users:
        directory=Path(data_root)/user/'Trajectory'
        files=sorted(directory.glob('*.plt'),key=lambda p:p.name) if directory.is_dir() else []
        records.append(dict(user_id=user,available_trajectory_count=len(files),
                            selected_filenames=[p.name for p in files[:5]],missing_user=not files))
    return records

def exclude_duplicates(inputs, prior):
    required=['user_id','trajectory_id','original_trajectory_id','content_fingerprint',
              'eligible_for_dataset','exclusion_reason','cleaned_row_count']
    if any(c not in inputs for c in required):
        raise ValueError('Incomplete trajectory manifest.')
    result=inputs.copy()
    result['quality_eligible_for_dataset']=result.eligible_for_dataset.astype(bool)
    result['dataset_split']='final'
    result['is_exact_duplicate']=False
    result['retained_trajectory_id']=''
    result['duplicate_excluded_rows']=0
    records=[]
    for fp, group in result.loc[result.content_fingerprint.ne('')].groupby('content_fingerprint',sort=True):
        ordered=group.sort_values(['original_trajectory_id','trajectory_id'],kind='stable')
        previous=prior.loc[prior.content_fingerprint.eq(fp)].sort_values(['original_trajectory_id','trajectory_id'],kind='stable')
        retained=str(previous.trajectory_id.iloc[0]) if len(previous) else str(ordered.trajectory_id.iloc[0])
        excluded=ordered if len(previous) else ordered.iloc[1:]
        if not len(excluded):
            continue
        result.loc[group.index,'retained_trajectory_id']=retained
        for idx, row in excluded.iterrows():
            reason='cross_stage_exact_duplicate_trajectory' if len(previous) else 'exact_duplicate_trajectory'
            result.loc[idx,'is_exact_duplicate']=True
            result.loc[idx,'eligible_for_dataset']=False
            result.loc[idx,'duplicate_excluded_rows']=int(row.cleaned_row_count)
            result.loc[idx,'exclusion_reason']=';'.join(filter(None,[str(row.exclusion_reason),reason]))
            records.append(dict(duplicate_fingerprint=str(fp),user_id=str(row.user_id),trajectory_id=str(row.trajectory_id),
                retained_trajectory=retained,excluded_trajectory=str(row.trajectory_id),split='final',
                prior_split=str(previous.dataset_split.iloc[0]) if len(previous) else '',
                excluded_rows=int(row.cleaned_row_count),exclusion_reason=reason))
    return result,pd.DataFrame(records,columns=EXCLUSION_COLUMNS)

def read_final_csv(path):
    # The historical reader supplies audited ID/boolean semantics. Preserve float bits too.
    f=read_dataset_csv(path)
    cols=[c for c in ['latitude','longitude',*FEATURE_COLUMNS] if c in f]
    if cols:
        exact=pd.read_csv(path,usecols=cols,float_precision='round_trip')
        f[cols]=exact[cols]
    return f

def validate_cohort(normal,synthetic,inputs,prior):
    if not set(normal.user_id.astype(str)).issubset(contract.USERS):
        raise ValueError('Unexpected final user.')
    if normal.duplicated(KEY).any() or synthetic.duplicated(['sample_id',*KEY]).any():
        raise ValueError('Duplicate final lineage.')
    expected=normal.user_id.astype(str)+'__'+normal.original_trajectory_id.astype(str)
    if not normal.source_trajectory_id.astype(str).eq(expected).all():
        raise ValueError('Final identity mismatch.')
    if normal.is_synthetic.any() or normal.anomaly_label.ne(0).any():
        raise ValueError('Original labels changed.')
    if not synthetic.is_synthetic.all() or not synthetic.anomaly_label.isin([0,1]).all():
        raise ValueError('Invalid synthetic labels.')
    joined=attach_source_quality(synthetic,normal)
    for c in QUALITY_FIELDS:
        if not joined[c].reset_index(drop=True).equals(synthetic[c].reset_index(drop=True)):
            raise ValueError('Source quality lineage changed: '+c)
    for source, group in synthetic.groupby('source_trajectory_id',sort=True):
        original=normal.loc[normal.source_trajectory_id.eq(source)]
        if group.sample_id.nunique()!=4 or set(group.anomaly_type)!=set(contract.ANOMALY_TYPES):
            raise ValueError('Synthetic sample type/count mismatch.')
        for _, sample in group.groupby('sample_id',sort=True):
            if set(sample.source_point_index)!=set(original.source_point_index):
                raise ValueError('Incomplete synthetic lineage.')
            if sample.anomaly_type.nunique()!=1 or not sample.anomaly_label.eq(1).any():
                raise ValueError('Invalid anomaly segment.')
    retained=inputs.loc[inputs.eligible_for_dataset]
    if set(normal.source_trajectory_id)!=set(retained.trajectory_id):
        raise ValueError('Excluded trajectory entered final dataset.')
    if retained.content_fingerprint.isin(prior.content_fingerprint[prior.content_fingerprint.ne('')]).any():
        raise ValueError('Cross-stage duplicate remains.')
    if retained.content_fingerprint.duplicated().any():
        raise ValueError('Within-final duplicate remains.')
    for source, group in normal.groupby('source_trajectory_id',sort=True):
        fp=trajectory_fingerprint(group)
        if fp!=inputs.loc[inputs.trajectory_id.eq(source),'content_fingerprint'].iloc[0]:
            raise ValueError('CSV round-trip fingerprint changed.')
    validate_numeric_values(normal); validate_numeric_values(synthetic)
    return dict(prior_user_overlap=0,cross_stage_trajectory_duplicate_after_exclusions=0,
                within_final_duplicate_after_exclusions=0,lineage_errors=0,nonfinite_feature_count=0)

def prepare(root=contract.ROOT,data_root=DEFAULT_DATA_ROOT):
    root=Path(root)
    protocol,receipt=contract.verify_freeze(root)
    out=root/contract.DATA
    if out.exists():
        raise FileExistsError('Final cohort already exists; no overwrite/replacement.')
    selection=select_first5(data_root)
    missing=[r['user_id'] for r in selection if r['missing_user']]
    if missing:
        contract.save(root/contract.METRICS/'final_cohort_integrity.json',
                      dict(requested_users=list(contract.USERS),missing_users=missing,
                           selection=selection,status='missing_user_fail',replacement=False))
        raise ValueError('Missing fixed final users: '+','.join(missing))
    featured,inputs=load_multiuser_features(Path(data_root),contract.USERS,5)
    validate_numeric_values(featured)
    checked=add_quality_flags(featured)
    summaries=summarize_trajectories(checked)
    quality=inputs.merge(summaries,on='trajectory_id',how='left',validate='one_to_one')
    empty=quality.load_status.eq('all_rows_invalid')
    quality.loc[empty,'eligible_for_dataset']=False
    quality.loc[empty,'exclusion_reason']='all_rows_invalid'
    quality['eligible_for_dataset']=quality.eligible_for_dataset.fillna(False).astype(bool)
    quality['exclusion_reason']=quality.exclusion_reason.fillna('')
    quality['eligible_normal_row_count']=quality.trajectory_id.map(
        checked.groupby('trajectory_id').is_training_eligible.sum()).fillna(0).astype(int)
    prior=read_dataset_csv(root/single_data_path()/'input_manifest.csv')
    inputs,excluded=exclude_duplicates(quality,prior)
    eligible=set(inputs.loc[inputs.eligible_for_dataset,'trajectory_id'])
    if not eligible:
        raise ValueError('No eligible final trajectories; no replacement.')
    normal=add_normal_metadata(checked.loc[checked.trajectory_id.isin(eligible)].copy())
    normal['source_quality_valid']=~normal.is_low_quality & normal.is_training_eligible
    normal['synthetic_value_valid']=True
    normal['dataset_split']='test'  # Historical evaluation APIs use the Test semantic tag.
    normal=add_bearing_features(normal)
    assignments={u:'test' for u in contract.USERS}
    synthetic,manifest=generate_evaluation_synthetic(normal,assignments,protocol['synthetic_seed'])
    synthetic=add_bearing_features(synthetic)
    evaluation=evaluation_rows(pd.concat([normal,synthetic],ignore_index=True))
    checks=validate_cohort(normal,synthetic,inputs,prior)
    from src.audit_alert_aggregation import build_catalog
    route=synthetic.loc[synthetic.anomaly_type.eq('route_deviation')]
    route_catalog=build_catalog(route,manifest.loc[manifest.anomaly_type.eq('route_deviation')],
                               evaluation.loc[evaluation.anomaly_type.eq('route_deviation')])
    if route_catalog.evaluated_point_count.eq(0).any():
        raise ValueError('Unscorable route event: frozen Stage6 event protocol requires observed predictions; report and stop before scoring.')
    out.mkdir(parents=True)
    frames={'final_normal':normal,'final_synthetic':synthetic,'final_evaluation':evaluation,
            'input_manifest':inputs,'excluded_duplicates':excluded,'quality_summary':inputs,
            'synthetic_anomaly_manifest':manifest}
    for name, frame in frames.items():
        frame.to_csv(out/(name+'.csv'),index=False)
    reread={k:read_final_csv(out/(k+'.csv')) for k in ['final_normal','final_synthetic','input_manifest']}
    checks=validate_cohort(reread['final_normal'],reread['final_synthetic'],reread['input_manifest'],prior)
    actual=read_final_csv(out/'final_evaluation.csv')
    expected=evaluation_rows(pd.concat([reread['final_normal'],reread['final_synthetic']],ignore_index=True))
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
    stats=dict(requested_users=list(contract.USERS),available_users=[r['user_id'] for r in selection],
        missing_users=[],selection=selection,trajectory_count=len(inputs),eligible_trajectory_count=len(eligible),
        quality_eligible_trajectory_count=int(inputs.quality_eligible_for_dataset.sum()),
        excluded_quality_trajectories=int((~inputs.quality_eligible_for_dataset).sum()),
        excluded_duplicate_count=len(excluded),excluded_duplicate_rows=int(excluded.excluded_rows.sum()),
        cross_stage_duplicate_exclusions=int(excluded.exclusion_reason.eq('cross_stage_exact_duplicate_trajectory').sum()),
        within_final_duplicate_exclusions=int(excluded.exclusion_reason.eq('exact_duplicate_trajectory').sum()),
        normal_full_rows=len(normal),normal_rows=int((~evaluation.is_synthetic).sum()),
        synthetic_samples=synthetic.sample_id.nunique(),synthetic_full_rows=len(synthetic),
        full_labelled_anomaly_rows=int(synthetic.anomaly_label.sum()),
        anomaly_labelled_rows=int(evaluation.anomaly_label.sum()),route_event_count=len(route_catalog),
        per_user=[dict(user_id=u,input_trajectories=int(inputs.user_id.eq(u).sum()),
            eligible_trajectories=int((inputs.user_id.eq(u)&inputs.eligible_for_dataset).sum()),
            normal_rows=int((evaluation.user_id.eq(u)&~evaluation.is_synthetic).sum()),
            anomaly_rows=int((evaluation.user_id.eq(u)&evaluation.is_synthetic).sum())) for u in contract.USERS],
        **checks,protocol_sha256=receipt['protocol_sha256'],created_at=contract.now(),
        duplicate_exclusions=excluded.to_dict('records'),no_replacement=True,
        nan_inf_count=0,quality_filter='Stage5 unchanged; source quality, not generated-effect quality')
    file_hashes={p.name:contract.sha(p) for p in out.glob('*.csv')}
    contract.save(out/'final_manifest.json',dict(protocol_sha256=receipt['protocol_sha256'],
        freeze_commit_sha=receipt['protocol_freeze_commit_sha'],cohort=stats,file_hashes=file_hashes))
    contract.save(root/contract.METRICS/'final_cohort_integrity.json',stats)
    contract.verify_freeze(root)
    print('Final cohort created; no scoring:',len(eligible),'trajectories;',stats['normal_rows'],
          'normal rows;',stats['anomaly_labelled_rows'],'anomaly rows;',len(excluded),'duplicates excluded')
    return stats

def single_data_path():
    from src.train_multiuser_autoencoder import DEFAULT_DATA_DIR
    return DEFAULT_DATA_DIR.relative_to(contract.ROOT)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT)
    prepare(data_root=ap.parse_args().data_root)
