"""GPS CSV demo using existing frozen artifacts. No training or tuning."""
from __future__ import annotations
import argparse, html, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src import stage7_protocol as p
from src.evaluate_final_validation import load_frozen_detector,infer
from src.feature_engineering import generate_trajectory_features
from src.data_quality import add_quality_flags
from src.train_autoencoder import add_bearing_features
from src.alert_policy import AlertPolicy,apply_policy
from src.visualize_route import create_route_map

FAMILIES={'rule':'statistical_rule','if':'isolation_forest','ae':'autoencoder'}

def validate_input(frame):
    needed=['timestamp','latitude','longitude']
    if frame.empty or any(c not in frame for c in needed):
        raise ValueError('Nonempty CSV with timestamp,latitude,longitude required.')
    f=frame.copy()
    for key,default in [('user_id','demo'),('trajectory_id','example')]:
        if key not in f:f[key]=default
        if f[key].isna().any() or f[key].astype(str).str.strip().eq('').any():
            raise ValueError('Empty identity: '+key)
        f[key]=f[key].astype(str)
    if 'altitude' not in f:f['altitude']=0.
    f['timestamp']=pd.to_datetime(f.timestamp,errors='raise',format='mixed')
    for c in ['latitude','longitude','altitude']:f[c]=pd.to_numeric(f[c],errors='raise')
    if f.timestamp.isna().any() or not np.isfinite(f[['latitude','longitude','altitude']]).all().all():
        raise ValueError('NaN/inf GPS input.')
    if not f.latitude.between(-90,90).all() or not f.longitude.between(-180,180).all():
        raise ValueError('GPS coordinate range invalid.')
    f['trajectory_id']=f.user_id+'__'+f.trajectory_id
    for _,g in f.groupby('trajectory_id',sort=False):
        if not g.timestamp.diff().dropna().gt(pd.Timedelta(0)).all():
            raise ValueError('Timestamp must strictly increase within each trajectory.')
    return f

def load_demo_detector(detector='ae',seed=42,root=p.ROOT):
    root=Path(root)
    if detector not in FAMILIES or seed not in p.SEEDS:raise ValueError('Unsupported detector/seed.')
    protocol=p.validate_protocol(p.read(root/p.PROTOCOL))
    receipt=p.read(root/p.METRICS/'final_protocol_manifest.json')
    if p.sha(root/p.PROTOCOL)!=receipt['protocol_sha256']:
        raise ValueError('Frozen protocol modified.')
    if p.sha(root/p.LOCK)!=protocol['alert_lock_sha256']:
        raise ValueError('Frozen G0_C60 lock modified.')
    p.assert_hashes(root,protocol['evaluator_source_hashes'])
    p.assert_hashes(root,protocol['generator_source_hashes'])
    family=FAMILIES[detector]
    record=next(m for m in protocol['model_manifest']['models']
                if m['detector']==family and m['seed']==(None if detector=='rule' else seed))
    return load_frozen_detector(record,root)

def predict_csv(input_path,detector='ae',seed=42,root=p.ROOT):
    raw=pd.read_csv(input_path,dtype={'user_id':str,'trajectory_id':str},float_precision='round_trip')
    clean=validate_input(raw)
    featured,removed=generate_trajectory_features(clean)
    if removed:raise ValueError('Feature construction unexpectedly removed rows.')
    featured=add_bearing_features(add_quality_flags(featured))
    featured['source_trajectory_id']=featured.trajectory_id
    featured['source_point_index']=featured.groupby('trajectory_id',sort=False).cumcount()
    featured['sample_id']=featured.trajectory_id+'__demo'
    featured['quality_valid_for_inference']=~featured.is_low_quality & featured.is_training_eligible
    valid=featured.loc[featured.quality_valid_for_inference].copy()
    if valid.empty:raise ValueError('No quality-valid inference points.')
    torch.set_num_threads(1)
    model=load_demo_detector(detector,seed,root)
    scored=infer(model,valid)
    replays=[apply_policy(g,AlertPolicy('G0',60)) for _,g in scored.groupby('source_trajectory_id',sort=True)]
    replay=pd.concat(replays,ignore_index=True)
    outputs=featured.merge(replay[['source_trajectory_id','source_point_index','anomaly_score',
                                   'predicted_anomaly','notification_emitted']],
                           on=['source_trajectory_id','source_point_index'],how='left',validate='one_to_one')
    outputs['point_prediction']=outputs.predicted_anomaly.astype('Int64')
    outputs['notification']=outputs.notification_emitted.astype('boolean').fillna(False).astype(bool)
    outputs['prediction_status']=np.where(outputs.quality_valid_for_inference,'scored','quality_unknown')
    columns=['timestamp','latitude','longitude','anomaly_score','point_prediction','notification',
             'trajectory_id','source_point_index','distance_m','speed_mps','quality_valid_for_inference',
             'prediction_status','quality_reason']
    return outputs[columns],model['record']

def create_demo_map(output,path):
    source=output.trajectory_id.iloc[0]
    route=output.loc[output.trajectory_id.eq(source)]
    create_route_map(source,route,path)
    text=path.read_text(encoding='utf-8')
    match=re.search(r'var (map_[a-f0-9]+) = L.map',text)
    if not match:raise ValueError('Existing Folium map template unavailable.')
    map_name=match.group(1);markers=[]
    for row in route.itertuples():
        pred=None if pd.isna(row.point_prediction) else int(row.point_prediction)
        color='#f59e0b' if row.notification else '#ef4444' if pred==1 else '#94a3b8' if pred is None else '#2563eb'
        label='notification' if row.notification else 'anomaly point' if pred==1 else 'quality unknown' if pred is None else 'normal point'
        markers.append(f'L.circleMarker([{row.latitude},{row.longitude}],'
            +json.dumps(dict(radius=6 if row.notification else 3,color=color,fill=True,fillOpacity=.8))
            +f').addTo({map_name}).bindTooltip('+json.dumps(label)+');')
    legend='<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:12px">Blue: normal; red: anomaly; amber: notification; gray: quality unknown</div>'
    path.write_text(text+legend+'\n<script>\n'+'\n'.join(markers)+'\n</script>\n',encoding='utf-8')

def run(input_path,output_dir,detector='ae',seed=42,make_map=False,root=p.ROOT):
    out=Path(output_dir)
    if (out/'demo_predictions.csv').exists() or (out/'demo_manifest.json').exists():
        raise FileExistsError('Demo output exists; choose a new output directory.')
    output,record=predict_csv(input_path,detector,seed,root)
    out.mkdir(parents=True,exist_ok=True)
    output.to_csv(out/'demo_predictions.csv',index=False)
    if make_map:create_demo_map(output,out/'demo_route.html')
    p.save(out/'demo_manifest.json',dict(detector=record['detector'],seed=record['seed'],
        model_sha256=record['model_sha256'],scaler_sha256=record['scaler_sha256'],threshold=record['threshold'],
        alert_policy='G0_C60',canonical_seed_reason='42 is the existing Stage4/5 canonical seed, not selected on Final results',
        input_sha256=p.sha(input_path),output_sha256=p.sha(out/'demo_predictions.csv'),
        rows=len(output),quality_valid_rows=int(output.quality_valid_for_inference.sum()),
        notification_count=int(output.notification.sum()),training=False,threshold_recalculation=False,
        map_first_trajectory_only=make_map))
    print('Demo saved:',out/'demo_predictions.csv')
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=p.ROOT/'outputs/demo')
    ap.add_argument('--detector',choices=FAMILIES,default='ae')
    ap.add_argument('--seed',type=int,choices=p.SEEDS,default=42)
    ap.add_argument('--map',action='store_true')
    a=ap.parse_args()
    run(a.input,a.output_dir,a.detector,a.seed,a.map)
