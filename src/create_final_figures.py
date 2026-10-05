"""Publication figures from frozen result tables only; no metric/inference calls."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import stage7_protocol as p
from src.evaluate_final_validation import verify_results

MODELS=['statistical_rule','isolation_forest','autoencoder']
LABELS=['Rule','IF (5 seeds)','AE (5 seeds)']
COLORS=['#64748b','#2563eb','#ea580c']

def create(root=p.ROOT):
    root=Path(root);results=verify_results(root);out=root/'outputs/figures/stage7/final'
    out.mkdir(parents=True,exist_ok=True)
    docs=root/'docs/images/stage7';docs.mkdir(parents=True,exist_ok=True)
    tables={name:pd.read_csv(root/p.METRICS/(name+'.csv'),dtype={'user_id':str})
        for name in ['final_point_metrics','final_user_metrics','final_anomaly_type_metrics',
                     'stage5_vs_stage7','final_event_summary','final_alert_summary']}
    def values(table,metric,evaluation=None):
        f=tables[table];f=f.loc[f.metric.eq(metric)]
        if evaluation is not None:f=f.loc[f.evaluation.eq(evaluation)]
        f=f.set_index('model').reindex(MODELS)
        return f['mean'].to_numpy(float),f.sample_std.fillna(0).to_numpy(float)
    files=[]
    def finish(fig,name):
        target=out/name
        if target.exists():raise FileExistsError('Figure already exists: '+name)
        fig.savefig(target,dpi=170,bbox_inches='tight');plt.close(fig)
        shutil.copyfile(target,docs/name);files.append(target)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,label in zip(axes,['f1_score','roc_auc','average_precision'],['F1','ROC-AUC','Average precision']):
        y,e=values('final_point_metrics',metric);ax.bar(LABELS,y,yerr=e,color=COLORS,capsize=4)
        ax.set_ylim(0,1);ax.set_title(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Untouched cohort: point discrimination (seed sample std; Rule deterministic)')
    finish(fig,'final_detector_comparison.png')
    fig,ax=plt.subplots(figsize=(13,4),layout='constrained')
    f=tables['final_user_metrics'];f=f.loc[f.metric.eq('f1_score')]
    for model,label,color in zip(MODELS,LABELS,COLORS):
        g=f.loc[f.model.eq(model)].set_index('user_id').reindex(p.USERS)
        ax.errorbar(range(20),g['mean'],yerr=g.sample_std.fillna(0),label=label,color=color,marker='o',capsize=2)
    ax.set_xticks(range(20),p.USERS,rotation=45);ax.set_ylim(0,1);ax.set_ylabel('F1')
    ax.set_title('Per-user F1: 021 has no quality-eligible rows (NA, retained in manifest)')
    ax.legend();ax.grid(alpha=.2);finish(fig,'final_per_user_f1.png')
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    f=tables['final_anomaly_type_metrics'];f=f.loc[f.metric.eq('recall')]
    types=['route_deviation','abnormal_speed','long_stop','direction_change']
    for i,(model,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):
        g=f.loc[f.model.eq(model)].set_index('anomaly_type').reindex(types)
        ax.bar(np.arange(4)+(i-1)*.24,g['mean'],.24,yerr=g.sample_std.fillna(0),label=label,color=color,capsize=3)
    ax.set_xticks(range(4),['Route','Speed','Long stop','Direction']);ax.set_ylim(0,1)
    ax.set_ylabel('Labelled-point recall');ax.legend();ax.grid(axis='y',alpha=.2)
    finish(fig,'final_anomaly_type_recall.png')
    fig,axes=plt.subplots(1,5,figsize=(18,4),layout='constrained')
    f=tables['stage5_vs_stage7']
    for ax,metric,label in zip(axes,['f1_score','recall','false_positive_rate','roc_auc','average_precision'],
                               ['F1','Recall','FPR (lower better)','ROC-AUC','AP']):
        g=f.loc[f.metric.eq(metric)].set_index('model').reindex(MODELS)
        for i,(stage,color) in enumerate([('stage5','#94a3b8'),('stage7','#2563eb')]):
            ax.bar(np.arange(3)+(i-.5)*.32,g['mean_'+stage],.32,
                   yerr=g['sample_std_'+stage].fillna(0),label=stage.replace('stage','Stage '),color=color,capsize=3)
        ax.set_xticks(range(3),['Rule','IF','AE']);ax.set_title(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Same frozen detectors; different cohorts/prevalence (no model improvement claim)')
    finish(fig,'stage5_vs_stage7_generalization.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,point_metric,notification_metric,title in [
        (axes[0],'event_detection_rate','notification_event_detection_rate','Route event detection'),
        (axes[1],'early_detection_25_rate','notification_early25_rate','Early detection @25%')]:
        configurations=[('final_event_summary',point_metric,None,'Any raw point','#94a3b8'),
            ('final_alert_summary',notification_metric,'RAW','RAW notification','#2563eb'),
            ('final_alert_summary',notification_metric,'LOCKED','G0_C60 notification','#ea580c')]
        for i,(table,metric,evaluation,label,color) in enumerate(configurations):
            y,e=values(table,metric,evaluation);ax.bar(np.arange(3)+(i-1)*.24,y,.24,yerr=e,label=label,color=color,capsize=3)
        ax.set_xticks(range(3),['Rule','IF','AE']);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);finish(fig,'final_event_detection.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,metric,eval_,title in [(axes[0],'raw_fp_per_1000','RAW','False positive points /1000 normal rows'),
        (axes[1],'false_notifications_per_1000_normal_points','LOCKED','G0_C60 notifications /1000 normal rows')]:
        y,e=values('final_alert_summary',metric,eval_);ax.bar(LABELS,y,yerr=e,color=COLORS,capsize=4)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Separate point and notification units; only original normals')
    finish(fig,'final_false_alert_burden.png')
    fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
    for ax,metric,title in zip(axes,['false_notifications_per_1000_normal_points','notification_event_detection_rate',
                                  'notification_early25_rate'],['Notification burden /1000','Notification EDR','Notification early@25']):
        for i,(evaluation,label,color) in enumerate([('RAW','RAW G0_C0','#2563eb'),('LOCKED','Frozen G0_C60','#ea580c')]):
            y,e=values('final_alert_summary',metric,evaluation)
            ax.bar(np.arange(3)+(i-.5)*.32,y,.32,yerr=e,label=label,color=color,capsize=3)
        ax.set_xticks(range(3),['Rule','IF','AE']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend();finish(fig,'final_raw_vs_locked_alert.png')
    record=dict(final_results_manifest_sha256=p.sha(root/p.METRICS/'final_results_manifest.json'),
        figure_file_hashes={f.relative_to(root).as_posix():p.sha(f) for f in files},
        portfolio_file_hashes={(docs/f.name).relative_to(root).as_posix():p.sha(docs/f.name) for f in files},
        figure_source_hash=p.sha(Path(__file__)),metric_recomputation=False,figure_count=len(files),
        source_table_hashes={name:results['result_file_hashes'][(p.METRICS/(name+'.csv')).as_posix()] for name in tables})
    p.save(root/p.METRICS/'final_figure_manifest.json',record)
    print('Created',len(files),'figures from frozen summaries only')
    return record

if __name__=='__main__':create()
