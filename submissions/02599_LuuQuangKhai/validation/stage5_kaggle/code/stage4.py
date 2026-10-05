"""Three independent training ablations and a validation-selected combination; test locked."""
from dataclasses import asdict, replace
import gc,json,time,traceback,zipfile,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train
from stage3 import make_configs as backbone_configs, verified_metrics, file_sha

AXES = {'A': ('init',), 'B': ('aug', 'mix'), 'C': ('loss',)}
VARIANTS = [
    ('T01_scratch','A',{'init':'scratch'}),
    ('T02_frozen','A',{'init':'frozen'}),
    ('T03_randaug','B',{'aug':'randaug'}),
    ('T04_cutmix','B',{'mix':'cutmix'}),
    ('T05_smoothing','C',{'loss':'ls'}),
    ('T06_focal','C',{'loss':'focal'}),
]

def make_configs(images_dir, labels_dir, output, receipt=None, device='cuda:0'):
    base=replace(backbone_configs(images_dir,labels_dir,output,receipt,device)[1],exp_id='T00')
    return [base]+[replace(base,exp_id=exp,**changes) for exp,axis,changes in VARIANTS]


def validate_plan(configs):
    if [c.exp_id for c in configs] != ['T00']+[x[0] for x in VARIANTS]:
        raise ValueError('Use the declared seven-run single-axis plan')
    base=asdict(configs[0])
    for cfg, (exp,axis,changes) in zip(configs[1:],VARIANTS):
        changed={k for k,v in asdict(cfg).items() if v!=base[k]}-{'exp_id'}
        if changed!=set(changes):
            raise ValueError(f'{exp} must differ only in {changes}; got {changed}')
    for cfg in configs:
        if cfg.save_test_predictions or cfg.debug_train_per_class or cfg.debug_val_per_class:
            raise ValueError('Full train/val only; test locked')
        if (cfg.backbone,cfg.seed,cfg.fold,cfg.epochs,cfg.batch_size,cfg.img_size)!=(
                'convnext_tiny.fb_in1k',0,0,10,32,224):
            raise ValueError('Stage 4 recipe changed')


def completed_metrics(cfg):
    path=train.run_dir(cfg)/'summary.json'
    if not path.exists():
        return None
    summary=json.loads(path.read_text())
    if summary['status']!='completed' or summary['epochs_completed']!=cfg.epochs:
        return None
    if summary.get('test_evaluated') or summary.get('debug_subset'):
        raise ValueError('Invalid ablation artifact')
    return verified_metrics(cfg,summary)


def choose_combination(configs, metrics):
    """Keep baseline on ties; select improvements independently against the same T00."""
    if len(metrics)!=7 or any(c.exp_id not in metrics for c in configs):
        raise ValueError('Complete all seven single-axis runs before selecting combination')
    base=configs[0];score=metrics['T00']['macro_f1'];updates={};choices={}
    for axis,keys in AXES.items():
        candidates=[c for c in configs[1:] if next(v[1] for v in VARIANTS if v[0]==c.exp_id)==axis]
        winner=max(candidates,key=lambda c:metrics[c.exp_id]['macro_f1'])
        if metrics[winner.exp_id]['macro_f1']>score+1e-12:
            updates.update({k:getattr(winner,k) for k in keys})
            choices[axis]=winner.exp_id
        else:
            choices[axis]='T00'
    combo=replace(base,exp_id='T07_combined',**updates)
    decision={'selection_split':'val','criterion':'macro_f1','axis_choices':choices,
              'changes_from_T00':{k:v for k,v in updates.items() if v!=getattr(base,k)},
              'single_axis_macro_f1':{k:float(v['macro_f1']) for k,v in metrics.items()},
              'all_axes_kept_baseline':not updates,
              'note':'Seed-0 screening; tiny improvements are candidates, not statistically established gains.'}
    return combo,decision


def summarize(configs, output, combo=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];baseline=completed_metrics(configs[0]);all_configs=configs+([combo] if combo else [])
    for cfg in all_configs:
        axis='baseline' if cfg.exp_id=='T00' else ('combined' if cfg.exp_id=='T07_combined' else next(x[1] for x in VARIANTS if x[0]==cfg.exp_id))
        changes={k:v for k,v in asdict(cfg).items() if v!=asdict(configs[0])[k] and k not in ('exp_id','max_run_seconds')}
        row={'exp_id':cfg.exp_id,'axis':axis,'changes_from_T00':json.dumps(changes,sort_keys=True),
             'backbone':cfg.backbone,'seed':cfg.seed,'init':cfg.init,'aug':cfg.aug,'mix':cfg.mix,
             'loss':cfg.loss,'epochs_planned':cfg.epochs,'status':'pending','eligible':False}
        directory=train.run_dir(cfg);path=directory/'summary.json'
        if path.exists():
            s=json.loads(path.read_text());m=verified_metrics(cfg,s)
            row.update(status=s['status'],epochs_completed=s['epochs_completed'],best_epoch=s['best_epoch'],
                       eligible=s['status']=='completed' and s['epochs_completed']==cfg.epochs,
                       train_val_seconds_per_epoch=s['epoch_seconds_mean'],params_m=s['params_m'])
            for k in ('macro_f1','top1','balanced_acc','ece','nll'):
                row['val_'+k]=m[k]
            for k in range(9):
                row[f'val_f1_class{k}']=float(m['f1'][k]);row[f'val_recall_class{k}']=float(m['recall'][k])
            row['delta_macro_f1_vs_T00']=m['macro_f1']-baseline['macro_f1'] if baseline else np.nan
            curves=output/'curves';curves.mkdir(exist_ok=True)
            if (directory/'curves.png').exists():
                shutil.copyfile(directory/'curves.png',curves/f'{cfg.exp_id}_seed{cfg.seed}.png')
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(output/'training.csv',index=False)
    ranked=table[table.eligible].copy()
    if len(ranked):
        ranked=ranked.sort_values(['val_macro_f1','exp_id'],ascending=[False,True])
        ranked.to_csv(output/'training_ranked.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5));ax.bar(ranked.exp_id,ranked.delta_macro_f1_vs_T00)
        ax.axhline(0,color='black',linewidth=.7);ax.tick_params(axis='x',labelrotation=30)
        ax.set(ylabel='Validation macro-F1 change from T00',title='ConvNeXt-Tiny training ablations, seed 0')
        fig.tight_layout();fig.savefig(output/'training_ablation.png',dpi=150);plt.close(fig)
    report={'status':'complete' if len(ranked)==8 else 'incomplete','completed':len(ranked),'planned':8,
            'val_leader':ranked.iloc[0].exp_id if len(ranked) else None,'test_evaluated':False,
            'seed_count':1,'final_recipe_frozen':False,
            'note':'Training candidates for inference screening; final and baseline need >=3 seeds later.'}
    train.write_json(output/'stage4_summary.json',report)
    return report


def export_results(output, destination, code_dir):
    output,destination,code_dir=map(Path,(output,destination,code_dir));tmp=destination.with_suffix('.tmp')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix not in ('.pt','.tmp'):
                z.write(path,path.relative_to(output).as_posix())
        for path in sorted(code_dir.glob('*.py')):
            z.write(path,'code/'+path.name)
        manifest=code_dir.parent/'stage4_manifest.json'
        if manifest.exists():z.write(manifest,manifest.name)
    tmp.replace(destination)


def run_suite(configs, output, result_zip, code_dir, budget_seconds=21600):
    validate_plan(configs)
    if budget_seconds<900:raise ValueError('Budget must be at least 900 seconds')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);started=time.monotonic();combo=None
    identity={'runs':{c.exp_id:train.signature(c) for c in configs},
              'stage4_sha256':file_sha(__file__),'stage3_sha256':file_sha(Path(__file__).with_name('stage3.py'))}
    path=output/'suite_identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:
        raise ValueError('Resume requires identical stage-4 code/config/CSV')
    train.write_json(path,identity);train.write_json(output/'single_axis_configs.json',[asdict(c) for c in configs])
    def execute(cfg):
        # Run-specific time budgets are operational, excluded from train.signature.
        remaining=budget_seconds-(time.monotonic()-started)
        if remaining<900:return False
        active=replace(cfg,max_run_seconds=remaining-300)
        print(f'Starting/resuming {cfg.exp_id}, remaining {remaining/60:.1f} min',flush=True)
        summary=train.run(active);gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        summarize(configs,output,combo);export_results(output,result_zip,code_dir)
        return summary['status']=='completed'
    try:
        for cfg in configs:
            if not execute(cfg):break
        metrics={c.exp_id:completed_metrics(c) for c in configs}
        if all(v is not None for v in metrics.values()):
            combo,decision=choose_combination(configs,metrics)
            selected={'decision':decision,'signature':train.signature(combo)}
            selection_path=output/'combination_selection.json'
            if selection_path.exists() and json.loads(selection_path.read_text())!=selected:
                raise ValueError('Combination selection changed; do not resume inconsistent artifacts')
            train.write_json(selection_path,selected);train.write_json(output/'combined_config.json',asdict(combo))
            print(json.dumps(decision,indent=2),flush=True);execute(combo)
    except Exception:
        (output/'last_error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        try:report=summarize(configs,output,combo)
        finally:export_results(output,result_zip,code_dir)
    print(json.dumps(report,indent=2),flush=True)
    return report
