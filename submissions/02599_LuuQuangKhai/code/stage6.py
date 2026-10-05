"""Frozen final/baseline >=3 seeds. Validation calibration precedes one test pass per run."""
from dataclasses import asdict,replace
import gc,hashlib,json,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import dataset,model,train
from eval import compute_metrics,save_predictions,read_pred,check_against_csv,CLASS_NAMES
from inference import apply_temperature,fit_temperature
from stage3 import file_sha
from stage4 import make_configs as ablation_configs
from stage5 import make_loader,forward_method,measure

SEEDS=(0,1,2)
FINAL_METHOD=('F01_res288','single','prob',288,'fp32')
BASE_METHOD=('T00_I00','single','prob',224,'fp32')


def make_configs(images,labels,output,receipt=None,device='cuda:0'):
    base=ablation_configs(images,labels,output,receipt,device)[0]
    return [replace(base,seed=s,exp_id=exp,loss='ls' if exp=='F01' else 'ce')
            for s in SEEDS for exp in ('T00','F01')]


def validate_configs(configs):
    if {(c.exp_id,c.seed) for c in configs}!={(e,s) for s in SEEDS for e in ('T00','F01')} or len(configs)!=6:
        raise ValueError('Need exactly T00/F01 at seeds0,1,2')
    for c in configs:
        expected=ablation_configs(c.images_dir,c.labels_dir,Path(c.out_dir).parent,c.verified_summary,c.device)[0]
        expected=replace(expected,seed=c.seed,exp_id=c.exp_id,loss='ls' if c.exp_id=='F01' else 'ce')
        operational={'out_dir','pred_dir','images_dir','labels_dir','verified_summary','device','resume','stop_after_epochs','max_run_seconds'}
        for key,value in asdict(expected).items():
            if key not in operational and getattr(c,key)!=value:
                raise ValueError(f'Frozen training value changed: {key}')
        if c.save_test_predictions or c.debug_train_per_class or c.debug_val_per_class:
            raise ValueError('Training remains train/val only; no subset')
        if (c.backbone,c.init,c.aug,c.mix,c.epochs,c.batch_size,c.img_size,c.loss)!=(
                'convnext_tiny.fb_in1k','finetune','basic',None,10,32,224,'ls' if c.exp_id=='F01' else 'ce'):
            raise ValueError('Frozen training recipe changed')


def atomic_npz(path,**data):
    if 'filenames' in data:data['filenames']=np.asarray(data['filenames'],dtype=str)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(path)


def freeze(configs,selection,output,code):
    validate_configs(configs)
    if (selection['training_source'],selection['inference_resolution'],selection['dtype'],selection['seeds'])!=(
            'T05_smoothing',288,'fp32',[0,1,2]):raise ValueError('Selection must remain fixed from stage5 val')
    plan={'selection':selection,'training':{f'{c.exp_id}_seed{c.seed}':train.signature(c) for c in configs},
          'inference':{'F01':list(FINAL_METHOD),'T00':list(BASE_METHOD),'temperature_fit':'full_val_per_seed_before_test',
                       'baseline_temperature':1,'test_policy':'one complete forward traversal per run, uncal/cal share logits'},
          'code_sha256':{p.name:file_sha(p) for p in Path(code).glob('*.py')}}
    path=Path(output)/'frozen_recipe.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Frozen recipe/code/CSV changed. Never retune after seeing test.')
    train.write_json(path,plan);return file_sha(path)


def load_net(cfg):
    saved=torch.load(train.run_dir(cfg)/'best.pt',map_location='cpu',weights_only=False)
    if saved['signature']!=train.signature(cfg):raise ValueError('Checkpoint signature mismatch')
    net=model.build_model(cfg.backbone,pretrained=False,init=cfg.init,head_init_std=cfg.head_init_std).to(cfg.device).eval()
    net.load_state_dict(saved['model']);return net,saved


def evaluate_logits(net,frame,cfg,saved,method):
    names=[];truth=[];logits=[]
    loader=make_loader(frame,cfg.images_dir,saved['mean'],saved['std'],method)
    for index,(x,y,n) in enumerate(loader,1):
        l,_=forward_method(net,x.to(cfg.device),method)
        names.extend(n);truth.extend(y.numpy());logits.append(l[0].cpu().numpy())
        if index%50==0:print(f'  eval batch{index}/{len(loader)}',flush=True)
    return np.asarray(names),np.asarray(truth),np.concatenate(logits)


def metric_json(truth,probs):
    m=compute_metrics(truth,probs.argmax(1),probs)
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()}


def prepare_val(cfg,frame,output,frozen_sha):
    directory=train.run_dir(cfg)/'final_evaluation';directory.mkdir(exist_ok=True)
    checkpoint=file_sha(train.run_dir(cfg)/'best.pt')
    identity={'frozen_sha256':frozen_sha,'checkpoint_sha256':checkpoint,'exp_id':cfg.exp_id,'seed':cfg.seed}
    method=FINAL_METHOD if cfg.exp_id=='F01' else BASE_METHOD
    report_path=directory/'val_report.json';raw_path=directory/'val_logits.npz'
    if report_path.exists():
        r=json.loads(report_path.read_text())
        if r['identity']!=identity or file_sha(raw_path)!=r['val_logits_sha256']:raise ValueError('Validation receipt mismatch')
        return r
    net,saved=load_net(cfg)
    names,truth,logits=evaluate_logits(net,frame,cfg,saved,method)
    temperature=fit_temperature(logits,truth) if cfg.exp_id=='F01' else 1.
    probs=apply_temperature(logits,temperature);uncal=apply_temperature(logits,1.)
    if not np.array_equal(probs.argmax(1),uncal.argmax(1)):raise ValueError('Temperature must preserve argmax')
    atomic_npz(raw_path,filenames=names,y_true=truth,logits=logits)
    save_predictions(train.pred_path(cfg,'val'),names,truth,probs)
    if cfg.exp_id=='F01':save_predictions(Path(cfg.pred_dir)/f'F01_uncal_seed{cfg.seed}_val.csv',names,truth,uncal)
    report={'identity':identity,'val_logits_sha256':file_sha(raw_path),'temperature':temperature,
            'temperature_fit_split':'val' if cfg.exp_id=='F01' else None,'inference_method':list(method),
            'metrics':metric_json(truth,probs),'uncal_metrics':metric_json(truth,uncal),
            'latency_batch1':measure(net,method,temperature,1,torch.device(cfg.device)),
            'latency_batch32':measure(net,method,temperature,32,torch.device(cfg.device)),
            'latency_protocol':{'warmup':10,'measured_iterations':100,'sync_before_after':True,'preprocessing_included':False},
            'calibration_note':'Apparent val fit; temperature refit separately per seed before test.'}
    train.write_json(report_path,report);del net;gc.collect();torch.cuda.empty_cache();return report


def test_state(directory,identity):
    """A started pass without a complete raw-logit cache cannot be silently rerun."""
    directory=Path(directory);marker=directory/'test_started.json';raw=directory/'test_logits.npz'
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError('Test identity changed; forbidden to retune')
        if not raw.exists():raise RuntimeError('Test pass already started but no complete logits. Stop: record interruption; do not automatically rerun test.')
        with np.load(raw) as a:
            if str(a['identity_sha256'])!=hash_identity(identity):raise ValueError('Test raw-cache identity mismatch')
        return 'cached'
    if raw.exists():raise ValueError('Test logits exist without started receipt')
    return 'new'


def hash_identity(identity):
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def evaluate_test_once(cfg,frame,val_report,frozen_sha):
    directory=train.run_dir(cfg)/'final_evaluation'
    identity={'frozen_sha256':frozen_sha,'checkpoint_sha256':file_sha(train.run_dir(cfg)/'best.pt'),
              'val_report_sha256':file_sha(directory/'val_report.json'),'exp_id':cfg.exp_id,'seed':cfg.seed,
              'temperature':val_report['temperature'],'method':val_report['inference_method']}
    state=test_state(directory,identity);raw=directory/'test_logits.npz'
    if state=='new':
        net,saved=load_net(cfg)
        # Mark before opening the first test image. Recovery only from a complete raw cache.
        train.write_json(directory/'test_started.json',identity)
        names,truth,logits=evaluate_logits(net,frame,cfg,saved,tuple(val_report['inference_method']))
        atomic_npz(raw,filenames=names,y_true=truth,logits=logits,identity_sha256=np.asarray(hash_identity(identity)))
        del net;gc.collect();torch.cuda.empty_cache()
    with np.load(raw) as a:names,truth,logits=a['filenames'],a['y_true'],a['logits']
    if list(names)!=list(frame.Filename) or not np.array_equal(truth,frame.Label.to_numpy()):raise ValueError('Test cache names/labels mismatch')
    probs=apply_temperature(logits,val_report['temperature']);uncal=apply_temperature(logits,1)
    if not np.array_equal(probs.argmax(1),uncal.argmax(1)):raise ValueError('Temperature changed test argmax')
    save_predictions(train.pred_path(cfg,'test'),names,truth,probs)
    if cfg.exp_id=='F01':save_predictions(Path(cfg.pred_dir)/f'F01_uncal_seed{cfg.seed}_test.csv',names,truth,uncal)
    report={'identity':identity,'test_logits_sha256':file_sha(raw),'metrics':metric_json(truth,probs),
            'uncal_metrics':metric_json(truth,uncal),'test_forward_passes':1,'export_from_cache':state=='cached'}
    # Preserve the first receipt on repeated exports, so resume never rewrites history as another pass.
    path=directory/'test_report.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['identity']!=identity or previous['test_logits_sha256']!=file_sha(raw):raise ValueError('Test receipt mismatch')
        return previous
    train.write_json(path,report);return report


def summarize(configs,output):
    output=Path(output);rows=[];classes=[];complete=0
    for cfg in configs:
        d=train.run_dir(cfg)/'final_evaluation'
        if (d/'test_report.json').exists():complete+=1
        for split in ('val','test'):
            path=d/f'{split}_report.json'
            if not path.exists():continue
            r=json.loads(path.read_text());pred=read_pred(str(train.pred_path(cfg,split)))
            check_against_csv(pred,str(Path(cfg.labels_dir)/f'{split}_subset0.csv'),split)
            m=compute_metrics(pred.y_true,pred.y_pred,pred.probs)
            for key in ('macro_f1','top1','ece','nll'):
                if not np.isclose(m[key],r['metrics'][key],atol=1e-7,rtol=0):raise ValueError('Prediction/report mismatch')
            rows.append({'exp_id':cfg.exp_id,'seed':cfg.seed,'split':split,**{k:m[k] for k in ('macro_f1','top1','balanced_acc','ece','nll')}})
            for k,name in enumerate(CLASS_NAMES):classes.append({'exp_id':cfg.exp_id,'seed':cfg.seed,'split':split,'class_id':k,'class_name':name,
                       'f1':m['f1'][k],'recall':m['recall'][k],'precision':m['precision'][k],'support':int(m['support'][k])})
    table=pd.DataFrame(rows);table.to_csv(output/'final_per_seed.csv',index=False)
    pc=pd.DataFrame(classes);pc.to_csv(output/'final_per_class.csv',index=False)
    if len(rows):
        aggregates=[]
        for (exp,split),group in table.groupby(['exp_id','split']):
            record={'exp_id':exp,'split':split,'n_seeds':len(group)}
            for key in ('macro_f1','top1','balanced_acc','ece','nll'):
                record[key+'_mean']=float(group[key].mean());record[key+'_std']=float(group[key].std(ddof=1))
            aggregates.append(record)
        pd.DataFrame(aggregates).to_csv(output/'final_mean_std.csv',index=False)
        pc.groupby(['exp_id','split','class_id','class_name'])[['f1','recall','precision']].agg(['mean','std']).to_csv(output/'final_per_class_mean_std.csv')
    r={'status':'complete' if complete==6 and (output/'final_complete.json').exists() else 'incomplete','test_runs_completed':complete,'planned_test_runs':6,
       'seeds':list(SEEDS),'ddof':1,'config_frozen':True,'no_retuning_after_test':True}
    train.write_json(output/'stage6_summary.json',r);return r


def plot_confusions(configs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for cfg in configs:
        r=json.loads((train.run_dir(cfg)/'final_evaluation/test_report.json').read_text());cm=np.asarray(r['metrics']['confusion'])
        fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
        ax.set(xticks=range(9),yticks=range(9),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,xlabel='Predicted',ylabel='True',title=f'{cfg.exp_id} seed{cfg.seed} test')
        plt.setp(ax.get_xticklabels(),rotation=40,ha='right')
        for i in range(9):
            for j in range(9):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()/2 else 'black')
        fig.tight_layout();p=Path(output)/'confusions';p.mkdir(exist_ok=True);fig.savefig(p/f'{cfg.exp_id}_seed{cfg.seed}_test.png',dpi=140);plt.close(fig)


def official_reports(configs,output,code):
    output,code=Path(output),Path(code);dest=output/'official';dest.mkdir(exist_ok=True)
    pred=output/'predictions';labels=Path(configs[0].labels_dir)
    for exp in ('T00','F01','F01_uncal'):
        command=[sys.executable,str(code/'eval.py'),'score','--pred',str(pred/f'{exp}_seed*_test.csv'),
                 '--test-csv',str(labels/'test_subset0.csv'),'--labels',str(labels/'labels.csv'),'--tag',exp,'--out',str(dest)]
        result=subprocess.run(command,capture_output=True,text=True);(dest/f'{exp}_score.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Official score failed')
    p95=max(json.loads((train.run_dir(c)/'final_evaluation/val_report.json').read_text())['latency_batch1']['p95'] for c in configs if c.exp_id=='F01')
    command=[sys.executable,str(code/'eval.py'),'grade','--final',str(pred/'F01_seed*_test.csv'),'--baseline',str(pred/'T00_seed*_test.csv'),
             '--uncal',str(pred/'F01_uncal_seed*_test.csv'),'--final-val',str(pred/'F01_seed*_val.csv'),
             '--test-csv',str(labels/'test_subset0.csv'),'--val-csv',str(labels/'val_subset0.csv'),'--labels',str(labels/'labels.csv'),
             '--latency-p95-ms',str(p95),'--latency-method','proper','--out',str(dest)]
    result=subprocess.run(command,capture_output=True,text=True);(dest/'grade.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Official grade failed')
    train.write_json(dest/'latency_grade_input.json',{'p95_ms':p95,'aggregation':'max of measured final p95 across3 seeds',
                       'condition':'batch1 FP32 288 + temperature; 10warmup100measure; preprocessing excluded'})


def export(output,destination,code):
    output,destination,code=map(Path,(output,destination,code));tmp=destination.with_suffix('.tmp');destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.tmp'):z.write(p,p.relative_to(output).as_posix())
        for p in sorted(code.glob('*.py')):z.write(p,'code/'+p.name)
        manifest=code.parent/'stage6_manifest.json'
        if manifest.exists():z.write(manifest,manifest.name)
    tmp.replace(destination)


def run_suite(configs,selection,output,code,result_zip,budget_seconds=21600):
    if budget_seconds<900:raise ValueError('At least15min budget required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    frozen_sha=freeze(configs,selection,output,code)
    train.write_json(output/'suite_config.json',[asdict(c) for c in configs])
    def remaining():return budget_seconds-(time.monotonic()-start)
    try:
        for cfg in configs:
            if remaining()<900:break
            print('Training/resuming',cfg.exp_id,'seed',cfg.seed,flush=True)
            r=train.run(replace(cfg,max_run_seconds=remaining()-300))
            gc.collect();torch.cuda.empty_cache();summarize(configs,output);export(output,result_zip,code)
            if r['status']!='completed':break
        all_trained=all((train.run_dir(c)/'summary.json').exists() and json.loads((train.run_dir(c)/'summary.json').read_text())['status']=='completed' for c in configs)
        val_reports={}
        if all_trained:
            _,val_frame,test_frame=dataset.load_split(configs[0].labels_dir)
            for cfg in configs:
                if remaining()<900:break
                val_reports[(cfg.exp_id,cfg.seed)]=prepare_val(cfg,val_frame,output,frozen_sha)
                summarize(configs,output);export(output,result_zip,code)
            # No test is touched until all6 runs are trained and all6 val calibrations/latencies saved.
            if len(val_reports)==6:
                for cfg in configs:
                    if remaining()<900:break
                    print('Final test export',cfg.exp_id,'seed',cfg.seed,flush=True)
                    evaluate_test_once(cfg,test_frame,val_reports[(cfg.exp_id,cfg.seed)],frozen_sha)
                    summarize(configs,output);export(output,result_zip,code)
                if all((train.run_dir(c)/'final_evaluation/test_report.json').exists() for c in configs):
                    official_reports(configs,output,code);plot_confusions(configs,output)
                    train.write_json(output/'final_complete.json',{'official_score_grade':'PASS','confusions_exported':True,'frozen_sha256':frozen_sha})
    except Exception:
        (output/'last_error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        try:report=summarize(configs,output)
        finally:export(output,result_zip,code)
    print(json.dumps(report,indent=2),flush=True);return report
