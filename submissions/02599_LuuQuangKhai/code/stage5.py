"""Stage-5 validation inference screening; no training and no test loader."""
import gc,json,zipfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
import dataset,model,train
from inference import views_multicrop,fit_temperature
from benchmark import bench
from stage3 import file_sha
from eval import compute_metrics,save_predictions

METHODS=[
 ('I00','single','prob',224,'fp32'),
 ('I01_hflip','flip','prob',224,'fp32'),
 ('I02_5crop_prob','five','prob',224,'fp32'),
 ('I03_5crop_logit','five','logit',224,'fp32'),
 ('I04_res256','single','prob',256,'fp32'),
 ('I04_res288','single','prob',288,'fp32'),
 ('I07_temperature','single','prob',224,'fp32'),
 ('I08_amp','single','prob',224,'amp'),
]

@torch.inference_mode()
def forward_method(net,x,method,temperature=1.):
    _,view,space,size,dtype=method
    views=[x] if view=='single' else ([x,x.flip(-1)] if view=='flip' else views_multicrop(x,size))
    with torch.autocast(x.device.type,enabled=dtype=='amp'):
        logits=torch.stack([net(v).float() for v in views])
    probs=(logits.mean(0)/temperature).softmax(-1) if space=='logit' or temperature!=1 else logits.softmax(-1).mean(0)
    return logits,probs


def make_loader(frame,images,mean,std,method):
    if method[1]=='five':
        transform=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(mean,std)])
    else:transform=dataset.build_transforms(False,method[3],'basic',mean,std)
    return dataset.make_loader(frame,images,transform,32,False,num_workers=2,seed=1)


def measure(net,method,temperature,batch,device):
    side=256 if method[1]=='five' else method[3]
    x=torch.randn(batch,3,side,side,device=device)
    with torch.inference_mode():
        report=bench(lambda:forward_method(net,x,method,temperature),10,100,lambda:torch.cuda.synchronize(device))
    return {**report,'batch':batch,'images_per_s':batch/(report['p50']/1000),'gpu':torch.cuda.get_device_name(device),
            'torch':torch.__version__,'dtype':method[4],'resolution':method[3],
            'views':5 if method[1]=='five' else (2 if method[1]=='flip' else 1),
            'preprocessing_included':False,'host_to_device_included':False,'view_operations_and_aggregation_included':True,'bn_fused':False}


def export(output,destination,code):
    output,destination,code=map(Path,(output,destination,code));tmp=destination.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix!='.tmp':z.write(p,p.relative_to(output).as_posix())
        for p in sorted(code.glob('*.py')):z.write(p,'code/'+p.name)
        manifest=code.parent/'stage5_manifest.json'
        if manifest.exists():z.write(manifest,manifest.name)
    tmp.replace(destination)


def summarize(output):
    output=Path(output);rows=[]
    for p in sorted((output/'methods').glob('*/*/report.json')):
        r=json.loads(p.read_text());base=output/'methods'/r['training_exp']/'I00/report.json'
        b=json.loads(base.read_text())['latency_batch1']['p50'] if base.exists() else float('nan')
        rows.append({'training_exp':r['training_exp'],'method':r['method'],'val_macro_f1':r['metrics']['macro_f1'],
                     'val_top1':r['metrics']['top1'],'val_ece':r['metrics']['ece'],'val_nll':r['metrics']['nll'],
                     'p50_ms':r['latency_batch1']['p50'],'p95_ms':r['latency_batch1']['p95'],'p99_ms':r['latency_batch1']['p99'],
                     'batch32_images_per_s':r['latency_batch32']['images_per_s'],'relative_cost':r['latency_batch1']['p50']/b,
                     'temperature':r['temperature'],'dtype':r['latency_batch1']['dtype'],'views':r['latency_batch1']['views']})
    table=pd.DataFrame(rows);table.to_csv(output/'inference.csv',index=False)
    report={'status':'complete' if len(rows)==16 else 'incomplete','completed':len(rows),'planned':16,'test_evaluated':False,
            'final_recipe_frozen':False,'calibration_metrics':'Temperature fitted and evaluated on full val; apparent fit, not independent calibration evaluation.'}
    if len(rows):
        ranked=table.sort_values(['val_macro_f1','p95_ms'],ascending=[False,True]);ranked.to_csv(output/'inference_ranked.csv',index=False)
        report['val_leader']={'training_exp':ranked.iloc[0].training_exp,'method':ranked.iloc[0].method}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,6))
        for exp,group in table.groupby('training_exp'):
            ax.scatter(group.p95_ms,group.val_macro_f1,label=exp)
            for r in group.itertuples():ax.annotate(r.method,(r.p95_ms,r.val_macro_f1),fontsize=7,xytext=(3,3),textcoords='offset points')
        ax.set(xlabel='Batch1 p95 inference latency (ms)',ylabel='Validation macro-F1',title='Stage5: T00/T05, seed0, measured GPU');ax.legend();fig.tight_layout();fig.savefig(output/'inference_tradeoff.png',dpi=150);plt.close(fig)
    train.write_json(output/'stage5_summary.json',report);return report


def run_suite(source,images,labels,output,code,result_zip,reference):
    source,output,reference=map(Path,(source,output,reference));output.mkdir(parents=True,exist_ok=True)
    train.set_seed(0);device=torch.device('cuda:0')
    configs={c['exp_id']:c for c in json.loads((source/'single_axis_configs.json').read_text())}
    identity={'code':{p.name:file_sha(p) for p in Path(code).glob('*.py')},'labels':dataset.csv_hashes(labels,0),
              'checkpoints':{exp:file_sha(source/f'runs/{exp}/seed0/best.pt') for exp in ('T00','T05_smoothing')}}
    path=output/'suite_identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Stage5 source/checkpoint changed; use original notebook or new output')
    train.write_json(path,identity)
    # Read official val only. The integrity check uses CSV metadata, never test images.
    train_df,val_df,test_df=dataset.load_split(labels);dataset.check_split(train_df,val_df,test_df,images,verify_files=False)
    try:
        for exp in ('T00','T05_smoothing'):
            cfg=train.Config(**configs[exp]);cfg=replace(cfg,labels_dir=str(labels))
            saved=torch.load(source/f'runs/{exp}/seed0/best.pt',map_location='cpu',weights_only=False)
            assert saved['signature']==train.signature(cfg),'Checkpoint signature mismatch'
            net=model.build_model(cfg.backbone,pretrained=False,init=cfg.init).to(device).eval();net.load_state_dict(saved['model'])
            mean,std=saved['mean'],saved['std'];temperature=1.
            for method in METHODS:
                name=method[0];directory=output/'methods'/exp/name;directory.mkdir(parents=True,exist_ok=True)
                completed=directory/'report.json'
                if completed.exists():
                    if name=='I07_temperature':temperature=json.loads(completed.read_text())['temperature']
                    print('Already completed',exp,name,flush=True);continue
                print('Evaluating',exp,name,flush=True)
                if name=='I07_temperature':
                    cached=np.load(output/f'methods/{exp}/I00/views_logits.npz')
                    temperature=fit_temperature(cached['logits'][0],cached['y_true'])
                else:temperature=1.
                names=[];truth=[];logits=[];probs=[]
                for x,y,n in make_loader(val_df,images,mean,std,method):
                    l,p=forward_method(net,x.to(device),method,temperature)
                    names.extend(n);truth.extend(y.numpy());logits.append(l.cpu().numpy());probs.append(p.cpu().numpy())
                truth=np.asarray(truth);l=np.concatenate(logits,axis=1);p=np.concatenate(probs)
                if name=='I00':
                    expected=np.load(reference/f'{exp}_val_logits.npz')
                    assert list(expected['filenames'])==names and np.array_equal(expected['y_true'],truth)
                    assert np.allclose(expected['logits'],l[0],atol=1e-3,rtol=1e-4),'I00 differs from stage4 checkpoint predictions'
                if name=='I07_temperature':
                    original=np.load(output/f'methods/{exp}/I00/views_logits.npz')['logits'][0]
                    assert np.array_equal(original.argmax(1),p.argmax(1)),'Temperature changed argmax'
                np.savez_compressed(directory/'views_logits.npz',filenames=np.asarray(names),y_true=truth,logits=l)
                save_predictions(output/'predictions'/f'{exp}_{name}_seed0_val.csv',names,truth,p)
                m=compute_metrics(truth,p.argmax(1),p)
                scalars={k:float(m[k]) for k in ('macro_f1','top1','ece','nll','balanced_acc')}
                report={'training_exp':exp,'method':name,'temperature':temperature,'metrics':scalars,
                        'f1':m['f1'].tolist(),'recall':m['recall'].tolist(),'checkpoint_sha256':identity['checkpoints'][exp],
                        'latency_batch1':measure(net,method,temperature,1,device),'latency_batch32':measure(net,method,temperature,32,device),
                        'temperature_fit_split':'full_val' if name=='I07_temperature' else None,'aggregation':method[2]}
                train.write_json(completed,report);summarize(output);export(output,result_zip,code)
            del net;gc.collect();torch.cuda.empty_cache()
    finally:
        report=summarize(output);export(output,result_zip,code)
    return report
