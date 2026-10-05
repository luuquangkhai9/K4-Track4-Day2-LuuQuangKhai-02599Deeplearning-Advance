"""Audit final returned results; recompute from official prediction files, never rerun models."""
import hashlib,json,sys,zipfile,shutil,subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SUB=ROOT/'submissions/02599_LuuQuangKhai'
sys.path.insert(0,str(SUB/'code'))
import train,stage6
from eval import read_pred,check_against_csv,compute_metrics
from inference import apply_temperature,fit_temperature
archive=ROOT/'stage6_results.zip';out=SUB/'validation/stage6_kaggle'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts
        if name.startswith('code/') and name.endswith('.py'):assert z.read(name)==(SUB/name).read_bytes(),name
    assert z.read('code/eval.py')==(ROOT/'eval.py').read_bytes()
    out.mkdir(parents=True,exist_ok=True);z.extractall(out)
configs=[train.Config(**c) for c in json.loads((out/'suite_config.json').read_text())]
stage6.validate_configs(configs)
summary=json.loads((out/'stage6_summary.json').read_text());assert summary['status']=='complete' and summary['test_runs_completed']==6
frozen=json.loads((out/'frozen_recipe.json').read_text());frozen_sha=stage6.file_sha(out/'frozen_recipe.json')
for name,sha in frozen['code_sha256'].items():assert sha==stage6.file_sha(out/'code'/name)
assert json.loads((out/'final_complete.json').read_text())['frozen_sha256']==frozen_sha
rows=[];checks=[]
for cfg in configs:
    cfg=replace(cfg,labels_dir=str(SUB/'labels'))
    assert frozen['training'][f'{cfg.exp_id}_seed{cfg.seed}']==train.signature(cfg)
    d=out/f'runs/{cfg.exp_id}/seed{cfg.seed}';e=d/'final_evaluation'
    history=pd.read_csv(d/'history.csv');s=json.loads((d/'summary.json').read_text())
    assert s['status']=='completed' and list(history.epoch)==list(range(1,11))
    assert s['best_epoch']==int(history.loc[history.val_macro_f1.idxmax(),'epoch'])
    v=json.loads((e/'val_report.json').read_text());t=json.loads((e/'test_report.json').read_text());started=json.loads((e/'test_started.json').read_text())
    assert t['identity']==started and t['test_forward_passes']==1
    assert started['frozen_sha256']==frozen_sha and started['val_report_sha256']==stage6.file_sha(e/'val_report.json')
    assert v['val_logits_sha256']==stage6.file_sha(e/'val_logits.npz') and t['test_logits_sha256']==stage6.file_sha(e/'test_logits.npz')
    assert v['identity']['checkpoint_sha256']==started['checkpoint_sha256']
    for split,r in (('val',v),('test',t)):
        a=np.load(e/f'{split}_logits.npz');p=read_pred(str(out/f'predictions/{cfg.exp_id}_seed{cfg.seed}_{split}.csv'))
        check_against_csv(p,str(SUB/f'labels/{split}_subset0.csv'),split)
        assert list(a['filenames'])==list(p.filenames) and np.array_equal(a['y_true'],p.y_true)
        if split=='test':assert str(a['identity_sha256'])==stage6.hash_identity(started)
        probs=apply_temperature(a['logits'],v['temperature']);assert np.allclose(probs,p.probs,atol=1e-7,rtol=0)
        m=compute_metrics(p.y_true,p.y_pred,p.probs)
        for k in ('macro_f1','top1','balanced_acc','ece','nll'):assert np.isclose(m[k],r['metrics'][k],atol=1e-7,rtol=0)
        for k in ('precision','recall','f1','support','confusion'):assert np.allclose(m[k],r['metrics'][k],atol=1e-7,rtol=0)
        if cfg.exp_id=='F01':
            u=read_pred(str(out/f'predictions/F01_uncal_seed{cfg.seed}_{split}.csv'))
            assert np.allclose(u.probs,apply_temperature(a['logits'],1),atol=1e-7,rtol=0)
            assert np.array_equal(u.y_pred,p.y_pred)
            if split=='val':assert np.isclose(v['temperature'],fit_temperature(a['logits'],p.y_true),atol=1e-10)
        else:assert v['temperature']==1
        rows.append({'exp_id':cfg.exp_id,'seed':cfg.seed,'split':split,**{k:m[k] for k in ('macro_f1','top1','balanced_acc','ece','nll')}})
    for batch in (1,32):
        l=v[f'latency_batch{batch}'];assert l['batch']==batch and l['n']==100 and len(l['samples_ms'])==100
        for percentile in (50,95,99):assert np.isclose(l[f'p{percentile}'],np.percentile(l['samples_ms'],percentile))
    checks.append({'exp_id':cfg.exp_id,'seed':cfg.seed,'test_forward_passes_receipt':1,'temperature':v['temperature'],'checkpoint_sha256_receipt':started['checkpoint_sha256']})
actual=pd.DataFrame(rows);saved=pd.read_csv(out/'final_mean_std.csv')
for row in saved.itertuples():
    group=actual[(actual.exp_id==row.exp_id)&(actual.split==row.split)];assert len(group)==3
    for k in ('macro_f1','top1','balanced_acc','ece','nll'):
        assert np.isclose(getattr(row,k+'_mean'),group[k].mean(),atol=1e-7,rtol=0)
        assert np.isclose(getattr(row,k+'_std'),group[k].std(ddof=1),atol=1e-7,rtol=0)
n=json.loads((SUB/'code/kaggle_stage6.ipynb').read_text(encoding='utf-8'));assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
for log in ('test_pipeline.py','test_stage5.py','test_stage6.py'):
    assert (out/(log+'.log')).read_text().rstrip().endswith('OK')
# Assemble public grading artifacts without overwriting evidence snapshots.
(SUB/'predictions').mkdir(exist_ok=True)
for stage in (3,4,5,6):
    src=SUB/f'validation/stage{stage}_kaggle/predictions'
    for p in src.glob('*.csv'):shutil.copyfile(p,SUB/'predictions'/p.name)
(SUB/'curves').mkdir(exist_ok=True)
for stage in (3,4,6):
    for p in (SUB/f'validation/stage{stage}_kaggle/runs').glob('*/seed*/curves.png'):
        exp,seed=p.parent.parent.name,p.parent.name
        name=f'{exp}_{seed}.png' if not (stage==4 and exp=='T00') else f'T00_stage4_{seed}.png'
        shutil.copyfile(p,SUB/'curves'/name)
for name in ('official','confusions'):
    shutil.copytree(out/name,SUB/name,dirs_exist_ok=True)
# Re-run score/grade locally on existing CSVs only, compare official proposed score.
local=SUB/'validation/stage7_audit';local.mkdir(exist_ok=True)
for exp in ('T00','F01','F01_uncal'):
    result=subprocess.run([sys.executable,str(SUB/'code/eval.py'),'score','--pred',str(SUB/f'predictions/{exp}_seed*_test.csv'),
                          '--test-csv',str(SUB/'labels/test_subset0.csv'),'--labels',str(SUB/'labels/labels.csv'),'--tag',exp,'--out',str(local)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    (local/f'{exp}_score.txt').write_text(result.stdout,encoding='utf-8')
p95=json.loads((out/'official/latency_grade_input.json').read_text())['p95_ms']
result=subprocess.run([sys.executable,str(SUB/'code/eval.py'),'grade','--final',str(SUB/'predictions/F01_seed*_test.csv'),
                      '--baseline',str(SUB/'predictions/T00_seed*_test.csv'),'--uncal',str(SUB/'predictions/F01_uncal_seed*_test.csv'),
                      '--final-val',str(SUB/'predictions/F01_seed*_val.csv'),'--test-csv',str(SUB/'labels/test_subset0.csv'),
                      '--val-csv',str(SUB/'labels/val_subset0.csv'),'--labels',str(SUB/'labels/labels.csv'),
                      '--latency-p95-ms',str(p95),'--latency-method','proper','--out',str(local)],capture_output=True,text=True)
assert result.returncode==0,result.stderr
(local/'grade.txt').write_text(result.stdout,encoding='utf-8')
assert json.loads((local/'grade_I.json').read_text())==json.loads((out/'official/grade_I.json').read_text())
v={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc':'PASS','code_bytes_match':True,
'official_evaluator_unchanged':True,'all_final_and_baseline_predictions_logits_metrics_match':True,
'freeze_calibration_and_test_receipts_match':True,'mean_std_ddof1_match':True,'local_score_grade_match':True,
'proposed_rubric_I_score':json.loads((local/'grade_I.json').read_text())['total'],'checkpoints_in_small_zip':False,
'notebook_errors':0,'runs':checks,'training_curves_count':len(list((SUB/'curves').glob('*.png')))}
(out/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v,indent=2))
