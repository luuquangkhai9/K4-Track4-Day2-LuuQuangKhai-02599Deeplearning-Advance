"""Validate returned stage-4 results without running archive code or training."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SUB=ROOT/'submissions/02599_LuuQuangKhai'
sys.path.insert(0,str(SUB/'code'))
import train,stage4
from eval import read_pred,check_against_csv,compute_metrics
archive=ROOT/'stage4_results.zip';out=SUB/'validation/stage4_kaggle'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts
        if name.startswith('code/') and name.endswith('.py'):assert z.read(name)==(SUB/name).read_bytes(),name
    assert z.read('code/eval.py')==(ROOT/'eval.py').read_bytes()
    assert not any('_test.csv' in n for n in z.namelist())
    out.mkdir(parents=True,exist_ok=True);z.extractall(out)
report=json.loads((out/'stage4_summary.json').read_text())
assert report['status']=='complete' and report['completed']==8 and not report['test_evaluated']
configs=[train.Config(**c) for c in json.loads((out/'single_axis_configs.json').read_text())]
stage4.validate_plan(configs)
combo=train.Config(**json.loads((out/'combined_config.json').read_text()))
table=pd.read_csv(out/'training.csv').set_index('exp_id');assert len(table)==8
metrics={};checks=[]
for cfg in configs+[combo]:
    exp=cfg.exp_id;d=out/'runs'/exp/'seed0'
    s=json.loads((d/'summary.json').read_text());h=pd.read_csv(d/'history.csv')
    assert s['status']=='completed' and s['epochs_completed']==10 and not s['test_evaluated'] and not s['debug_subset']
    assert list(h.epoch)==list(range(1,11))
    assert s['best_epoch']==int(h.loc[h.val_macro_f1.idxmax(),'epoch'])
    p=read_pred(str(out/'predictions'/f'{exp}_seed0_val.csv'))
    check_against_csv(p,str(SUB/'labels/val_subset0.csv'),'val');m=compute_metrics(p.y_true,p.y_pred,p.probs)
    for key in ('macro_f1','top1'):
        assert np.isclose(m[key],s['val_'+key],atol=1e-7,rtol=0)
    for key in ('macro_f1','top1','balanced_acc','ece','nll'):
        assert np.isclose(m[key],table.loc[exp,'val_'+key],atol=1e-7,rtol=0)
    for k in range(9):
        for key in ('f1','recall'):
            assert np.isclose(m[key][k],table.loc[exp,f'val_{key}_class{k}'],atol=1e-7,rtol=0)
    a=np.load(d/'val_logits.npz');assert list(a['filenames'])==list(p.filenames) and np.array_equal(a['y_true'],p.y_true)
    l=a['logits'].astype(np.float64);q=np.exp(l-l.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
    assert np.allclose(q,p.probs,atol=1e-6,rtol=0)
    metrics[exp]=m;checks.append({'exp_id':exp,'macro_f1':m['macro_f1'],'top1':m['top1'],'ece':m['ece'],'best_epoch':s['best_epoch']})
for exp,m in metrics.items():
    assert np.isclose(table.loc[exp,'delta_macro_f1_vs_T00'],m['macro_f1']-metrics['T00']['macro_f1'],atol=1e-7,rtol=0)
expected,decision=stage4.choose_combination(configs,{c.exp_id:metrics[c.exp_id] for c in configs})
assert decision==json.loads((out/'combination_selection.json').read_text())['decision']
for k in ('init','aug','mix','loss'):assert getattr(expected,k)==getattr(combo,k)
n=json.loads((SUB/'code/kaggle_stage4.ipynb').read_text(encoding='utf-8'))
assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
logs={}
for name in ('test_pipeline.py','test_stage3.py','test_stage4.py'):
    log=(out/(name+'.log')).read_text();assert '\nOK\n' in log or log.rstrip().endswith('OK');logs[name]='PASS'
verification={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc':'PASS',
'code_bytes_match':True,'official_evaluator_unchanged':True,'full_val_predictions_and_logits_match':True,
'per_class_metrics_and_deltas_match':True,'combination_selection_verified':True,'kaggle_tests':logs,
'notebook_errors':0,'test_evaluated':False,'checkpoints_in_small_zip':False,'runs':checks,
'candidate':'T05_smoothing','baseline':'T00','combination_is_single_axis_repeat':True,
'note':'T05 and T07 use the same recipe and seed; they are not independent seeds or a multi-factor gain.'}
(out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
