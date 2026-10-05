"""Verify returned stage-3 artifacts with the unchanged official evaluator; no training."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SUB=ROOT/'submissions/02599_LuuQuangKhai'
sys.path.insert(0,str(SUB/'code'))
from eval import read_pred,check_against_csv,compute_metrics
archive=ROOT/'stage3_results.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        if name.startswith('code/') and name.endswith('.py'):
            assert z.read(name)==(SUB/name).read_bytes(),name
    assert z.read('code/eval.py')==(ROOT/'eval.py').read_bytes()
    out=SUB/'validation/stage3_kaggle'
    out.mkdir(parents=True,exist_ok=True)
    z.extractall(out)
    assert not any('_test.csv' in n for n in z.namelist())
report=json.loads((out/'stage3_summary.json').read_text())
assert report['status']=='complete' and not report['test_evaluated']
configs=json.loads((out/'suite_config.json').read_text())
assert len(configs)==5
ranked=pd.read_csv(out/'backbones_ranked.csv').set_index('exp_id')
checks=[]
for cfg in configs:
    exp=cfg['exp_id'];d=out/'runs'/exp/'seed0'
    s=json.loads((d/'summary.json').read_text());h=pd.read_csv(d/'history.csv')
    assert s['status']=='completed' and s['epochs_completed']==10 and len(h)==10
    assert not cfg['save_test_predictions'] and not s['test_evaluated']
    p=read_pred(str(out/'predictions'/f'{exp}_seed0_val.csv'))
    check_against_csv(p,str(SUB/'labels/val_subset0.csv'),'val')
    m=compute_metrics(p.y_true,p.y_pred,p.probs)
    for key in ('macro_f1','top1'):
        assert np.isclose(m[key],s['val_'+key],atol=1e-7,rtol=0)
    for key in ('macro_f1','top1','balanced_acc','ece'):
        assert np.isclose(m[key],ranked.loc[exp,'val_'+key],atol=1e-7,rtol=0)
    a=np.load(d/'val_logits.npz')
    assert list(a['filenames'])==list(p.filenames) and np.array_equal(a['y_true'],p.y_true)
    l=a['logits'].astype(np.float64);q=np.exp(l-l.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
    assert np.allclose(q,p.probs,atol=1e-6,rtol=0)
    checks.append({'exp_id':exp,'macro_f1':m['macro_f1'],'top1':m['top1'],'epochs':10})
leader=max(checks,key=lambda x:x['macro_f1'])['exp_id']
assert leader==report['val_leader']=='B02_convnext_tiny'
verification={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc':'PASS',
'code_bytes_match':True,'official_evaluator_unchanged':True,'full_val_predictions_and_logits_match':True,
'test_evaluated':False,'checkpoints_in_small_zip':False,'runs':checks,
'selected_backbone':'convnext_tiny.fb_in1k','selection_reason':'Highest validation macro-F1 among five completed seed-0 runs'}
(out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
