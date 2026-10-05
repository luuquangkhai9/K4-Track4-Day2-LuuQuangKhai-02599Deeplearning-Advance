"""Validate stage5 returned predictions, all views, calibration and measured latency."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SUB=ROOT/'submissions/02599_LuuQuangKhai'
sys.path.insert(0,str(SUB/'code'))
from eval import read_pred,check_against_csv,compute_metrics
from inference import aggregate_views,apply_temperature,fit_temperature
archive=ROOT/'stage5_results.zip';out=SUB/'validation/stage5_kaggle'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts
        if name.startswith('code/') and name.endswith('.py'):assert z.read(name)==(SUB/name).read_bytes(),name
    assert z.read('code/eval.py')==(ROOT/'eval.py').read_bytes()
    assert not any('_test.csv' in n for n in z.namelist())
    out.mkdir(parents=True,exist_ok=True);z.extractall(out)
summary=json.loads((out/'stage5_summary.json').read_text());assert summary['completed']==16 and summary['status']=='complete'
table=pd.read_csv(out/'inference.csv').set_index(['training_exp','method']);assert len(table)==16
checks=[]
for path in sorted((out/'methods').glob('*/*/report.json')):
    r=json.loads(path.read_text());exp,name=r['training_exp'],r['method']
    a=np.load(path.parent/'views_logits.npz');p=read_pred(str(out/'predictions'/f'{exp}_{name}_seed0_val.csv'))
    check_against_csv(p,str(SUB/'labels/val_subset0.csv'),'val')
    assert list(a['filenames'])==list(p.filenames) and np.array_equal(a['y_true'],p.y_true)
    if name=='I07_temperature':
        q=apply_temperature(a['logits'].mean(0),r['temperature'])
        assert np.isclose(r['temperature'],fit_temperature(a['logits'][0],p.y_true),atol=1e-10)
        assert np.array_equal(a['logits'][0].argmax(1),p.y_pred)
    else:q=aggregate_views(list(a['logits']),r['aggregation'])
    assert np.allclose(q,p.probs,atol=1e-6,rtol=0),(exp,name)
    m=compute_metrics(p.y_true,p.y_pred,p.probs)
    for k in ('macro_f1','top1','ece','nll'):
        assert np.isclose(m[k],r['metrics'][k],atol=1e-7,rtol=0)
        assert np.isclose(m[k],table.loc[(exp,name),'val_'+k],atol=1e-7,rtol=0)
    for k in ('f1','recall'):assert np.allclose(m[k],r[k],atol=1e-7,rtol=0)
    for batch in (1,32):
        l=r[f'latency_batch{batch}'];assert l['batch']==batch and l['warmup'] if 'warmup' in l else l['batch']==batch
        assert l['n']>=50 and len(l['samples_ms'])==l['n'] and min(l['samples_ms'])>0
        for percentile in (50,95,99):assert np.isclose(l['p'+str(percentile)],np.percentile(l['samples_ms'],percentile))
    checks.append({'training_exp':exp,'method':name,**{k:m[k] for k in ('macro_f1','top1','ece')}})
n=json.loads((SUB/'code/kaggle_stage5.ipynb').read_text(encoding='utf-8'))
assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
selection={'training_source':'T05_smoothing','backbone':'convnext_tiny.fb_in1k','loss':'ls','inference_method':'I04_res288',
'inference_resolution':288,'dtype':'fp32','temperature_policy':'Fit scalar T on final-resolution val logits separately per seed; never on test.',
'baseline':'T00 CE + I00 FP32 224 without calibration','seeds':[0,1,2],
'reason':'Highest stage5 val macro-F1; measured batch1 p95 15.4685 ms. Temperature is an accuracy-preserving calibration rule motivated by stage5 I07.'}
v={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc':'PASS','code_bytes_match':True,
'official_evaluator_unchanged':True,'all16_full_val_predictions_logits_metrics_match':True,
'latency_samples_percentiles_match':True,'notebook_errors':0,'test_evaluated':False,'runs':checks,'stage6_selection':selection}
(out/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
(SUB/'stage6_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
print(json.dumps(selection,indent=2));print('All16 inference artifacts verified.')
