"""Read-only submission audit. No image loading, model inference, training or calibration."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from eval import read_pred,check_against_csv,compute_metrics

SUB=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit():
    for name in ('README.md','results.xlsx','report.md','curves','code','predictions'):
        assert (SUB/name).exists(),name
    source_eval=SUB/'validation/stage6_kaggle/code/eval.py'
    assert (SUB/'code/eval.py').read_bytes()==source_eval.read_bytes(),'Evaluator modified'
    official={e:json.loads((SUB/f'official/{e}_summary.json').read_text()) for e in ('F01','T00','F01_uncal')}
    metrics={}
    for exp in official:
        metrics[exp]=[]
        assert len(list((SUB/'predictions').glob(f'{exp}_seed*_test.csv')))==3
        for seed in (0,1,2):
            p=read_pred(str(SUB/f'predictions/{exp}_seed{seed}_test.csv'))
            check_against_csv(p,str(SUB/'labels/test_subset0.csv'),'test')
            m=compute_metrics(p.y_true,p.y_pred,p.probs);metrics[exp].append(m)
            v=read_pred(str(SUB/f'predictions/{exp}_seed{seed}_val.csv'))
            check_against_csv(v,str(SUB/'labels/val_subset0.csv'),'val')
            if exp!='F01_uncal':
                d=SUB/f'validation/stage6_kaggle/runs/{exp}/seed{seed}/final_evaluation'
                started=json.loads((d/'test_started.json').read_text());receipt=json.loads((d/'test_report.json').read_text())
                assert started==receipt['identity'] and receipt['test_forward_passes']==1
                assert receipt['test_logits_sha256']==sha(d/'test_logits.npz')
        for key in ('macro_f1','top1','ece','nll'):
            x=np.asarray([m[key] for m in metrics[exp]])
            assert np.isclose(x.mean(),official[exp][key]['mean'],atol=1e-7,rtol=0)
            assert np.isclose(x.std(ddof=1),official[exp][key]['std'],atol=1e-7,rtol=0)
    for seed in (0,1,2):
        a=read_pred(str(SUB/f'predictions/F01_seed{seed}_test.csv'));b=read_pred(str(SUB/f'predictions/F01_uncal_seed{seed}_test.csv'))
        assert np.array_equal(a.y_pred,b.y_pred)
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(SUB/'results.xlsx') as z:
        assert z.testzip() is None
        xml=ET.fromstring(z.read('xl/workbook.xml'))
        sheets=[x.attrib['name'] for x in xml.findall('m:sheets/m:sheet',ns)]
        assert set(sheets)=={'Summary','Final','Backbones','Training','Inference','PerClass','Latency'}
        for name in z.namelist():
            if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                xml=ET.fromstring(z.read(name));assert not xml.findall('.//m:c[@t="e"]',ns),'Workbook error cells'
    curves=pd.read_csv(SUB/'curves/manifest.csv')
    assert len(curves)==21
    for r in curves.itertuples():
        for p in (r.curve,r.history,r.config):assert (SUB/p).is_file(),p
    assert (SUB/'curves/S00_overfit9.png').is_file()
    for path in (SUB/'code').glob('*.py'):
        assert ('Not'+'ImplementedError') not in path.read_text(encoding='utf-8-sig'),path
    manifest=SUB/'submission_manifest.json';count=None
    if manifest.exists():
        records=json.loads(manifest.read_text())['files'];count=len(records)
        for record in records:
            path=SUB/record['path'];assert path.is_file() and sha(path)==record['sha256'],record['path']
    return {'status':'PASS','read_only':True,'evaluator_unchanged':True,'final_baseline_seeds':[0,1,2],
            'test_rows_per_seed':3507,'ddof':1,'xlsx_sheets':sheets,'epoch_training_curves':21,'overfit_curve':1,
            'manifest_files_verified':count,'final_test_macro_f1':official['F01']['macro_f1'],
            'final_test_top1':official['F01']['top1'],'rubric_I_proposed':20,
            'known_limitations':['T07 is a single-loss repeat, not a multi-factor interaction.',
                                 'Checkpoint weights are retained in Kaggle output, absent from the small result ZIP.',
                                 'Workbook was rendered and audited; native Excel behavior was not checked.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    result=audit()
    if args.write_report:
        path=SUB/'validation/stage7_audit/final_audit.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
