"""Final recipe and exactly-one test traversal checks; synthetic fixtures only."""
import json,sys,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
SUB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SUB/'code'))
import stage6,train

class FinalTests(unittest.TestCase):
    def test_seed_plan_and_recipe_guard(self):
        configs=stage6.make_configs('images',SUB/'labels','outputs')
        stage6.validate_configs(configs)
        self.assertEqual({c.seed for c in configs},{0,1,2})
        for key,value in [('lr_head',.02),('label_smoothing',.2),('save_test_predictions',True),('epochs',11)]:
            changed=configs.copy();changed[0]=replace(changed[0],**{key:value})
            with self.assertRaises(ValueError):stage6.validate_configs(changed)
    def test_started_without_complete_cache_cannot_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);identity={'seed':0};train.write_json(d/'test_started.json',identity)
            with self.assertRaises(RuntimeError):stage6.test_state(d,identity)
            with self.assertRaises(ValueError):stage6.test_state(d,{'seed':1})
    def test_atomic_cache_recovery_requires_matching_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);identity={'seed':0};train.write_json(d/'test_started.json',identity)
            stage6.atomic_npz(d/'test_logits.npz',identity_sha256=np.asarray(stage6.hash_identity(identity)))
            self.assertEqual(stage6.test_state(d,identity),'cached')
            stage6.atomic_npz(d/'test_logits.npz',identity_sha256=np.asarray('wrong'))
            with self.assertRaises(ValueError):stage6.test_state(d,identity)
    def test_calibrated_uncalibrated_exports_share_one_forward_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=train.Config(exp_id='F01',seed=0,out_dir=str(root/'runs'),pred_dir=str(root/'predictions'),device='cpu')
            d=train.run_dir(cfg);e=d/'final_evaluation';e.mkdir(parents=True)
            (d/'best.pt').write_bytes(b'synthetic checkpoint identity, never loaded')
            val={'temperature':.7,'inference_method':list(stage6.FINAL_METHOD)};train.write_json(e/'val_report.json',val)
            frame=pd.DataFrame({'Filename':['a.jpg','b.jpg'],'Label':[0,1]})
            logits=np.zeros((2,9));logits[0,0]=3;logits[1,1]=2
            with patch.object(stage6,'load_net',return_value=(torch.nn.Identity(),{})),patch.object(stage6,'evaluate_logits',return_value=(frame.Filename.to_numpy(),frame.Label.to_numpy(),logits)) as call:
                first=stage6.evaluate_test_once(cfg,frame,val,'fixed')
                second=stage6.evaluate_test_once(cfg,frame,val,'fixed')
                self.assertEqual(call.call_count,1)
                self.assertEqual(first,second)
            a=pd.read_csv(root/'predictions/F01_seed0_test.csv');b=pd.read_csv(root/'predictions/F01_uncal_seed0_test.csv')
            np.testing.assert_array_equal(a.y_pred,b.y_pred)
            self.assertFalse(np.allclose(a.filter(regex='^p[0-8]$'),b.filter(regex='^p[0-8]$')))
    def test_freeze_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            configs=stage6.make_configs('images',SUB/'labels',tmp)
            selection={'training_source':'T05_smoothing','inference_resolution':288,'dtype':'fp32','seeds':[0,1,2]}
            a=stage6.freeze(configs,selection,tmp,SUB/'code');b=stage6.freeze(configs,selection,tmp,SUB/'code')
            self.assertEqual(a,b)
            with self.assertRaises(ValueError):stage6.freeze(configs,{**selection,'note':'changed'},tmp,SUB/'code')
    def test_summary_uses_sample_std_from_actual_prediction_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);labels=root/'labels';labels.mkdir()
            frame=pd.DataFrame({'Filename':[f'{i}.jpg' for i in range(9)],'Label':range(9)})
            frame.to_csv(labels/'val_subset0.csv',index=False);configs=[];scores=[]
            for seed in range(3):
                cfg=train.Config(exp_id='F01',seed=seed,labels_dir=str(labels),out_dir=str(root/'runs'),pred_dir=str(root/'predictions'))
                configs.append(cfg);probs=np.full((9,9),.01)
                guesses=np.arange(9);guesses[:seed+1]=(guesses[:seed+1]+1)%9
                probs[np.arange(9),guesses]=.92
                stage6.save_predictions(train.pred_path(cfg,'val'),frame.Filename,frame.Label,probs)
                metrics=stage6.metric_json(frame.Label.to_numpy(),probs);scores.append(metrics['macro_f1'])
                train.write_json(train.run_dir(cfg)/'final_evaluation/val_report.json',{'metrics':metrics})
            stage6.summarize(configs,root)
            result=pd.read_csv(root/'final_mean_std.csv').iloc[0]
            self.assertAlmostEqual(result.macro_f1_std,float(np.std(scores,ddof=1)))
            self.assertEqual(result.n_seeds,3)
    def test_all_training_and_val_finish_before_first_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);configs=stage6.make_configs('images',SUB/'labels',root);trace=[]
            def trained(cfg):
                trace.append('train');train.write_json(train.run_dir(cfg)/'summary.json',{'status':'completed'})
                return {'status':'completed'}
            def validated(*args):trace.append('val');return {}
            def tested(*args):trace.append('test');return {}
            with patch.object(stage6,'freeze',return_value='fixed'),patch.object(train,'run',side_effect=trained),patch.object(stage6,'prepare_val',side_effect=validated),patch.object(stage6,'evaluate_test_once',side_effect=tested),patch.object(stage6,'summarize',return_value={'status':'incomplete'}),patch.object(stage6,'export'):
                stage6.run_suite(configs,{},root,SUB/'code',root/'results.zip')
            self.assertEqual(trace,['train']*6+['val']*6+['test']*6)
    def test_official_cli_score_and_grade_use_exported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);labels=root/'labels';labels.mkdir()
            (labels/'labels.csv').write_bytes((SUB/'labels/labels.csv').read_bytes())
            names=[f'{i}.jpg' for i in range(9)];truth=np.arange(9)
            frame=pd.DataFrame({'Filename':names,'Label':truth})
            frame.to_csv(labels/'test_subset0.csv',index=False);frame.to_csv(labels/'val_subset0.csv',index=False)
            configs=stage6.make_configs('images',labels,root)
            probs=np.full((9,9),.01);probs[np.arange(9),truth]=.92
            for cfg in configs:
                for split in ('val','test'):
                    stage6.save_predictions(train.pred_path(cfg,split),names,truth,probs)
                    if cfg.exp_id=='F01':stage6.save_predictions(root/'predictions'/f'F01_uncal_seed{cfg.seed}_{split}.csv',names,truth,probs)
                train.write_json(train.run_dir(cfg)/'final_evaluation/val_report.json',{'latency_batch1':{'p95':12.}})
            stage6.official_reports(configs,root,SUB/'code')
            self.assertTrue((root/'official/grade_I.json').is_file())
            for exp in ('F01','T00','F01_uncal'):
                self.assertTrue((root/'official'/f'{exp}_summary.json').is_file())

if __name__=='__main__':unittest.main(verbosity=2)
