"""Stage-4 policy tests; synthetic scores exercise selection logic, never lab results."""
import sys,tempfile,unittest,zipfile
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import stage4

class Stage4Tests(unittest.TestCase):
    def setUp(self):
        self.configs=stage4.make_configs('images','labels','outputs')
    def test_independent_axes_and_shared_recipe(self):
        stage4.validate_plan(self.configs)
        self.assertEqual(len(self.configs),7)
        self.assertEqual({c.init for c in self.configs},{'scratch','frozen','finetune'})
        self.assertEqual({c.loss for c in self.configs},{'ce','ls','focal'})
        modified=self.configs.copy();modified[1]=replace(modified[1],lr_head=.1)
        with self.assertRaises(ValueError):stage4.validate_plan(modified)
    def test_combination_selects_each_axis_against_baseline(self):
        scores=dict(zip([c.exp_id for c in self.configs],[.8,.7,.81,.82,.83,.8,.79]))
        combo,decision=stage4.choose_combination(self.configs,{k:{'macro_f1':v} for k,v in scores.items()})
        self.assertEqual((combo.init,combo.aug,combo.mix,combo.loss),('frozen','basic','cutmix','ce'))
        self.assertEqual(decision['axis_choices'],{'A':'T02_frozen','B':'T04_cutmix','C':'T00'})
        self.assertFalse(combo.save_test_predictions)
    def test_ties_keep_baseline_and_partial_is_rejected(self):
        scores={c.exp_id:{'macro_f1':.8} for c in self.configs}
        combo,decision=stage4.choose_combination(self.configs,scores)
        self.assertTrue(decision['all_axes_kept_baseline'])
        self.assertEqual(combo.exp_id,'T07_combined')
        scores.pop('T06_focal')
        with self.assertRaises(ValueError):stage4.choose_combination(self.configs,scores)
    def test_no_test_or_subset(self):
        for key,value in [('save_test_predictions',True),('debug_train_per_class',1),('debug_val_per_class',1)]:
            modified=self.configs.copy();modified[0]=replace(modified[0],**{key:value})
            with self.assertRaises(ValueError):stage4.validate_plan(modified)
    def test_export_keeps_manifest_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'outputs';code=root/'code';out.mkdir();code.mkdir()
            (out/'best.pt').write_bytes(b'checkpoint');(out/'summary.json').write_text('{}')
            (code/'stage4.py').write_text('# test fixture');(root/'stage4_manifest.json').write_text('{}')
            destination=root/'stage4_results.zip';stage4.export_results(out,destination,code)
            with zipfile.ZipFile(destination) as z:
                self.assertEqual(set(z.namelist()),{'summary.json','code/stage4.py','stage4_manifest.json'})
            self.assertTrue((out/'best.pt').exists())

if __name__=='__main__':unittest.main(verbosity=2)
