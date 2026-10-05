"""Inference operator checks; no lab scores or training are produced."""
import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import inference as inf
import stage5

class InferenceTests(unittest.TestCase):
    def test_crop_coordinates_and_flip(self):
        x=torch.arange(36).reshape(1,1,6,6)
        crops=inf.views_multicrop(x,4)
        self.assertEqual([int(v[0,0,0,0]) for v in crops],[0,2,12,14,7])
        self.assertTrue(torch.equal(inf.view_hflip(inf.view_hflip(x)),x))
        with self.assertRaises(ValueError):inf.views_multicrop(x,7)
    def test_probability_vs_logit_aggregation(self):
        views=[np.array([[8.,0.,0.]]),np.array([[0.,1.,0.]])]
        a=inf.aggregate_views(views,'prob');b=inf.aggregate_views(views,'logit')
        self.assertFalse(np.allclose(a,b));self.assertAlmostEqual(float(a.sum()),1.)
        np.testing.assert_allclose(a,sum(inf.apply_temperature(v,1) for v in views)/2)
    def test_temperature_nll_and_argmax(self):
        x=np.array([[8.,0.],[8.,0.],[8.,0.],[0.,8.]])
        y=np.array([0,0,1,1]);t=inf.fit_temperature(x,y)
        p=inf.apply_temperature(x,t);base=inf.apply_temperature(x,1)
        self.assertGreater(t,1)
        self.assertLess(-np.log(p[np.arange(4),y]).mean(),-np.log(base[np.arange(4),y]).mean())
        np.testing.assert_array_equal(p.argmax(1),x.argmax(1))
        for t in (0,-1,float('nan')):
            with self.assertRaises(ValueError):inf.apply_temperature(x,t)
    def test_real_forward_view_and_aggregation(self):
        class Tiny(torch.nn.Module):
            def forward(self,x):
                a=x.mean((1,2,3));return torch.stack([a,-a],1)
        net=Tiny();x=torch.randn(3,3,256,256)
        for method in stage5.METHODS:
            if method[4]=='amp':continue
            values,probs=stage5.forward_method(net,x,method,1.3)
            self.assertEqual(values.shape[1:],(3,2));self.assertEqual(probs.shape,(3,2))
            torch.testing.assert_close(probs.sum(1),torch.ones(3))
        self.assertEqual(stage5.forward_method(net,x,stage5.METHODS[2])[0].shape[0],5)
    def test_partial_output_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=stage5.summarize(tmp)
            self.assertEqual(r['status'],'incomplete');self.assertFalse(r['test_evaluated'])
    def test_fusion_matches_explicit_sequential_graph(self):
        torch.manual_seed(1)
        net=torch.nn.Sequential(torch.nn.Conv2d(3,4,3),torch.nn.BatchNorm2d(4)).eval()
        x=torch.randn(2,3,12,12);fused=inf.fuse_conv_bn(net)
        torch.testing.assert_close(net(x),fused(x),atol=1e-5,rtol=1e-5)
        self.assertIsInstance(net[1],torch.nn.BatchNorm2d)
        self.assertIsInstance(fused[1],torch.nn.Identity)

if __name__=='__main__':unittest.main(verbosity=2)
