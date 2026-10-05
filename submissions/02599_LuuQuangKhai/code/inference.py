"""Deterministic inference views and scalar temperature calibration."""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F


def view_identity(x):return x

def view_hflip(x):return x.flip(-1)

def views_multicrop(x,crop):
    h,w=x.shape[-2:]
    if crop>min(h,w):raise ValueError('Crop exceeds image dimensions')
    return [x[...,y:y+crop,z:z+crop] for y,z in [(0,0),(0,w-crop),(h-crop,0),(h-crop,w-crop),((h-crop)//2,(w-crop)//2)]]

def views_multiscale(x,sizes):
    return [F.interpolate(x,size=(s,s),mode='bicubic',align_corners=False) for s in sizes]

def apply_temperature(logits,T):
    if not math.isfinite(T) or T<=0:raise ValueError('Positive finite temperature required')
    x=np.asarray(logits,dtype=np.float64)/T;x-=x.max(-1,keepdims=True)
    p=np.exp(x);return p/p.sum(-1,keepdims=True)

def aggregate_views(logits_per_view,space='prob'):
    a=np.stack(logits_per_view)
    if space=='prob':return apply_temperature(a,1).mean(0)
    if space=='logit':return apply_temperature(a.mean(0),1)
    raise ValueError('Use prob/logit')

def ensemble_probs(list_of_probs):
    a=np.stack(list_of_probs)
    if not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid probabilities')
    p=a.mean(0);return p/p.sum(-1,keepdims=True)

def fit_temperature(val_logits,val_labels):
    """Bounded golden-section search for scalar logT on val NLL, including T=1."""
    x=np.asarray(val_logits,dtype=np.float64);y=np.asarray(val_labels,dtype=int)
    if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all():raise ValueError('Invalid logits')
    def nll(logt):
        z=x/np.exp(logt);m=z.max(1)
        return float((m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]).mean())
    a,b=-4.,4.;r=(5**.5-1)/2;c=b-r*(b-a);d=a+r*(b-a)
    for _ in range(100):
        if nll(c)<nll(d):b,d=d,c;c=b-r*(b-a)
        else:a,c=c,d;d=a+r*(b-a)
    candidates=[0.,a,b,(a+b)/2]
    return float(np.exp(min(candidates,key=nll)))

@torch.inference_mode()
def predict_logits(model,loader,device,view=None):
    model.eval();names=[];truth=[];values=[]
    for x,y,n in loader:
        x=x.to(device);x=view(x) if view else x
        values.append(model(x).float().cpu().numpy());truth.extend(y.numpy());names.extend(n)
    return names,np.asarray(truth),np.concatenate(values)

def fuse_conv_bn(model):
    """Fuse explicit adjacent Conv/BN pairs in Sequential only; never guess graph adjacency."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    result=copy.deepcopy(model).eval()
    for module in result.modules():
        if isinstance(module,torch.nn.Sequential):
            children=list(module._modules.items())
            for (a,conv),(b,bn) in zip(children,children[1:]):
                if isinstance(conv,torch.nn.Conv2d) and isinstance(bn,torch.nn.BatchNorm2d):
                    module._modules[a]=fuse_conv_bn_eval(conv,bn);module._modules[b]=torch.nn.Identity()
    return result
