"""timm models with classifier LR and no weight decay for norm/bias."""
import copy
import torch
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50", "resnext50": "resnext50_32x4d", "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224", "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0", "mobilenetv3": "mobilenetv3_large_100",
}
def build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune", head_init_std=0.01):
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"Unknown initialization: {init}")
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    # A shared small random head avoids architecture-dependent initial logit scale.
    # timm MobileNet's default head init scales with output classes and is large for K=9.
    if head_init_std <= 0:
        raise ValueError("head_init_std must be positive")
    for layer in model.get_classifier().modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.normal_(layer.weight, std=head_init_std)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    if init == "frozen":
        freeze_backbone(model)
    return model

def freeze_backbone(model):
    model.requires_grad_(False)
    model.get_classifier().requires_grad_(True)
    model.eval()

def set_train_mode(model, init):
    if init == "frozen":
        model.eval()
        model.get_classifier().train()
    else:
        model.train()

def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = {id(p) for p in model.get_classifier().parameters()}
    no_decay = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = lr_head if id(p) in head else lr_backbone
        decay = 0.0 if p.ndim <= 1 or name.endswith(".bias") or name in no_decay else weight_decay
        buckets.setdefault((lr, decay), []).append(p)
    # Four groups when head has bias: exempt ALL norm/bias, including head bias.
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in buckets.items()]

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_gmacs(model, img_size=224):
    """thop MAC estimate; attention may require custom hooks. Never implies latency."""
    from thop import profile
    replica = copy.deepcopy(model).cpu().eval()
    with torch.inference_mode():
        macs, _ = profile(replica, inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)
    return float(macs / 1e9)

