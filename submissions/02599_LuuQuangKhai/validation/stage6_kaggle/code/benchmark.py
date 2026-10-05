"""Forward-only latency. Inputs already reside on device; preprocessing excluded."""
import copy
import time
import numpy as np
import torch


def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError("Use at least 10 warmup and 50 measured iterations")
    synchronize = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"p50": float(np.percentile(samples, 50)), "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)), "mean": float(np.mean(samples)),
            "n": iters, "samples_ms": samples}


def latency_report(model, batch_size, img_size, dtype="fp32", device="cuda", warmup=10, iters=100):
    device = torch.device(device)
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError("dtype must be fp32/amp/fp16")
    if dtype != "fp32" and device.type != "cuda":
        raise ValueError("AMP/FP16 benchmarks require CUDA")
    # Do not mutate training model's device, dtype or BN/dropout mode.
    measured = copy.deepcopy(model).to(device).eval()
    measured = measured.half() if dtype == "fp16" else measured.float()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    with torch.inference_mode(), torch.autocast(device.type, enabled=dtype == "amp"):
        result = bench(lambda: measured(x), warmup, iters, sync)
    return {**result, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, "warmup": warmup,
            "images_per_s": batch_size / (result["p50"] / 1000), "torch": torch.__version__,
            "preprocessing_included": False, "host_to_device_included": False, "bn_fused": False}


def tta_latency(model, k_views, **kw):
    """Measure actual supplied view callables and aggregation, with preloaded input."""
    views = kw.pop("views", None)
    if views is None or len(views) != k_views:
        raise ValueError("Provide one callable per view; latency is never inferred by multiplying K")
    device = torch.device(kw.pop("device", "cuda:0"))
    batch = kw.pop("batch_size", 1)
    size = kw.pop("img_size", 224)
    warmup, iters = kw.pop("warmup", 10), kw.pop("iters", 100)
    if kw:
        raise ValueError(f"Unsupported options: {list(kw)}")
    measured = copy.deepcopy(model).to(device).eval()
    x = torch.randn(batch, 3, size, size, device=device)
    def forward():
        return torch.stack([measured(view(x)).float().softmax(-1) for view in views]).mean(0)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    with torch.inference_mode():
        return bench(forward, warmup, iters, sync)
