"""Five-backbone screening on VAL only. Serial GPU runs with resumable output."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from timm.models.vision_transformer import Attention

import model as models
import train
from benchmark import latency_report
from eval import read_pred, check_against_csv, compute_metrics

PLAN = [
    ("B01_resnet50", "resnet50.a1_in1k"),
    ("B02_convnext_tiny", "convnext_tiny.fb_in1k"),
    ("B03_deit_small", "deit_small_patch16_224.fb_in1k"),
    ("B04_efficientnet_b0", "efficientnet_b0.ra_in1k"),
    ("B05_mobilenetv3", "mobilenetv3_large_100.ra_in1k"),
]


def make_configs(images_dir, labels_dir, output, receipt=None, device="cuda:0"):
    return [train.Config(exp_id=exp, backbone=backbone, seed=0, fold=0, init="finetune",
                         head_init_std=0.01, img_size=224, aug="basic", mix=None, loss="ce",
                         sampler=None, epochs=10, batch_size=32, warmup_epochs=1.0,
                         lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05, ema_decay=None,
                         amp=True, num_workers=2, device=device, images_dir=str(images_dir),
                         labels_dir=str(labels_dir), verified_summary=str(receipt) if receipt else None,
                         out_dir=str(Path(output) / "runs"), pred_dir=str(Path(output) / "predictions"))
            for exp, backbone in PLAN]


def mac_report(model, img_size=224):
    """MACs for Conv2d + Linear + DeiT QK^T/AV; scalar/norm/softmax ops excluded.

    One multiplication-accumulation is one MAC (not two FLOPs). Attention products
    are counted explicitly even when timm uses fused scaled-dot-product attention.
    Only the five supported architectures are intended for this counter.
    """
    counts = {"conv": 0, "linear": 0, "attention_products": 0}
    hooks = []
    modes = {module: module.training for module in model.modules()}
    def conv_hook(module, inputs, output):
        counts["conv"] += output.numel() * module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
    def linear_hook(module, inputs, output):
        counts["linear"] += output.numel() * module.in_features
    def attention_hook(module, inputs, output):
        batch, tokens, channels = inputs[0].shape
        counts["attention_products"] += 2 * batch * tokens * tokens * channels
    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, Attention):
                hooks.append(module.register_forward_hook(attention_hook))
        parameter = next(model.parameters())
        model.eval()
        with torch.inference_mode():
            result = model(torch.zeros(1, 3, img_size, img_size, device=parameter.device, dtype=parameter.dtype))
        if result.ndim != 2 or result.shape[0] != 1:
            raise ValueError("Expected classification output")
    finally:
        for hook in hooks:
            hook.remove()
        for module, mode in modes.items():
            module.training = mode
    return {"gmacs": sum(counts.values()) / 1e9, "mac_breakdown": counts,
            "method": "Conv2d/Linear hooks + explicit DeiT QK^T and AV",
            "excluded_ops": "bias, normalization, activation, pooling, softmax and other elementwise operations"}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_run(cfg):
    directory = train.run_dir(cfg)
    path = directory / "best.pt"
    identity = {"checkpoint_sha256": file_sha(path), "stage3_code_sha256": file_sha(__file__),
                "benchmark_code_sha256": file_sha(Path(__file__).with_name("benchmark.py"))}
    report_path = directory / "profile.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved.get("identity") == identity:
            return saved
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = models.build_model(cfg.backbone, pretrained=False, init=cfg.init,
                               head_init_std=cfg.head_init_std).eval()
    model.load_state_dict(saved["model"])
    macs = mac_report(model, cfg.img_size)
    latency = latency_report(model, batch_size=1, img_size=cfg.img_size, dtype="fp32",
                             device=cfg.device, warmup=10, iters=100)
    report = {"identity": identity, "backbone": cfg.backbone, "best_epoch": saved["epoch"],
              "params_m": models.count_params(model), **macs, "latency": latency}
    train.write_json(report_path, report)
    return report


def verified_metrics(cfg, summary):
    path = train.pred_path(cfg, "val")
    pred = read_pred(str(path))
    check_against_csv(pred, str(Path(cfg.labels_dir) / f"val_subset{cfg.fold}.csv"), "val")
    metrics = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
    for key in ("macro_f1", "top1"):
        if not np.isclose(metrics[key], summary["val_" + key], rtol=0, atol=1e-7):
            raise ValueError(f"Saved predictions disagree with summary: {cfg.exp_id}/{key}")
    return metrics


def summarize(configs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configs:
        directory = train.run_dir(cfg)
        row = {"exp_id": cfg.exp_id, "backbone": cfg.backbone, "seed": cfg.seed,
               "img_size": cfg.img_size, "epochs_planned": cfg.epochs, "batch_size": cfg.batch_size,
               "status": "pending", "eligible_for_ranking": False}
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            metrics = verified_metrics(cfg, summary)
            row.update(status=summary["status"], epochs_completed=summary["epochs_completed"],
                       best_epoch=summary["best_epoch"], val_macro_f1=metrics["macro_f1"],
                       val_top1=metrics["top1"], val_balanced_acc=metrics["balanced_acc"], val_ece=metrics["ece"],
                       params_m=summary["params_m"], train_val_seconds_per_epoch=summary["epoch_seconds_mean"])
            history = pd.read_csv(directory / "history.csv")
            row["train_val_seconds_total"] = float(history.epoch_seconds.sum())
            profile = directory / "profile.json"
            if summary["status"] == "completed" and profile.exists():
                data = json.loads(profile.read_text())
                latency = data["latency"]
                row.update(gmacs=data["gmacs"], latency_p50_ms=latency["p50"],
                           latency_p95_ms=latency["p95"], latency_p99_ms=latency["p99"],
                           latency_dtype=latency["dtype"], latency_gpu=latency["gpu"],
                           eligible_for_ranking=summary["epochs_completed"] == cfg.epochs)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "backbones.csv", index=False)
    ranked = table[table.eligible_for_ranking].copy()
    if len(ranked):
        ranked = ranked.sort_values(["val_macro_f1", "latency_p50_ms"], ascending=[False, True])
        ranked.to_csv(output / "backbones_ranked.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(ranked.latency_p50_ms, ranked.val_macro_f1, s=70)
        for row in ranked.itertuples():
            ax.annotate(row.exp_id, (row.latency_p50_ms, row.val_macro_f1), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="FP32 batch-1 forward latency p50 (ms)", ylabel="Validation macro-F1",
               title="Stage 3 — completed runs, seed 0; not test results")
        ax.margins(0.25)
        fig.tight_layout()
        fig.savefig(output / "backbone_tradeoff.png", dpi=150)
        plt.close(fig)
    report = {"completed_and_profiled": len(ranked), "planned": len(configs),
              "status": "complete" if len(ranked) == len(configs) else "incomplete",
              "test_evaluated": False, "selection_is_provisional": len(ranked) != len(configs),
              "val_leader": ranked.iloc[0].exp_id if len(ranked) else None,
              "note": "One-seed screening only; do not infer significance from small differences."}
    train.write_json(output / "stage3_summary.json", report)
    return report


def export_results(output, destination, code_dir):
    """Small result ZIP only; checkpoints remain in the saved notebook output."""
    output, destination, code_dir = Path(output), Path(destination), Path(code_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix not in (".pt", ".tmp"):
                archive.write(path, path.relative_to(output).as_posix())
        for path in sorted(code_dir.glob("*.py")):
            archive.write(path, "code/" + path.name)
        manifest = code_dir.parent / "stage3_manifest.json"
        if manifest.exists():
            archive.write(manifest, manifest.name)
    temporary.replace(destination)


def run_suite(configs, output, result_zip, code_dir, budget_seconds=21600):
    """Cooperative time budget, checked between epochs via train.run, never guarantees host uptime."""
    if budget_seconds < 900:
        raise ValueError("Provide at least 900 seconds; reserve time to export outputs")
    if len({c.exp_id for c in configs}) != len(configs):
        raise ValueError("Duplicate exp_id")
    if any(c.save_test_predictions or c.debug_train_per_class or c.debug_val_per_class for c in configs):
        raise ValueError("Stage 3 must use full train/val, no test")
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {c.exp_id: train.signature(c) for c in configs}
    plan_file = output / "suite_identity.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != identity:
        raise ValueError("Suite config/code/CSV changed; restore original notebook or use a new output directory")
    train.write_json(plan_file, identity)
    train.write_json(output / "suite_config.json", [asdict(c) for c in configs])
    try:
        for cfg in configs:
            remaining = budget_seconds - (time.monotonic() - started)
            if remaining < 900:
                print("Time budget reached; keeping outputs for the next session", flush=True)
                break
            cfg.max_run_seconds = remaining - 300  # leave time for val export and profiling
            print(f"Starting/resuming {cfg.exp_id}: {cfg.backbone}, budget remaining {remaining/60:.1f} min", flush=True)
            summary = train.run(cfg)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if summary["status"] == "completed":
                profile_run(cfg)
            summarize(configs, output)
            export_results(output, result_zip, code_dir)
            if summary["status"] != "completed":
                print("Paused at epoch boundary; resume with the SAME configuration", flush=True)
                break
    except Exception:
        (output / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        # Preserve partial results even if a later model fails. Do not silently alter batch/LR.
        try:
            report = summarize(configs, output)
        finally:
            export_results(output, result_zip, code_dir)
    print(json.dumps(report, indent=2), flush=True)
    return report
