"""Shared training loop. Stage 2 enables train/val only; test remains locked."""
from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass, asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import typing
import numpy as np
import pandas as pd
import torch
from torch import nn
import timm
import dataset
import model as models
import losses
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "mobilenetv3_large_100.ra_in1k"
    init: str = "finetune"
    drop_rate: float = 0.0
    head_init_std: float = 0.01
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 10
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False
    verified_summary: str | None = None
    device: str = "auto"
    resume: bool = True
    stop_after_epochs: int | None = None
    max_run_seconds: float | None = None
    debug_train_per_class: int | None = None
    debug_val_per_class: int | None = None
    measure_gmacs: bool = False


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg, split):
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn on unsupported deterministic kernels; versions/hardware are logged.
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(model, cfg):
    return torch.optim.AdamW(models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    warmup = min(round(cfg.warmup_epochs * steps_per_epoch), max(total - 1, 0))
    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(max((step - warmup) / max(total - warmup, 1), 0), 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between 0 and 1")
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # Average parameters; copy BN running stats/counters from live model.
        live_parameters = dict(model.named_parameters())
        for name, p in self.model.named_parameters():
            p.lerp_(live_parameters[name], 1 - self.decay)
        live = dict(model.named_buffers())
        for name, value in self.model.named_buffers():
            value.copy_(live[name])

    def copy_to(self, model):
        model.load_state_dict(self.model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    device = torch.device(device)
    models.set_train_mode(model, cfg.init)
    total_loss, count, steps = 0.0, 0, 0
    first_lr = optimizer.param_groups[0]["lr"]
    for batch_index, (x, y, _) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = y
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)
            loss = (losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite train loss; inspect input/LR/AMP")
        scaler.scale(loss).backward()
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        successful_step = scaler.get_scale() >= scale
        if successful_step:
            scheduler.step()
            if ema is not None:
                ema.update(model)
            steps += 1
        total_loss += float(loss.detach()) * len(y)
        count += len(y)
        if batch_index % 50 == 0:
            print(f"  train batch {batch_index}/{len(loader)} loss={total_loss/count:.4f}", flush=True)
    if count == 0 or steps == 0:
        raise RuntimeError("No successful optimizer steps")
    return {"train_loss": total_loss / count, "lr": first_lr,
            "lr_end": optimizer.param_groups[0]["lr"], "optimizer_steps": steps, "train_seen": count}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    filenames, targets, all_logits = [], [], []
    total, count = 0.0, 0
    for x, y, names in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x).float()  # FP32 val baseline, deterministic transforms.
        loss = criterion(logits, y)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation output")
        total += float(loss) * len(y)
        count += len(y)
        filenames.extend(names)
        targets.append(y.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    if not count:
        raise ValueError("Empty evaluation loader")
    return filenames, np.concatenate(targets), np.concatenate(all_logits), total / count


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label="train objective")
    axes[0].plot(frame.epoch, frame.val_loss, label="val CE")
    axes[0].legend()
    axes[0].set_ylabel("Loss")
    axes[1].plot(frame.epoch, frame.val_macro_f1, label="val macro-F1")
    axes[1].plot(frame.epoch, frame.val_top1, label="val top-1")
    axes[1].legend()
    axes[1].set_ylabel("Score")
    axes[2].plot(frame.epoch, frame.lr)
    axes[2].set_ylabel("LR (first parameter group, epoch start)")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_checkpoint(value, path):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state(train_loader, val_loader):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_loader": train_loader.generator.get_state(), "val_loader": val_loader.generator.get_state()}


def restore_rng(state, train_loader, val_loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["train_loader"])
    val_loader.generator.set_state(state["val_loader"])


def signature(cfg):
    values = asdict(cfg)
    for key in ("out_dir", "pred_dir", "images_dir", "labels_dir", "verified_summary", "resume",
                "stop_after_epochs", "max_run_seconds", "device"):
        values.pop(key)
    code = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ("dataset.py", "model.py", "losses.py", "train.py", "eval.py")}
    return {"config": values, "labels": dataset.csv_hashes(cfg.labels_dir, cfg.fold), "code": code}


def _subset(frame, per_class):
    if per_class is None:
        return frame
    return frame.groupby("Label", group_keys=False).head(per_class).reset_index(drop=True)


def run(cfg):
    if cfg.save_test_predictions:
        raise ValueError("Test is locked in stage 2. Final evaluation is implemented only after val selection.")
    if cfg.epochs < 1 or cfg.batch_size < 2 or cfg.num_workers < 0 or cfg.warmup_epochs < 0:
        raise ValueError("Invalid epochs/batch_size/workers/warmup")
    if cfg.stop_after_epochs is not None and cfg.stop_after_epochs < 1:
        raise ValueError("stop_after_epochs must be positive")
    if cfg.max_run_seconds is not None and cfg.max_run_seconds <= 0:
        raise ValueError("max_run_seconds must be positive")
    if not cfg.exp_id or Path(cfg.exp_id).name != cfg.exp_id or "\\" in cfg.exp_id:
        raise ValueError("exp_id must be a bare name")
    debug = cfg.debug_train_per_class is not None or cfg.debug_val_per_class is not None
    if debug and not cfg.exp_id.startswith("S"):
        raise ValueError("Subset debug runs must have exp_id starting with S")
    if any(n is not None and n < 1 for n in (cfg.debug_train_per_class, cfg.debug_val_per_class)):
        raise ValueError("Subset size must be positive")
    started = time.monotonic()
    directory = run_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    sig = signature(cfg)
    checkpoint_path = directory / "last.pt"
    summary_path = directory / "summary.json"
    checkpoint = None
    if checkpoint_path.exists():
        if not cfg.resume:
            raise FileExistsError("Run exists; use resume=True or a new exp_id")
        # Load only checkpoints produced by this workflow.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != sig:
            raise ValueError("Checkpoint config/code/CSV mismatch; use original settings or a new exp_id")
    elif (directory / "config.json").exists() and not cfg.resume:
        raise FileExistsError("Run folder exists; choose a new exp_id")
    if checkpoint is not None and checkpoint["epoch"] == cfg.epochs and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            print("Already completed:", directory)
            return previous
    set_seed(cfg.seed)
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    reuse = dataset.verified_stage1(cfg.verified_summary, cfg.images_dir, cfg.labels_dir, cfg.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir, verify_files=not reuse)
    write_json(directory / "split_check.json", {**split_report, "reused_stage1": reuse})
    # No test images or test loader are touched.
    train_df, val_df = _subset(train_df, cfg.debug_train_per_class), _subset(val_df, cfg.debug_val_per_class)
    model = models.build_model(cfg.backbone, pretrained=checkpoint is None, init=cfg.init,
                              drop_rate=cfg.drop_rate, head_init_std=cfg.head_init_std).to(device)
    pre_cfg = model.pretrained_cfg
    mean, std = tuple(pre_cfg.get("mean", dataset.IMAGENET_MEAN)), tuple(pre_cfg.get("std", dataset.IMAGENET_STD))
    train_loader = dataset.make_loader(train_df, cfg.images_dir,
        dataset.build_transforms(True, cfg.img_size, cfg.aug, mean, std),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers, cfg.seed)
    val_loader = dataset.make_loader(val_df, cfg.images_dir,
        dataset.build_transforms(False, cfg.img_size, "basic", mean, std),
        cfg.batch_size, False, num_workers=cfg.num_workers, seed=cfg.seed + 1)
    weight = None
    if cfg.loss == "ce_weighted":
        weight = losses.class_weights(train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy(),
                                       cfg.class_weight_beta or 0).to(device)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                     gamma=cfg.focal_gamma, weight=weight).to(device)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best_score, best_epoch, best_state = [], -1.0, 0, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema:
            ema.model.load_state_dict(checkpoint["ema"])
        history = checkpoint["history"]
        best_score, best_epoch, best_state = checkpoint["best_score"], checkpoint["best_epoch"], checkpoint["best_state"]
        restore_rng(checkpoint["rng"], train_loader, val_loader)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "timm": timm.__version__, "cuda": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    write_json(directory / "config.json", asdict(cfg))
    write_json(directory / "environment.json", environment)
    write_json(directory / "preprocessing.json", {"mean": mean, "std": std,
               "train": "RandomResizedCrop bicubic + horizontal flip; configured augmentation",
               "val": f"Resize shorter edge {round(cfg.img_size*256/224)} bicubic + CenterCrop {cfg.img_size}",
               "pretrained_cfg": pre_cfg})
    first_epoch = len(history)
    last_allowed = cfg.epochs if cfg.stop_after_epochs is None else min(cfg.epochs, first_epoch + cfg.stop_after_epochs)
    for epoch in range(first_epoch, last_allowed):
        # Predict next epoch cost; budget is cooperative, not protection against hard termination.
        if history and cfg.max_run_seconds is not None:
            estimate = max(h["epoch_seconds"] for h in history[-3:]) * 1.3 + 45
            if time.monotonic() - started + estimate >= cfg.max_run_seconds:
                break
        begin = time.monotonic()
        training = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        candidate = ema.model if ema else model
        names, truth, logits, val_loss = evaluate(candidate, val_loader, val_criterion, device)
        probabilities = torch.from_numpy(logits).softmax(1).numpy()
        metrics = compute_metrics(truth, probabilities.argmax(1), probabilities)
        history.append({"epoch": epoch + 1, **training, "val_loss": val_loss,
                        "val_macro_f1": metrics["macro_f1"], "val_top1": metrics["top1"],
                        "val_ece": metrics["ece"], "epoch_seconds": time.monotonic() - begin})
        if metrics["macro_f1"] > best_score:  # strict > keeps earlier epoch on ties
            best_score, best_epoch, best_state = metrics["macro_f1"], epoch + 1, cpu_state(candidate)
        state = {"signature": sig, "epoch": epoch + 1, "model": cpu_state(model),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "ema": cpu_state(ema.model) if ema else None, "history": history,
                 "best_score": best_score, "best_epoch": best_epoch, "best_state": best_state,
                 "rng": rng_state(train_loader, val_loader), "environment": environment}
        atomic_checkpoint(state, checkpoint_path)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"{cfg.exp_id} seed={cfg.seed} epoch={epoch+1}/{cfg.epochs}: "
              f"loss={training['train_loss']:.4f} val_F1={metrics['macro_f1']:.4f} "
              f"val_top1={metrics['top1']:.4f} seconds={history[-1]['epoch_seconds']:.1f}", flush=True)
    if not history:
        raise RuntimeError("No epoch completed")
    # Regenerate export artifacts from authoritative last.pt if interrupted after its atomic save.
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    atomic_checkpoint({"model": best_state, "epoch": best_epoch, "signature": sig,
                       "pretrained_cfg": pre_cfg, "mean": mean, "std": std}, directory / "best.pt")
    model.load_state_dict(best_state)
    names, truth, logits, val_loss = evaluate(model, val_loader, val_criterion, device)
    np.savez_compressed(directory / "val_logits.npz", filenames=np.asarray(names), y_true=truth, logits=logits)
    probabilities = torch.from_numpy(logits).softmax(1).numpy()
    save_predictions(pred_path(cfg, "val"), names, truth, probabilities)
    plot_curves(history, directory / "curves.png", f"{cfg.exp_id} / seed {cfg.seed} / {cfg.backbone}")
    summary = {"status": "completed" if len(history) == cfg.epochs else "paused",
               "exp_id": cfg.exp_id, "seed": cfg.seed, "debug_subset": debug,
               "epochs_completed": len(history), "best_epoch": best_epoch, "val_macro_f1": best_score,
               "val_top1": float((probabilities.argmax(1) == truth).mean()),
               "params_m": models.count_params(model), "epoch_seconds_mean": float(np.mean([h["epoch_seconds"] for h in history])),
               "gmacs": models.count_gmacs(model, cfg.img_size) if cfg.measure_gmacs else None,
               "gmacs_method": "thop estimate" if cfg.measure_gmacs else "not measured in stage 2",
               "train_images": len(train_df), "val_images": len(val_df), "test_evaluated": False,
               "environment": environment}
    write_json(summary_path, summary)
    return summary


def parse_overrides(pairs):
    hints = typing.get_type_hints(Config)
    result = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in hints:
            raise ValueError(f"Invalid override: {pair}")
        kind = hints[key]
        types = typing.get_args(kind) or (kind,)
        if value.lower() in ("none", "null") and type(None) in types:
            result[key] = None
        elif bool in types:
            if value.lower() not in ("true", "false"):
                raise ValueError(f"Use true/false: {pair}")
            result[key] = value.lower() == "true"
        else:
            cast = next(t for t in types if t is not type(None))
            result[key] = cast(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__":
    main()

