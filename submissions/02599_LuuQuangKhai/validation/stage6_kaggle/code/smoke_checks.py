"""GPU diagnostics on one fixed TRAIN batch; never a scored experiment."""
from pathlib import Path
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import dataset
import model as models
import train
from losses import FocalLoss, LabelSmoothingCE, mix_batch


def run_checks(images_dir, labels_dir, output, backbone="mobilenetv3_large_100.ra_in1k",
               device="cuda:0", max_steps=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train.set_seed(0)
    device = torch.device(device)
    model = models.build_model(backbone).to(device)
    mean = tuple(model.pretrained_cfg["mean"])
    std = tuple(model.pretrained_cfg["std"])
    df = dataset.load_split(labels_dir)[0].groupby("Label", group_keys=False).head(1).sort_values("Label")
    fixed = dataset.make_loader(df, images_dir, dataset.build_transforms(False, 224, mean=mean, std=std),
                                batch_size=9, train=False, num_workers=0)
    x, y, names = next(iter(fixed))
    pd.DataFrame({"Filename": names, "Label": y.numpy(), "split": "train"}).to_csv(
        output / "overfit_batch.csv", index=False)

    # Visual check of train augmentation and CutMix, with inverse normalization.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    augmented = dataset.make_loader(df, images_dir, dataset.build_transforms(True, 224, mean=mean, std=std),
                                    batch_size=9, train=False, num_workers=0)
    aug_x, aug_y, _ = next(iter(augmented))
    mixed, (ya, yb, lam) = mix_batch(aug_x, aug_y)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row, (batch, title) in enumerate(((x, "fixed"), (aug_x, "train aug"), (mixed, "CutMix"))):
        for col in range(3):
            view = batch[col] * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
            axes[row, col].imshow(view.permute(1, 2, 0).clamp(0, 1))
            label = f"{int(ya[col])}/{int(yb[col])}, lambda={lam:.3f}" if row == 2 else str(int(y[col]))
            axes[row, col].set_title(f"{title}: label {label}")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output / "augmentation_check.png", dpi=140)
    plt.close(fig)

    x, y = x.to(device), y.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        logits = model(x)
        initial_loss = float(criterion(logits, y))
        initial_logits_std = float(logits.std())
        uniform_ce = float(criterion(torch.zeros_like(logits), y))
        assert abs(uniform_ce - math.log(9)) < 1e-6
        torch.testing.assert_close(FocalLoss(0)(logits, y), criterion(logits, y), atol=1e-6, rtol=0)
        torch.testing.assert_close(LabelSmoothingCE(0)(logits, y), criterion(logits, y), atol=1e-6, rtol=0)
    print(f"Initial CE={initial_loss:.4f}; uniform reference ln(9)={math.log(9):.4f}", flush=True)
    if not math.isfinite(initial_loss):
        raise FloatingPointError("Nonfinite initial loss")
    # Diagnostic deliberately uses higher LR and no augmentation/decay.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history = []
    begin = time.monotonic()
    final_loss, final_accuracy = initial_loss, 0.0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite overfit loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        row = {"step": step, "train_loss": float(loss.detach())}
        if step % 10 == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                final_loss = float(criterion(logits, y))
                final_accuracy = float((logits.argmax(1) == y).float().mean())
            row.update(fixed_batch_eval_ce=final_loss, fixed_batch_accuracy=final_accuracy)
            print(f"Overfit step={step}: eval CE={final_loss:.4f}, acc={final_accuracy:.3f}", flush=True)
        history.append(row)
        if final_loss < 0.1 and final_accuracy == 1.0:
            break
    pd.DataFrame(history).to_csv(output / "overfit_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in history], [r["train_loss"] for r in history], label="fixed train batch")
    ax.set(xlabel="Optimizer step", ylabel="CE", title="Diagnostic only: overfit 9 train images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "overfit_curve.png", dpi=140)
    plt.close(fig)
    passed = final_loss < 0.1 and final_accuracy == 1.0
    result = {"status": "PASS" if passed else "FAIL", "initial_ce": initial_loss,
              "uniform_reference": math.log(9), "final_ce": final_loss, "accuracy": final_accuracy,
              "steps": step, "seconds": time.monotonic() - begin, "backbone": backbone,
              "device": str(device), "test_evaluated": False, "diagnostic_only": True}
    result.update(head_init_std=0.01, initial_logits_std=initial_logits_std, uniform_ce=uniform_ce)
    train.write_json(output / "smoke_summary.json", result)
    if not passed:
        raise RuntimeError("Overfit check failed; send smoke_summary.json and overfit_history.csv. Do not start full training.")
    return result
