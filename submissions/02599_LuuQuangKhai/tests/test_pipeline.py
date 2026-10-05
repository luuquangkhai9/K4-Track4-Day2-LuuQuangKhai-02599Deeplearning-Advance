"""CPU integration tests for implemented code (separate from starter stub tests)."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import dataset
import model as models
import losses
import train
from eval import read_pred, check_against_csv

torch.set_num_threads(1)


class TinyModel(nn.Module):
    """Fast test double: exercises BN, dropout, classifier and full training loop."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2))
        self.head = nn.Linear(8, 9)
        self.pretrained_cfg = {"mean": dataset.IMAGENET_MEAN, "std": dataset.IMAGENET_STD}

    def forward(self, x):
        return self.head(self.body(x))

    def get_classifier(self):
        return self.head


def tiny_model(*args, **kwargs):
    return TinyModel()


class PipelineTests(unittest.TestCase):
    def test_focal_and_smoothing_reduce_to_ce(self):
        torch.manual_seed(1)
        x = torch.randn(19, 9, requires_grad=True)
        y = torch.arange(19) % 9
        ce = nn.CrossEntropyLoss()(x, y)
        for loss in (losses.FocalLoss(gamma=0)(x, y), losses.LabelSmoothingCE(0)(x, y)):
            torch.testing.assert_close(loss, ce, atol=1e-6, rtol=0)
            torch.testing.assert_close(torch.autograd.grad(loss, x, retain_graph=True)[0],
                                       torch.autograd.grad(ce, x, retain_graph=True)[0])

    def test_cutmix_uses_actual_area(self):
        x = torch.stack((torch.zeros(3, 8, 8), torch.ones(3, 8, 8)))
        y = torch.tensor([0, 1])
        with patch("losses.np.random.beta", return_value=0.25), \
             patch("losses.np.random.randint", return_value=0), \
             patch("losses.torch.randperm", return_value=torch.tensor([1, 0])):
            mixed, (_, other, lam) = losses.mix_batch(x, y)
        self.assertGreater(lam, 0.25)  # clipped at upper-left image boundary
        self.assertAlmostEqual(float(mixed[0].mean()), 1 - lam)
        torch.testing.assert_close(other, y.flip(0))
        torch.testing.assert_close(x[0], torch.zeros_like(x[0]))

    def test_parameter_groups_and_frozen_bn(self):
        model = TinyModel()
        groups = models.param_groups(model, 1e-4, 1e-3, 0.05)
        assigned = {id(p): g for g in groups for p in g["params"]}
        self.assertEqual(sum(len(g["params"]) for g in groups), len(assigned))
        self.assertEqual(set(assigned), {id(p) for p in model.parameters()})
        for name, p in model.named_parameters():
            self.assertEqual(assigned[id(p)]["lr"], 1e-3 if name.startswith("head") else 1e-4)
            if p.ndim <= 1:
                self.assertEqual(assigned[id(p)]["weight_decay"], 0)
        models.freeze_backbone(model)
        models.set_train_mode(model, "frozen")
        before = model.body[1].running_mean.clone()
        model(torch.randn(4, 3, 16, 16)).sum().backward()
        torch.testing.assert_close(model.body[1].running_mean, before)
        self.assertIsNone(model.body[0].weight.grad)
        self.assertIsNotNone(model.head.weight.grad)

    def test_eval_repeatable_and_preserves_bn(self):
        model = TinyModel().eval()
        x, y = torch.randn(9, 3, 12, 12), torch.arange(9)
        loader = [(x, y, [f"{i}.jpg" for i in range(9)])]
        before = model.body[1].running_mean.clone()
        a = train.evaluate(model, loader, nn.CrossEntropyLoss(), "cpu")
        b = train.evaluate(model, loader, nn.CrossEntropyLoss(), "cpu")
        np.testing.assert_array_equal(a[2], b[2])
        self.assertEqual(a[0], loader[0][2])
        torch.testing.assert_close(model.body[1].running_mean, before)

    def test_ema_parameters_and_bn_buffers(self):
        model = TinyModel()
        ema = train.EMA(model, 0.9)
        before = ema.model.head.weight.detach().clone()
        with torch.no_grad():
            model.head.weight.add_(1)
            model.body[1].running_mean.add_(2)
        ema.update(model)
        torch.testing.assert_close(ema.model.head.weight, before + 0.1)
        torch.testing.assert_close(ema.model.body[1].running_mean, model.body[1].running_mean)

    def test_original_split_and_stage1_receipt(self):
        root = CODE.parent
        frames = dataset.load_split(root / "labels")
        report = dataset.check_split(*frames, root, verify_files=False)
        self.assertEqual(report["n"], {"train": 10501, "val": 3501, "test": 3507})
        receipt = root / "eda/kaggle/stage1_summary.json"
        summary = json.loads(receipt.read_text())
        self.assertTrue(dataset.verified_stage1(receipt, summary["images_source"], root / "labels", 0))
        self.assertFalse(dataset.verified_stage1(receipt, "different-images", root / "labels", 0))
        duplicate = frames[1].copy()
        duplicate.loc[0, "Filename"] = frames[0].iloc[0].Filename
        with self.assertRaises(ValueError):
            dataset.check_split(frames[0], duplicate, frames[2], root, verify_files=False)

    def test_resume_matches_uninterrupted_and_never_reads_test_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            rng = np.random.default_rng(7)
            all_rows = []
            for split, per_class in (("train", 2), ("val", 1), ("test", 1)):
                rows = []
                for label in range(9):
                    for i in range(per_class):
                        name = f"{split}_{label}_{i}.jpg"
                        rows.append({"Filename": name, "Label": label, "Species": str(label)})
                        if split != "test":  # Deliberately absent; train/val must never open them.
                            Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(images / name)
                pd.DataFrame(rows).to_csv(labels / f"{split}_subset0.csv", index=False)
                all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(labels / "labels.csv", index=False)
            cfg = train.Config(exp_id="S_resume", epochs=2, batch_size=9, img_size=16,
                               images_dir=str(images), labels_dir=str(labels), init="scratch",
                               num_workers=0, amp=False, device="cpu", mix="cutmix", ema_decay=0.9,
                               out_dir=str(root / "continuous"), pred_dir=str(root / "pred_a"))
            # Synthetic fixture is small; actual split invariants have a separate test above.
            with patch.object(dataset, "check_split", return_value={}), \
                 patch.object(models, "build_model", side_effect=tiny_model):
                continuous = train.run(cfg)
                partial_cfg = replace(cfg, out_dir=str(root / "resumed"), pred_dir=str(root / "pred_b"), stop_after_epochs=1)
                partial = train.run(partial_cfg)
                self.assertEqual(partial["status"], "paused")
                resumed = train.run(replace(partial_cfg, stop_after_epochs=None))
                self.assertEqual(resumed["status"], "completed")
                a = torch.load(train.run_dir(cfg) / "last.pt", weights_only=False)
                b = torch.load(train.run_dir(partial_cfg) / "last.pt", weights_only=False)
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
                for key in ("best_epoch", "best_score"):
                    self.assertEqual(a[key], b[key])
                np.testing.assert_array_equal(np.load(train.run_dir(cfg) / "val_logits.npz")["logits"],
                                              np.load(train.run_dir(partial_cfg) / "val_logits.npz")["logits"])
                check_against_csv(read_pred(str(train.pred_path(partial_cfg, "val"))),
                                  str(labels / "val_subset0.csv"), "val")
                with patch.object(models, "build_model", side_effect=AssertionError("Should skip complete run")):
                    self.assertEqual(train.run(partial_cfg)["status"], "completed")
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    train.run(replace(partial_cfg, lr_head=0.02))
                with self.assertRaisesRegex(ValueError, "Test is locked"):
                    train.run(replace(cfg, save_test_predictions=True))

    def test_real_timm_mobile_forward(self):
        model = models.build_model("mobilenetv3_large_100", pretrained=False, init="scratch").eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 9))
        self.assertTrue(torch.isfinite(output).all())

    def test_override_types(self):
        self.assertEqual(train.parse_overrides(["seed=3", "amp=false", "ema_decay=none"]),
                         {"seed": 3, "amp": False, "ema_decay": None})
        with self.assertRaises(ValueError):
            train.parse_overrides(["amp=yes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
