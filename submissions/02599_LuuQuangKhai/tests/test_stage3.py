"""Checks for model profiling and suite control; no real benchmark results are fabricated."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import stage3
import benchmark
from eval import save_predictions, compute_metrics

torch.set_num_threads(2)


class Stage3Tests(unittest.TestCase):
    def test_five_backbones_same_recipe(self):
        configs = stage3.make_configs("images", "labels", "output")
        self.assertEqual(len(configs), 5)
        recipes = []
        for cfg in configs:
            values = asdict(cfg)
            values.pop("exp_id")
            values.pop("backbone")
            recipes.append(values)
        self.assertTrue(all(r == recipes[0] for r in recipes))
        self.assertEqual(configs[0].epochs, 10)
        self.assertFalse(configs[0].save_test_predictions)

    def test_mac_counter_known_convolution_and_linear(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(4, 9, bias=False)).train()
        report = stage3.mac_report(model, 8)
        self.assertEqual(report["mac_breakdown"]["conv"], 6 * 6 * 4 * 3 * 3 * 3)
        self.assertEqual(report["mac_breakdown"]["linear"], 4 * 9)
        self.assertTrue(model.training)
        self.assertFalse(any(m._forward_hooks for m in model.modules()))

    def test_real_backbones_and_attention_count(self):
        for exp, name in stage3.PLAN:
            with self.subTest(backbone=name):
                model = stage3.models.build_model(name, pretrained=False)
                report = stage3.mac_report(model, 224)
                self.assertGreater(report["gmacs"], 0.1)
                self.assertLess(report["gmacs"], 6)
                if "deit" in name:
                    expected = 12 * 2 * 197 * 197 * 384
                    self.assertEqual(report["mac_breakdown"]["attention_products"], expected)
                else:
                    self.assertEqual(report["mac_breakdown"]["attention_products"], 0)
                print(name, "GMAC:", round(report["gmacs"], 6), flush=True)
                del model

    def test_benchmark_sync_and_percentiles(self):
        calls, syncs = [], []
        times = np.arange(100, dtype=float) * 0.001
        with patch.object(benchmark.time, "perf_counter", side_effect=times):
            result = benchmark.bench(lambda: calls.append(1), 10, 50, lambda: syncs.append(1))
        self.assertEqual(len(calls), 60)
        self.assertEqual(len(syncs), 100)
        self.assertAlmostEqual(result["p50"], 1.0)
        self.assertEqual(result["n"], 50)
        with self.assertRaises(ValueError):
            benchmark.bench(lambda: None, warmup=0)

    def test_suite_exports_partial_results_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs"
            code = root / "code"
            code.mkdir()
            (code / "dummy.py").write_text("x=1")
            configs = stage3.make_configs("images", "labels", output)[:2]
            def fake_run(cfg):
                if cfg.exp_id == configs[1].exp_id:
                    raise RuntimeError("Simulated second-model failure")
                (output / "first_model.txt").write_text("completed")
                (output / "checkpoint.pt").write_bytes(b"test-only checkpoint")
                return {"status": "completed"}
            with patch.object(stage3.train, "signature", return_value={"test_fixture": True}), \
                 patch.object(stage3.train, "run", side_effect=fake_run), \
                 patch.object(stage3, "profile_run"), \
                 patch.object(stage3, "summarize", return_value={"status": "incomplete"}):
                with self.assertRaisesRegex(RuntimeError, "Simulated"):
                    stage3.run_suite(configs, output, root / "results.zip", code, 1800)
            with zipfile.ZipFile(root / "results.zip") as archive:
                self.assertIn("first_model.txt", archive.namelist())
                self.assertIn("last_error.txt", archive.namelist())
                self.assertNotIn("checkpoint.pt", archive.namelist())
            self.assertTrue((output / "checkpoint.pt").exists())

    def test_ranking_excludes_partial_runs_and_rechecks_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = stage3.make_configs("images", CODE.parent / "labels", root / "outputs")[:2]
            val = pd.read_csv(CODE.parent / "labels/val_subset0.csv")
            truth = val.Label.to_numpy()
            # Synthetic predictions in a temporary test fixture, never reported as lab results.
            probs = np.full((len(val), 9), 0.01)
            probs[np.arange(len(val)), truth] = 0.92
            metrics = compute_metrics(truth, probs.argmax(1), probs)
            for index, cfg in enumerate(configs):
                directory = stage3.train.run_dir(cfg)
                directory.mkdir(parents=True)
                epochs = 10 if index == 0 else 1
                summary = {"status": "completed" if index == 0 else "paused", "epochs_completed": epochs,
                           "best_epoch": 1, "val_macro_f1": metrics["macro_f1"], "val_top1": metrics["top1"],
                           "params_m": 1, "epoch_seconds_mean": 1}
                stage3.train.write_json(directory / "summary.json", summary)
                pd.DataFrame({"epoch": range(1, epochs + 1), "epoch_seconds": [1] * epochs}).to_csv(directory / "history.csv", index=False)
                save_predictions(stage3.train.pred_path(cfg, "val"), val.Filename, truth, probs)
                stage3.train.write_json(directory / "profile.json", {"gmacs": 1, "latency": {
                    "p50": 1, "p95": 2, "p99": 3, "dtype": "fp32", "gpu": "TEST FIXTURE"}})
            result = stage3.summarize(configs, root / "outputs")
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["completed_and_profiled"], 1)
            ranked = pd.read_csv(root / "outputs/backbones_ranked.csv")
            self.assertEqual(ranked.exp_id.tolist(), [configs[0].exp_id])
            path = stage3.train.pred_path(configs[0], "val")
            tampered = pd.read_csv(path)
            tampered.loc[0, "Filename"] = "wrong.jpg"
            tampered.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                stage3.summarize(configs, root / "outputs")

    def test_short_budget_never_starts_an_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = root / "code"
            code.mkdir()
            configs = stage3.make_configs("images", "labels", root / "outputs")
            with patch.object(stage3.train, "signature", return_value={}), \
                 patch.object(stage3.train, "run") as run, \
                 patch.object(stage3, "summarize", return_value={"status": "incomplete"}), \
                 patch.object(stage3.time, "monotonic", side_effect=[0, 100]):
                result = stage3.run_suite(configs, root / "outputs", root / "results.zip", code, 900)
            run.assert_not_called()
            self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
