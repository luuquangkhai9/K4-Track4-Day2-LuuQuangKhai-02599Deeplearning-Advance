"""Stage 1: validate the official fold and export EDA; never train a model."""
from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

MD5 = "b7b30f96d466fba86016aa5a26606e0f"
BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
LABEL_FILES = ["labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_labels(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in LABEL_FILES:
        target = folder / name
        if not target.exists():
            temporary = folder / (name + ".download")
            urlretrieve(f"{BASE}/{name}", temporary)
            temporary.replace(target)
    return folder


def run(images, labels, output):
    images, labels, output = Path(images), Path(labels), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(labels / "labels.csv")
    frames = {s: pd.read_csv(labels / f"{s}_subset0.csv") for s in ("train", "val", "test")}
    for name, frame in {"all": master, **frames}.items():
        required = ["Filename", "Label", "Species"] if name == "all" else ["Filename", "Label"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Missing CSV columns: {name}")
        if frame[required].isna().any().any():
            raise ValueError(f"Null fields: {name}")
        if frame.Filename.duplicated().any() or not frame.Label.isin(range(9)).all():
            raise ValueError(f"Duplicate filenames or invalid labels: {name}")
    if len(master) != 17509 or set(master.Label) != set(range(9)):
        raise ValueError("Expected 17509 master records and classes 0..8")
    names = {s: set(df.Filename) for s, df in frames.items()}
    overlap = {f"{a}_{b}": len(names[a] & names[b]) for a, b in
               [("train", "val"), ("train", "test"), ("val", "test")]}
    if any(overlap.values()) or set.union(*names.values()) != set(master.Filename):
        raise ValueError(f"Split overlap or coverage failure: {overlap}")
    reference = master.set_index("Filename")
    class_names = master.groupby("Label").Species.first().reindex(range(9))
    mismatches = []
    for split, df in frames.items():
        master_labels = reference.loc[df.Filename, "Label"].to_numpy()
        different = df.Label.to_numpy() != master_labels
        for filename, split_label, master_label in zip(df.Filename[different], df.Label[different], master_labels[different]):
            mismatches.append({"split": split, "Filename": filename,
                               "split_label": int(split_label), "master_label": int(master_label)})
        # Official split CSVs have only Filename,Label. Enrich in memory, never rewrite CSVs.
        df["Species"] = df.Label.map(class_names)
        expected = 0.6 if split == "train" else 0.2
        if abs(len(df) / len(master) - expected) > 0.01:
            raise ValueError(f"Unexpected split ratio: {split}")
    pd.DataFrame(mismatches, columns=["split", "Filename", "split_label", "master_label"]).to_csv(
        output / "label_discrepancies.csv", index=False)

    archive = None
    try:
        if images.is_file():
            actual_md5 = digest(images, "md5")
            if actual_md5 != MD5:
                raise ValueError(f"Image ZIP checksum mismatch: {actual_md5}")
            archive = zipfile.ZipFile(images)
            available = set(archive.namelist())
            missing = set(master.Filename) - available
            def read_image(name):
                return Image.open(io.BytesIO(archive.read(name)))
        else:
            actual_md5 = None
            missing = {name for name in master.Filename if not (images / name).is_file()}
            def read_image(name):
                return Image.open(images / name)
        if missing:
            raise ValueError(f"Missing {len(missing)} images: {sorted(missing)[:5]}")

        # Decode all images to detect corruption. This checks data integrity only.
        geometry = Counter()
        for index, name in enumerate(master.Filename, 1):
            with read_image(name) as im:
                im.load()
                geometry[f"{im.width}x{im.height}/{im.mode}"] += 1
            if index % 5000 == 0:
                print(f"Decoded {index}/{len(master)} images", flush=True)

        class_names = master.groupby("Label").Species.first().reindex(range(9))
        counts = pd.DataFrame({s: df.Label.value_counts().reindex(range(9), fill_value=0)
                               for s, df in frames.items()}).astype(int)
        counts["total"] = counts.sum(axis=1)
        counts["master_total"] = master.Label.value_counts().reindex(range(9), fill_value=0)
        counts.insert(0, "species", class_names)
        counts.index.name = "label"
        counts.to_csv(output / "class_counts.csv")
        ax = counts.set_index("species")[["train", "val", "test"]].plot.bar(figsize=(13, 6))
        ax.set(ylabel="Images", title="DeepWeeds — official fold 0")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output / "class_distribution.png", dpi=150)
        plt.close()

        # Qualitative exploration uses train only; no test-guided decisions.
        fig, axes = plt.subplots(9, 3, figsize=(10, 27))
        samples = []
        for label in range(9):
            selected = frames["train"].query("Label == @label").sample(3, random_state=0)
            for col, row in enumerate(selected.itertuples()):
                with read_image(row.Filename) as im:
                    axes[label, col].imshow(im.convert("RGB"))
                axes[label, col].set_title(f"{row.Species}\n{row.Filename}", fontsize=8)
                axes[label, col].axis("off")
                samples.append({"Filename": row.Filename, "Label": label, "split": "train"})
        fig.tight_layout()
        fig.savefig(output / "train_samples.png", dpi=110)
        plt.close(fig)
        pd.DataFrame(samples).to_csv(output / "sample_manifest.csv", index=False)
    finally:
        if archive is not None:
            archive.close()

    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    try:
        import torch
        environment.update(torch=torch.__version__, cuda=torch.version.cuda,
                           gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except ImportError:
        environment.update(torch=None, gpu=[])
    summary = {"status": "PASS", "fold": 0, "seed_for_samples": 0,
               "label_discrepancies": mismatches,
               "label_policy": "Keep official split labels unchanged; report discrepancies with labels.csv.",
               "n": {s: len(df) for s, df in frames.items()}, "overlap": overlap,
               "union": len(master), "image_geometry": dict(geometry), "zip_md5": actual_md5,
               "imbalance_ratio": float(counts.total.max() / counts.total.min()),
               "labels_sha256": {name: digest(labels / name) for name in LABEL_FILES},
               "images_source": str(images), "environment": environment}
    (output / "stage1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(counts.to_string())
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="images.zip or directory of JPGs")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.images, prepare_labels(args.labels), args.output)
