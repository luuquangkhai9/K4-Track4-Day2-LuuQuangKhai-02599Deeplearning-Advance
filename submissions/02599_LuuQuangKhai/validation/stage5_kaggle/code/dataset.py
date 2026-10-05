"""Official splits and reproducible image loading; preserve original split labels."""
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
               "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_split(labels_dir, fold=0):
    return tuple(pd.read_csv(Path(labels_dir) / f"{s}_subset{fold}.csv")
                 for s in ("train", "val", "test"))

def check_split(train_df, val_df, test_df, images_dir, verify_files=True):
    frames = dict(zip(("train", "val", "test"), (train_df, val_df, test_df)))
    names = {}
    for split, df in frames.items():
        if not {"Filename", "Label"}.issubset(df.columns):
            raise ValueError(f"Missing split columns: {split}")
        if df[["Filename", "Label"]].isna().any().any() or df.Filename.duplicated().any():
            raise ValueError(f"Null/duplicate records: {split}")
        if set(df.Label) != set(range(NUM_CLASSES)):
            raise ValueError(f"Expected labels 0..8: {split}")
        if any(Path(n).name != n or "\\" in n for n in df.Filename):
            raise ValueError("Expected bare image filenames")
        names[split] = set(df.Filename)
    overlap = {f"{a}_{b}": len(names[a] & names[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    union = set.union(*names.values())
    if any(overlap.values()) or len(union) != 17509:
        raise ValueError(f"Invalid split coverage/overlap: {len(union)}, {overlap}")
    for split, df in frames.items():
        if abs(len(df) / 17509 - (0.6 if split == "train" else 0.2)) > 0.01:
            raise ValueError(f"Invalid split ratio: {split}")
    if verify_files:
        missing = [n for n in union if not (Path(images_dir) / n).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images: {missing[:5]}")
    result = {"n": {s: len(df) for s, df in frames.items()}, "overlap": overlap,
              "union": len(union), "verified_all_file_paths": verify_files,
              "per_class": {s: df.Label.value_counts().sort_index().to_dict() for s, df in frames.items()}}
    print("Split:", result["n"], "overlap:", overlap)
    return result

def csv_hashes(labels_dir, fold=0):
    files = ["labels.csv"] + [f"{s}_subset{fold}.csv" for s in ("train", "val", "test")]
    return {n: hashlib.sha256((Path(labels_dir) / n).read_bytes()).hexdigest() for n in files}

def verified_stage1(path, images_dir, labels_dir, fold):
    """Reuse exhaustive decode only for unchanged source path and CSV bytes."""
    if not path or fold != 0:
        return False
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    return (summary.get("status") == "PASS" and summary.get("union") == 17509
            and summary.get("labels_sha256") == csv_hashes(labels_dir, fold)
            and Path(summary.get("images_source", "")).as_posix() == Path(images_dir).as_posix())

def build_transforms(train, img_size=224, aug="basic", mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if aug not in ("basic", "color", "trivial", "randaug"):
        raise ValueError(f"Unknown augmentation: {aug}")
    if train:
        steps = [T.RandomResizedCrop(img_size, interpolation=T.InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip()]
        if aug == "color":
            steps += [T.ColorJitter(0.2, 0.2, 0.2, 0.05)]
        elif aug == "trivial":
            steps += [T.TrivialAugmentWide()]
        elif aug == "randaug":
            steps += [T.RandAugment()]
    else:
        steps = [T.Resize(round(img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
                 T.CenterCrop(img_size)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])

class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir, self.transform = Path(images_dir), transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row.Label), row.Filename

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2, seed=0):
    if sampler not in (None, "balanced") or (sampler and not train):
        raise ValueError("Sampler only supports None or balanced on train")
    generator = torch.Generator().manual_seed(seed)
    weighted = None
    if sampler == "balanced":
        counts = df.Label.value_counts()
        weights = torch.tensor([1 / counts[y] for y in df.Label], dtype=torch.double)
        weighted = WeightedRandomSampler(weights, len(df), replacement=True, generator=generator)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=train and weighted is None, sampler=weighted,
                      drop_last=train and len(df) % batch_size == 1 and len(df) > batch_size,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator, persistent_workers=False)

