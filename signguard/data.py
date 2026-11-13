import argparse
import csv
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .config import DATA_DIR, DEFAULT_CACHE, IMG_SIZE


def _track_group(path, label, track_index):
    parts = Path(path).stem.split("_")
    track = parts[track_index] if len(parts) > track_index + 1 else Path(path).stem
    return f"{label}_{track}"


def torchvision_records(root):
    from torchvision.datasets import GTSRB

    records = {}
    for split in ("train", "test"):
        dataset = GTSRB(root=str(root), split=split, download=True)
        samples = getattr(dataset, "_samples", None)
        if samples is None:
            raise RuntimeError("This torchvision version does not expose the GTSRB sample list, use --source kaggle.")
        records[split] = [
            (str(path), int(label), _track_group(path, int(label), 0) if split == "train" else None)
            for path, label in samples
        ]
    return records


def kaggle_records(kaggle_dir):
    kaggle_dir = Path(kaggle_dir)
    train_dir = kaggle_dir / "Train"
    test_csv = kaggle_dir / "Test.csv"
    if not train_dir.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Could not find Train/ and Test.csv in {kaggle_dir}.")
    train = []
    for class_dir in sorted(train_dir.iterdir(), key=lambda p: p.name):
        if not class_dir.is_dir():
            continue
        label = int(class_dir.name)
        for path in sorted(class_dir.glob("*.png")):
            train.append((str(path), label, _track_group(path, label, 1)))
    test = []
    with open(test_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            test.append((str(kaggle_dir / row["Path"]), int(row["ClassId"]), None))
    return {"train": train, "test": test}


def load_images(records, size, workers=8):
    def load(record):
        with Image.open(record[0]) as image:
            return np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrays = list(tqdm(pool.map(load, records), total=len(records), desc="images"))
    x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor([record[1] for record in records], dtype=torch.long)
    return x, y


def group_split(records, val_fraction, seed):
    rng = random.Random(seed)
    groups_by_class = defaultdict(set)
    for _, label, group in records:
        groups_by_class[label].add(group)
    val_groups = set()
    for label in sorted(groups_by_class):
        groups = sorted(groups_by_class[label])
        rng.shuffle(groups)
        val_groups.update(groups[: max(1, round(len(groups) * val_fraction))])
    train_idx = [i for i, record in enumerate(records) if record[2] not in val_groups]
    val_idx = [i for i, record in enumerate(records) if record[2] in val_groups]
    return train_idx, val_idx


def channel_stats(x):
    means, stds = [], []
    for channel in range(x.shape[1]):
        values = x[:, channel].float().div(255)
        means.append(values.mean().item())
        stds.append(values.std().item())
    return torch.tensor(means), torch.tensor(stds)


def prepare_cache(source="torchvision", kaggle_dir=None, size=IMG_SIZE, val_fraction=0.1, seed=0, out=None):
    out = Path(out) if out else (DEFAULT_CACHE if size == IMG_SIZE else DATA_DIR / f"gtsrb_{size}.pt")
    records = kaggle_records(kaggle_dir) if source == "kaggle" else torchvision_records(DATA_DIR)
    train_idx, val_idx = group_split(records["train"], val_fraction, seed)
    x_all, y_all = load_images(records["train"], size)
    x_test, y_test = load_images(records["test"], size)
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    mean, std = channel_stats(x_train)
    cache = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "mean": mean,
        "std": std,
        "size": size,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print(f"Mean: {mean.tolist()}  Std: {std.tolist()}")
    print(f"Cache saved: {out}")
    return out


def load_cache(path=DEFAULT_CACHE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m signguard.data' first.")
    return torch.load(path, map_location="cpu", weights_only=True)


class GPUAugment(nn.Module):
    def __init__(self, rotate=15.0, scale=0.15, translate=0.1, shear=0.1, brightness=0.3, contrast=0.3):
        super().__init__()
        self.rotate = math.radians(rotate)
        self.scale = scale
        self.translate = translate
        self.shear = shear
        self.brightness = brightness
        self.contrast = contrast

    def _uniform(self, n, device, magnitude):
        return (torch.rand(n, device=device) * 2 - 1) * magnitude

    def forward(self, x):
        n, device = x.size(0), x.device
        angle = self._uniform(n, device, self.rotate)
        scale = 1 + self._uniform(n, device, self.scale)
        shear = self._uniform(n, device, self.shear)
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(n, 2, 3, device=device, dtype=x.dtype)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = (cos * shear - sin) / scale
        theta[:, 1, 0] = sin / scale
        theta[:, 1, 1] = (sin * shear + cos) / scale
        theta[:, 0, 2] = self._uniform(n, device, self.translate * 2)
        theta[:, 1, 2] = self._uniform(n, device, self.translate * 2)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        brightness = (1 + self._uniform(n, device, self.brightness)).view(n, 1, 1, 1)
        contrast = (1 + self._uniform(n, device, self.contrast)).view(n, 1, 1, 1)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * contrast + mean
        return (x * brightness).clamp(0.0, 1.0)


class TensorLoader:
    def __init__(self, x, y, batch_size, shuffle, device, augment=None, drop_last=False, channels_last=False):
        self.x = x.to(device)
        self.y = y.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.drop_last = drop_last
        self.channels_last = channels_last

    def __len__(self):
        n = len(self.y)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(len(self)):
            index = order[i * self.batch_size : (i + 1) * self.batch_size]
            xb = self.x[index].float().div_(255)
            if self.augment is not None:
                xb = self.augment(xb)
            if self.channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yield xb, self.y[index]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Downloads GTSRB and builds a tensor cache for fast training.")
    parser.add_argument("--source", choices=["torchvision", "kaggle"], default="torchvision")
    parser.add_argument("--kaggle-dir", default=str(DATA_DIR / "kaggle"))
    parser.add_argument("--size", type=int, default=IMG_SIZE)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    prepare_cache(args.source, args.kaggle_dir, args.size, args.val_fraction, args.seed, args.out)


if __name__ == "__main__":
    main()
