import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import CLASS_NAMES, DATA_DIR, DEFAULT_CACHE, FIG_DIR, NUM_CLASSES, STOP_CLASS
from .data import GPUAugment, load_cache


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def class_distribution(cache, path):
    index = np.arange(NUM_CLASSES)
    bottom = np.zeros(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for split in ("train", "val", "test"):
        counts = torch.bincount(cache[f"y_{split}"], minlength=NUM_CLASSES).numpy()
        ax.bar(index, counts, bottom=bottom, label=split)
        bottom += counts
    ax.set_xticks(index, [f"{i}: {CLASS_NAMES[i]}" for i in index], rotation=90, fontsize=7)
    ax.set_ylabel("Number of images")
    ax.set_title("GTSRB class distribution")
    ax.legend()
    save(fig, path)


def class_gallery(cache, path):
    rows, cols = 6, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.1))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= NUM_CLASSES:
            continue
        matches = (cache["y_train"] == i).nonzero()
        if len(matches) == 0:
            continue
        ax.imshow(cache["x_train"][int(matches[0])].permute(1, 2, 0).numpy())
        ax.set_title(f"{i}: {CLASS_NAMES[i]}", fontsize=6)
    fig.suptitle("One training image per class (48x48 after resizing)")
    fig.tight_layout()
    save(fig, path)


def track_strip(raw_dir, label, path):
    files = sorted((Path(raw_dir) / f"{label:05d}").glob("*.ppm"))
    if not files:
        print(f"Raw images not found in {raw_dir}, skipping the track figure.")
        return
    track = files[0].stem.split("_")[0]
    frames = [f for f in files if f.stem.split("_")[0] == track][:30]
    cols = 10
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, frame in zip(axes.flat, frames):
        with Image.open(frame) as image:
            ax.imshow(np.asarray(image.convert("RGB")))
            ax.set_title(f"{frame.name}\n{image.width}x{image.height}", fontsize=6)
    fig.suptitle(f"One physical '{CLASS_NAMES[label]}' sign seen as the car approaches (track {track}, original sizes)")
    fig.tight_layout()
    save(fig, path)


def augmentation_examples(cache, path, n_variants=7):
    torch.manual_seed(0)
    labels = [label for label in (STOP_CLASS, 3, 33, 25) if (cache["y_train"] == label).any()]
    if not labels:
        return
    augment = GPUAugment()
    fig, axes = plt.subplots(len(labels), n_variants + 1, figsize=((n_variants + 1) * 1.6, len(labels) * 1.8), squeeze=False)
    for row, label in enumerate(labels):
        sample = int((cache["y_train"] == label).nonzero()[0])
        x = cache["x_train"][sample].float().div(255).unsqueeze(0)
        variants = augment(x.repeat(n_variants, 1, 1, 1))
        images = [x[0]] + list(variants)
        for col, image in enumerate(images):
            ax = axes[row][col]
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title("original" if col == 0 else f"augmented {col}", fontsize=7)
    fig.suptitle("On-the-fly data augmentation (no horizontal flips)")
    fig.tight_layout()
    save(fig, path)


def print_summary(cache):
    print(f"Image tensor shape (train): {tuple(cache['x_train'].shape)}  dtype: {cache['x_train'].dtype}")
    for split in ("train", "val", "test"):
        counts = torch.bincount(cache[f"y_{split}"], minlength=NUM_CLASSES)
        most, least = int(counts.argmax()), int(counts.argmin())
        print(
            f"{split:<5} {len(cache[f'y_{split}']):>6} images | most: {CLASS_NAMES[most]} ({int(counts[most])}) "
            f"| least: {CLASS_NAMES[least]} ({int(counts[least])})"
        )
    print(f"Channel mean: {[round(v, 4) for v in cache['mean'].tolist()]}  std: {[round(v, 4) for v in cache['std'].tolist()]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explore the GTSRB dataset and save overview figures.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--raw-dir", default=str(DATA_DIR / "gtsrb" / "GTSRB" / "Training"))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    args = parser.parse_args(argv)

    cache = load_cache(args.cache)
    fig_dir = Path(args.fig_dir)
    print_summary(cache)
    class_distribution(cache, fig_dir / "data_class_distribution.png")
    class_gallery(cache, fig_dir / "data_class_gallery.png")
    track_strip(args.raw_dir, STOP_CLASS, fig_dir / "data_track_example.png")
    augmentation_examples(cache, fig_dir / "data_augmentation.png")


if __name__ == "__main__":
    main()
