from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES


def to_hwc(tensor):
    return tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def perturbation_image(x, x_adv, eps):
    delta = (x_adv - x).detach().float().cpu()
    return (delta / (2 * max(eps, 1e-8)) + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_cam(image, cam, alpha=0.45):
    heat = plt.get_cmap("jet")(np.asarray(cam))[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def label_text(index, prob=None):
    name = CLASS_NAMES[index]
    return name if prob is None else f"{name} ({float(prob) * 100:.0f}%)"


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_robustness_curves(results, eps_list, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, attack, title in zip(axes, ("fgsm", "pgd"), ("FGSM", "PGD")):
        for name, res in results.items():
            ax.plot(eps_list, [res[attack][str(e)] * 100 for e in eps_list], marker="o", label=name)
        ax.set_title(f"Accuracy under {title} attack")
        ax.set_xlabel("ε (x/255)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].legend()
    return _save(fig, path)


def plot_success_curves(rates, eps_list, path, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, values in rates.items():
        ax.plot(eps_list, [values[str(e)] * 100 for e in eps_list], marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("ε (x/255)")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_transfer_matrix(names, matrix, path, title):
    data = np.asarray(matrix) * 100
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 1.3 * len(names) + 1.5))
    image = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Target model (evaluated)")
    ax.set_ylabel("Source model (attack crafted on)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Accuracy (%)")
    ax.set_title(title)
    return _save(fig, path)


def plot_image_grid(rows, col_titles, path, suptitle=None, cell=2.3):
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell * n_cols, cell * n_rows * 1.2), squeeze=False)
    for i, row in enumerate(rows):
        for j, (image, caption) in enumerate(row):
            ax = axes[i][j]
            ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if caption:
                ax.set_xlabel(caption, fontsize=7)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_history(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["val_clean"] * 100 for h in history], label=name)
        axes[1].plot(epochs, [h["val_robust"] * 100 for h in history], label=name)
    axes[0].set_title("Validation: clean accuracy")
    axes[1].set_title("Validation: accuracy under PGD")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path)
