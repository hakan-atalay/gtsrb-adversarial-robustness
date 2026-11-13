import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import CKPT_DIR, DEFAULT_CACHE, FIG_DIR, IMG_SIZE, RESULTS_DIR
from .models import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


def save_checkpoint(path, model, arch, mean, std, num_classes, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mean": list(mean),
        "std": list(std),
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
    }
    payload.update(extra or {})
    torch.save(payload, path)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["arch"], checkpoint["num_classes"], checkpoint["mean"], checkpoint["std"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def find_checkpoints(models=None):
    if models:
        paths = [Path(m) if m.endswith(".pt") else CKPT_DIR / f"{m}.pt" for m in models]
    else:
        paths = sorted(CKPT_DIR.glob("*.pt"))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint not found: {', '.join(missing)}")
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {CKPT_DIR}. Train the models first.")
    return paths


def subset(x, y, n, seed=0):
    if n is None or n >= len(y):
        return x, y
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(len(y), generator=generator)[:n]
    return x[index], y[index]


def parse_eps(text):
    return [float(value) for value in text.split(",") if value.strip()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_common_args(parser, models=True):
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    if models:
        parser.add_argument("--models", nargs="*", default=None)
