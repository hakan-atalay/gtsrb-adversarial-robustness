import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .attacks import pgd
from .config import STOP_CLASS
from .data import load_cache
from .utils import add_common_args, find_checkpoints, get_device, load_model
from .visualize import label_text, overlay_cam, plot_image_grid, to_hwc


def grad_cam(model, x, class_idx=None):
    store = {}

    def forward_hook(module, inputs, output):
        store["activations"] = output
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = model.cam_layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            logits = model(x)
            index = logits.argmax(1) if class_idx is None else class_idx
            score = logits.gather(1, index.view(-1, 1)).sum()
            model.zero_grad(set_to_none=True)
            score.backward()
    finally:
        handle.remove()
    activations = store["activations"].detach()
    weights = store["gradients"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam - cam.amin(dim=(2, 3), keepdim=True)
    cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
    model.zero_grad(set_to_none=True)
    return cam[:, 0].detach(), logits.detach()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grad-CAM comparison before and after the attack.")
    add_common_args(parser)
    parser.add_argument("--classes", default=f"{STOP_CLASS},3,17,13,25,38")
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)

    device = get_device(args.device)
    cache = load_cache(args.cache)
    classes = [int(c) for c in args.classes.split(",") if c.strip()]
    indices = [int((cache["y_test"] == c).nonzero()[0]) for c in classes if (cache["y_test"] == c).any()]
    x = cache["x_test"][indices].float().div(255).to(device)
    y = cache["y_test"][indices].to(device)

    columns = {}
    for path in find_checkpoints(args.models):
        model, _ = load_model(path, device)
        x_adv = pgd(model, x, y, args.eps / 255, steps=args.steps)
        cam_clean, logits_clean = grad_cam(model, x)
        cam_adv, logits_adv = grad_cam(model, x_adv)
        columns[path.stem] = (x_adv, cam_clean.cpu(), cam_adv.cpu(), F.softmax(logits_clean, 1).cpu(), F.softmax(logits_adv, 1).cpu())

    rows = []
    for i in range(len(x)):
        row = [(to_hwc(x[i]), label_text(y[i].item()))]
        for x_adv, cam_clean, cam_adv, p_clean, p_adv in columns.values():
            clean_pred, adv_pred = p_clean[i].argmax().item(), p_adv[i].argmax().item()
            row.append((overlay_cam(to_hwc(x[i]), cam_clean[i].numpy()), label_text(clean_pred, p_clean[i, clean_pred])))
            row.append((overlay_cam(to_hwc(x_adv[i]), cam_adv[i].numpy()), label_text(adv_pred, p_adv[i, adv_pred])))
        rows.append(row)
    col_titles = ["Original"]
    for name in columns:
        col_titles += [f"{name}\nGrad-CAM clean", f"{name}\nGrad-CAM adversarial"]
    output = plot_image_grid(
        rows, col_titles, Path(args.fig_dir) / "gradcam.png", f"Where does the model look? (PGD ε={args.eps:g}/255)"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
