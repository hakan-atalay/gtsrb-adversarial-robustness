import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .attacks import pgd
from .config import CLASS_NAMES, SPEED60_CLASS, STOP_CLASS
from .data import TensorLoader, load_cache
from .metrics import targeted_success
from .utils import add_common_args, find_checkpoints, get_device, load_model, parse_eps, subset, write_json
from .visualize import label_text, perturbation_image, plot_image_grid, plot_success_curves, to_hwc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Targeted attack: by default makes the model read a STOP sign as a 60 km/h speed limit.")
    add_common_args(parser)
    parser.add_argument("--source-class", type=int, default=STOP_CLASS)
    parser.add_argument("--target-class", type=int, default=SPEED60_CLASS)
    parser.add_argument("--eps-list", default="1,2,4,8,12,16")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-show", type=int, default=5)
    parser.add_argument("--show-eps", type=float, default=8.0)
    args = parser.parse_args(argv)

    device = get_device(args.device)
    cache = load_cache(args.cache)
    mask = cache["y_test"] == args.source_class
    x_src, y_src = subset(cache["x_test"][mask], cache["y_test"][mask], args.n_samples, args.seed)
    if len(y_src) == 0:
        raise SystemExit("No test samples found for the source class.")
    loader = TensorLoader(x_src, y_src, args.batch_size, False, device)
    eps_list = parse_eps(args.eps_list)
    source_name, target_name = CLASS_NAMES[args.source_class], CLASS_NAMES[args.target_class]

    x_show = x_src[: args.n_show].float().div(255).to(device)
    target_show = torch.full((len(x_show),), args.target_class, dtype=torch.long, device=device)
    rates, columns = {}, {}
    for path in find_checkpoints(args.models):
        model, _ = load_model(path, device)
        name = path.stem
        rates[name] = {}
        for eps in eps_list:
            rates[name][str(eps)] = targeted_success(model, loader, pgd, args.target_class, eps=eps / 255, steps=args.steps)
            print(f"{name:<22} ε={eps:>4g}/255  '{source_name}' -> '{target_name}' success: {rates[name][str(eps)]:.4f}")
        x_adv = pgd(model, x_show, target_show, args.show_eps / 255, steps=args.steps, targeted=True)
        with torch.no_grad():
            p_clean = F.softmax(model(x_show), dim=1).cpu()
            p_adv = F.softmax(model(x_adv), dim=1).cpu()
        columns[name] = (x_adv, p_clean, p_adv)

    rows = []
    eps_show = args.show_eps / 255
    for i in range(len(x_show)):
        row = [(to_hwc(x_show[i]), "")]
        for x_adv, p_clean, p_adv in columns.values():
            before, after = p_clean[i].argmax().item(), p_adv[i].argmax().item()
            row.append((perturbation_image(x_show[i], x_adv[i], eps_show), ""))
            row.append((to_hwc(x_adv[i]), f"before: {label_text(before, p_clean[i, before])}\nafter: {label_text(after, p_adv[i, after])}"))
        rows.append(row)
    col_titles = ["Original"]
    for name in columns:
        col_titles += [f"{name}\nperturbation", f"{name}\nadversarial"]

    fig_dir = Path(args.fig_dir)
    plot_image_grid(
        rows,
        col_titles,
        fig_dir / "targeted_examples.png",
        f"Targeted PGD: '{source_name}' → '{target_name}' (ε={args.show_eps:g}/255)",
    )
    plot_success_curves(rates, eps_list, fig_dir / "targeted_success.png", f"'{source_name}' → '{target_name}' attack success rate")
    write_json(
        Path(args.results_dir) / "targeted.json",
        {"source": source_name, "target": target_name, "steps": args.steps, "n": len(y_src), "success": rates},
    )
    return rates


if __name__ == "__main__":
    main()
