import argparse
from pathlib import Path

import torch

from .attacks import pgd
from .data import TensorLoader, load_cache
from .utils import add_common_args, find_checkpoints, get_device, load_model, subset, write_json
from .visualize import plot_transfer_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description="Black-box scenario: do attacks crafted on one model fool the others?")
    add_common_args(parser)
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args(argv)

    device = get_device(args.device)
    cache = load_cache(args.cache)
    x_test, y_test = subset(cache["x_test"], cache["y_test"], args.n_samples, args.seed)
    loader = TensorLoader(x_test, y_test, args.batch_size, False, device)
    paths = find_checkpoints(args.models)
    names = [p.stem for p in paths]
    models = [load_model(p, device)[0] for p in paths]

    matrix = []
    for source_name, source in zip(names, models):
        adversarial = [(pgd(source, x, y, args.eps / 255, steps=args.steps), y) for x, y in loader]
        row = []
        for target in models:
            correct = total = 0
            with torch.no_grad():
                for x_adv, y in adversarial:
                    correct += target(x_adv).argmax(1).eq(y).sum().item()
                    total += y.numel()
            row.append(correct / total)
        matrix.append(row)
        print(f"source {source_name:<22} " + "  ".join(f"{n}: {v:.4f}" for n, v in zip(names, row)))

    plot_transfer_matrix(
        names, matrix, Path(args.fig_dir) / "transfer_matrix.png", f"Transfer attack (PGD-{args.steps}, ε={args.eps:g}/255)"
    )
    write_json(Path(args.results_dir) / "transfer.json", {"eps": args.eps, "steps": args.steps, "models": names, "matrix": matrix})
    return names, matrix


if __name__ == "__main__":
    main()
