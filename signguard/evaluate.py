import argparse
import json
from pathlib import Path

from .attacks import autoattack_accuracy, fgsm, pgd
from .data import TensorLoader, load_cache
from .metrics import clean_accuracy, robust_accuracy
from .utils import add_common_args, find_checkpoints, get_device, load_model, parse_eps, subset, write_json
from .visualize import plot_robustness_curves, plot_training_history


def sanity_checks(model, loader, res, eps_list, steps):
    fgsm_values = [res["fgsm"][str(e)] for e in eps_list]
    pgd_values = [res["pgd"][str(e)] for e in eps_list]
    huge_eps = robust_accuracy(model, loader, pgd, eps=64 / 255, steps=steps)
    return {
        "pgd_stronger_than_fgsm": all(p <= f + 0.01 for p, f in zip(pgd_values, fgsm_values)),
        "accuracy_decreases_with_eps": all(a >= b - 0.01 for a, b in zip(pgd_values, pgd_values[1:])),
        "large_eps_breaks_model": huge_eps < 0.05,
        "pgd_eps64_accuracy": huge_eps,
    }


def summary_table(results, report_eps, steps, autoattack):
    key = str(report_eps)
    eps_text = f"{report_eps:g}/255"
    header = f"| Model | Clean accuracy | FGSM ε={eps_text} | PGD-{steps} ε={eps_text} |"
    divider = "|---|---|---|---|"
    if autoattack:
        header += f" AutoAttack ε={eps_text} |"
        divider += "---|"
    lines = [header, divider]
    for name, res in results.items():
        row = f"| {name} | {res['clean'] * 100:.2f}% | {res['fgsm'][key] * 100:.2f}% | {res['pgd'][key] * 100:.2f}% |"
        if autoattack:
            row += f" {res['autoattack'] * 100:.2f}% |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluates models with FGSM, PGD and optionally AutoAttack.")
    add_common_args(parser)
    parser.add_argument("--eps-list", default="0,1,2,4,6,8,12,16")
    parser.add_argument("--report-eps", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--autoattack", action="store_true")
    parser.add_argument("--no-sanity", action="store_true")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    cache = load_cache(args.cache)
    x_test, y_test = subset(cache["x_test"], cache["y_test"], args.n_samples, args.seed)
    loader = TensorLoader(x_test, y_test, args.batch_size, False, device)
    eps_list = sorted(set(parse_eps(args.eps_list) + [args.report_eps]))

    results = {}
    for path in find_checkpoints(args.models):
        model, _ = load_model(path, device)
        name = path.stem
        res = {"clean": clean_accuracy(model, loader), "fgsm": {}, "pgd": {}}
        for eps in eps_list:
            res["fgsm"][str(eps)] = robust_accuracy(model, loader, fgsm, eps=eps / 255)
            res["pgd"][str(eps)] = robust_accuracy(model, loader, pgd, eps=eps / 255, steps=args.steps, restarts=args.restarts)
            print(f"{name:<22} ε={eps:>4g}/255  FGSM {res['fgsm'][str(eps)]:.4f}  PGD-{args.steps} {res['pgd'][str(eps)]:.4f}")
        if args.autoattack:
            x_all = x_test.float().div(255).to(device)
            res["autoattack"] = autoattack_accuracy(model, x_all, y_test.to(device), args.report_eps / 255, args.batch_size)
            print(f"{name:<22} AutoAttack ε={args.report_eps:g}/255  {res['autoattack']:.4f}")
        if not args.no_sanity:
            res["sanity"] = sanity_checks(model, loader, res, eps_list, args.steps)
            print(f"{name:<22} gradient masking checks: {res['sanity']}")
        results[name] = res

    results_dir, fig_dir = Path(args.results_dir), Path(args.fig_dir)
    write_json(results_dir / "robustness.json", {"steps": args.steps, "eps_list": eps_list, "results": results})
    table = summary_table(results, args.report_eps, args.steps, args.autoattack)
    (results_dir / "summary.md").write_text(table, encoding="utf-8")
    plot_robustness_curves(results, eps_list, fig_dir / "robustness_curves.png")

    histories = {}
    for name in results:
        history_path = results_dir / f"train_{name}.json"
        if history_path.exists():
            histories[name] = json.loads(history_path.read_text(encoding="utf-8"))["history"]
    if histories:
        plot_training_history(histories, fig_dir / "training_history.png")

    print("\n" + table)
    return results


if __name__ == "__main__":
    main()
