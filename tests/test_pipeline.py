import json

import torch

from signguard import evaluate, explore, gradcam, targeted, train, transfer


def make_cache(path):
    torch.manual_seed(0)

    def split(n):
        y = torch.randint(0, 43, (n,))
        y[:6] = 14
        return torch.randint(0, 256, (n, 3, 48, 48), dtype=torch.uint8), y

    x_train, y_train = split(64)
    x_val, y_val = split(24)
    x_test, y_test = split(24)
    torch.save(
        {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "mean": torch.tensor([0.5, 0.5, 0.5]),
            "std": torch.tensor([0.29, 0.29, 0.29]),
            "size": 48,
        },
        path,
    )


def test_full_pipeline(tmp_path):
    cache = tmp_path / "cache.pt"
    make_cache(cache)
    common = ["--cache", str(cache), "--device", "cpu", "--results-dir", str(tmp_path / "results")]
    train_common = common + ["--ckpt-dir", str(tmp_path / "ckpt"), "--epochs", "1", "--batch-size", "16", "--no-amp"]
    fast = ["--steps", "2", "--val-steps", "2", "--val-robust-samples", "16"]

    standard = train.main(train_common + ["--mode", "standard", "--name", "standard"] + fast)
    robust = train.main(train_common + ["--mode", "pgd", "--name", "pgd_at"] + fast)
    trades = train.main(train_common + ["--mode", "trades", "--name", "trades"] + fast)
    fast_at = train.main(train_common + ["--mode", "fast", "--name", "fast_at"] + fast)
    for path in (standard, robust, trades, fast_at):
        assert path.exists()

    models = ["--models", str(standard), str(robust), str(trades)]
    eval_common = common + ["--fig-dir", str(tmp_path / "fig"), "--batch-size", "8"] + models
    results = evaluate.main(eval_common + ["--eps-list", "0,8", "--steps", "2"])
    assert set(results) == {"standard", "pgd_at", "trades"}
    assert (tmp_path / "results" / "summary.md").exists()
    assert (tmp_path / "fig" / "robustness_curves.png").exists()

    targeted.main(eval_common + ["--eps-list", "8", "--steps", "2", "--n-show", "2"])
    assert json.loads((tmp_path / "results" / "targeted.json").read_text(encoding="utf-8"))["source"] == "Stop"

    names, matrix = transfer.main(eval_common + ["--steps", "2", "--n-samples", "16"])
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)

    explore.main(["--cache", str(cache), "--fig-dir", str(tmp_path / "fig"), "--raw-dir", str(tmp_path / "missing")])
    assert (tmp_path / "fig" / "data_class_distribution.png").exists()

    gradcam.main(eval_common + ["--steps", "2", "--classes", "14"])
    assert (tmp_path / "fig" / "gradcam.png").exists()
