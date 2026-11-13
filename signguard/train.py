import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import autocast_ctx, pgd, trades_loss
from .config import CKPT_DIR, DEFAULT_CACHE, NUM_CLASSES, RESULTS_DIR
from .data import GPUAugment, TensorLoader, load_cache
from .metrics import clean_accuracy, robust_accuracy
from .models import ARCHS, build_model, count_parameters
from .utils import ModelEMA, get_device, save_checkpoint, seed_everything, subset, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Standard or adversarial training on GTSRB.")
    parser.add_argument("--mode", choices=["standard", "fast", "pgd", "trades"], default="standard")
    parser.add_argument("--arch", choices=ARCHS, default="signnet")
    parser.add_argument("--name", default=None)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--eps-warmup", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--val-robust-samples", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--max-train-batches", type=int, default=None)
    args = parser.parse_args(argv)
    if args.name is None:
        args.name = args.mode if args.arch == "signnet" else f"{args.arch}_{args.mode}"
    return args


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (no_decay if param.ndim <= 1 else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def train_one_epoch(args, model, net, loader, optimizer, scheduler, scaler, ema, eps, alpha, use_amp, epoch):
    model.train()
    net.train()
    loss_sum, correct, seen = 0.0, 0, 0
    steps = min(len(loader), args.max_train_batches or len(loader))
    bar = tqdm(loader, total=steps, desc=f"epoch {epoch}/{args.epochs}", leave=False)
    for i, (x, y) in enumerate(bar):
        if i >= steps:
            break
        if args.mode in ("pgd", "fast"):
            attack_steps, attack_alpha = (1, 1.25 * eps) if args.mode == "fast" else (args.steps, alpha)
            net.eval()
            x_adv = pgd(net, x, y, eps, alpha=attack_alpha, steps=attack_steps, amp=use_amp)
            net.train()
            with autocast_ctx(x.device, use_amp):
                logits = net(x_adv)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)
        elif args.mode == "trades":
            loss, logits = trades_loss(net, x, y, eps, alpha, args.steps, args.beta, amp=use_amp)
        else:
            with autocast_ctx(x.device, use_amp):
                logits = net(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n = y.numel()
        loss_sum += loss.item() * n
        correct += logits.argmax(1).eq(y).sum().item()
        seen += n
        bar.set_postfix(loss=f"{loss_sum / seen:.3f}", acc=f"{correct / seen:.3f}")
    return {"loss": loss_sum / max(1, seen), "acc": correct / max(1, seen)}


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"
    channels_last = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cache = load_cache(args.cache)
    mean, std = cache["mean"].tolist(), cache["std"].tolist()
    model = build_model(args.arch, NUM_CLASSES, mean, std).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    net = torch.compile(model) if args.compile else model
    print(f"Arch: {args.arch}  Params: {count_parameters(model) / 1e6:.2f}M  Mode: {args.mode}  Device: {device}")

    augment = None if args.no_augment else GPUAugment()
    train_loader = TensorLoader(
        cache["x_train"], cache["y_train"], args.batch_size, True, device, augment, drop_last=True, channels_last=channels_last
    )
    val_loader = TensorLoader(cache["x_val"], cache["y_val"], args.eval_batch_size, False, device, channels_last=channels_last)
    x_rob, y_rob = subset(cache["x_val"], cache["y_val"], args.val_robust_samples, args.seed)
    val_robust_loader = TensorLoader(x_rob, y_rob, args.eval_batch_size, False, device, channels_last=channels_last)

    steps_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    if steps_per_epoch == 0:
        raise ValueError("Batch size is larger than the training set.")
    optimizer = torch.optim.SGD(param_groups(model, args.weight_decay), lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    eps_full, alpha_full = args.eps / 255, args.alpha / 255
    ckpt_path = Path(args.ckpt_dir) / f"{args.name}.pt"
    history, best_score, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        ramp = min(1.0, epoch / args.eps_warmup) if args.eps_warmup > 0 else 1.0
        eps, alpha = eps_full * ramp, alpha_full * ramp
        start = time.time()
        train_stats = train_one_epoch(args, model, net, train_loader, optimizer, scheduler, scaler, ema, eps, alpha, use_amp, epoch)
        eval_model = ema.ema if ema is not None else model
        val_clean = clean_accuracy(eval_model, val_loader)
        val_robust = robust_accuracy(eval_model, val_robust_loader, pgd, eps=eps_full, steps=args.val_steps, amp=use_amp)
        score = val_clean if args.mode == "standard" else val_robust
        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_clean": val_clean,
            "val_robust": val_robust,
            "eps": eps * 255,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - start,
        }
        history.append(record)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            save_checkpoint(
                ckpt_path,
                eval_model,
                args.arch,
                mean,
                std,
                NUM_CLASSES,
                {"mode": args.mode, "epoch": epoch, "val_clean": val_clean, "val_robust": val_robust, "args": vars(args)},
            )
        print(
            f"[{epoch:03d}] loss {record['train_loss']:.4f} | train acc {record['train_acc']:.4f} | "
            f"val clean {val_clean:.4f} | PGD-{args.val_steps} {val_robust:.4f} | "
            f"{record['seconds']:.1f}s{'  *' if improved else ''}"
        )

    write_json(
        Path(args.results_dir) / f"train_{args.name}.json",
        {"args": vars(args), "best_epoch": best_epoch, "best_score": best_score, "history": history},
    )
    print(f"Best epoch: {best_epoch}  score: {best_score:.4f}  checkpoint: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    main()
