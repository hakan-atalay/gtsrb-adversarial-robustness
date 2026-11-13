from contextlib import nullcontext

import torch
import torch.nn.functional as F

GRAD_SCALE = 1024.0


def autocast_ctx(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _project(x, delta, eps):
    delta = delta.clamp(-eps, eps)
    return (x + delta).clamp(0.0, 1.0) - x


def _input_grad(loss, delta):
    (grad,) = torch.autograd.grad(loss * GRAD_SCALE, delta)
    return torch.nan_to_num(grad)


@torch.enable_grad()
def fgsm(model, x, y, eps, targeted=False, amp=False):
    x = x.detach()
    if eps <= 0:
        return x.clone()
    delta = torch.zeros_like(x, requires_grad=True)
    with autocast_ctx(x.device, amp):
        loss = F.cross_entropy(model(x + delta), y)
    grad = _input_grad(loss, delta)
    step = -grad.sign() if targeted else grad.sign()
    return (x + _project(x, eps * step, eps)).detach()


@torch.enable_grad()
def pgd(model, x, y, eps, alpha=None, steps=10, restarts=1, random_start=True, targeted=False, amp=False):
    x = x.detach()
    if eps <= 0 or steps <= 0:
        return x.clone()
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    best, found = None, None
    for _restart in range(max(1, restarts)):
        if random_start:
            delta = torch.empty_like(x).uniform_(-eps, eps)
        else:
            delta = torch.zeros_like(x)
        delta = _project(x, delta, eps)
        for _step in range(steps):
            delta.requires_grad_(True)
            with autocast_ctx(x.device, amp):
                loss = F.cross_entropy(model(x + delta), y)
            grad = _input_grad(loss, delta)
            step = -grad.sign() if targeted else grad.sign()
            delta = _project(x, delta.detach() + alpha * step, eps)
        x_adv = (x + delta).detach()
        if restarts <= 1:
            return x_adv
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        success = pred.eq(y) if targeted else pred.ne(y)
        if best is None:
            best, found = x_adv.clone(), success.clone()
        else:
            update = success & ~found
            best[update] = x_adv[update]
            found |= success
    return best


def trades_loss(model, x, y, eps, alpha, steps, beta, amp=False):
    x = x.detach()
    model.eval()
    with torch.no_grad(), autocast_ctx(x.device, amp):
        p_natural = F.softmax(model(x).float(), dim=1)
    delta = _project(x, 0.001 * torch.randn_like(x), eps)
    for _step in range(steps):
        delta.requires_grad_(True)
        with autocast_ctx(x.device, amp):
            logits = model(x + delta).float()
        loss = F.kl_div(F.log_softmax(logits, dim=1), p_natural, reduction="batchmean")
        grad = _input_grad(loss, delta)
        delta = _project(x, delta.detach() + alpha * grad.sign(), eps)
    model.train()
    x_adv = (x + delta).detach()
    with autocast_ctx(x.device, amp):
        logits_natural = model(x).float()
        logits_adv = model(x_adv).float()
    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1), reduction="batchmean")
    return loss_natural + beta * loss_robust, logits_natural.detach()


def autoattack_accuracy(model, x, y, eps, batch_size=500, version="standard"):
    try:
        from autoattack import AutoAttack
    except ImportError as error:
        raise ImportError("AutoAttack is not installed: pip install git+https://github.com/fra31/auto-attack") from error
    model.eval()
    adversary = AutoAttack(model, norm="Linf", eps=eps, version=version, verbose=False)
    x_adv = adversary.run_standard_evaluation(x, y, bs=batch_size)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            correct += model(x_adv[i : i + batch_size]).argmax(1).eq(y[i : i + batch_size]).sum().item()
    return correct / len(y)
