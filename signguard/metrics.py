import torch


def clean_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model(x).argmax(1).eq(y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def robust_accuracy(model, loader, attack, **kwargs):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x_adv = attack(model, x, y, **kwargs)
        with torch.no_grad():
            correct += model(x_adv).argmax(1).eq(y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def targeted_success(model, loader, attack, target, **kwargs):
    model.eval()
    success = total = 0
    for x, _ in loader:
        t = torch.full((x.size(0),), target, dtype=torch.long, device=x.device)
        x_adv = attack(model, x, t, targeted=True, **kwargs)
        with torch.no_grad():
            success += model(x_adv).argmax(1).eq(t).sum().item()
        total += t.numel()
    return success / max(1, total)
