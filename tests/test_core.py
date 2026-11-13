import pytest
import torch

from signguard.attacks import fgsm, pgd, trades_loss
from signguard.data import GPUAugment, TensorLoader, group_split
from signguard.gradcam import grad_cam
from signguard.models import ARCHS, build_model

MEAN, STD = [0.34, 0.31, 0.32], [0.27, 0.26, 0.27]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model("signnet", 43, MEAN, STD).eval()


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.rand(4, 3, 48, 48), torch.randint(0, 43, (4,))


@pytest.mark.parametrize("arch", ARCHS)
def test_model_output_shape(arch):
    net = build_model(arch, 43, MEAN, STD).eval()
    assert net(torch.rand(2, 3, 48, 48)).shape == (2, 43)


def test_fgsm_zero_eps_is_identity(model, batch):
    x, y = batch
    assert torch.equal(fgsm(model, x, y, 0.0), x)


@pytest.mark.parametrize("targeted", [False, True])
def test_pgd_respects_constraints(model, batch, targeted):
    x, y = batch
    eps = 8 / 255
    x_adv = pgd(model, x, y, eps, steps=3, targeted=targeted, restarts=2)
    assert (x_adv - x).abs().max() <= eps + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_pgd_increases_loss(model, batch):
    x, y = batch
    x_adv = pgd(model, x, y, 8 / 255, steps=5)
    with torch.no_grad():
        clean = torch.nn.functional.cross_entropy(model(x), y)
        adv = torch.nn.functional.cross_entropy(model(x_adv), y)
    assert torch.equal(pgd(model, x, y, 0.0), x)
    assert adv >= clean


def test_trades_loss_is_finite(batch):
    net = build_model("signnet", 43, MEAN, STD)
    x, y = batch
    loss, logits = trades_loss(net, x, y, 8 / 255, 2 / 255, 2, 6.0)
    assert torch.isfinite(loss) and logits.shape == (4, 43)
    loss.backward()


def test_gradcam_shape_and_range(model, batch):
    x, _ = batch
    cam, logits = grad_cam(model, x)
    assert cam.shape == (4, 48, 48) and logits.shape == (4, 43)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_augment_keeps_shape_and_range(batch):
    x, _ = batch
    out = GPUAugment()(x)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1


def test_tensor_loader_batches():
    x = torch.randint(0, 255, (10, 3, 48, 48), dtype=torch.uint8)
    y = torch.arange(10)
    loader = TensorLoader(x, y, 4, False, torch.device("cpu"))
    batches = list(loader)
    assert len(batches) == 3 and batches[-1][0].shape[0] == 2
    assert batches[0][0].dtype == torch.float32 and batches[0][0].max() <= 1


def test_group_split_has_no_track_leakage():
    records = [(f"{c}_{t}_{i}", c, f"{c}_{t}") for c in range(3) for t in range(10) for i in range(5)]
    train_idx, val_idx = group_split(records, 0.2, 0)
    train_groups = {records[i][2] for i in train_idx}
    val_groups = {records[i][2] for i in val_idx}
    assert train_groups.isdisjoint(val_groups)
    assert len(train_idx) + len(val_idx) == len(records)
