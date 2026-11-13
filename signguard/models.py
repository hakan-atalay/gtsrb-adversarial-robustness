import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep)
    return x * mask / keep


class PreActSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_path_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.se = SqueezeExcite(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        out = self.act(self.bn1(x))
        identity = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.act(self.bn2(out)))
        out = self.se(out)
        return identity + drop_path(out, self.drop_path_rate, self.training)


class SignNet(nn.Module):
    def __init__(self, num_classes, widths, depths, mean, std, drop_path_rate=0.05, dropout=0.2):
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(inplace=True),
        )
        blocks = []
        in_channels = widths[0]
        total = sum(depths)
        index = 0
        for stage, (width, depth) in enumerate(zip(widths, depths)):
            for block in range(depth):
                stride = 2 if block == 0 and stage > 0 else 1
                rate = drop_path_rate * index / max(1, total - 1)
                blocks.append(PreActSEBlock(in_channels, width, stride, rate))
                in_channels = width
                index += 1
        self.stages = nn.Sequential(*blocks)
        self.head_norm = nn.Sequential(nn.BatchNorm2d(in_channels), nn.SiLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    @property
    def cam_layer(self):
        return self.head_norm

    def features(self, x):
        return self.head_norm(self.stages(self.stem(self.normalize(x))))

    def forward(self, x):
        return self.fc(self.dropout(torch.flatten(self.pool(self.features(x)), 1)))


class SmallInputResNet18(nn.Module):
    def __init__(self, num_classes, mean, std):
        super().__init__()
        from torchvision.models import resnet18

        self.normalize = Normalize(mean, std)
        self.net = resnet18(weights=None, num_classes=num_classes)
        self.net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.net.maxpool = nn.Identity()

    @property
    def cam_layer(self):
        return self.net.layer4

    def forward(self, x):
        return self.net(self.normalize(x))


ARCHS = ("signnet", "signnet_wide", "resnet18")


def build_model(arch, num_classes, mean, std):
    if arch == "signnet":
        return SignNet(num_classes, (32, 64, 128), (2, 2, 2), mean, std)
    if arch == "signnet_wide":
        return SignNet(num_classes, (64, 128, 256), (2, 2, 2), mean, std)
    if arch == "resnet18":
        return SmallInputResNet18(num_classes, mean, std)
    raise ValueError(f"Unknown architecture: {arch}. Options: {', '.join(ARCHS)}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
