import copy
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from .config import NUM_CLASSES

MSF_STAGES = (5, 6, 7, 8)
MSF_CHANNELS = (112, 192, 320, 1280)
UNFREEZE_MIN = {"all": 0, "from3": 3, "from5": 5, "from6": 6, "head": 9}


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        x = x.float().clamp(min=self.eps).pow(p)
        return F.avg_pool2d(x, x.shape[-2:]).pow(1.0 / p).flatten(1)


class MSFHead(nn.Module):
    """Multi-scale feature fusion head using GeM pooling across intermediate stages."""

    def __init__(self, channels: Sequence[int], hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.pools = nn.ModuleList([GeM() for _ in channels])
        in_dim = int(sum(channels))
        self.norm = nn.LayerNorm(in_dim)
        self.drop1 = nn.Dropout(dropout * 0.5)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        pooled = [pool(feat) for pool, feat in zip(self.pools, feats)]
        z = self.norm(torch.cat(pooled, dim=1))
        z = self.drop2(self.act(self.fc1(self.drop1(z))))
        return self.fc2(z)


class EffNetB0Net(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        head_type: str = "paper",
        hidden_dim: int = 256,
        dropout: float = 0.5,
        drop_path: float = 0.2,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights, stochastic_depth_prob=float(drop_path))
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.head_type = head_type

        if head_type == "paper":
            self.classifier = nn.Sequential(
                nn.Linear(1280, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), num_classes),
            )
        elif head_type == "msf_gem":
            self.classifier = MSFHead(MSF_CHANNELS, int(hidden_dim), num_classes, float(dropout))
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "paper":
            x = self.avgpool(self.features(x))
            return self.classifier(torch.flatten(x, 1))

        feats = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in MSF_STAGES:
                feats.append(x)
        return self.classifier(feats)


def apply_freezing(model: EffNetB0Net, unfreeze: str) -> None:
    min_stage = UNFREEZE_MIN[unfreeze]
    for idx, block in enumerate(model.features):
        for param in block.parameters():
            param.requires_grad = idx >= min_stage


def set_bn_mode(model: EffNetB0Net, unfreeze: str, bn_mode: str) -> None:
    # Keep BatchNorm running stats frozen on locked stages (or globally when bn_mode='freeze').
    min_stage = UNFREEZE_MIN[unfreeze]
    for idx, block in enumerate(model.features):
        if idx < min_stage or bn_mode == "freeze":
            for module in block.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


def build_model(cfg: Dict, device: torch.device) -> EffNetB0Net:
    if not cfg["pretrained"] and cfg["unfreeze"] != "all":
        raise ValueError("Training from scratch requires unfreeze='all'.")

    model = EffNetB0Net(
        pretrained=cfg["pretrained"],
        head_type=cfg["head_type"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        drop_path=cfg["drop_path"],
    )
    apply_freezing(model, cfg["unfreeze"])
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = float(decay)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        src_state = model.state_dict()
        for key, dst_val in self.module.state_dict().items():
            if dst_val.dtype.is_floating_point:
                dst_val.mul_(decay).add_(src_state[key].detach(), alpha=1.0 - decay)
            else:
                dst_val.copy_(src_state[key])
