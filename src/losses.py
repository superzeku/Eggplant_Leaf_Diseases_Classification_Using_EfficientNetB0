from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES

VALID_LOSS_TYPES = (
    "ce",
    "weighted_ce",
    "cb_ce",
    "focal",
    "weighted_focal",
    "cb_focal",
)


def class_weights_from_counts(
    counts: np.ndarray,
    loss_type: str,
    power: float = 0.5,
    cb_beta: float = 0.999,
) -> Optional[torch.Tensor]:
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    if loss_type in ("weighted_ce", "weighted_focal"):
        weights = (counts.sum() / (len(counts) * counts)) ** power
    elif loss_type in ("cb_ce", "cb_focal"):
        # Effective number of samples from Cui et al. (CVPR 2019)
        weights = (1.0 - cb_beta) / (1.0 - np.power(cb_beta, counts))
    else:
        return None
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


class ImbalanceLoss(nn.Module):
    def __init__(
        self,
        loss_type: str,
        counts: np.ndarray,
        power: float = 0.5,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        cb_beta: float = 0.999,
    ):
        super().__init__()
        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.focal = loss_type.endswith("focal")
        self.gamma = float(gamma)
        self.label_smoothing = 0.0 if self.focal else float(label_smoothing)

        weights = class_weights_from_counts(counts, loss_type, power=power, cb_beta=cb_beta)
        self.use_weights = weights is not None
        self.register_buffer("w", weights if weights is not None else torch.ones(len(counts)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if not self.focal:
            return F.cross_entropy(
                logits,
                targets,
                weight=self.w if self.use_weights else None,
                label_smoothing=self.label_smoothing,
            )

        log_pt = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        # Clamp away from zero so gamma < 1 does not produce infinite gradients when pt -> 1.
        modulating = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)
        loss = -modulating * log_pt
        if self.use_weights:
            loss = loss * self.w[targets]
        return loss.mean()


def sample_epoch_order(y: np.ndarray, strategy: str, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    if strategy == "none":
        return rng.permutation(n)
    if strategy not in ("sqrt", "balanced"):
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    per_class = 1.0 / np.sqrt(safe_counts) if strategy == "sqrt" else 1.0 / safe_counts
    probs = per_class[y]
    return rng.choice(n, size=n, replace=True, p=probs / probs.sum())
