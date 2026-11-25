import random
from typing import Any, Dict

import numpy as np
import torch

CLASSES = [
    "Healthy Leaf",
    "Insect Pest Disease",
    "Leaf Spot Disease",
    "Mosaic Virus Disease",
    "White Mold Disease",
    "Wilt Disease",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

BASE_SIZE = 224
HIGH_SIZE = 320
IMAGE_SIZES = (BASE_SIZE, HIGH_SIZE)

TEST_FRAC = 0.15
VAL_FRAC = 0.15

DEFAULT_CFG: Dict[str, Any] = {
    "pretrained": True,
    "img_size": BASE_SIZE,
    "head_type": "paper",
    "hidden_dim": 256,
    "dropout": 0.5,
    "drop_path": 0.2,
    "unfreeze": "all",
    "bn_mode": "train",
    "optimizer": "adamw",
    "backbone_lr": 1e-4,
    "head_lr": 1e-3,
    "weight_decay": 1e-4,
    "lr_decay": 1.0,
    "sam_rho": 0.0,
    "grad_clip": 1.0,
    "scheduler": "cosine",
    "warmup_epochs": 2,
    "plateau_factor": 0.3,
    "plateau_patience": 3,
    "batch_size": 32,
    "epochs": 40,
    "patience": 12,
    "loss_type": "ce",
    "class_weight_power": 0.5,
    "focal_gamma": 2.0,
    "label_smoothing": 0.0,
    "cb_beta": 0.999,
    "sampling": "none",
    "aug": "basic",
    "mix_prob": 0.0,
    "mixup_alpha": 0.4,
    "cutmix_alpha": 1.0,
    "ema_decay": 0.0,
}


def get_budget(smoke_test: bool = False) -> Dict[str, Any]:
    if smoke_test:
        return {
            "A_epochs": 2,
            "B_epochs": 2,
            "C_n_iter": 3,
            "C_search_epochs": 2,
            "C_final_epochs": 3,
            "C_topk": 2,
            "D_epochs": 3,
            "D_longer_epochs": 4,
            "D_folds": 2,
            "per_class_cap": 24,
        }
    return {
        "A_epochs": 40,
        "B_epochs": 100,
        "C_n_iter": 40,
        "C_search_epochs": 25,
        "C_final_epochs": 60,
        "C_topk": 3,
        "D_epochs": 60,
        "D_longer_epochs": 90,
        "D_folds": 5,
        "per_class_cap": None,
    }


def to_python_scalar(val: Any) -> Any:
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (list, tuple)):
        return type(val)(to_python_scalar(v) for v in val)
    return val


def merge_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    unknown = set(overrides) - set(DEFAULT_CFG)
    if unknown:
        raise KeyError(f"Unknown configuration keys: {sorted(unknown)}")
    return {**DEFAULT_CFG, **{k: to_python_scalar(v) for k, v in overrides.items()}}


def config_diff(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = merge_config(cfg)
    return {k: v for k, v in merged.items() if v != DEFAULT_CFG[k]}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_config():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device, use_amp, amp_dtype
