import hashlib
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

from .config import CLASS_TO_IDX, CLASSES, IMAGE_SIZES, NUM_CLASSES, TEST_FRAC, VAL_FRAC

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _norm_name(name: str) -> str:
    return str(name).strip().casefold().replace("_", " ").replace("-", " ")


def find_dataset_dir(explicit_dir: Optional[Path] = None) -> Path:
    candidates = []
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir))
    candidates.extend([
        Path("Dataset") / "Original_dataset",
        Path("Original_dataset"),
    ])
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()

    skip_prefixes = (".", "cache_", "models_", "results_", "__pycache__")
    for base in (Path.cwd(), Path.cwd().parent):
        for root, dirs, _ in os.walk(base):
            depth = len(Path(root).relative_to(base).parts)
            if depth > 3:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith(skip_prefixes)]
            if Path(root).name.lower() == "original_dataset":
                return Path(root).resolve()

    raise FileNotFoundError(
        "Could not locate 'Original_dataset'. Pass --dataset-dir or place the dataset under Dataset/Original_dataset."
    )


def collect_dataset_records(source_dir: Path) -> pd.DataFrame:
    folders = {_norm_name(p.name): p for p in source_dir.iterdir() if p.is_dir()}
    missing = [cls for cls in CLASSES if _norm_name(cls) not in folders]
    if missing:
        available = [p.name for p in source_dir.iterdir() if p.is_dir()]
        raise FileNotFoundError(f"Missing class folders: {missing}. Available: {available}")

    records = []
    for cls in CLASSES:
        for path in sorted(folders[_norm_name(cls)].rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                records.append({
                    "path": str(path.resolve()),
                    "class_name": cls,
                    "label": CLASS_TO_IDX[cls],
                })

    df = pd.DataFrame(records).reset_index(drop=True)
    df["idx"] = np.arange(len(df))
    return df


def _decode_one(path: str, sizes: Sequence[int]) -> Optional[Dict[int, np.ndarray]]:
    try:
        with Image.open(path) as im:
            try:
                im.draft("RGB", (max(sizes), max(sizes)))
            except Exception:
                pass
            im = im.convert("RGB")
            return {
                s: np.asarray(im.resize((s, s), Image.Resampling.BILINEAR), dtype=np.uint8)
                for s in sizes
            }
    except Exception:
        return None


def build_or_load_cache(
    paths: Sequence[str],
    cache_dir: Path,
    sizes: Sequence[int] = IMAGE_SIZES,
    workers: Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = hashlib.md5("\n".join(paths).encode("utf-8")).hexdigest()[:12]
    files = {s: cache_dir / f"img_{key}_{s}.npy" for s in sizes}
    valid_file = cache_dir / f"valid_{key}.npy"

    if all(f.exists() for f in files.values()) and valid_file.exists():
        print(f"Loaded cached images from {cache_dir}")
        return {s: np.load(f) for s, f in files.items()}, np.load(valid_file)

    n = len(paths)
    arrays = {s: np.zeros((n, s, s, 3), dtype=np.uint8) for s in sizes}
    valid = np.zeros(n, dtype=bool)
    workers = workers or min(8, os.cpu_count() or 4)

    t0 = time.time()
    print(f"Decoding {n} images using {workers} threads...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(lambda p: _decode_one(p, sizes), paths)):
            if res is not None:
                valid[i] = True
                for s in sizes:
                    arrays[s][i] = res[s]
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)")

    for s in sizes:
        np.save(files[s], arrays[s])
    np.save(valid_file, valid)
    return arrays, valid


def make_or_load_splits(
    df: pd.DataFrame,
    results_dir: Path,
    seed: int = 42,
    resume: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    split_files = {k: results_dir / f"split6_{k}.csv" for k in ("train", "val", "test")}

    def _load_one(csv_path: Path) -> pd.DataFrame:
        saved = pd.read_csv(csv_path)
        return df.merge(saved[["path"]], on="path", how="inner").reset_index(drop=True)

    loaded_ok = False
    if resume and all(p.exists() for p in split_files.values()):
        train_df = _load_one(split_files["train"])
        val_df = _load_one(split_files["val"])
        test_df = _load_one(split_files["test"])
        loaded_ok = (len(train_df) + len(val_df) + len(test_df)) == len(df)
        if loaded_ok:
            print("Loaded existing stratified split.")
        else:
            print("Saved split did not match active dataset; regenerating.")

    if not loaded_ok:
        dev_df, test_df = train_test_split(
            df, test_size=TEST_FRAC, stratify=df["label"], random_state=seed
        )
        train_df, val_df = train_test_split(
            dev_df,
            test_size=VAL_FRAC / (1.0 - TEST_FRAC),
            stratify=dev_df["label"],
            random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        for name, frame in zip(("train", "val", "test"), (train_df, val_df, test_df)):
            frame[["path", "class_name", "label", "idx"]].to_csv(split_files[name], index=False)
        print("Created and saved new stratified split.")

    assert set(train_df["path"]).isdisjoint(val_df["path"])
    assert set(train_df["path"]).isdisjoint(test_df["path"])
    assert set(val_df["path"]).isdisjoint(test_df["path"])
    return train_df, val_df, test_df


def df_to_arrays(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "idx": frame["idx"].to_numpy(np.int64),
        "y": frame["label"].to_numpy(np.int64),
    }


def batch_to_device(X: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    xb = torch.from_numpy(X[idx]).to(device, non_blocking=True)
    return xb.permute(0, 3, 1, 2).float().div_(255.0).contiguous()


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _uniform(batch_size: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    if lo == hi:
        return torch.full((batch_size,), float(lo), device=device)
    return torch.empty(batch_size, device=device).uniform_(float(lo), float(hi))


def random_geometric(
    x: torch.Tensor,
    rot_deg: float = 0.0,
    area: Tuple[float, float] = (1.0, 1.0),
    ratio: Tuple[float, float] = (1.0, 1.0),
    hflip: float = 0.5,
    vflip: float = 0.0,
) -> torch.Tensor:
    b = x.size(0)
    dev = x.device
    ang = _uniform(b, -rot_deg, rot_deg, dev) * (math.pi / 180.0)
    scale_area = _uniform(b, area[0], area[1], dev)
    aspect = torch.exp(_uniform(b, math.log(ratio[0]), math.log(ratio[1]), dev))
    w = torch.sqrt(scale_area * aspect).clamp(max=1.0)
    h = torch.sqrt(scale_area / aspect).clamp(max=1.0)
    tx = (torch.rand(b, device=dev) * 2.0 - 1.0) * (1.0 - w)
    ty = (torch.rand(b, device=dev) * 2.0 - 1.0) * (1.0 - h)
    fx = 1.0 - 2.0 * (torch.rand(b, device=dev) < hflip).float()
    fy = 1.0 - 2.0 * (torch.rand(b, device=dev) < vflip).float()

    cos_a, sin_a = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0] = cos_a * w * fx
    theta[:, 0, 1] = -sin_a * h * fy
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a * w * fx
    theta[:, 1, 1] = cos_a * h * fy
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def gaussian_blur(x: torch.Tensor, sigma: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    active = sigma > 1e-4
    if not active.any():
        return x

    b, c, h, w = x.shape
    r = kernel_size // 2
    coords = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    ker = torch.exp(-(coords[None, :] ** 2) / (2.0 * sigma[:, None].clamp_min(1e-3) ** 2))
    ker = ker / ker.sum(1, keepdim=True)

    kx = ker.repeat_interleave(c, 0).view(b * c, 1, 1, kernel_size)
    ky = ker.repeat_interleave(c, 0).view(b * c, 1, kernel_size, 1)
    y = x.reshape(1, b * c, h, w)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), kx, groups=b * c)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), ky, groups=b * c)
    return y.view(b, c, h, w)


def add_noise(x: torch.Tensor, max_std: float) -> torch.Tensor:
    std = _uniform(x.size(0), 0.0, max_std, x.device).view(-1, 1, 1, 1)
    return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)


def color_jitter(x: torch.Tensor, b: float = 0.3, c: float = 0.3, s: float = 0.3) -> torch.Tensor:
    bs = x.size(0)
    dev = x.device
    if b > 0:
        x = x * _uniform(bs, 1.0 - b, 1.0 + b, dev).view(bs, 1, 1, 1)
    if c > 0:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * _uniform(bs, 1.0 - c, 1.0 + c, dev).view(bs, 1, 1, 1) + mean
    if s > 0:
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = gray + (x - gray) * _uniform(bs, 1.0 - s, 1.0 + s, dev).view(bs, 1, 1, 1)
    return x.clamp(0.0, 1.0)


def random_erase(
    x: torch.Tensor,
    p: float = 0.3,
    area: Tuple[float, float] = (0.02, 0.15),
    ratio: Tuple[float, float] = (0.3, 3.3),
) -> torch.Tensor:
    b, _, h, w = x.shape
    dev = x.device
    enabled = torch.rand(b, device=dev) < p
    if not enabled.any():
        return x

    target_area = _uniform(b, area[0], area[1], dev) * (h * w)
    aspect = torch.exp(_uniform(b, math.log(ratio[0]), math.log(ratio[1]), dev))
    eh = torch.sqrt(target_area * aspect).clamp(max=h - 1)
    ew = torch.sqrt(target_area / aspect).clamp(max=w - 1)
    cy = torch.rand(b, device=dev) * h
    cx = torch.rand(b, device=dev) * w

    ys = torch.arange(h, device=dev, dtype=x.dtype).view(1, h, 1)
    xs = torch.arange(w, device=dev, dtype=x.dtype).view(1, 1, w)
    mask = (
        ((ys - cy.view(b, 1, 1)).abs() < eh.view(b, 1, 1) / 2.0)
        & ((xs - cx.view(b, 1, 1)).abs() < ew.view(b, 1, 1) / 2.0)
        & enabled.view(b, 1, 1)
    )
    return torch.where(mask.unsqueeze(1), torch.rand_like(x), x)


def augment_batch(x: torch.Tensor, level: str) -> torch.Tensor:
    b = x.size(0)
    dev = x.device
    if level == "none":
        return x
    if level == "basic":
        x = random_geometric(x, rot_deg=15.0, hflip=0.5)
        x = gaussian_blur(x, _uniform(b, 0.0, 1.0, dev))
        x = add_noise(x, 0.05)
        return (x * _uniform(b, 0.8, 1.2, dev).view(b, 1, 1, 1)).clamp(0.0, 1.0)
    if level == "strong":
        x = random_geometric(
            x, rot_deg=30.0, area=(0.45, 1.0), ratio=(0.75, 4.0 / 3.0), hflip=0.5, vflip=0.5
        )
        x = color_jitter(x, 0.3, 0.3, 0.3)
        sigma = _uniform(b, 0.1, 1.5, dev) * (torch.rand(b, device=dev) < 0.3).float()
        x = gaussian_blur(x, sigma)
        x = add_noise(x, 0.04)
        return random_erase(x, p=0.3)
    raise ValueError(f"Unsupported augmentation level: {level}")


def apply_mix(
    x: torch.Tensor, y: torch.Tensor, cfg: Dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    prob = float(cfg["mix_prob"])
    if prob <= 0.0 or x.size(0) < 2 or random.random() >= prob:
        return x, y, y, 1.0

    perm = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    lam = float(np.random.beta(cfg["cutmix_alpha"], cfg["cutmix_alpha"]))
    _, _, h, w = x.shape
    cut_ratio = math.sqrt(1.0 - lam)
    ch, cw = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(h * w))
    return mixed, y, y[perm], lam
