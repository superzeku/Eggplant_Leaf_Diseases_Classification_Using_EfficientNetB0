import gc
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from .config import NUM_CLASSES, merge_config, set_seed
from .data import apply_mix, augment_batch, batch_to_device, normalize_batch
from .losses import ImbalanceLoss, sample_epoch_order
from .models import EffNetB0Net, ModelEMA, build_model, count_params, set_bn_mode


def _split_decay_params(module: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    decay, no_decay = [], []
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
    no_decay_ids = set()

    for m in module.modules():
        if isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                no_decay_ids.add(id(p))

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_ids or param.ndim < 2 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def make_optimizer(model: EffNetB0Net, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    num_stages = len(model.features)
    wd = float(cfg["weight_decay"])
    groups: List[Dict[str, Any]] = []

    for idx, block in enumerate(model.features):
        lr = cfg["backbone_lr"] * (cfg["lr_decay"] ** (num_stages - 1 - idx))
        dec, nodec = _split_decay_params(block)
        if dec:
            groups.append({"params": dec, "lr": lr, "weight_decay": wd})
        if nodec:
            groups.append({"params": nodec, "lr": lr, "weight_decay": 0.0})

    head_dec, head_nodec = _split_decay_params(model.classifier)
    if head_dec:
        groups.append({"params": head_dec, "lr": cfg["head_lr"], "weight_decay": wd})
    if head_nodec:
        groups.append({"params": head_nodec, "lr": cfg["head_lr"], "weight_decay": 0.0})

    name = cfg["optimizer"]
    lr_mult = {"sgd": 20.0, "rmsprop": 0.5}.get(name, 1.0)
    for g in groups:
        g["lr"] *= lr_mult
        g["base_lr"] = g["lr"]

    if name == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    if name == "adam":
        return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, alpha=0.99, eps=1e-6, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def tta_views(x: torch.Tensor, n_views: int) -> List[torch.Tensor]:
    views = [x]
    if n_views >= 2:
        views.append(x.flip(3))
    if n_views >= 4:
        views.extend([x.flip(2), x.flip(2).flip(3)])
    if n_views >= 8:
        transposed = x.transpose(2, 3)
        views.extend([
            transposed,
            transposed.flip(3),
            transposed.flip(2),
            transposed.flip(2).flip(3),
        ])
    if x.device.type == "cuda":
        views = [v.contiguous(memory_format=torch.channels_last) for v in views]
    return views


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    X: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    tta: int = 1,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(idx), batch_size):
        xb = normalize_batch(batch_to_device(X, idx[start : start + batch_size], device))
        views = tta_views(xb, tta)
        prob_sum = 0
        for view in views:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(view)
            prob_sum = prob_sum + torch.softmax(logits.float(), dim=1)
        outputs.append((prob_sum / len(views)).cpu())
    return torch.cat(outputs, dim=0).numpy()


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )
    clipped = np.clip(probs[np.arange(len(y_true)), y_true], 1e-9, 1.0)
    nll = float(-np.log(clipped).mean())
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "nll": nll,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    pred: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = rng.integers(0, n, size=(n_boot, n))
    accs = (y_true[samples] == pred[samples]).mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def _make_batch_slices(order: np.ndarray, batch_size: int) -> List[np.ndarray]:
    slices = [order[s : s + batch_size] for s in range(0, len(order), batch_size)]
    if len(slices) > 1 and len(slices[-1]) < 2:
        slices.pop()
    return slices


def train_model(
    cfg: Dict[str, Any],
    tr: Dict[str, np.ndarray],
    va: Dict[str, np.ndarray],
    image_cache: Dict[int, np.ndarray],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    tag: str = "",
    verbose: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    cfg = merge_config(cfg)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    X_img = image_cache[cfg["img_size"]]
    model = build_model(cfg, device)
    total_params, train_params = count_params(model)

    criterion = ImbalanceLoss(
        cfg["loss_type"],
        np.bincount(tr["y"], minlength=NUM_CLASSES),
        power=cfg["class_weight_power"],
        gamma=cfg["focal_gamma"],
        label_smoothing=cfg["label_smoothing"],
        cb_beta=cfg["cb_beta"],
    ).to(device)

    optimizer = make_optimizer(model, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]

    n_samples = len(tr["y"])
    bs = int(cfg["batch_size"])
    steps_per_epoch = len(_make_batch_slices(np.arange(n_samples), bs))
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    if cfg["scheduler"] == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg["plateau_factor"],
            patience=cfg["plateau_patience"],
            min_lr=1e-7,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg['scheduler']}")

    use_sam = cfg["sam_rho"] > 0.0
    amp_enabled = use_amp
    use_scaler = amp_enabled and amp_dtype == torch.float16
    if use_sam and use_scaler:
        amp_enabled, use_scaler = False, False
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = ModelEMA(model, cfg["ema_decay"]) if cfg["ema_decay"] > 0.0 else None

    history = {
        k: []
        for k in (
            "epoch",
            "train_loss",
            "train_acc",
            "val_acc",
            "val_f1",
            "val_nll",
            "lr_head",
            "seconds",
        )
    }
    init_eval = ema.module if ema is not None else model
    best = {
        "acc": -1.0,
        "f1": -1.0,
        "nll": 1e9,
        "epoch": 0,
        "state": {k: v.detach().cpu().clone() for k, v in init_eval.state_dict().items()},
        "metrics": None,
    }
    bad_epochs = 0
    global_step = 0
    t_start = time.time()

    def forward_loss(xn: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor, lam: float):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(xn)
        logits = logits.float()
        if lam >= 1.0:
            loss = criterion(logits, ya)
        else:
            loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
        return logits, loss

    for epoch in range(1, cfg["epochs"] + 1):
        t_ep = time.time()
        model.train()
        set_bn_mode(model, cfg["unfreeze"], cfg["bn_mode"])

        order = sample_epoch_order(tr["y"], cfg["sampling"], rng)
        batches = _make_batch_slices(order, bs)
        run_loss, run_correct, seen = 0.0, 0, 0

        for sel in batches:
            if cfg["scheduler"] == "plateau" and global_step < warmup_steps:
                warm_scale = (global_step + 1) / max(1, warmup_steps)
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"] * warm_scale

            x = batch_to_device(X_img, tr["idx"][sel], device)
            y = torch.from_numpy(tr["y"][sel]).to(device)
            x = augment_batch(x, cfg["aug"])
            x, ya, yb, lam = apply_mix(x, y, cfg)
            xn = normalize_batch(x)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = forward_loss(xn, ya, yb, lam)
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if use_sam:
                grad_params = [
                    p for g in optimizer.param_groups for p in g["params"] if p.grad is not None
                ]
                grad_norm = torch.norm(
                    torch.stack([p.grad.detach().norm(2) for p in grad_params]), 2
                )
                scale = cfg["sam_rho"] / (grad_norm + 1e-12)
                perturbations = []
                with torch.no_grad():
                    for p in grad_params:
                        eps_w = p.grad * scale
                        p.add_(eps_w)
                        perturbations.append(eps_w)

                optimizer.zero_grad(set_to_none=True)
                _, loss_second = forward_loss(xn, ya, yb, lam)
                loss_second.backward()

                with torch.no_grad():
                    for p, eps_w in zip(grad_params, perturbations):
                        p.sub_(eps_w)

            if use_scaler:
                scaler.unscale_(optimizer)
            if cfg["grad_clip"] > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if cfg["scheduler"] == "cosine":
                scheduler.step()
            if ema is not None:
                ema.update(model)

            global_step += 1
            batch_len = len(sel)
            run_loss += float(loss.item()) * batch_len
            run_correct += int((logits.argmax(1) == ya).sum().item())
            seen += batch_len

        eval_model = ema.module if ema is not None else model
        val_probs = predict_probs(
            eval_model, X_img, va["idx"], device, use_amp=use_amp, amp_dtype=amp_dtype, tta=1
        )
        metrics = compute_metrics(va["y"], val_probs)

        if cfg["scheduler"] == "plateau" and global_step >= warmup_steps:
            scheduler.step(metrics["accuracy"])

        history["epoch"].append(epoch)
        history["train_loss"].append(run_loss / max(1, seen))
        history["train_acc"].append(run_correct / max(1, seen))
        history["val_acc"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["macro_f1"])
        history["val_nll"].append(metrics["nll"])
        history["lr_head"].append(optimizer.param_groups[-1]["lr"])
        history["seconds"].append(time.time() - t_ep)

        improved = (metrics["accuracy"] > best["acc"] + 1e-6) or (
            abs(metrics["accuracy"] - best["acc"]) <= 1e-6
            and (
                metrics["macro_f1"] > best["f1"] + 1e-6
                or (
                    abs(metrics["macro_f1"] - best["f1"]) <= 1e-6
                    and metrics["nll"] < best["nll"]
                )
            )
        )
        if improved or best["metrics"] is None:
            best.update(
                acc=metrics["accuracy"],
                f1=metrics["macro_f1"],
                nll=metrics["nll"],
                epoch=epoch,
                metrics=metrics,
                state={k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()},
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose >= 2 or (verbose == 1 and (epoch <= 2 or epoch % 5 == 0)):
            print(
                f"  [{tag}] ep {epoch:03d}/{cfg['epochs']} | loss {history['train_loss'][-1]:.4f} "
                f"| val_acc {metrics['accuracy']:.4f} | val_f1 {metrics['macro_f1']:.4f} "
                f"| best {best['acc']:.4f}@{best['epoch']} | {history['seconds'][-1]:.1f}s"
            )
        if bad_epochs >= cfg["patience"]:
            if verbose:
                print(f"  [{tag}] early stopping at epoch {epoch} (best epoch {best['epoch']}).")
            break

    final_model = ema.module if ema is not None else model
    final_model.load_state_dict(best["state"])
    final_model.eval()

    if ema is not None:
        del model
    del optimizer, scheduler, criterion
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    if verbose:
        print(
            f"  [{tag}] finished: best val_acc={best['acc']:.4f} (epoch {best['epoch']}), "
            f"{elapsed / 60:.1f} min, params {total_params:,} (trainable {train_params:,})"
        )
    return {
        "model": final_model,
        "history": history,
        "best_epoch": best["epoch"],
        "val": best["metrics"],
        "seconds": elapsed,
        "cfg": cfg,
        "n_params": total_params,
    }


def train_or_load(
    tag: str,
    cfg: Dict[str, Any],
    tr: Dict[str, np.ndarray],
    va: Dict[str, np.ndarray],
    image_cache: Dict[int, np.ndarray],
    models_dir: Path,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    resume: bool = True,
    verbose: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"{tag}.pt"

    if resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model({**blob["cfg"], "pretrained": False}, device)
        model.load_state_dict(blob["state"])
        model.eval()
        print(f"[{tag}] loaded checkpoint {ckpt_path.name} (val_acc={blob['val']['accuracy']:.4f})")
        return {
            "model": model,
            "history": blob["history"],
            "best_epoch": blob["best_epoch"],
            "val": blob["val"],
            "seconds": blob["seconds"],
            "cfg": blob["cfg"],
            "n_params": blob.get("n_params", 0),
        }

    res = train_model(
        cfg,
        tr,
        va,
        image_cache,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        tag=tag,
        verbose=verbose,
        seed=seed,
    )
    torch.save(
        {
            "state": res["model"].state_dict(),
            "cfg": res["cfg"],
            "history": res["history"],
            "best_epoch": res["best_epoch"],
            "val": res["val"],
            "seconds": res["seconds"],
            "n_params": res["n_params"],
        },
        ckpt_path,
    )
    return res


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> Tuple[EffNetB0Net, Dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = merge_config({**blob["cfg"], "pretrained": False, "unfreeze": "all"})
    model = build_model(cfg, device)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, cfg


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
