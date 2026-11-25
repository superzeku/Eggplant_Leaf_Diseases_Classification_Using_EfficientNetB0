from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .config import CLASSES, NUM_CLASSES
from .engine import bootstrap_ci, compute_metrics, predict_probs


def plot_confusion(
    y_true: np.ndarray,
    pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> np.ndarray:
    cm = confusion_matrix(y_true, pred, labels=np.arange(NUM_CLASSES))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title(f"{title} (Counts)")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title(f"{title} (Row-Normalized Recall)")

    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return cm


def plot_history(
    res: Dict[str, Any],
    title: str,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    h = pd.DataFrame(res["history"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(h["epoch"], h["train_loss"], label="train loss")
    axes[0].set_title(f"{title} - Train Loss")

    axes[1].plot(h["epoch"], h["train_acc"], label="train acc")
    axes[1].plot(h["epoch"], h["val_acc"], label="val acc")
    axes[1].axvline(res["best_epoch"], color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Accuracy")

    axes[2].plot(h["epoch"], h["val_f1"], label="val macro-F1", color="tab:green")
    norm_lr = h["lr_head"] / max(float(h["lr_head"].max()), 1e-12)
    axes[2].plot(h["epoch"], norm_lr, label="lr (normalized)", color="tab:orange")
    axes[2].set_title("Validation Macro-F1 / LR")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def register_test_result(
    results_store: Dict[str, Any],
    name: str,
    y_test: np.ndarray,
    probs_single: np.ndarray,
    probs_tta: np.ndarray,
    seed: int = 42,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "probs": probs_single,
        "probs_tta": probs_tta,
        "metrics": compute_metrics(y_test, probs_single),
        "metrics_tta": compute_metrics(y_test, probs_tta),
        "ci": bootstrap_ci(y_test, probs_single.argmax(1), seed=seed),
        "ci_tta": bootstrap_ci(y_test, probs_tta.argmax(1), seed=seed),
        "extra": extra or {},
    }
    results_store[name] = entry
    return entry


def save_classification_report(
    name: str,
    entry: Dict[str, Any],
    y_test: np.ndarray,
    results_dir: Path,
    use_tta: bool = False,
    plot: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    probs = entry["probs_tta"] if use_tta else entry["probs"]
    pred = probs.argmax(1)
    suffix = "tta" if use_tta else "single"

    report_dict = classification_report(
        y_test,
        pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(results_dir / f"{name}_{suffix}_report.csv")

    if plot:
        title = f"{name} (TTA x8)" if use_tta else name
        plot_confusion(
            y_test,
            pred,
            title=title,
            save_path=results_dir / f"{name}_{suffix}_confusion.png",
            show=show,
        )
    return report_df


def evaluate_model_on_test(
    name: str,
    res: Dict[str, Any],
    test_split: Dict[str, np.ndarray],
    image_cache: Dict[int, np.ndarray],
    results_store: Dict[str, Any],
    results_dir: Path,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    tta: int = 8,
    seed: int = 42,
    plot: bool = True,
    show: bool = False,
) -> Dict[str, Any]:
    img_size = res["cfg"]["img_size"]
    probs_1 = predict_probs(
        res["model"],
        image_cache[img_size],
        test_split["idx"],
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        tta=1,
    )
    probs_tta = predict_probs(
        res["model"],
        image_cache[img_size],
        test_split["idx"],
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        tta=tta,
    )

    entry = register_test_result(
        results_store,
        name,
        test_split["y"],
        probs_1,
        probs_tta,
        seed=seed,
        extra={
            "best_epoch": res["best_epoch"],
            "val": res["val"],
            "seconds": res["seconds"],
            "cfg": res["cfg"],
        },
    )

    m1, mt = entry["metrics"], entry["metrics_tta"]
    print(f"\n[{name}] Test evaluation:")
    print(
        f"  Single view: acc={m1['accuracy']:.4f} (95% CI {entry['ci'][0]:.3f}-{entry['ci'][1]:.3f}) "
        f"macro-F1={m1['macro_f1']:.4f} balanced-acc={m1['balanced_accuracy']:.4f}"
    )
    print(
        f"  TTA x{tta:<6d}: acc={mt['accuracy']:.4f} (95% CI {entry['ci_tta'][0]:.3f}-{entry['ci_tta'][1]:.3f}) "
        f"macro-F1={mt['macro_f1']:.4f} balanced-acc={mt['balanced_accuracy']:.4f}"
    )
    save_classification_report(
        name, entry, test_split["y"], results_dir, use_tta=False, plot=plot, show=show
    )
    return entry
