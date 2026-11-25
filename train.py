import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, StratifiedKFold

from src.config import (
    BASE_SIZE,
    CLASSES,
    HIGH_SIZE,
    IMAGE_SIZES,
    NUM_CLASSES,
    config_diff,
    get_budget,
    get_device_config,
    set_seed,
    to_python_scalar,
)
from src.data import (
    build_or_load_cache,
    collect_dataset_records,
    df_to_arrays,
    find_dataset_dir,
    make_or_load_splits,
)
from src.engine import compute_metrics, free_memory, predict_probs, train_model, train_or_load
from src.evaluation import (
    evaluate_model_on_test,
    plot_history,
    register_test_result,
    save_classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

SEARCH_KEYS = [
    "backbone_lr",
    "head_lr",
    "weight_decay",
    "dropout",
    "hidden_dim",
    "unfreeze",
    "bn_mode",
    "batch_size",
    "optimizer",
    "sam_rho",
    "scheduler",
    "loss_type",
    "focal_gamma",
    "class_weight_power",
    "cb_beta",
    "label_smoothing",
    "sampling",
    "epochs",
    "patience",
]


class EffNetSearchEstimator(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        backbone_lr=1e-4,
        head_lr=1e-3,
        weight_decay=1e-4,
        dropout=0.5,
        hidden_dim=256,
        unfreeze="all",
        bn_mode="train",
        batch_size=32,
        optimizer="adamw",
        sam_rho=0.0,
        scheduler="cosine",
        loss_type="ce",
        focal_gamma=2.0,
        class_weight_power=0.5,
        cb_beta=0.999,
        label_smoothing=0.0,
        sampling="none",
        epochs=25,
        patience=8,
        fixed_cfg=None,
        pool_data=None,
        val_data=None,
        n_pool_train=0,
        image_cache=None,
        cache_dir=None,
        device=None,
        use_amp=False,
        amp_dtype=torch.float16,
        resume=True,
        seed=42,
    ):
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.unfreeze = unfreeze
        self.bn_mode = bn_mode
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.sam_rho = sam_rho
        self.scheduler = scheduler
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.class_weight_power = class_weight_power
        self.cb_beta = cb_beta
        self.label_smoothing = label_smoothing
        self.sampling = sampling
        self.epochs = epochs
        self.patience = patience
        self.fixed_cfg = fixed_cfg
        self.pool_data = pool_data
        self.val_data = val_data
        self.n_pool_train = n_pool_train
        self.image_cache = image_cache
        self.cache_dir = cache_dir
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.resume = resume
        self.seed = seed

    def _build_cfg(self) -> Dict[str, Any]:
        base = dict(self.fixed_cfg or {})
        return {**base, **{k: to_python_scalar(getattr(self, k)) for k in SEARCH_KEYS}}

    def fit(self, X, y):
        rows = np.asarray(X).reshape(-1).astype(int)
        if (rows >= self.n_pool_train).any():
            raise RuntimeError("Validation indices leaked into training fold.")

        cfg = self._build_cfg()
        row_hash = hashlib.md5(rows.tobytes()).hexdigest()[:8]
        payload = json.dumps({"cfg": cfg, "seed": self.seed, "rows": row_hash}, sort_keys=True, default=str)
        key = hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]
        cache_file = Path(self.cache_dir) / f"{key}.json"

        if self.resume and cache_file.exists():
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
            self.val_pred_ = np.asarray(blob["val_pred"], dtype=np.int64)
            self.val_metrics_ = blob["val_metrics"]
            self.best_epoch_ = int(blob["best_epoch"])
            print(f"  [cached trial] val_acc={blob['val_metrics']['accuracy']:.4f}")
        else:
            t0 = time.time()
            tr_sub = {"idx": self.pool_data["idx"][rows], "y": self.pool_data["y"][rows]}
            try:
                res = train_model(
                    cfg,
                    tr_sub,
                    self.val_data,
                    self.image_cache,
                    device=self.device,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    tag="C-trial",
                    verbose=1,
                    seed=self.seed,
                )
                probs = predict_probs(
                    res["model"],
                    self.image_cache[BASE_SIZE],
                    self.val_data["idx"],
                    device=self.device,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    tta=1,
                )
            except Exception:
                traceback.print_exc()
                free_memory()
                raise

            self.val_pred_ = probs.argmax(1).astype(np.int64)
            self.val_metrics_ = res["val"]
            self.best_epoch_ = int(res["best_epoch"])
            cache_file.write_text(
                json.dumps(
                    {
                        "cfg": cfg,
                        "val_pred": self.val_pred_.tolist(),
                        "val_metrics": res["val"],
                        "best_epoch": self.best_epoch_,
                        "minutes": (time.time() - t0) / 60.0,
                    },
                    default=str,
                ),
                encoding="utf-8",
            )
            del res
            free_memory()

        self.classes_ = np.arange(NUM_CLASSES)
        return self

    def predict(self, X):
        rows = np.asarray(X).reshape(-1)
        if len(rows) != len(self.val_pred_):
            raise RuntimeError("predict() only supports the validation fold in PredefinedSplit.")
        return self.val_pred_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate EfficientNet-B0 on 6-class eggplant leaf disease data."
    )
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help="Experiment stage to run (default: all). Note: D uses best config from C if available.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Path to Original_dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test with tiny budget.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and retrain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume = not args.no_resume
    set_seed(args.seed)

    device, use_amp, amp_dtype = get_device_config()
    budget = get_budget(args.smoke_test)

    out_tag = "smoke" if args.smoke_test else "full"
    results_dir = Path(f"results_6class_{out_tag}")
    models_dir = Path(f"models_6class_{out_tag}")
    cache_dir = Path("cache_6class")
    search_cache_dir = results_dir / "search_trials"
    for folder in (results_dir, models_dir, cache_dir, search_cache_dir):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Python  : {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch : {torch.__version__} | Device: {device} | AMP: {use_amp} ({amp_dtype})")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU     : {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    print(f"Mode    : {out_tag} | Resume: {resume}")

    source_dir = find_dataset_dir(args.dataset_dir)
    print(f"Dataset : {source_dir}")
    df_all = collect_dataset_records(source_dir)
    image_cache, valid_mask = build_or_load_cache(df_all["path"].tolist(), cache_dir, IMAGE_SIZES)

    df_all["valid"] = valid_mask
    bad_df = df_all[~df_all["valid"]]
    if len(bad_df):
        bad_df.to_csv(results_dir / "bad_images.csv", index=False)
        print(f"Skipped {len(bad_df)} unreadable images.")

    df = df_all[df_all["valid"]].copy()
    if budget["per_class_cap"]:
        df = df.groupby("label", group_keys=False).head(budget["per_class_cap"])
    df = df.reset_index(drop=True)

    train_df, val_df, test_df = make_or_load_splits(df, results_dir, seed=args.seed, resume=resume)
    print(f"Split   : train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    TR = df_to_arrays(train_df)
    VA = df_to_arrays(val_df)
    TE = df_to_arrays(test_df)
    DEV = {
        "idx": np.concatenate([TR["idx"], VA["idx"]]),
        "y": np.concatenate([TR["y"], VA["y"]]),
    }

    results_store: Dict[str, Any] = {}
    cfg_A = {
        "pretrained": True,
        "img_size": BASE_SIZE,
        "head_type": "paper",
        "hidden_dim": 256,
        "dropout": 0.5,
        "unfreeze": "all",
        "bn_mode": "train",
        "optimizer": "adamw",
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "warmup_epochs": 2,
        "batch_size": 32,
        "epochs": budget["A_epochs"],
        "patience": 12,
        "loss_type": "ce",
        "sampling": "none",
        "aug": "basic",
    }

    if args.experiment in ("A", "all"):
        print("\nExperiment A: Pretrained EfficientNet-B0 baseline")
        res_A = train_or_load(
            "A_pretrained", cfg_A, TR, VA, image_cache, models_dir,
            device, use_amp, amp_dtype, resume=resume, seed=args.seed,
        )
        plot_history(res_A, "Experiment A (Pretrained)", results_dir / "A_history.png")
        evaluate_model_on_test(
            "A_pretrained", res_A, TE, image_cache, results_store,
            results_dir, device, use_amp, amp_dtype, seed=args.seed,
        )
        free_memory()

    if args.experiment in ("B", "all"):
        print("\nExperiment B: Training EfficientNet-B0 from scratch")
        cfg_B = {
            **cfg_A,
            "pretrained": False,
            "backbone_lr": 1e-3,
            "head_lr": 1e-3,
            "warmup_epochs": 5,
            "epochs": budget["B_epochs"],
            "patience": 30,
        }
        res_B = train_or_load(
            "B_scratch", cfg_B, TR, VA, image_cache, models_dir,
            device, use_amp, amp_dtype, resume=resume, seed=args.seed,
        )
        plot_history(res_B, "Experiment B (From Scratch)", results_dir / "B_history.png")
        evaluate_model_on_test(
            "B_scratch", res_B, TE, image_cache, results_store,
            results_dir, device, use_amp, amp_dtype, seed=args.seed,
        )
        free_memory()

    res_C = None
    if args.experiment in ("C", "all"):
        print("\nExperiment C: Hyperparameter & imbalance search (RandomizedSearchCV)")
        c_fixed = {
            "pretrained": True,
            "img_size": BASE_SIZE,
            "head_type": "paper",
            "aug": "basic",
            "warmup_epochs": 2,
        }
        n_pool_train = len(TR["y"])
        c_X = np.arange(len(DEV["y"])).reshape(-1, 1)
        c_split = PredefinedSplit(
            test_fold=np.concatenate([np.full(n_pool_train, -1), np.zeros(len(VA["y"]), dtype=int)])
        )
        param_dist = {
            "backbone_lr": loguniform(3e-5, 1e-3),
            "head_lr": loguniform(2e-4, 5e-3),
            "weight_decay": loguniform(1e-6, 5e-2),
            "dropout": uniform(0.10, 0.50),
            "hidden_dim": [128, 256, 512],
            "unfreeze": ["all", "from3", "from5"],
            "bn_mode": ["train", "freeze"],
            "batch_size": [16, 32],
            "optimizer": ["adamw", "adam", "sgd", "rmsprop"],
            "sam_rho": [0.0, 0.0, 0.05],
            "scheduler": ["cosine", "plateau"],
            "loss_type": ["ce", "weighted_ce", "cb_ce", "focal", "weighted_focal", "cb_focal"],
            "focal_gamma": [0.5, 1.0, 2.0, 3.0],
            "class_weight_power": [0.25, 0.5, 0.75, 1.0],
            "cb_beta": [0.99, 0.999, 0.9999],
            "label_smoothing": [0.0, 0.05, 0.1],
            "sampling": ["none", "sqrt", "balanced"],
        }

        estimator = EffNetSearchEstimator(
            epochs=budget["C_search_epochs"],
            patience=8,
            fixed_cfg=c_fixed,
            pool_data=DEV,
            val_data=VA,
            n_pool_train=n_pool_train,
            image_cache=image_cache,
            cache_dir=search_cache_dir,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            resume=resume,
            seed=args.seed,
        )
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=budget["C_n_iter"],
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            refit=False,
            cv=c_split,
            random_state=47,
            n_jobs=1,
            verbose=1,
            error_score=np.nan,
            return_train_score=False,
        )
        search.fit(c_X, DEV["y"])

        cv_df = pd.DataFrame(search.cv_results_)
        if cv_df["mean_test_accuracy"].isna().all():
            raise RuntimeError("All hyperparameter search trials failed.")
        cv_df = cv_df.sort_values(
            ["mean_test_accuracy", "mean_test_macro_f1"], ascending=False
        ).reset_index(drop=True)
        cv_df.drop(columns=["params"]).to_csv(
            results_dir / "C_randomized_search_results.csv", index=False
        )

        final_runs = []
        for k in range(min(budget["C_topk"], len(cv_df))):
            params = {kk: to_python_scalar(vv) for kk, vv in cv_df.loc[k, "params"].items()}
            cfg_k = {**c_fixed, **params, "epochs": budget["C_final_epochs"], "patience": 15}
            print(f"\nRefitting Top-{k + 1} candidate (search val_acc={cv_df.loc[k, 'mean_test_accuracy']:.4f})")
            run_k = train_or_load(
                f"C_final_rank{k + 1}", cfg_k, TR, VA, image_cache, models_dir,
                device, use_amp, amp_dtype, resume=resume, seed=args.seed + k,
            )
            final_runs.append(run_k)
            free_memory()

        best_idx = max(
            range(len(final_runs)),
            key=lambda i: (final_runs[i]["val"]["accuracy"], final_runs[i]["val"]["macro_f1"]),
        )
        res_C = final_runs[best_idx]
        with open(results_dir / "C_best_cfg.json", "w", encoding="utf-8") as f:
            json.dump({k: to_python_scalar(v) for k, v in res_C["cfg"].items()}, f, indent=2)

        plot_history(res_C, "Experiment C (Best Config)", results_dir / "C_history.png")
        evaluate_model_on_test(
            "C_search", res_C, TE, image_cache, results_store,
            results_dir, device, use_amp, amp_dtype, seed=args.seed,
        )
        free_memory()

    if args.experiment in ("D", "all"):
        print("\nExperiment D: Greedy architectural ablation & 5-fold ensemble")
        if res_C is None:
            best_c_path = results_dir / "C_best_cfg.json"
            if best_c_path.exists():
                with open(best_c_path, "r", encoding="utf-8") as f:
                    base_d_cfg = json.load(f)
            else:
                base_d_cfg = dict(cfg_A)
            res_C = train_or_load(
                "D0_base", {**base_d_cfg, "epochs": budget["D_epochs"]}, TR, VA,
                image_cache, models_dir, device, use_amp, amp_dtype, resume=resume, seed=args.seed,
            )
            evaluate_model_on_test(
                "C_search", res_C, TE, image_cache, results_store,
                results_dir, device, use_amp, amp_dtype, seed=args.seed,
            )

        stages = [
            ("D1 strong aug + MixUp/CutMix", lambda c: {"aug": "strong", "mix_prob": 0.5}),
            ("D2 320px input resolution", lambda c: {"img_size": HIGH_SIZE}),
            (
                "D3 MSF-GeM head + drop_path",
                lambda c: {
                    "head_type": "msf_gem",
                    "hidden_dim": 512,
                    "dropout": max(float(c["dropout"]), 0.3),
                    "drop_path": 0.3,
                },
            ),
            (
                "D4 full fine-tuning + layer-wise LR decay",
                lambda c: {"unfreeze": "all", "bn_mode": "train", "lr_decay": 0.85},
            ),
            ("D5 weight EMA", lambda c: {"ema_decay": 0.998}),
            (
                "D6 longer cosine schedule",
                lambda c: {
                    "scheduler": "cosine",
                    "epochs": budget["D_longer_epochs"],
                    "patience": 25,
                },
            ),
        ]

        cur_cfg = {**res_C["cfg"], "epochs": budget["D_epochs"]}
        cur_val = (res_C["val"]["accuracy"], res_C["val"]["macro_f1"])
        best_D_res = res_C
        stage_rows = [
            {
                "stage": "D0 base (best C config)",
                "accepted": True,
                "val_acc": cur_val[0],
                "val_f1": cur_val[1],
                "best_epoch": res_C["best_epoch"],
                "minutes": res_C["seconds"] / 60.0,
            }
        ]

        for idx, (stage_name, override_fn) in enumerate(stages, start=1):
            cand_cfg = {**cur_cfg, **override_fn(cur_cfg)}
            diff = {k: v for k, v in cand_cfg.items() if cur_cfg.get(k) != v}
            print(f"\nStage {stage_name} | overrides: {diff}")
            run_stage = train_or_load(
                f"D{idx}_stage", cand_cfg, TR, VA, image_cache, models_dir,
                device, use_amp, amp_dtype, resume=resume, seed=args.seed,
            )
            val_score = (run_stage["val"]["accuracy"], run_stage["val"]["macro_f1"])
            accepted = val_score >= cur_val
            stage_rows.append(
                {
                    "stage": stage_name,
                    "accepted": accepted,
                    "val_acc": val_score[0],
                    "val_f1": val_score[1],
                    "best_epoch": run_stage["best_epoch"],
                    "minutes": run_stage["seconds"] / 60.0,
                }
            )
            status = "accepted" if accepted else "rejected"
            print(
                f"  val_acc={val_score[0]:.4f}, macro_f1={val_score[1]:.4f} "
                f"(baseline {cur_val[0]:.4f} / {cur_val[1]:.4f}) -> {status}"
            )
            if accepted:
                cur_cfg, cur_val = dict(run_stage["cfg"]), val_score
                best_D_res = run_stage
            else:
                del run_stage
            free_memory()

        pd.DataFrame(stage_rows).to_csv(results_dir / "D_stage_ablation.csv", index=False)
        with open(results_dir / "D_best_cfg.json", "w", encoding="utf-8") as f:
            json.dump({k: to_python_scalar(v) for k, v in cur_cfg.items()}, f, indent=2)
        print("Selected D configuration:", config_diff(cur_cfg))

        evaluate_model_on_test(
            "D_single", best_D_res, TE, image_cache, results_store,
            results_dir, device, use_amp, amp_dtype, seed=args.seed,
        )
        plot_history(best_D_res, "Experiment D (Single Model)", results_dir / "D_single_history.png")

        skf = StratifiedKFold(n_splits=budget["D_folds"], shuffle=True, random_state=args.seed)
        size_d = cur_cfg["img_size"]
        fold_p1, fold_pt, fold_rows = [], [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(DEV["idx"], DEV["y"]), start=1):
            tr_f = {"idx": DEV["idx"][tr_idx], "y": DEV["y"][tr_idx]}
            va_f = {"idx": DEV["idx"][va_idx], "y": DEV["y"][va_idx]}
            print(f"\nFold {fold_idx}/{budget['D_folds']} (train={len(tr_idx)}, val={len(va_idx)})")
            run_f = train_or_load(
                f"D_fold{fold_idx}", cur_cfg, tr_f, va_f, image_cache, models_dir,
                device, use_amp, amp_dtype, resume=resume, seed=args.seed + 100 + fold_idx,
            )
            p1 = predict_probs(
                run_f["model"], image_cache[size_d], TE["idx"],
                device=device, use_amp=use_amp, amp_dtype=amp_dtype, tta=1,
            )
            pt = predict_probs(
                run_f["model"], image_cache[size_d], TE["idx"],
                device=device, use_amp=use_amp, amp_dtype=amp_dtype, tta=8,
            )
            fold_p1.append(p1)
            fold_pt.append(pt)
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "val_acc": run_f["val"]["accuracy"],
                    "val_f1": run_f["val"]["macro_f1"],
                    "best_epoch": run_f["best_epoch"],
                    "minutes": run_f["seconds"] / 60.0,
                    "test_acc": compute_metrics(TE["y"], p1)["accuracy"],
                    "test_acc_tta": compute_metrics(TE["y"], pt)["accuracy"],
                }
            )
            del run_f
            free_memory()

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(results_dir / "D_fold_results.csv", index=False)

        ens_entry = register_test_result(
            results_store,
            "D_ensemble",
            TE["y"],
            np.mean(fold_p1, axis=0),
            np.mean(fold_pt, axis=0),
            seed=args.seed,
            extra={"folds": fold_rows, "cfg": cur_cfg},
        )
        m1, mt = ens_entry["metrics"], ens_entry["metrics_tta"]
        print("\n[D_ensemble] Test evaluation:")
        print(
            f"  Ensemble (1 view): acc={m1['accuracy']:.4f} macro-F1={m1['macro_f1']:.4f} "
            f"balanced-acc={m1['balanced_accuracy']:.4f}"
        )
        print(
            f"  Ensemble + TTA x8: acc={mt['accuracy']:.4f} macro-F1={mt['macro_f1']:.4f} "
            f"balanced-acc={mt['balanced_accuracy']:.4f} "
            f"(95% CI {ens_entry['ci_tta'][0]:.3f}-{ens_entry['ci_tta'][1]:.3f})"
        )
        save_classification_report(
            "D_ensemble", ens_entry, TE["y"], results_dir, use_tta=True, plot=True
        )

    if len(results_store) > 1:
        summary_rows = []
        for key, entry in results_store.items():
            summary_rows.append(
                {
                    "experiment": key,
                    "acc_single": entry["metrics"]["accuracy"],
                    "ci_95": f"{entry['ci'][0]:.3f}-{entry['ci'][1]:.3f}",
                    "macro_f1": entry["metrics"]["macro_f1"],
                    "balanced_acc": entry["metrics"]["balanced_accuracy"],
                    "acc_tta8": entry["metrics_tta"]["accuracy"],
                    "macro_f1_tta8": entry["metrics_tta"]["macro_f1"],
                }
            )
        summary_df = pd.DataFrame(summary_rows).set_index("experiment")
        summary_df.to_csv(results_dir / "FINAL_comparison.csv")
        print("\nSummary across evaluated experiments:")
        print(summary_df.round(4).to_string())


if __name__ == "__main__":
    main()
