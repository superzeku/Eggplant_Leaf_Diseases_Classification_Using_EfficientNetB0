import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import CLASSES, IDX_TO_CLASS, get_device_config
from src.data import IMAGE_EXTS, normalize_batch
from src.engine import load_checkpoint_model, tta_views


def resolve_images(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(
            p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No supported image files found in {target}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {target}")


def resolve_checkpoints(paths: List[Path]) -> List[Path]:
    ckpts: List[Path] = []
    for p in paths:
        if p.is_file():
            ckpts.append(p)
        elif p.is_dir():
            ckpts.extend(sorted(p.glob("*.pt")))
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {p}")
    if not ckpts:
        raise FileNotFoundError("No .pt checkpoint files found.")
    return ckpts


def load_image_tensor(path: Path, img_size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.uint8)
    tensor = torch.from_numpy(arr).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    return normalize_batch(tensor)


@torch.no_grad()
def predict_single_model(
    model: torch.nn.Module,
    image_paths: List[Path],
    img_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    tta: int,
) -> np.ndarray:
    probs_all = []
    for img_path in image_paths:
        xb = load_image_tensor(img_path, img_size, device)
        views = tta_views(xb, tta)
        prob_sum = 0
        for view in views:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(view)
            prob_sum = prob_sum + torch.softmax(logits.float(), dim=1)
        probs_all.append((prob_sum / len(views)).squeeze(0).cpu().numpy())
    return np.stack(probs_all, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on an eggplant leaf image or folder using trained checkpoints."
    )
    parser.add_argument("input", type=Path, help="Path to an image file or a directory of images.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more .pt checkpoint files (or a directory of fold checkpoints for ensemble).",
    )
    parser.add_argument(
        "--tta",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="Number of test-time augmentation views (default: 1).",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes to display.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV path to save predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, use_amp, amp_dtype = get_device_config()
    image_paths = resolve_images(args.input)
    ckpt_paths = resolve_checkpoints(args.checkpoint)

    print(f"Running inference on {len(image_paths)} image(s) with {len(ckpt_paths)} checkpoint(s) on {device}")

    ensemble_probs = []
    for ckpt in ckpt_paths:
        model, cfg = load_checkpoint_model(ckpt, device)
        probs = predict_single_model(
            model,
            image_paths,
            img_size=int(cfg["img_size"]),
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            tta=args.tta,
        )
        ensemble_probs.append(probs)
        del model

    mean_probs = np.mean(ensemble_probs, axis=0)
    top_k = min(max(1, args.top_k), len(CLASSES))

    records = []
    for path, prob_vec in zip(image_paths, mean_probs):
        order = np.argsort(prob_vec)[::-1]
        pred_idx = int(order[0])
        pred_label = IDX_TO_CLASS[pred_idx]
        confidence = float(prob_vec[pred_idx])

        top_summary = ", ".join(
            f"{IDX_TO_CLASS[int(i)]}: {prob_vec[int(i)] * 100:.1f}%" for i in order[:top_k]
        )
        print(f"{path.name}: {pred_label} ({confidence * 100:.2f}%) | [{top_summary}]")

        row = {
            "image": str(path),
            "predicted_class": pred_label,
            "confidence": round(confidence, 6),
        }
        for cls_idx, cls_name in enumerate(CLASSES):
            row[f"prob_{cls_name}"] = round(float(prob_vec[cls_idx]), 6)
        records.append(row)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
