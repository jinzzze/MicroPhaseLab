"""Checkpoint inference, evaluation and visual comparison for U-Net predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from .baseline import binary_metrics
from .torch_dataset import ManifestSegmentationDataset
from .unet import TinyUNet


def _device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(device_name)


def load_checkpoint(
    checkpoint_path: str | Path, *, device: str = "auto"
) -> tuple[TinyUNet, dict[str, Any], torch.device]:
    """Restore a teaching U-Net checkpoint produced by train_model."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    resolved_device = _device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary produced by MicroPhaseLab")
    state = checkpoint.get("model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("Checkpoint is missing model_state_dict or config")
    base_channels = checkpoint.get("base_channels", config.get("base_channels"))
    if not isinstance(base_channels, int) or base_channels <= 0:
        raise ValueError("Checkpoint has an invalid base_channels value")
    model = TinyUNet(base_channels=base_channels).to(resolved_device)
    model.load_state_dict(state)
    model.eval()
    return model, config, resolved_device


def predict_manifest(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.5,
    batch_size: int = 4,
    device: str = "auto",
) -> dict[str, object]:
    """Write thresholded 0/1 masks at original image resolution."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model, checkpoint_config, resolved_device = load_checkpoint(checkpoint_path, device=device)
    image_size = checkpoint_config.get("image_size")
    if not isinstance(image_size, int) or image_size < 32:
        raise ValueError("Checkpoint has an invalid image_size value")

    dataset = ManifestSegmentationDataset(manifest_path, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    destination = Path(output_dir)
    predictions_dir = destination / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(resolved_device)
            probabilities = torch.sigmoid(model(images))
            image_ids = batch["image_id"]
            group_ids = batch["group_id"]
            image_paths = batch["image_path"] if "image_path" in batch else None
            mask_paths = batch["mask_path"] if "mask_path" in batch else None
            if image_paths is not None or mask_paths is not None:
                raise RuntimeError("Unexpected metadata in dataset batch")
            for index, image_id in enumerate(image_ids):
                source_row = dataset.manifest.iloc[len(rows)]
                target_path = Path(source_row["mask_path"])
                with Image.open(target_path) as target_file:
                    target_width, target_height = target_file.size
                resized = functional.interpolate(
                    probabilities[index : index + 1],
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
                prediction = (resized >= threshold).to(torch.uint8).squeeze().cpu().numpy()
                prediction_path = predictions_dir / f"{image_id}_unet_mask.png"
                Image.fromarray(prediction, mode="L").save(prediction_path)
                rows.append(
                    {
                        "image_id": image_id,
                        "group_id": group_ids[index],
                        "image_path": source_row["image_path"],
                        "mask_path": source_row["mask_path"],
                        "prediction_path": str(prediction_path.resolve()),
                    }
                )

    pd.DataFrame(rows).to_csv(destination / "prediction_manifest.csv", index=False)
    return {
        "images": len(rows),
        "threshold": threshold,
        "device": str(resolved_device),
        "output_dir": str(destination.resolve()),
    }


def _read_prediction(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction mask does not exist: {path}")
    with Image.open(path) as image_file:
        prediction = np.asarray(image_file.convert("L"))
    values = set(np.unique(prediction).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Prediction {path} must contain only 0 and 1; got {sorted(values)}")
    return prediction


def evaluate_predictions(
    manifest_path: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Evaluate prediction masks against the expert masks recorded in a manifest."""
    manifest = pd.read_csv(manifest_path)
    required = {"image_id", "image_path", "mask_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if manifest.empty:
        raise ValueError("Manifest is empty")
    prediction_root = Path(predictions_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}

    for _, row in manifest.iterrows():
        with Image.open(row["mask_path"]) as target_file:
            target = np.asarray(target_file.convert("L"))
        prediction = _read_prediction(prediction_root / f"{row['image_id']}_unet_mask.png")
        metrics = binary_metrics(target > 0, prediction > 0)
        for key in totals:
            totals[key] += int(metrics[key])
        rows.append({"image_id": row["image_id"], **metrics})

    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "metrics_per_image.csv", index=False)
    true_positive = totals["true_positive"]
    false_positive = totals["false_positive"]
    false_negative = totals["false_negative"]
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    summary: dict[str, object] = {
        "images": len(frame),
        "mean_dice": float(frame["dice"].mean()),
        "mean_iou": float(frame["iou"].mean()),
        "mean_precision": float(frame["precision"].mean()),
        "mean_recall": float(frame["recall"].mean()),
        "mean_area_fraction_absolute_error": float(
            frame["area_fraction_absolute_error"].mean()
        ),
        "micro_dice": float(
            1.0 if dice_denominator == 0 else 2 * true_positive / dice_denominator
        ),
        "micro_iou": float(1.0 if union == 0 else true_positive / union),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def save_prediction_overlays(
    manifest_path: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 20,
) -> int:
    """Save image, expert, prediction and error comparison figures."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/microphaselab-matplotlib")
    import matplotlib.pyplot as plt

    manifest = pd.read_csv(manifest_path).head(limit)
    prediction_root = Path(predictions_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    for _, row in manifest.iterrows():
        with Image.open(row["image_path"]) as image_file:
            image = np.asarray(image_file.convert("L"))
        with Image.open(row["mask_path"]) as target_file:
            target = np.asarray(target_file.convert("L")) > 0
        prediction = _read_prediction(prediction_root / f"{row['image_id']}_unet_mask.png") > 0
        false_positive = np.logical_and(prediction, ~target)
        false_negative = np.logical_and(~prediction, target)
        error = np.zeros((*target.shape, 3), dtype=np.float32)
        error[false_positive] = (1.0, 0.0, 0.0)
        error[false_negative] = (0.0, 1.0, 1.0)

        figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("SEM image")
        axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Expert mask")
        axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("U-Net prediction")
        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(error, alpha=0.75)
        axes[3].set_title("Error: red=FP, cyan=FN")
        for axis in axes:
            axis.axis("off")
        figure.savefig(destination / f"{row['image_id']}_comparison.png", dpi=150)
        plt.close(figure)
        written += 1
    return written
