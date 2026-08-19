"""Classical Otsu and morphology baseline for binary MA segmentation."""

from __future__ import annotations

import json
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    closing,
    disk,
    opening,
    remove_small_holes,
    remove_small_objects,
)


def _remove_small_objects(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    if "max_size" in signature(remove_small_objects).parameters:
        return remove_small_objects(mask, max_size=max(0, minimum_size - 1))
    return remove_small_objects(mask, min_size=minimum_size)


def _remove_small_holes(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    if "max_size" in signature(remove_small_holes).parameters:
        return remove_small_holes(mask, max_size=max(0, minimum_size - 1))
    return remove_small_holes(mask, area_threshold=minimum_size)


def segment_otsu_morphology(
    image: np.ndarray,
    *,
    gaussian_sigma: float = 1.0,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_object_size: int = 32,
    min_hole_size: int = 32,
) -> tuple[np.ndarray, float]:
    """Segment bright MA candidates and return a 0/1 mask plus Otsu threshold."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image; got shape={image.shape}")
    if gaussian_sigma < 0:
        raise ValueError("gaussian_sigma cannot be negative")
    if min(opening_radius, closing_radius, min_object_size, min_hole_size) < 0:
        raise ValueError("Morphology sizes cannot be negative")

    normalized = image.astype(np.float32)
    if normalized.size == 0:
        raise ValueError("Image cannot be empty")
    normalized -= float(normalized.min())
    maximum = float(normalized.max())
    if maximum > 0:
        normalized /= maximum
    smoothed = gaussian(normalized, sigma=gaussian_sigma, preserve_range=True)
    threshold = float(threshold_otsu(smoothed)) if np.ptp(smoothed) > 0 else 1.0
    prediction = smoothed > threshold

    if opening_radius:
        prediction = opening(prediction, disk(opening_radius))
    if closing_radius:
        prediction = closing(prediction, disk(closing_radius))
    if min_object_size:
        prediction = _remove_small_objects(prediction, min_object_size)
    if min_hole_size:
        prediction = _remove_small_holes(prediction, min_hole_size)
    return prediction.astype(np.uint8), threshold


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Calculate pixel metrics with explicit behavior for empty masks."""
    target_bool = np.asarray(target, dtype=bool)
    prediction_bool = np.asarray(prediction, dtype=bool)
    if target_bool.shape != prediction_bool.shape:
        raise ValueError(
            f"Target and prediction shapes differ: {target_bool.shape} != {prediction_bool.shape}"
        )

    true_positive = int(np.logical_and(target_bool, prediction_bool).sum())
    false_positive = int(np.logical_and(~target_bool, prediction_bool).sum())
    false_negative = int(np.logical_and(target_bool, ~prediction_bool).sum())
    target_positive = true_positive + false_negative
    predicted_positive = true_positive + false_positive
    union = true_positive + false_positive + false_negative

    dice_denominator = target_positive + predicted_positive
    dice = 1.0 if dice_denominator == 0 else 2 * true_positive / dice_denominator
    iou = 1.0 if union == 0 else true_positive / union
    precision = (
        1.0
        if predicted_positive == 0 and target_positive == 0
        else (true_positive / predicted_positive if predicted_positive else 0.0)
    )
    recall = 1.0 if target_positive == 0 else true_positive / target_positive
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "target_fraction": float(target_bool.mean()),
        "prediction_fraction": float(prediction_bool.mean()),
        "area_fraction_absolute_error": float(abs(target_bool.mean() - prediction_bool.mean())),
    }


def run_baseline(
    manifest_path: Path,
    output_dir: Path,
    *,
    gaussian_sigma: float = 1.0,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_object_size: int = 32,
    min_hole_size: int = 32,
) -> dict[str, object]:
    """Run the baseline for every manifest row and save predictions and metrics."""
    manifest = pd.read_csv(manifest_path)
    required = {"image_id", "image_path", "mask_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Manifest is missing columns: {missing}")
    if manifest.empty:
        raise ValueError("Manifest is empty")

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}

    for _, row in manifest.iterrows():
        with Image.open(row["image_path"]) as image_file:
            image = np.asarray(image_file.convert("L"))
        with Image.open(row["mask_path"]) as mask_file:
            target = np.asarray(mask_file) > 0
        prediction, threshold = segment_otsu_morphology(
            image,
            gaussian_sigma=gaussian_sigma,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            min_object_size=min_object_size,
            min_hole_size=min_hole_size,
        )
        metrics = binary_metrics(target, prediction)
        for key in totals:
            totals[key] += int(metrics[key])
        prediction_path = predictions_dir / f"{row['image_id']}_otsu_mask.png"
        Image.fromarray(prediction, mode="L").save(prediction_path)
        metric_rows.append(
            {
                "image_id": row["image_id"],
                "threshold": threshold,
                "prediction_path": str(prediction_path.resolve()),
                **metrics,
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(output_dir / "metrics_per_image.csv", index=False)

    tp = totals["true_positive"]
    fp = totals["false_positive"]
    fn = totals["false_negative"]
    micro_union = tp + fp + fn
    micro_dice_denominator = 2 * tp + fp + fn
    summary: dict[str, object] = {
        "images": len(metrics_frame),
        "mean_dice": float(metrics_frame["dice"].mean()),
        "mean_iou": float(metrics_frame["iou"].mean()),
        "mean_precision": float(metrics_frame["precision"].mean()),
        "mean_recall": float(metrics_frame["recall"].mean()),
        "mean_area_fraction_absolute_error": float(
            metrics_frame["area_fraction_absolute_error"].mean()
        ),
        "micro_dice": float(
            1.0 if micro_dice_denominator == 0 else 2 * tp / micro_dice_denominator
        ),
        "micro_iou": float(1.0 if micro_union == 0 else tp / micro_union),
        "parameters": {
            "gaussian_sigma": gaussian_sigma,
            "opening_radius": opening_radius,
            "closing_radius": closing_radius,
            "min_object_size": min_object_size,
            "min_hole_size": min_hole_size,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
