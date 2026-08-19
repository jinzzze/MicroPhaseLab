"""Reproducible CPU-first training for the teaching U-Net."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .torch_dataset import ManifestSegmentationDataset, assert_disjoint_groups
from .unet import TinyUNet


@dataclass(frozen=True)
class TrainingConfig:
    """The complete, serializable configuration for one training run."""

    train_manifest: Path
    val_manifest: Path
    output_dir: Path
    test_manifest: Path | None = None
    seed: int = 42
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3
    image_size: int = 128
    base_channels: int = 16
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32 for the four-level U-Net")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }


def _resolve_path(config_path: Path, value: object | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a flat YAML training configuration, resolving paths from its location."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a YAML mapping")
    allowed = set(TrainingConfig.__dataclass_fields__)
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError(f"Training config has unknown keys: {unexpected}")
    required = {"train_manifest", "val_manifest", "output_dir"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Training config is missing keys: {missing}")
    paths = {
        key: _resolve_path(config_path, raw.pop(key, None))
        for key in ("train_manifest", "val_manifest", "test_manifest", "output_dir")
    }
    return TrainingConfig(**paths, **raw)


def seed_everything(seed: int) -> None:
    """Set the random seeds used by Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device_from_config(config: TrainingConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(config.device)


def _dice_loss(logits: Tensor, targets: Tensor) -> Tensor:
    probabilities = torch.sigmoid(logits)
    numerator = 2 * (probabilities * targets).sum(dim=(1, 2, 3)) + 1.0
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1.0
    return 1.0 - (numerator / denominator).mean()


def _metric_counts(logits: Tensor, targets: Tensor) -> dict[str, float]:
    prediction = logits > 0
    target = targets > 0.5
    true_positive = float(torch.logical_and(prediction, target).sum().item())
    false_positive = float(torch.logical_and(prediction, ~target).sum().item())
    false_negative = float(torch.logical_and(~prediction, target).sum().item())
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "pixels": float(target.numel()),
    }


def _metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    predicted_positive = true_positive + false_positive
    target_positive = true_positive + false_negative
    return {
        "dice": 1.0 if dice_denominator == 0 else 2 * true_positive / dice_denominator,
        "iou": 1.0 if union == 0 else true_positive / union,
        "precision": (
            1.0
            if predicted_positive == 0 and target_positive == 0
            else (true_positive / predicted_positive if predicted_positive else 0.0)
        ),
        "recall": 1.0 if target_positive == 0 else true_positive / target_positive,
        "area_fraction_absolute_error": (
            abs(predicted_positive - target_positive) / counts["pixels"]
        ),
    }


def _run_epoch(
    model: TinyUNet,
    loader: DataLoader[dict[str, Tensor | str]],
    *,
    device: torch.device,
    optimizer: Adam | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    metric_counts = {
        "true_positive": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
        "pixels": 0.0,
    }

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = bce(logits, masks) + _dice_loss(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = images.shape[0]
        counts = _metric_counts(logits.detach(), masks)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        for key, value in counts.items():
            metric_counts[key] += value

    if total_samples == 0:
        raise ValueError("DataLoader produced no batches")
    return {
        "loss": total_loss / total_samples,
        **_metrics_from_counts(metric_counts),
    }


def train_model(config: TrainingConfig) -> dict[str, object]:
    """Train a small U-Net and write reproducible artifacts to output_dir."""
    split_paths = [config.train_manifest, config.val_manifest]
    if config.test_manifest is not None:
        split_paths.append(config.test_manifest)
    assert_disjoint_groups(split_paths)
    seed_everything(config.seed)
    device = _device_from_config(config)
    train_dataset = ManifestSegmentationDataset(config.train_manifest, image_size=config.image_size)
    val_dataset = ManifestSegmentationDataset(config.val_manifest, image_size=config.image_size)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model = TinyUNet(base_channels=config.base_channels).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=True), encoding="utf-8"
    )

    history: list[dict[str, float | int]] = []
    best_dice = float("-inf")
    best_epoch = 0
    best_summary: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        train_result = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        with torch.no_grad():
            validation_result = _run_epoch(model, val_loader, device=device, optimizer=None)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": validation_result["loss"],
            **{f"val_{key}": value for key, value in validation_result.items() if key != "loss"},
        }
        history.append(row)
        if validation_result["dice"] > best_dice:
            best_dice = validation_result["dice"]
            best_epoch = epoch
            best_summary = validation_result
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": config.base_channels,
                    "epoch": epoch,
                    "validation": validation_result,
                    "config": config.to_dict(),
                },
                config.output_dir / "best.pt",
            )

    pd.DataFrame(history).to_csv(config.output_dir / "metrics.csv", index=False)
    summary: dict[str, object] = {
        "device": str(device),
        "epochs": config.epochs,
        "best_epoch": best_epoch,
        "best_validation": best_summary,
        "output_dir": str(config.output_dir.resolve()),
    }
    (config.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def train_from_config(config_path: str | Path) -> dict[str, object]:
    """Load configuration and run training."""
    return train_model(load_training_config(config_path))
