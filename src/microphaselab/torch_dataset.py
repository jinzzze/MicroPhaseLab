"""PyTorch dataset helpers for manifest-based binary segmentation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

REQUIRED_MANIFEST_COLUMNS = {"image_id", "group_id", "image_path", "mask_path"}


def _read_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {missing}")
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return manifest


def assert_disjoint_groups(manifest_paths: Iterable[str | Path]) -> None:
    """Raise when a sample group appears in more than one supplied split."""
    seen: dict[str, Path] = {}
    for path in manifest_paths:
        manifest_path = Path(path)
        manifest = _read_manifest(manifest_path)
        for group_id in manifest["group_id"].astype(str).unique():
            previous = seen.get(group_id)
            if previous is not None:
                raise ValueError(
                    f"Group leakage detected for {group_id!r}: {previous} and {manifest_path}"
                )
            seen[group_id] = manifest_path


class ManifestSegmentationDataset(Dataset[dict[str, Tensor | str]]):
    """Read grayscale image/mask pairs from a checked manifest."""

    def __init__(self, manifest_path: str | Path, *, image_size: int | None = None) -> None:
        self.manifest = _read_manifest(manifest_path)
        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive when provided")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.manifest.iloc[index]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Manifest row {index} has missing files: image={image_path}, mask={mask_path}"
            )

        with Image.open(image_path) as image_file:
            image = image_file.convert("L")
            if self.image_size is not None:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")
            if self.image_size is not None:
                mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask_array = np.asarray(mask, dtype=np.uint8)

        if image_array.shape != mask_array.shape:
            raise ValueError(
                f"Image and mask shapes differ for {row['image_id']}: "
                f"{image_array.shape} != {mask_array.shape}"
            )
        unique_values = set(np.unique(mask_array).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Mask for {row['image_id']} must contain only 0 and 1; got {sorted(unique_values)}"
            )

        return {
            "image": torch.from_numpy(image_array).unsqueeze(0),
            "mask": torch.from_numpy(mask_array.astype(np.float32)).unsqueeze(0),
            "image_id": str(row["image_id"]),
            "group_id": str(row["group_id"]),
        }
