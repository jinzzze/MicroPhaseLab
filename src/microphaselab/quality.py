"""Data integrity checks for image-mask manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REQUIRED_COLUMNS = {
    "image_id",
    "group_id",
    "image_path",
    "mask_path",
    "width",
    "height",
    "object_count",
    "foreground_fraction",
}


def check_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    errors: list[dict[str, object]] = []
    if missing_columns:
        return {
            "ok": False,
            "rows": len(manifest),
            "missing_columns": missing_columns,
            "errors": [],
        }

    for index, row in manifest.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(
                {
                    "row": int(index),
                    "error": "missing_file",
                    "image_exists": image_path.is_file(),
                    "mask_exists": mask_path.is_file(),
                }
            )
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
            if image.size != mask_image.size:
                errors.append(
                    {
                        "row": int(index),
                        "error": "size_mismatch",
                        "sizes": [image.size, mask_image.size],
                    }
                )
            mask = np.asarray(mask_image)
        unique = set(np.unique(mask).tolist())
        if not unique.issubset({0, 1}):
            errors.append(
                {"row": int(index), "error": "non_binary_mask", "values": sorted(unique)[:20]}
            )

    return {
        "ok": not errors,
        "rows": len(manifest),
        "groups": int(manifest["group_id"].nunique()),
        "missing_columns": [],
        "errors": errors,
        "foreground_fraction": {
            "min": float(manifest["foreground_fraction"].min()) if len(manifest) else None,
            "mean": float(manifest["foreground_fraction"].mean()) if len(manifest) else None,
            "max": float(manifest["foreground_fraction"].max()) if len(manifest) else None,
        },
    }


def write_quality_report(result: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
