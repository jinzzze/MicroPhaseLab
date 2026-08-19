"""Generate a tiny synthetic dataset for validating the installation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from .pipeline import prepare_dataset
from .splits import create_group_splits


def _synthetic_sem(size: tuple[int, int], seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    height, width = size[1], size[0]
    base = rng.normal(loc=115, scale=22, size=(height, width))
    base = np.clip(base, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="L").filter(ImageFilter.GaussianBlur(radius=1.1))
    draw = ImageDraw.Draw(image)
    for _ in range(45):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, 5))
        shade = int(rng.integers(70, 170))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=shade)
    return image


def create_demo(root: Path) -> None:
    raw = root / "raw"
    images_dir = raw / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    polygons = {
        "demo_a.png": [
            [(18, 18), (52, 14), (63, 38), (40, 61), (15, 48)],
            [(75, 65), (106, 58), (118, 86), (95, 108), (70, 91)],
        ],
        "demo_b.png": [
            [(35, 70), (55, 45), (86, 48), (99, 74), (73, 101), (46, 95)],
        ],
    }
    annotation_rows = []
    for index, (name, image_polygons) in enumerate(polygons.items()):
        image = _synthetic_sem((128, 128), seed=100 + index)
        draw = ImageDraw.Draw(image)
        for polygon in image_polygons:
            draw.polygon(polygon, fill=205)
            annotation_rows.append({"Image_url": name, "polygon": repr(polygon)})
        image.save(images_dir / name)

    pd.DataFrame(annotation_rows).to_csv(raw / "annotations.csv", index=False)
    pd.DataFrame(
        [
            {"Image_url": "demo_a.png", "sample_id": "steel_A", "temperature": 400},
            {"Image_url": "demo_b.png", "sample_id": "steel_B", "temperature": 500},
        ]
    ).to_csv(raw / "metadata.csv", index=False)

    prepare_dataset(
        images_dir=images_dir,
        annotations_path=raw / "annotations.csv",
        metadata_path=raw / "metadata.csv",
        output_dir=root / "processed",
    )
    create_group_splits(
        root / "processed" / "manifest.csv",
        root / "splits",
        train=0.5,
        val=0.0,
        test=0.5,
        seed=42,
    )
