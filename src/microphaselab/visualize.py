"""Create consistent image/mask/overlay quality-control figures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/microphaselab-matplotlib")


def save_overlays(
    manifest_path: Path,
    output_dir: Path,
    *,
    limit: int = 20,
    random: bool = False,
    seed: int = 42,
) -> int:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    manifest = pd.read_csv(manifest_path)
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if random and len(manifest) > limit:
        manifest = manifest.sample(n=limit, random_state=seed)
    else:
        manifest = manifest.head(limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest[["image_id", "image_path", "mask_path"]].to_csv(
        output_dir / "qc_selection.csv", index=False
    )
    written = 0
    for _, row in manifest.iterrows():
        with Image.open(row["image_path"]) as image_file:
            image = np.asarray(image_file.convert("L"))
        with Image.open(row["mask_path"]) as mask_file:
            mask = np.asarray(mask_file) > 0

        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("SEM image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Expert mask")
        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.55)
        axes[2].set_title(f"Overlay | MA fraction={row['foreground_fraction']:.3f}")
        for axis in axes:
            axis.axis("off")
        figure.savefig(output_dir / f"{row['image_id']}_overlay.png", dpi=150)
        plt.close(figure)
        written += 1
    return written
