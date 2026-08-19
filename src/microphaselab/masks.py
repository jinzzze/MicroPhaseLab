"""Rasterize expert polygons into binary semantic-segmentation masks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .annotations import Point


@dataclass(frozen=True)
class RasterizationSummary:
    object_count: int
    clipped_vertex_count: int
    foreground_pixels: int
    foreground_fraction: float


def _clip_points(
    points: Sequence[Point], width: int, height: int, strict_bounds: bool
) -> tuple[list[Point], int]:
    clipped: list[Point] = []
    clipped_count = 0
    for x, y in points:
        outside = x < 0 or y < 0 or x > width - 1 or y > height - 1
        if outside and strict_bounds:
            raise ValueError(
                f"Polygon coordinate ({x}, {y}) is outside image bounds {width}x{height}"
            )
        clipped_x = min(max(x, 0.0), float(width - 1))
        clipped_y = min(max(y, 0.0), float(height - 1))
        clipped_count += int((clipped_x, clipped_y) != (x, y))
        clipped.append((clipped_x, clipped_y))
    return clipped, clipped_count


def polygons_to_mask(
    polygons: Iterable[Sequence[Point]],
    image_size: tuple[int, int],
    *,
    strict_bounds: bool = True,
) -> tuple[np.ndarray, RasterizationSummary]:
    """Return a uint8 mask containing only values 0 and 1.

    ``image_size`` follows Pillow convention: ``(width, height)``.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    object_count = 0
    clipped_vertex_count = 0

    for points in polygons:
        clipped, clipped_count = _clip_points(points, width, height, strict_bounds)
        draw.polygon(clipped, fill=1, outline=1)
        object_count += 1
        clipped_vertex_count += clipped_count

    mask = np.asarray(image, dtype=np.uint8)
    foreground_pixels = int(mask.sum())
    summary = RasterizationSummary(
        object_count=object_count,
        clipped_vertex_count=clipped_vertex_count,
        foreground_pixels=foreground_pixels,
        foreground_fraction=foreground_pixels / mask.size,
    )
    return mask, summary


def save_binary_mask(mask: np.ndarray, path: str) -> None:
    """Save a 0/1 array as lossless PNG while retaining binary pixel values."""
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Mask is not binary: found values {sorted(unique)}")
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path, format="PNG")
