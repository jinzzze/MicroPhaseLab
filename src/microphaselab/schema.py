"""Column-name and filename normalization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_COLUMN_CANDIDATES = (
    "Image_url",
    "image_url",
    "image",
    "image_name",
    "filename",
    "file_name",
)
POLYGON_COLUMN_CANDIDATES = ("polygon", "poly", "points", "vertices", "poly_shapely")
GROUP_COLUMN_CANDIDATES = (
    "sample_id",
    "sample",
    "specimen_id",
    "specimen",
    "steel_sample",
    "Type",
    "type",
)


def resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: tuple[str, ...],
    purpose: str,
) -> str:
    if requested:
        if requested not in frame.columns:
            raise KeyError(
                f"{purpose} column {requested!r} not found; columns={list(frame.columns)}"
            )
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lowered = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    raise KeyError(f"Could not infer {purpose} column; columns={list(frame.columns)}")


def image_key(value: object) -> str:
    """Normalize an annotation image reference to a case-insensitive stem."""
    text = str(value).replace("\\", "/")
    return Path(text).stem.casefold()


def find_images(images_dir: Path) -> dict[str, Path]:
    supported = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    result: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.casefold() in supported:
            key = path.stem.casefold()
            if key in result:
                duplicates.setdefault(key, [result[key]]).append(path)
            else:
                result[key] = path
    if duplicates:
        examples = "; ".join(f"{key}: {paths}" for key, paths in list(duplicates.items())[:3])
        raise ValueError(f"Duplicate image stems are ambiguous: {examples}")
    return result
