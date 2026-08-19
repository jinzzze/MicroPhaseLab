"""End-to-end conversion from annotation rows to masks and a manifest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .annotations import AnnotationFormatError, parse_polygon
from .masks import polygons_to_mask, save_binary_mask
from .schema import (
    GROUP_COLUMN_CANDIDATES,
    IMAGE_COLUMN_CANDIDATES,
    POLYGON_COLUMN_CANDIDATES,
    find_images,
    image_key,
    resolve_column,
)


@dataclass(frozen=True)
class PreparationReport:
    annotation_rows: int
    annotated_images: int
    images_found: int
    masks_written: int
    missing_images: list[str]
    missing_group_metadata: list[str]
    invalid_annotations: list[dict[str, Any]]
    group_strategy: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    # The official Aachen-Heerlen tables use semicolons between columns because
    # point and polygon values contain many unquoted commas. Files created by
    # MicroPhaseLab itself use ordinary comma-separated CSV, so infer the
    # delimiter from the header instead of assuming one format.
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        header = source.readline()
    delimiters = (",", ";", "\t")
    delimiter = max(delimiters, key=header.count)
    if header.count(delimiter) == 0:
        raise ValueError(
            f"Could not detect a CSV delimiter in {path}; "
            "expected a comma-, semicolon-, or tab-separated header."
        )
    return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")


def _metadata_group_map(
    metadata_path: Path | None,
    requested_image_column: str | None,
    requested_group_column: str | None,
) -> tuple[dict[str, str], str]:
    if metadata_path is None:
        return {}, "image_stem_fallback:no_metadata"

    metadata = _read_csv(metadata_path)
    image_column = resolve_column(
        metadata, requested_image_column, IMAGE_COLUMN_CANDIDATES, "metadata image"
    )
    if requested_group_column and "," in requested_group_column:
        group_columns = [column.strip() for column in requested_group_column.split(",")]
        missing = [column for column in group_columns if column not in metadata.columns]
        if missing:
            raise KeyError(f"Group columns {missing!r} not found; columns={list(metadata.columns)}")
    else:
        try:
            group_columns = [
                resolve_column(metadata, requested_group_column, GROUP_COLUMN_CANDIDATES, "group")
            ]
        except KeyError:
            return {}, "image_stem_fallback:no_group_column"

    mapping: dict[str, str] = {}
    selected_columns = [image_column, *group_columns]
    for _, row in metadata[selected_columns].dropna().iterrows():
        group_id = "|".join(f"{column}={row[column]}" for column in group_columns)
        mapping[image_key(row[image_column])] = group_id
    return mapping, f"metadata:{'+'.join(group_columns)}"


def prepare_dataset(
    *,
    images_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
    image_column: str | None = None,
    polygon_column: str | None = None,
    metadata_image_column: str | None = None,
    group_column: str | None = None,
    strict_bounds: bool = True,
) -> PreparationReport:
    """Create masks, ``manifest.csv`` and ``preparation_report.json``."""
    annotations = _read_csv(annotations_path)
    annotation_image_column = resolve_column(
        annotations, image_column, IMAGE_COLUMN_CANDIDATES, "annotation image"
    )
    annotation_polygon_column = resolve_column(
        annotations, polygon_column, POLYGON_COLUMN_CANDIDATES, "polygon"
    )

    images = find_images(images_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    group_map, group_strategy = _metadata_group_map(
        metadata_path, metadata_image_column, group_column
    )
    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    missing_group_metadata: list[str] = []
    invalid_annotations: list[dict[str, Any]] = []

    grouped = annotations.groupby(
        annotations[annotation_image_column].map(image_key), sort=True, dropna=False
    )
    for key, group in grouped:
        image_path = images.get(str(key))
        if image_path is None:
            missing_images.append(str(group.iloc[0][annotation_image_column]))
            continue

        polygons = []
        for row_index, raw_polygon in group[annotation_polygon_column].items():
            try:
                polygons.append(parse_polygon(raw_polygon))
            except AnnotationFormatError as exc:
                invalid_annotations.append(
                    {"row": int(row_index), "image_key": str(key), "error": str(exc)}
                )

        if not polygons:
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        mask, summary = polygons_to_mask(polygons, (width, height), strict_bounds=strict_bounds)
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        save_binary_mask(mask, str(mask_path))
        group_id = group_map.get(str(key))
        if group_id is None:
            group_id = image_path.stem
            if metadata_path is not None:
                missing_group_metadata.append(image_path.name)
        rows.append(
            {
                "image_id": image_path.stem,
                "group_id": group_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "width": width,
                "height": height,
                "object_count": summary.object_count,
                "foreground_pixels": summary.foreground_pixels,
                "foreground_fraction": summary.foreground_fraction,
                "clipped_vertex_count": summary.clipped_vertex_count,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    report = PreparationReport(
        annotation_rows=len(annotations),
        annotated_images=len(grouped),
        images_found=len(images),
        masks_written=len(rows),
        missing_images=missing_images,
        missing_group_metadata=missing_group_metadata,
        invalid_annotations=invalid_annotations,
        group_strategy=group_strategy,
    )
    (output_dir / "preparation_report.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
