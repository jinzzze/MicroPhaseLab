"""Read polygon annotations without trusting or executing CSV content."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

Point = tuple[float, float]


class AnnotationFormatError(ValueError):
    """Raised when a polygon cannot be converted to a valid point sequence."""


def _unwrap_polygon(value: Any) -> Any:
    """Unwrap common GeoJSON/Shapely-like containers into coordinate data."""
    if hasattr(value, "exterior") and hasattr(value.exterior, "coords"):
        return list(value.exterior.coords)

    if isinstance(value, dict):
        if "coordinates" in value:
            coordinates = value["coordinates"]
            if value.get("type", "").lower() == "polygon" and coordinates:
                return coordinates[0]
            return coordinates
        for key in ("polygon", "points", "vertices"):
            if key in value:
                return value[key]
    return value


def _parse_wkt(text: str) -> list[Point] | None:
    match = re.fullmatch(r"\s*POLYGON\s*\(\((.+)\)\)\s*", text, flags=re.IGNORECASE)
    if not match:
        return None
    points: list[Point] = []
    for pair in match.group(1).split(","):
        values = pair.strip().split()
        if len(values) < 2:
            raise AnnotationFormatError(f"Invalid WKT coordinate: {pair!r}")
        points.append((float(values[0]), float(values[1])))
    return points


def _decode_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise AnnotationFormatError("Polygon value is empty")

    wkt = _parse_wkt(stripped)
    if wkt is not None:
        return wkt

    for decoder in (json.loads, ast.literal_eval):
        try:
            return decoder(stripped)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    raise AnnotationFormatError("Polygon is not valid JSON, Python literal, or POLYGON WKT")


def parse_polygon(value: Any) -> list[Point]:
    """Convert a polygon value into finite ``(x, y)`` points.

    Supported inputs include nested lists/tuples, JSON or Python-literal strings,
    GeoJSON-style dictionaries, WKT ``POLYGON`` strings and Shapely polygons.
    A repeated closing vertex is removed because Pillow closes polygons itself.
    """
    if isinstance(value, str):
        value = _decode_text(value)
    value = _unwrap_polygon(value)

    # GeoJSON Polygon coordinates have one extra ring dimension.
    if (
        isinstance(value, Sequence)
        and value
        and isinstance(value[0], Sequence)
        and value[0]
        and isinstance(value[0][0], Sequence)
    ):
        value = value[0]

    if not isinstance(value, Iterable) or isinstance(value, (bytes, str, dict)):
        raise AnnotationFormatError("Polygon must be an iterable of coordinate pairs")

    points: list[Point] = []
    for index, point in enumerate(value):
        if not isinstance(point, Sequence) or isinstance(point, (bytes, str)) or len(point) < 2:
            raise AnnotationFormatError(f"Point {index} is not an (x, y) coordinate")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise AnnotationFormatError(f"Point {index} contains non-numeric coordinates") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise AnnotationFormatError(f"Point {index} contains a non-finite coordinate")
        points.append((x, y))

    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise AnnotationFormatError("A polygon needs at least three distinct vertices")
    if len(set(points)) < 3:
        raise AnnotationFormatError("A polygon needs at least three unique vertices")
    return points
