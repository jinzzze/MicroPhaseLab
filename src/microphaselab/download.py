"""Download pinned Figshare files with resume support and integrity checks."""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadItem:
    name: str
    url: str
    filename: str
    size: int
    md5: str
    archive: bool = False


ITEMS = {
    "annotations": DownloadItem(
        "annotations",
        "https://ndownloader.figshare.com/files/25214918",
        "annotations.csv",
        3_744_453,
        "e56d19f16c82ae23faeeb77577579e1e",
    ),
    "metadata": DownloadItem(
        "metadata",
        "https://ndownloader.figshare.com/files/25214927",
        "metadata.csv",
        136_915,
        "f16113f032e6d984faac4836eb0cb0e1",
    ),
    "images": DownloadItem(
        "images",
        "https://ndownloader.figshare.com/files/26094926",
        "images.zip",
        1_223_165_391,
        "09306cad9a754ae5a963d3f5c67f8f30",
        archive=True,
    ),
}


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract an archive while rejecting absolute paths and ``..`` traversal."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename!r}")
        archive.extractall(destination)


class DownloadValidationError(RuntimeError):
    """Raised when a response does not match the pinned Figshare file."""


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid(path: Path, item: DownloadItem) -> bool:
    return path.is_file() and path.stat().st_size == item.size and _md5(path) == item.md5


def _discard_invalid_archive_prefix(temporary: Path, item: DownloadItem) -> None:
    """Do not resume a ZIP after an HTML/XML challenge was saved as its prefix."""
    if not item.archive or not temporary.is_file() or temporary.stat().st_size == 0:
        return
    with temporary.open("rb") as source:
        signature = source.read(4)
    if signature not in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}:
        temporary.unlink()


def _download(item: DownloadItem, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    _discard_invalid_archive_prefix(temporary, item)
    offset = temporary.stat().st_size if temporary.is_file() else 0
    if offset > item.size:
        temporary.unlink()
        offset = 0
    headers = {"User-Agent": "MicroPhaseLab/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(item.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
        status = response.getcode()
        content_type = response.headers.get_content_type()
        if status not in {200, 206}:
            raise DownloadValidationError(
                f"Figshare returned HTTP {status} for {item.name}; no dataset file was saved."
            )
        if content_type in {"text/html", "application/xml", "text/xml"}:
            raise DownloadValidationError(
                f"Figshare returned {content_type} instead of {item.filename}. "
                "Try again or download the file in a browser."
            )
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        with response, temporary.open(mode) as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        actual_size = temporary.stat().st_size
        if actual_size != item.size:
            raise DownloadValidationError(
                f"Incomplete {item.filename}: got {actual_size:,} of {item.size:,} bytes. "
                f"Partial data remains at {temporary}; rerun the command to resume."
            )
        actual_md5 = _md5(temporary)
        if actual_md5 != item.md5:
            raise DownloadValidationError(
                f"Checksum mismatch for {item.filename}: expected {item.md5}, got {actual_md5}. "
                f"Remove {temporary} and retry."
            )
        temporary.replace(destination)
    except Exception:
        # Keep the partial file: a later call can resume when the server supports Range.
        raise


def download_dataset(output_dir: Path, *, include_images: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [ITEMS["annotations"], ITEMS["metadata"]]
    if include_images:
        selected.append(ITEMS["images"])

    written: list[Path] = []
    for item in selected:
        destination = output_dir / item.filename
        if not _is_valid(destination, item):
            if destination.exists():
                print(f"Ignoring invalid existing file: {destination}", file=sys.stderr)
            print(
                f"Downloading {item.filename} ({item.size / (1024**2):.1f} MiB)...",
                file=sys.stderr,
            )
            _download(item, destination)
        written.append(destination)
        if item.archive:
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            _safe_extract(destination, images_dir)
    return written
