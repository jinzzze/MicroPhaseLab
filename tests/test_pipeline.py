import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from microphaselab.download import (
    DownloadItem,
    _discard_invalid_archive_prefix,
    _is_valid,
    _safe_extract,
)
from microphaselab.pipeline import _read_csv, prepare_dataset
from microphaselab.quality import check_manifest
from microphaselab.splits import create_group_splits


class PipelineTests(unittest.TestCase):
    def test_csv_reader_supports_official_semicolon_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            path.write_text(
                "Image_url;point;polygon\n"
                "image_a.png;(4, 5);[(1, 2), (3, 4), (5, 6)]\n"
                "image_b.png;(7, 8);[(10, 11), (12, 13), (14, 15), (16, 17)]\n",
                encoding="utf-8",
            )

            frame = _read_csv(path)

            self.assertEqual(list(frame.columns), ["Image_url", "point", "polygon"])
            self.assertEqual(len(frame), 2)
            self.assertEqual(
                frame.loc[1, "polygon"],
                "[(10, 11), (12, 13), (14, 15), (16, 17)]",
            )

    def test_csv_reader_keeps_comma_csv_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            pd.DataFrame([{"image_id": "a", "group_id": "sample_1"}]).to_csv(
                path, index=False
            )

            frame = _read_csv(path)

            self.assertEqual(
                frame.to_dict("records"), [{"image_id": "a", "group_id": "sample_1"}]
            )

    def test_download_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_bytes(b"image,polygon\n")
            digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
            item = DownloadItem(
                "sample",
                "https://example.invalid",
                path.name,
                path.stat().st_size,
                digest,
            )
            self.assertTrue(_is_valid(path, item))
            path.write_bytes(b"not the pinned file")
            self.assertFalse(_is_valid(path, item))

    def test_html_archive_prefix_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.zip.part"
            path.write_bytes(b"<html>AWS challenge</html>")
            item = DownloadItem(
                "images",
                "https://example.invalid",
                "images.zip",
                100,
                "0" * 32,
                archive=True,
            )
            _discard_invalid_archive_prefix(path, item)
            self.assertFalse(path.exists())

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaises(ValueError):
                _safe_extract(archive_path, root / "images")

    def test_prepare_check_and_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            annotation_rows = []
            metadata_rows = []
            for index in range(6):
                name = f"image_{index}.png"
                Image.new("L", (32, 24), color=80 + index).save(images / name)
                annotation_rows.append(
                    {"Image_url": name, "polygon": "[(3, 3), (15, 3), (15, 14), (3, 14)]"}
                )
                metadata_rows.append({"Image_url": name, "sample_id": f"sample_{index // 2}"})

            annotations = root / "annotations.csv"
            metadata = root / "metadata.csv"
            pd.DataFrame(annotation_rows).to_csv(annotations, index=False)
            pd.DataFrame(metadata_rows).to_csv(metadata, index=False)

            processed = root / "processed"
            report = prepare_dataset(
                images_dir=images,
                annotations_path=annotations,
                metadata_path=metadata,
                output_dir=processed,
            )
            self.assertEqual(report.masks_written, 6)
            self.assertEqual(report.group_strategy, "metadata:sample_id")
            result = check_manifest(processed / "manifest.csv")
            self.assertTrue(result["ok"])

            splits = root / "splits"
            create_group_splits(
                processed / "manifest.csv", splits, train=1 / 3, val=1 / 3, test=1 / 3
            )
            group_sets = []
            for name in ("train", "val", "test"):
                frame = pd.read_csv(splits / f"{name}.csv")
                group_sets.append(set(frame["group_id"].astype(str)))
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])

    def test_composite_metadata_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("L", (16, 16), color=100).save(images / "image.png")
            annotations = root / "annotations.csv"
            metadata = root / "metadata.csv"
            pd.DataFrame(
                [{"Image_url": "image.png", "polygon": "[(2, 2), (8, 2), (8, 8)]"}]
            ).to_csv(annotations, index=False)
            pd.DataFrame([{"Image_url": "image.png", "Type": "A", "Temperature": 400}]).to_csv(
                metadata, index=False
            )

            report = prepare_dataset(
                images_dir=images,
                annotations_path=annotations,
                metadata_path=metadata,
                output_dir=root / "processed",
                group_column="Type,Temperature",
            )
            manifest = pd.read_csv(root / "processed" / "manifest.csv")
            self.assertEqual(report.group_strategy, "metadata:Type+Temperature")
            self.assertEqual(manifest.loc[0, "group_id"], "Type=A|Temperature=400")


if __name__ == "__main__":
    unittest.main()
