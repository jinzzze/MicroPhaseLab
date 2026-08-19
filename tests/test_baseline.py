import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from microphaselab.baseline import binary_metrics, run_baseline, segment_otsu_morphology


class BaselineTests(unittest.TestCase):
    def test_binary_metrics_exact_match(self):
        target = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
        metrics = binary_metrics(target, target)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)

    def test_otsu_segments_bright_object(self):
        image = np.full((64, 64), 30, dtype=np.uint8)
        image[20:44, 20:44] = 220
        prediction, threshold = segment_otsu_morphology(
            image,
            gaussian_sigma=0,
            opening_radius=0,
            closing_radius=0,
            min_object_size=0,
            min_hole_size=0,
        )
        self.assertGreater(threshold, 0)
        self.assertEqual(prediction[30, 30], 1)
        self.assertEqual(prediction[0, 0], 0)

    def test_end_to_end_baseline_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((32, 32), 20, dtype=np.uint8)
            image[8:24, 8:24] = 220
            target = np.zeros((32, 32), dtype=np.uint8)
            target[8:24, 8:24] = 1
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(target).save(mask_path)
            manifest = root / "manifest.csv"
            pd.DataFrame(
                [{"image_id": "sample", "image_path": image_path, "mask_path": mask_path}]
            ).to_csv(manifest, index=False)

            summary = run_baseline(
                manifest,
                root / "baseline",
                gaussian_sigma=0,
                opening_radius=0,
                closing_radius=0,
                min_object_size=0,
                min_hole_size=0,
            )
            self.assertEqual(summary["micro_iou"], 1.0)
            self.assertTrue((root / "baseline" / "summary.json").is_file())
            self.assertTrue((root / "baseline" / "predictions" / "sample_otsu_mask.png").is_file())


if __name__ == "__main__":
    unittest.main()
