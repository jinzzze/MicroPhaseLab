import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from microphaselab.masks import polygons_to_mask, save_binary_mask


class MaskTests(unittest.TestCase):
    def test_rasterization_is_binary(self):
        mask, summary = polygons_to_mask([[(1, 1), (6, 1), (6, 6), (1, 6)]], (8, 8))
        self.assertEqual(set(np.unique(mask).tolist()), {0, 1})
        self.assertEqual(summary.object_count, 1)
        self.assertGreater(summary.foreground_fraction, 0)

    def test_bounds_can_fail_or_clip(self):
        polygon = [[(-2, 1), (5, 1), (5, 5)]]
        with self.assertRaises(ValueError):
            polygons_to_mask(polygon, (8, 8), strict_bounds=True)
        _, summary = polygons_to_mask(polygon, (8, 8), strict_bounds=False)
        self.assertEqual(summary.clipped_vertex_count, 1)

    def test_saved_mask_retains_zero_one_values(self):
        mask = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            save_binary_mask(mask, str(path))
            with Image.open(path) as loaded:
                self.assertEqual(set(np.unique(np.asarray(loaded)).tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
