import importlib.util
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch optional dependency is not installed")

import numpy as np
import pandas as pd
from PIL import Image

from microphaselab.cli import main
from microphaselab.inference import evaluate_predictions, predict_manifest, save_prediction_overlays
from microphaselab.torch_dataset import ManifestSegmentationDataset, assert_disjoint_groups
from microphaselab.training import TrainingConfig, train_model
from microphaselab.unet import TinyUNet


def _write_pair(root: Path, image_id: str, group_id: str) -> dict[str, object]:
    image = np.full((32, 32), 30, dtype=np.uint8)
    image[8:24, 8:24] = 220
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    image_path = root / f"{image_id}.png"
    mask_path = root / f"{image_id}_mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return {
        "image_id": image_id,
        "group_id": group_id,
        "image_path": image_path,
        "mask_path": mask_path,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TorchDatasetTests(unittest.TestCase):
    def test_dataset_returns_normalized_single_channel_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root / "manifest.csv", [_write_pair(root, "a", "steel_a")])
            dataset = ManifestSegmentationDataset(manifest, image_size=64)

            sample = dataset[0]

            self.assertEqual(tuple(sample["image"].shape), (1, 64, 64))
            self.assertEqual(tuple(sample["mask"].shape), (1, 64, 64))
            self.assertAlmostEqual(float(sample["image"].max()), 220 / 255)
            self.assertEqual(set(sample["mask"].unique().tolist()), {0.0, 1.0})

    def test_group_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_manifest(root / "train.csv", [_write_pair(root, "a", "steel_a")])
            second = _write_manifest(root / "val.csv", [_write_pair(root, "b", "steel_a")])

            with self.assertRaisesRegex(ValueError, "Group leakage"):
                assert_disjoint_groups([first, second])


class UNetTrainingTests(unittest.TestCase):
    def test_unet_preserves_spatial_shape(self):
        import torch

        model = TinyUNet(base_channels=2)
        output = model(torch.zeros(1, 1, 128, 128))
        self.assertEqual(tuple(output.shape), (1, 1, 128, 128))

    def test_cpu_smoke_training_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_manifest(root / "train.csv", [_write_pair(root, "train", "steel_a")])
            val = _write_manifest(root / "val.csv", [_write_pair(root, "val", "steel_b")])
            test = _write_manifest(root / "test.csv", [_write_pair(root, "test", "steel_c")])
            output_dir = root / "run"
            result = train_model(
                TrainingConfig(
                    train_manifest=train,
                    val_manifest=val,
                    test_manifest=test,
                    output_dir=output_dir,
                    epochs=1,
                    batch_size=1,
                    image_size=32,
                    base_channels=2,
                    device="cpu",
                )
            )

            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["best_epoch"], 1)
            self.assertTrue((output_dir / "config.yaml").is_file())
            self.assertTrue((output_dir / "metrics.csv").is_file())
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "validation_summary.json").is_file())
            prediction_dir = root / "prediction"
            prediction_result = predict_manifest(
                output_dir / "best.pt",
                test,
                prediction_dir,
                batch_size=1,
                device="cpu",
            )
            evaluation_dir = root / "evaluation"
            evaluation = evaluate_predictions(
                test,
                prediction_dir / "predictions",
                evaluation_dir,
            )
            overlays = save_prediction_overlays(
                test,
                prediction_dir / "predictions",
                root / "overlays",
                limit=1,
            )

            self.assertEqual(prediction_result["images"], 1)
            self.assertEqual(evaluation["images"], 1)
            self.assertTrue((prediction_dir / "predictions" / "test_unet_mask.png").is_file())
            self.assertTrue((evaluation_dir / "summary.json").is_file())
            self.assertEqual(overlays, 1)

    def test_train_cli_runs_from_yaml_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_manifest(root / "train.csv", [_write_pair(root, "train", "steel_a")])
            val = _write_manifest(root / "val.csv", [_write_pair(root, "val", "steel_b")])
            test = _write_manifest(root / "test.csv", [_write_pair(root, "test", "steel_c")])
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        f"train_manifest: {train.name}",
                        f"val_manifest: {val.name}",
                        f"test_manifest: {test.name}",
                        "output_dir: run",
                        "epochs: 1",
                        "batch_size: 1",
                        "image_size: 32",
                        "base_channels: 2",
                        "device: cpu",
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = main(["train", "--config", str(config)])
            prediction_dir = root / "prediction"
            prediction_exit_code = main(
                [
                    "predict",
                    "--checkpoint",
                    str(root / "run" / "best.pt"),
                    "--manifest",
                    str(test),
                    "--output-dir",
                    str(prediction_dir),
                    "--device",
                    "cpu",
                ]
            )
            evaluation_exit_code = main(
                [
                    "evaluate",
                    "--manifest",
                    str(test),
                    "--predictions",
                    str(prediction_dir / "predictions"),
                    "--output-dir",
                    str(root / "evaluation"),
                    "--overlay-dir",
                    str(root / "overlays"),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(prediction_exit_code, 0)
            self.assertEqual(evaluation_exit_code, 0)
            self.assertTrue((root / "run" / "best.pt").is_file())
            self.assertTrue((root / "evaluation" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
