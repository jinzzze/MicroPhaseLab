"""Command-line entry point for MicroPhaseLab."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .baseline import run_baseline
from .demo import create_demo
from .download import download_dataset
from .pipeline import prepare_dataset
from .quality import check_manifest, write_quality_report
from .splits import create_group_splits
from .visualize import save_overlays


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microphaselab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download pinned Figshare dataset files")
    download.add_argument("--output-dir", type=_path, default=Path("data/raw"))
    download.add_argument("--include-images", action="store_true")

    prepare = subparsers.add_parser("prepare", help="Convert polygon CSV rows to binary masks")
    prepare.add_argument("--images-dir", type=_path, required=True)
    prepare.add_argument("--annotations", type=_path, required=True)
    prepare.add_argument("--metadata", type=_path)
    prepare.add_argument("--output-dir", type=_path, required=True)
    prepare.add_argument("--image-column")
    prepare.add_argument("--polygon-column")
    prepare.add_argument("--metadata-image-column")
    prepare.add_argument("--group-column")
    prepare.add_argument(
        "--clip-out-of-bounds",
        action="store_true",
        help="Clip vertices to image bounds instead of failing",
    )

    split = subparsers.add_parser("split", help="Create deterministic group-aware CSV splits")
    split.add_argument("--manifest", type=_path, required=True)
    split.add_argument("--output-dir", type=_path, required=True)
    split.add_argument("--group-column", default="group_id")
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)

    check = subparsers.add_parser("check", help="Validate image-mask pairs")
    check.add_argument("--manifest", type=_path, required=True)
    check.add_argument("--report", type=_path, required=True)

    visualize = subparsers.add_parser("visualize", help="Save image-mask-overlay figures")
    visualize.add_argument("--manifest", type=_path, required=True)
    visualize.add_argument("--output-dir", type=_path, required=True)
    visualize.add_argument("--limit", type=int, default=20)
    visualize.add_argument("--random", action="store_true")
    visualize.add_argument("--seed", type=int, default=42)

    baseline = subparsers.add_parser(
        "baseline", help="Run Otsu plus morphology and evaluate against expert masks"
    )
    baseline.add_argument("--manifest", type=_path, required=True)
    baseline.add_argument("--output-dir", type=_path, required=True)
    baseline.add_argument("--gaussian-sigma", type=float, default=1.0)
    baseline.add_argument("--opening-radius", type=int, default=1)
    baseline.add_argument("--closing-radius", type=int, default=2)
    baseline.add_argument("--min-object-size", type=int, default=32)
    baseline.add_argument("--min-hole-size", type=int, default=32)

    demo = subparsers.add_parser("demo", help="Run the complete data pipeline offline")
    demo.add_argument("--output-dir", type=_path, default=Path("examples/demo"))

    train = subparsers.add_parser("train", help="Train the optional PyTorch U-Net")
    train.add_argument("--config", type=_path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        paths = download_dataset(args.output_dir, include_images=args.include_images)
        print(json.dumps({"downloaded": [str(path) for path in paths]}, indent=2))
    elif args.command == "prepare":
        report = prepare_dataset(
            images_dir=args.images_dir,
            annotations_path=args.annotations,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            image_column=args.image_column,
            polygon_column=args.polygon_column,
            metadata_image_column=args.metadata_image_column,
            group_column=args.group_column,
            strict_bounds=not args.clip_out_of_bounds,
        )
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    elif args.command == "split":
        summary = create_group_splits(
            args.manifest,
            args.output_dir,
            group_column=args.group_column,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "check":
        result = check_manifest(args.manifest)
        write_quality_report(result, args.report)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    elif args.command == "visualize":
        written = save_overlays(
            args.manifest,
            args.output_dir,
            limit=args.limit,
            random=args.random,
            seed=args.seed,
        )
        print(json.dumps({"figures_written": written}, indent=2))
    elif args.command == "baseline":
        result = run_baseline(
            args.manifest,
            args.output_dir,
            gaussian_sigma=args.gaussian_sigma,
            opening_radius=args.opening_radius,
            closing_radius=args.closing_radius,
            min_object_size=args.min_object_size,
            min_hole_size=args.min_hole_size,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "demo":
        create_demo(args.output_dir)
        print(json.dumps({"demo_created": str(args.output_dir)}, indent=2))
    elif args.command == "train":
        try:
            from .training import train_from_config
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise SystemExit(
                    "PyTorch is required for training. Install it with: "
                    "python -m pip install -e \".[torch]\""
                ) from error
            raise
        result = train_from_config(args.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
