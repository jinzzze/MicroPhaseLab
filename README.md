# MicroPhaseLab

MicroPhaseLab is an English-language teaching toolkit for materials and mechanical
engineering students. It demonstrates how to prepare, validate, segment, evaluate,
and critically interpret steel microstructure images.

The current package release is v0.2.2. It includes a classical segmentation workflow
and an optional PyTorch U-Net workflow for binary Martensite-Austenite (MA)
segmentation in the Aachen-Heerlen annotated steel microstructure dataset.

## Learning outcomes

- Distinguish SEM images, polygon annotations, and binary masks.
- Validate image-mask pairs before training.
- Create group-aware splits that reduce sample leakage.
- Interpret Dice, IoU, precision, recall, and MA area-fraction error.
- Compare a classical Otsu baseline with a reproducible U-Net experiment.

## Quick start on Windows

Run these PowerShell commands from the project root. The policy change applies only to
the current terminal session:

~~~powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

On macOS or Linux:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

## Lesson 1: offline workflow

No GPU or official image download is required:

~~~powershell
microphaselab demo
microphaselab check --manifest examples/demo/processed/manifest.csv --report examples/demo/processed/quality_report.json
microphaselab visualize --manifest examples/demo/processed/manifest.csv --output-dir examples/demo/figures --limit 4
microphaselab baseline --manifest examples/demo/processed/manifest.csv --output-dir outputs/baseline/demo
python -m pytest -q
~~~

Open examples/demo/figures and confirm that the masks cover the synthetic bright
regions. The deterministic synthetic run is expected to obtain approximately Dice
0.997 and IoU 0.993. These numbers verify the pipeline only; they are not a
performance claim for real steel micrographs.

## Official dataset workflow

Download annotations and metadata first. The optional PNG archive is about 1.14 GB and
contains 1,705 images.

~~~powershell
microphaselab download --output-dir data/raw
microphaselab download --output-dir data/raw --include-images
microphaselab prepare --images-dir data/raw/images --annotations data/raw/annotations.csv --metadata data/raw/metadata.csv --output-dir data/processed --group-column Type,Temperature
microphaselab check --manifest data/processed/manifest.csv --report data/processed/quality_report.json
microphaselab visualize --manifest data/processed/manifest.csv --output-dir outputs/figures/official_qc_seed42 --limit 20 --random --seed 42
microphaselab split --manifest data/processed/manifest.csv --output-dir data/splits --group-column group_id --seed 42
~~~

Inspect at least 20 overlays before splitting. Check for swapped axes, scaling errors,
translations, out-of-bounds polygons, and image-mask mismatches. A group must appear
in only one split. Do not continue if the preparation report lists missing images,
missing group metadata, or invalid annotations.

The data source is https://doi.org/10.6084/m9.figshare.c.5185004.

## Classical baseline

Tune on validation data only, then freeze settings before running the test set once:

~~~powershell
microphaselab baseline --manifest data/splits/val.csv --output-dir outputs/baseline/otsu_val
microphaselab baseline --manifest data/splits/test.csv --output-dir outputs/baseline/otsu_test
~~~

The global Otsu baseline does not use the expert-provided point of interest. It is a
teaching reference and is not directly comparable with studies that use a known point
of interest.

## Optional U-Net workflow

Install PyTorch after data validation and group-aware splitting:

~~~powershell
python -m pip install -e ".[torch]"
microphaselab train --config configs/unet_demo.yaml
microphaselab predict --checkpoint outputs/unet/run_001/best.pt --manifest data/splits/test.csv --output-dir outputs/unet/run_001/test_predictions
microphaselab evaluate --manifest data/splits/test.csv --predictions outputs/unet/run_001/test_predictions/predictions --output-dir outputs/unet/run_001/test_evaluation --overlay-dir outputs/unet/run_001/test_overlays --limit 20
~~~

Choose epochs and thresholds on validation data only. Run the test split once after
those choices are frozen. Comparison figures use red for false positives and cyan for
false negatives. Load only checkpoints that you trained or received from a trusted
source.

For a specific CPU or CUDA build, use the official PyTorch selector:
https://pytorch.org/get-started/locally/

## Metrics and scientific limitations

- Precision measures the correctness of predicted MA pixels; low precision indicates
  false positives.
- Recall measures recovered expert-labelled MA; low recall indicates false negatives.
- Dice and IoU measure foreground overlap.
- Area-fraction absolute error compares 2D MA fractions but cannot validate boundary
  location.
- The label is MA microconstituent, not every material phase.
- Expert polygons are approximate ground truth; a 2D area fraction is not
  automatically a 3D volume fraction.

This project is for teaching and research. It does not replace standard metallography
or expert judgement.

## Development

~~~powershell
python -m pytest -q
ruff check .
~~~

See docs/v0.3_unet_plan.md for the model design and acceptance criteria, and
docs/experiment_report_template.md for a real-data experiment report.

## Citation and license

If you use this software, cite MicroPhaseLab and:

Iren, D. et al. Aachen-Heerlen annotated steel microstructure dataset. Scientific
Data 8, 140 (2021). https://doi.org/10.1038/s41597-021-00926-7

Dataset files are labelled CC0. This repository's code is released under the MIT
License.

## Contributing

Please read CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, and SUPPORT.md before
opening an issue or pull request.
