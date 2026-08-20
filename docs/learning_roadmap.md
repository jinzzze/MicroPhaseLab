# Learning Roadmap: From First Clone to a Defensible Result

This roadmap is the student-facing route through MicroPhaseLab. Follow each lesson in
order. Do not begin model training before the quality checks and visual inspection in
Lesson B pass.

## Before you start

You need Python 3.10 or newer, Git, and a copy of this repository. Begin with the
[README](../README.md): clone the repository, create a virtual environment, and run
the offline Lesson A workflow. Install the teaching dependencies with:

~~~powershell
python -m pip install -e ".[dev,notebook]"
~~~

Run every command below from the project root. Keep original downloaded data read-only.
The optional official PNG archive is about 1.14 GB; ensure that you have a stable
connection and enough free disk space before downloading it.

## At a glance

| Lesson | Question answered | Main evidence | Continue when |
| --- | --- | --- | --- |
| A. Data representations | What are an image, polygon, mask, and prediction? | Offline demo, notebook, and overlays | You can explain a 0/1 mask and a false positive/negative. |
| B. Quality and leakage | Can these labelled images support a fair experiment? | Quality report, 20 overlays, group-aware CSV splits | The report is `ok: true` and groups do not cross splits. |
| C. Classical baseline | What can a transparent non-ML method achieve? | Validation metrics and frozen baseline settings | Parameters were selected using validation data only. |
| D. U-Net training | Can a learned model improve on the baseline? | Training metrics and `best.pt` selected by validation Dice | The checkpoint and settings are frozen. |
| E. Scientific interpretation | What does the final test result support? | Test metrics, 20 comparison figures, and a report | You have stated limitations as well as results. |

## Lesson A: data representations — completed first

**Goal:** understand the relationship between a micrograph, expert polygon, binary
mask, prediction, and segmentation metric.

1. Run the offline commands in the README: `demo`, `check`, `visualize`, and
   `baseline`.
2. Launch `jupyter lab` and complete
   [`notebooks/01_data_pipeline.ipynb`](../notebooks/01_data_pipeline.ipynb).
3. Confirm that the notebook's white/red/blue error image makes sense: white is a
   correct MA prediction, red is an extra MA prediction, and blue is missed MA.

**Pass condition:** the demo quality report contains `"ok": true`; the notebook runs
without an error; and you can explain why pixel accuracy alone is insufficient.

**Next:** Lesson B uses the real dataset and introduces data quality and leakage.

## Lesson B: data quality and leakage

**Goal:** turn the official polygons into checked masks and make splits that keep
related images from the same steel condition together.

### B1. Download source files

First download annotations and metadata:

~~~powershell
microphaselab download --output-dir data/raw
~~~

Then download the image archive when you are ready for the approximately 1.14 GB
download:

~~~powershell
microphaselab download --output-dir data/raw --include-images
~~~

**Expected files:** `data/raw/annotations.csv`, `data/raw/metadata.csv`, and
`data/raw/images/` containing PNG files.

**If this fails:** check your internet connection, then run the same command again.
Do not create placeholder images or edit the downloaded source files.

### B2. Prepare masks and the manifest

~~~powershell
microphaselab prepare --images-dir data/raw/images --annotations data/raw/annotations.csv --metadata data/raw/metadata.csv --output-dir data/processed --group-column Type,Temperature
~~~

This converts expert polygons into binary masks and writes
`data/processed/manifest.csv`. The command also prints a preparation report.

**Pass condition:** the report does not list missing images, invalid annotations, or
unexpected preparation errors. If it does, stop here and investigate before training.

### B3. Check data and inspect overlays

~~~powershell
microphaselab check --manifest data/processed/manifest.csv --report data/processed/quality_report.json
microphaselab visualize --manifest data/processed/manifest.csv --output-dir outputs/figures/official_qc_seed42 --limit 20 --random --seed 42
~~~

Open all 20 figures in `outputs/figures/official_qc_seed42`. Look for shifted masks,
swapped axes, incorrect scaling, polygons outside the image, and mismatched image-mask
pairs.

**Pass condition:** `quality_report.json` records `"ok": true`, and your visual
inspection finds no systematic error. A table cannot replace this inspection.

### B4. Create leakage-resistant splits

~~~powershell
microphaselab split --manifest data/processed/manifest.csv --output-dir data/splits --group-column group_id --seed 42
~~~

**Expected outputs:** `data/splits/train.csv`, `data/splits/val.csv`, and
`data/splits/test.csv`, plus a printed split summary.

`group_id` combines the selected material metadata fields. Every group must occur in
only one split. Randomly splitting individual images can place very similar images in
both training and test sets, giving a misleadingly high test score.

**Pass condition:** the split command succeeds and `split_summary.json` records the
number of groups and rows in each split. The command assigns every unique group once;
Lesson D repeats a group-leakage check before training. Do not change split ratios or
the seed after you begin comparing methods unless you document the new experiment as a
separate run.

**Next:** Lesson C establishes a transparent reference before machine learning.

## Lesson C: classical baseline

**Goal:** measure what brightness thresholding plus morphology can achieve, and avoid
claiming that a learned model is useful without a reference method.

### C1. Run the baseline on validation data

~~~powershell
microphaselab baseline --manifest data/splits/val.csv --output-dir outputs/baseline/otsu_val
~~~

Read `outputs/baseline/otsu_val/summary.json` and
`outputs/baseline/otsu_val/metrics_per_image.csv`. Inspect selected prediction masks
with a plotting tool or create comparison figures before deciding whether a setting is
reasonable.

### C2. Select settings only on validation data

The baseline accepts these transparent settings:

~~~text
--gaussian-sigma
--opening-radius
--closing-radius
--min-object-size
--min-hole-size
~~~

Change one setting at a time, write each result to a new validation output directory,
and record the setting and metrics. For example:

~~~powershell
microphaselab baseline --manifest data/splits/val.csv --output-dir outputs/baseline/otsu_val_sigma_1p5 --gaussian-sigma 1.5
~~~

Choose a setting using validation Dice, IoU, precision, recall, visual errors, and
area-fraction error together. A setting that improves one metric while producing poor
boundary errors is not automatically better.

### C3. Freeze settings and evaluate the test set once

Replace the values below with the settings selected in C2:

~~~powershell
microphaselab baseline --manifest data/splits/test.csv --output-dir outputs/baseline/otsu_test --gaussian-sigma 1.0 --opening-radius 1 --closing-radius 2 --min-object-size 32 --min-hole-size 32
~~~

**Expected outputs:** `summary.json`, `metrics_per_image.csv`, and a `predictions/`
folder in the selected output directory.

**Pass condition:** the test command is run only after settings are frozen. Record the
parameters and all reported metrics. The global Otsu baseline is a teaching reference;
it does not use any expert-provided point of interest.

**Next:** Lesson D trains a U-Net under the same split discipline.

## Lesson D: optional U-Net training

**Goal:** train a compact neural network using only the training split, choose a
checkpoint on validation data, and compare it fairly with Lesson C.

### D1. Install PyTorch

Install a CPU or CUDA build appropriate for your computer, using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) when necessary.
Then install the project extra:

~~~powershell
python -m pip install -e ".[torch]"
~~~

**Pass condition:** verify the installation with:

~~~powershell
python -c "import torch; print(torch.__version__)"
~~~

### D2. Read and keep a run configuration

Start from [`configs/unet_demo.yaml`](../configs/unet_demo.yaml). It records the
train/validation/test manifests, seed, epoch count, image size, batch size, learning
rate, model width, output directory, and device. Copy it to a new configuration file
before changing settings, so each experiment remains reproducible.

Do not alter `test_manifest` to make a result look better. The test split is not an
input to model selection.

### D3. Train and select a validation checkpoint

~~~powershell
microphaselab train --config configs/unet_demo.yaml
~~~

**Expected outputs:** `outputs/unet/run_001/best.pt`, training metrics, and the saved
run configuration. `best.pt` is selected by validation Dice.

**Pass condition:** training completes, the split-leakage check does not fail, and you
record the selected checkpoint, configuration, validation metric, and epoch. Do not
select a different checkpoint after looking at test performance.

### D4. Select an inference threshold on validation data only

Keep the default threshold of 0.5 unless you have a reason to compare alternatives.
If you do compare thresholds, predict and evaluate the validation split for each
candidate, then record one frozen choice before touching the test split. For example:

~~~powershell
microphaselab predict --checkpoint outputs/unet/run_001/best.pt --manifest data/splits/val.csv --output-dir outputs/unet/run_001/val_predictions --threshold 0.5
microphaselab evaluate --manifest data/splits/val.csv --predictions outputs/unet/run_001/val_predictions/predictions --output-dir outputs/unet/run_001/val_evaluation --overlay-dir outputs/unet/run_001/val_overlays --limit 20
~~~

**Pass condition:** checkpoint, threshold, and other inference settings are recorded
before Lesson E. Do not change them after reading test metrics.

**Next:** Lesson E predicts and evaluates the frozen checkpoint and threshold on the
held-back test set.

## Lesson E: final evaluation and materials interpretation

**Goal:** make one final test-set measurement, inspect errors, and state what the
result can and cannot support.

### E1. Predict the held-back test set

~~~powershell
microphaselab predict --checkpoint outputs/unet/run_001/best.pt --manifest data/splits/test.csv --output-dir outputs/unet/run_001/test_predictions --threshold 0.5
~~~

**Expected output:** `outputs/unet/run_001/test_predictions/predictions/` and a
prediction manifest.

### E2. Evaluate and create comparison figures

~~~powershell
microphaselab evaluate --manifest data/splits/test.csv --predictions outputs/unet/run_001/test_predictions/predictions --output-dir outputs/unet/run_001/test_evaluation --overlay-dir outputs/unet/run_001/test_overlays --limit 20
~~~

Open at least 20 figures in `outputs/unet/run_001/test_overlays`. In these figures,
red indicates false positives and cyan indicates false negatives.

**Expected outputs:** `summary.json`, `metrics_per_image.csv`, and the requested
comparison figures. Read Dice, IoU, precision, recall, and area-fraction error
together; none is sufficient alone.

### E3. Write the experiment report

Complete [`docs/experiment_report_template.md`](experiment_report_template.md). Record
the Git commit, dataset version, group definition, split seed, configuration,
checkpoint, final metrics, visual observations, and limitations.

**Final pass condition:** your report explains the result without overclaiming it.
MA labels are expert-derived approximations, and a 2D area fraction is not
automatically a 3D volume fraction. The model supports consistent image analysis; it
does not replace metallographic expertise.

## Final submission checklist

- [ ] Lesson A demo, notebook, and tests completed.
- [ ] Quality report passes and 20 real-data overlays were inspected.
- [ ] No group appears in more than one split.
- [ ] Baseline parameters were selected on validation data only.
- [ ] U-Net checkpoint and threshold were selected on validation data only.
- [ ] The final test split was evaluated after settings were frozen.
- [ ] At least 20 final comparison figures were reviewed.
- [ ] The experiment report records results, limitations, and reproduction details.

If a checklist item is not complete, describe the work as an incomplete learning
exercise rather than a finished materials-analysis result.
