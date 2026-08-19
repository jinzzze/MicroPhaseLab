# MicroPhaseLab Real-Data Experiment Report Template

> Complete this report only after data quality checks, inspection of 20 overlays,
> group-aware splitting, and validation-based model selection. A 2D area fraction is
> not automatically a 3D volume fraction.

## 1. Experiment information

- Student or team:
- Date:
- MicroPhaseLab Git commit:
- Python, PyTorch, and device:
- Dataset version and citation:

## 2. Data and quality control

- Raw image count and generated-mask count:
- Group count and group definition:
- Value of ok in quality_report.json:
- Missing or invalid entries from preparation_report.json:
- Directory for 20 visual QC overlays and observed issues:

Training may proceed only when image-mask dimensions match, masks contain 0 and 1,
and visual inspection finds no coordinate or pairing error.

## 3. Leakage-resistant split

| Split | Images | Groups | Group overlap with another split |
| --- | ---: | ---: | --- |
| Train |  |  | No |
| Validation |  |  | No |
| Test |  |  | No |

Explain why this group definition was selected and why per-image random splitting
would be unsafe.

## 4. Training and model selection

- Training configuration file:
- Random seed:
- Input size, batch size, epochs, learning rate, and base channels:
- Validation metric used to select best.pt:
- Parameters tried on validation data and final frozen values:

Do not use test metrics to select a model, threshold, or epoch.

## 5. Frozen test result

| Metric | Macro mean | Micro |
| --- | ---: | ---: |
| Dice |  |  |
| IoU |  |  |
| Precision |  |  |
| Recall |  |  |
| Area-fraction absolute error |  | Not applicable |

Attach summary.json, metrics_per_image.csv, and at least 20 comparison figures.

## 6. Materials interpretation

1. Is precision or recall lower? Which MA false-positive or false-negative mechanism
   could explain that pattern?
2. Do area-fraction error and IoU agree? If not, what boundary or shape error could
   explain the difference?
3. Compared with the Otsu plus morphology baseline, where does the U-Net improve and
   where does it still fail?
4. Which results may be influenced by imaging conditions, annotation boundaries,
   class imbalance, or sample distribution?

## 7. Limitations and reproducibility

- The task segments MA microconstituents, not every material phase.
- Expert polygons approximate boundaries and are not error-free ground truth.
- Record the commit, configuration, checkpoint, commands, and random seed needed to
  reproduce the result.
