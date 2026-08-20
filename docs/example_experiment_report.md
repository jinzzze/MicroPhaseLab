# Example: Completed MicroPhaseLab Experiment Report

This completed teaching run shows how to use
[`experiment_report_template.md`](experiment_report_template.md). The downloaded data,
model checkpoint, and output images are not stored in the repository. Students should
run the workflow themselves and record their own results.

## Aim

This experiment segments MA regions in steel SEM micrographs. It compares a classical
Otsu-threshold baseline with a small U-Net model, using group-aware splitting to test
generalisation to unseen sample groups.

## Reproducibility Record

| Item | Value |
|---|---|
| Code commit | `07723ad` |
| Random seed | `42` |
| Python | `3.12.10` |
| PyTorch | `2.13.0+cu130` |
| Training device | NVIDIA GeForce RTX 3080 Ti Laptop GPU |
| Device selected by program | CUDA |
| U-Net checkpoint | `outputs/unet/run_001/best.pt` |
| Prediction threshold | `0.5` |

## Data Preparation and Quality Control

The project downloader retrieved the official annotation CSV, metadata CSV, and image
archive. The archive contained 1,705 PNG images. Polygon annotations were converted to
binary masks. The metadata columns `type` and `temperature` were combined to create a
group identifier.

Some polygon coordinates extended outside image bounds. The documented
`--clip-out-of-bounds` option was used so that the valid in-image part of each
annotation was retained and that decision was recorded.

Quality control passed:

- Images and masks in the manifest: 1,705
- Sample groups: 14
- Missing required columns: none
- Reported data errors: none
- Foreground-fraction range: 0.000858 to 0.364408

Twenty randomly selected image-mask overlays were inspected. No systematic image-mask
displacement or obvious mask-conversion failure was observed.

## Group-Aware Split

Images were split by `group_id`, not randomly by image. This prevents closely related
images from the same material-condition group appearing in both training and evaluation.

| Split | Groups | Images |
|---|---:|---:|
| Train | 10 | 1,468 |
| Validation | 2 | 100 |
| Test | 2 | 137 |

An independent overlap check found zero shared groups between train, validation, and
test sets.

## Classical Baseline

The baseline used Gaussian smoothing, Otsu thresholding, and morphological
post-processing. Default parameters were Gaussian sigma 1.0, opening radius 1,
closing radius 2, minimum object size 32, and minimum hole size 32. A validation-only
trial using `gaussian_sigma = 1.5` did not improve Dice, so the default configuration
was frozen before testing.

### Test-set baseline results

| Metric | Result |
|---|---:|
| Mean Dice | 0.0901 |
| Mean IoU | 0.0481 |
| Mean precision | 0.0508 |
| Mean recall | 0.6270 |
| Mean area-fraction absolute error | 0.3056 |

The baseline over-segmented bright laths, interfaces, and texture features. Its recall
was relatively high, but precision and MA area estimation were poor.

## U-Net Training and Validation

| Parameter | Value |
|---|---:|
| Epochs | 10 |
| Batch size | 4 |
| Learning rate | 0.001 |
| Image size | 128 x 128 |
| Base channels | 16 |
| Data-loader workers | 0 |
| Device | `auto` -> CUDA |

The checkpoint was selected by validation performance only.

| Validation result | Value |
|---|---:|
| Best epoch | 8 |
| Best training-validation Dice | 0.4340 |
| Separately evaluated mean Dice | 0.4250 |
| Separately evaluated mean IoU | 0.2913 |
| Separately evaluated mean precision | 0.4242 |
| Separately evaluated mean recall | 0.5242 |
| Separately evaluated area-fraction absolute error | 0.0178 |

The small difference between the training summary and separate validation evaluation is
expected because they aggregate image results differently.

## Final Test-Set Evaluation

The checkpoint and threshold were frozen before the test set was evaluated. The test set
was predicted and evaluated once.

| Metric | Otsu baseline | U-Net |
|---|---:|---:|
| Mean Dice | 0.0901 | **0.3284** |
| Mean IoU | 0.0481 | **0.2157** |
| Mean precision | 0.0508 | **0.3582** |
| Mean recall | **0.6270** | 0.4021 |
| Mean area-fraction absolute error | 0.3056 | **0.0179** |

The U-Net substantially improved Dice, IoU, precision, and MA area-fraction estimation.
Its lower recall shows that it rejected many false positives but still missed some true
MA regions.

## Visual Error Analysis

Twenty test-set comparison figures were inspected. There was no global spatial offset,
scale mismatch, or repeated full-image prediction failure.

- **False negatives (cyan):** true MA regions missed by the model, especially in
  complex lath-like regions.
- **False positives (red):** non-MA structures predicted as MA where local texture
  resembles annotated MA.
- **Mixed errors:** some regions were partly detected but had inaccurate boundaries or
  nearby extra predictions.

The figures agree with the numerical result: U-Net is substantially better than global
thresholding, but it is not yet reliable enough for unsupervised materials conclusions.

## Interpretation and Limitations

Unlike a global intensity threshold, U-Net can use local image context and shape
information. However, its mean Dice dropped from 0.4250 on validation images to 0.3284
on fully unseen test groups. This demonstrates the difficulty of generalising across
material-condition groups.

These values are a reproducible teaching benchmark, not a final validated materials
model. Future work could add training groups, use augmentation, increase image or model
size, use class-imbalance-aware losses, and select all thresholds on validation data.

## Conclusion

The full workflow was reproduced: data preparation, mask quality control, group-aware
splitting, baseline analysis, GPU U-Net training, and a single frozen-model test
evaluation. The U-Net achieved mean test Dice **0.3284**, compared with **0.0901** for
the classical baseline. It demonstrates the value of learned image context while also
showing why careful validation and further development remain necessary.
