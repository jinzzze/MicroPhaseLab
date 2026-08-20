# Student Learning Guide: Machine Learning for Microstructure Analysis

## Who is this guide for?

This guide is for students who are new to machine learning, image analysis, or both.
You do not need previous experience with neural networks to begin. Start with the
offline demo, inspect the generated images, and work through the steps in order.

## How to begin

Follow the Windows or macOS/Linux setup in the [README](../README.md), including the
`notebook` extra. Complete the offline workflow, then launch JupyterLab from the
project root with `jupyter lab` and open
[`notebooks/01_data_pipeline.ipynb`](../notebooks/01_data_pipeline.ipynb). The notebook
is Lesson A; it explains the files created by the commands before you move to Lesson B.
For the exact commands, expected outputs, pass conditions, and next steps for Lessons
B–E, follow the [Learning Roadmap](learning_roadmap.md).

## The learning journey

MicroPhaseLab follows the same reasoning process used in a responsible materials
informatics experiment:

1. **Understand the scientific question.** In this project, the target is the
   Martensite-Austenite (MA) microconstituent in steel micrographs. It is not a model
   for every material phase.
2. **Represent the expert knowledge.** An expert polygon traces a region of interest.
   The project rasterizes each polygon into a binary mask: 1 for MA and 0 for
   background.
3. **Validate the data.** Before training, check that each image has a matching mask,
   that both have the same dimensions, and that masks contain only 0 and 1.
4. **Inspect examples visually.** Overlay images reveal swapped axes, incorrect
   scaling, shifted polygons, and image-mask mismatches that a table may not show.
5. **Prevent data leakage.** Similar images from one steel sample must stay in one
   split. Splitting individual images at random can make test performance look much
   better than it really is.
6. **Build a simple baseline.** Otsu thresholding plus morphology provides a
   transparent reference method. A machine learning model should be compared with a
   baseline, not evaluated in isolation.
7. **Train a segmentation model.** The optional U-Net learns a mapping from grayscale
   micrographs to MA masks using checked training images.
8. **Choose settings on validation data.** Use the validation split to select an
   epoch, threshold, or model setting. Do not use the test split for these choices.
9. **Evaluate once on the frozen test split.** Report Dice, IoU, precision, recall,
   and area-fraction error together with prediction comparison figures.
10. **State the limits of the conclusion.** A good score does not remove uncertainty
    in the images, labels, material conditions, or interpretation.

## Machine learning without the jargon

Machine learning is a way to make a program improve at a task by studying examples
instead of following only hand-written rules.

Imagine a geography exercise using aerial photographs of London. For each photograph,
the teacher colours every public park green and leaves roads, buildings, and the River
Thames uncoloured. A student first tries to mark the parks in a new photograph, then
compares their map with the teacher's answer. With many examples from different parts
of London, the student learns that parks tend to have particular colours, textures,
shapes, and neighbouring features. A machine learning model follows the same pattern
mathematically: it receives an image and the teacher's correct pixel mask, measures
its mistakes, and adjusts its internal numerical settings so that the next predicted
mask is closer to the correct one.

In MicroPhaseLab, the aerial photograph is replaced by a micrograph, and the park mask
is replaced by the expert MA mask. The learning process is the same: predict a region,
compare it with a trusted reference, and improve from many examples.

In this project:

| Term | Meaning in MicroPhaseLab |
| --- | --- |
| Input | A grayscale SEM-style micrograph |
| Label or target | An expert-derived binary MA mask |
| Model | A mathematical function, such as the compact U-Net |
| Prediction | The model's estimated MA mask |
| Training | Adjusting model parameters so predictions resemble expert masks |
| Loss | A number describing how different a prediction is from the target |
| Validation | Data used to choose settings during development |
| Test | Held-back data used once for the final evaluation |

### What does a U-Net learn?

A U-Net is a neural network designed for image segmentation. It does not simply
memorize a brightness threshold. During training, it learns many small numerical
filters that can respond to local texture, contrast, shape, and surrounding context.

The first half of the network summarizes increasingly broad image context. The second
half combines that context with fine image detail to produce one prediction for each
pixel. Its output is a probability-like score at every pixel; MicroPhaseLab converts
that score into a binary mask using a selected threshold.

The model can be useful when MA regions cannot be described reliably by one global
brightness rule. It can still fail when the real image differs from the training
examples, when labels are uncertain, or when important sample conditions are absent
from the training data.

## Why not use pixel accuracy alone?

Suppose MA occupies 5% of an image. A model that predicts background everywhere would
still obtain 95% pixel accuracy, but it would detect no MA at all. Segmentation needs
metrics that focus on the foreground region:

- **Precision:** Of all pixels predicted as MA, how many are truly MA?
- **Recall:** Of all expert-labelled MA pixels, how many were found?
- **Dice:** How strongly do predicted and expert MA regions overlap?
- **IoU:** A stricter overlap measure than Dice.
- **Area-fraction absolute error:** How different are the predicted and expert 2D MA
  fractions?

Read these metrics together. A small area-fraction error can coexist with a poor IoU
if the prediction has the correct total area but the wrong boundary or location.

## How machine learning supports materials science

Microstructure images contain information about phases, constituents, grain features,
porosity, inclusions, and defects. Machine learning can help turn repeated visual
inspection into a consistent and measurable workflow:

1. **Segmentation:** Identify pixels belonging to a selected phase or constituent.
2. **Quantification:** Measure area fraction, object count, size distribution, shape,
   or spatial arrangement after segmentation.
3. **Comparison:** Compare microstructures across compositions, heat treatments,
   processing routes, or imaging conditions.
4. **Screening:** Help experts prioritize images that need review or detect unusual
   regions in large image collections.
5. **Research support:** Create structured image-derived measurements that can be
   related to processing parameters or measured material properties.

Machine learning is not a substitute for metallographic expertise. It is most useful
when experts define the scientific target, inspect data quality, and interpret errors.
The output is only as trustworthy as the image acquisition, labels, split design, and
evaluation procedure behind it.

## Recommended study sequence

### Lesson A: data representations

Run the offline demo and open the generated image, polygon CSV, mask, and overlay.
Then complete [`01_data_pipeline.ipynb`](../notebooks/01_data_pipeline.ipynb). Explain
how one polygon becomes many pixel labels and how a baseline prediction is compared
with the reference mask.

### Lesson B: quality and leakage

Read the quality report and create group-aware splits. Explain why images from the
same material sample should not appear in both training and test data.

### Lesson C: classical baseline

Run the Otsu plus morphology baseline. Change one parameter on validation data and
observe how false positives and false negatives change. Then complete
[`02_classical_baseline_analysis.ipynb`](../notebooks/02_classical_baseline_analysis.ipynb)
to inspect per-image metrics and coloured false-positive/false-negative errors.

### Lesson D: neural-network segmentation

Install the optional PyTorch dependency and train the compact U-Net. Read metrics.csv,
identify the best validation epoch, then freeze the setting.

### Lesson E: scientific interpretation

Predict and evaluate on the test split once. Review at least 20 comparison figures and
complete docs/experiment_report_template.md. Describe what the model can support and
what it cannot justify.

## Final checklist

Before reporting a result, confirm all of the following:

- The data quality report passes.
- At least 20 overlays were reviewed.
- Groups do not overlap between splits.
- Model choices were made using validation data only.
- The test split was evaluated after settings were frozen.
- Metrics and visual examples are reported together.
- The report states data, label, and dimensionality limitations.

Following this workflow helps students learn more than how to run a model: it teaches
how to make a reproducible, evidence-based materials image-analysis claim.
