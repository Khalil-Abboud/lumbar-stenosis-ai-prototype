# Lumbar Stenosis AI — Article 2 Research Prototype

This repository contains the practical software prototype for the second
research article. It implements a reproducible pipeline:

```text
raster image or individual DICOM slice
        ↓
safe loading and preprocessing
        ↓
ResNet18 feature extractor (512 values)
        ↓
training-only min/max scaling and complement coding
        ↓
supervised Fuzzy ART-MAP classifier
        ↓
class prediction, ART match diagnostics, and test metrics
```

> **Research use only.** This software is not a medical device. It must not be
> used to diagnose lumbar stenosis or decide treatment or surgery.

## What is implemented

- Raster (`JPG`, `PNG`, `TIFF`, …) and individual MR DICOM-slice loading.
- DICOM modality/VOI transforms, windowing, `MONOCHROME1` handling, and
  explicit rejection of non-MR or multi-frame input.
- ImageNet-pretrained ResNet18 as a 512-dimensional feature extractor.
- A real incremental **Fuzzy ART-MAP** implementation with:
  - min/max scaling fitted on training data only;
  - complement coding;
  - choice and match functions;
  - vigilance and match tracking;
  - supervised category mapping;
  - `fit`, `partial_fit`, and model persistence.
- Patient-level splitting that prevents slices from one patient appearing in
  both training and test sets.
- One explicit experiment stratum per model (`plane`, `level`, and `sequence`)
  so anatomically different inputs are not pooled silently.
- Accuracy, balanced accuracy, sensitivity/recall, specificity, precision,
  F1, and a confusion matrix on unseen test samples only.
- A command-line interface for validation, feature extraction, training,
  incremental updating, prediction, and a non-medical synthetic self-check.

## Project structure

```text
lumbar_stenosis_ai/
├── artifacts.py          model, feature, prediction, and metric files
├── cli.py                command-line interface
├── config.py             reproducible run configuration
├── data/
│   ├── image_io.py       raster and DICOM readers
│   └── manifest.py       anonymized dataset contract and validation
├── features/
│   └── resnet18.py       batched 512-feature extraction
├── models/
│   └── fuzzy_artmap.py   supervised incremental Fuzzy ART-MAP
├── evaluation.py         dependency-free classification metrics
├── pipeline.py           training and inference orchestration
└── splitting.py          leakage-safe patient splitting
tests/                    automated unit tests
run_pipeline.py           thin launcher
```

## Installation

Activate the project virtual environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The same CLI is available through either command:

```powershell
python run_pipeline.py --help
python -m lumbar_stenosis_ai --help
```

## Fast technical self-check

This checks ART-MAP training, prediction, metrics, and serialization using
small synthetic vectors. It is **not** a medical result:

```powershell
python run_pipeline.py self-check
```

## Dataset manifest

Experiments use an anonymized UTF-8 CSV with exactly these columns:

```text
sample_id,patient_id,image_path,label,split,plane,level,sequence
```

- `sample_id`: anonymous slice/sample identifier.
- `patient_id`: anonymous patient identifier; never use a name or medical ID.
- `image_path`: path to a raster image or one DICOM slice.
- `label`: supervised target defined with medical collaborators.
- `split`: `train` or `test`; leave every row blank for automatic
  patient-level splitting.
- `plane`: for example `sagittal` or `axial`.
- `level`: for example `L4-L5` or `L5-S1`.
- `sequence`: for the current data, `T2`, confirmed by the medical team.

Every training run must contain one plane, one lumbar level, and one sequence.
Train separate model bundles for different anatomical strata.

Do not add names, dates of birth, phone numbers, addresses, passport data, or
other identifying columns. See `examples/manifest_example.csv` for the format.

Validate a manifest before running an experiment:

```powershell
python run_pipeline.py validate --manifest path\to\manifest.csv
```

## Train and evaluate

```powershell
python run_pipeline.py train `
  --manifest path\to\manifest.csv `
  --output-dir runs\experiment_001
```

The output directory must be new or empty. This prevents files from an older
experiment being mixed with a new model.

The output directory contains:

- `metadata.json`: configuration, software versions, split method, seed, and
  ART-MAP category information, plus the exact ResNet weights/preprocessing
  identifiers and model-state checksum;
- `model_state.npz`: model/scaler arrays;
- `manifest_snapshot.csv`: deidentified, path-free sample/label/anatomy and
  image-hash snapshot (patient identifiers are stored only as salted hashes);
- `train_features.npz` and `test_features.npz`;
- `test_predictions.csv`, `patient_test_predictions.csv`, and `metrics.json`
  when an unseen test set exists. Metrics contain both slice-level results and
  patient-level majority-vote results; tied patient votes are rejected.

The `match` value in predictions is an ART similarity diagnostic, **not a
probability**. If no ART category reaches the configured vigilance threshold,
the prediction is rejected (`predicted_label: null`, `requires_review: true`).

## Predict new slices

```powershell
python run_pipeline.py predict `
  --model-dir runs\experiment_001 `
  --image path\to\new_slice.dcm
```

## Add newly labeled cases incrementally

ART-MAP can learn new labeled observations without discarding the categories
already learned. The update is deliberately written to a new directory so the
previous model remains reproducible:

```powershell
python run_pipeline.py update `
  --model-dir runs\experiment_001 `
  --manifest path\to\new_labeled_cases.csv `
  --output-dir runs\experiment_001_update_01
```

The original min/max feature bounds remain fixed during incremental learning.
The run metadata reports how many new feature values fell outside those bounds
and were clipped to the valid fuzzy-input interval. An original test patient,
duplicate sample, or duplicate image cannot be added later through `update`.
The initial training set must therefore be a representative calibration set;
if clipping is substantial, create a new full training run instead of treating
the incremental result as reliable.

## Current scientific boundary

This version establishes the article-2 architecture and program product. It
does **not yet** implement automatic vertebral-level localization, dural-sac
segmentation, cross-sectional-area measurement, stenosis grading, or surgical
decision support. Those modules require labeled T2 sagittal/axial studies and
medical validation and are deliberately preserved for the next research
stages.
