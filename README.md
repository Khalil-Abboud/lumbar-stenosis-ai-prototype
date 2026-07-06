# Lumbar Stenosis AI Prototype

## Project idea

This project is an early software prototype for an intelligent decision-support system for lumbar spinal stenosis analysis based on MRI images.

The current version is not a medical diagnostic system. It demonstrates the technical pipeline of image preprocessing, feature extraction, and preliminary online classification.

## Current pipeline

1. Load MRI or medical image files from the `data` folder.
2. Preprocess images using the transformation pipeline required by ResNet18.
3. Extract a 512-dimensional feature vector using pretrained ResNet18.
4. Save extracted features together with image labels into `image_features.csv`.
5. Load the feature table and train a simplified online prototype classifier.
6. Save classification results into `classification_results.csv`.

## Files

- `main.py`  
  Extracts image features using ResNet18 and saves them to `image_features.csv`.

- `classifier_demo.py`  
  Reads extracted features and labels, trains a simplified online classifier, and saves the results.

- `labels.csv`  
  Contains image names and their class labels.

- `image_features.csv`  
  Contains image names, labels, and 512 extracted features for each image.

- `classification_results.csv`  
  Contains true labels, predicted labels, distances, and correctness of predictions.

## Current limitations

- The current labels are experimental and are not medically validated.
- The classifier is a simplified online prototype, not a full ART-MAP implementation.
- Testing is currently performed on the same images used for learning.
- Medical validation requires a real labeled MRI dataset provided by specialists.

## Planned development

The next stage is to replace or extend the simplified online classifier with an ART-MAP-based model. The goal is to support incremental learning from heterogeneous medical data, including retrospective cases and newly collected patient data.