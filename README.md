# Deep Space, Deeper Learning
## Galaxy Morphology Classifier with Convolutional Neural Networks

**Authors:** Jeet Parikh, Abheek Dhawan  
**Course:** CPSC 3810

---

## Overview

This project trains convolutional neural networks to classify galaxy images from the
Sloan Digital Sky Survey (SDSS) into three morphological types using citizen-labeled
data from Galaxy Zoo 2.

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Smooth | Elliptical / lenticular — featureless, round |
| 1 | Disk | Spiral — clearly shows disk structure or arms |
| 2 | Irregular | Mergers, peculiar, or ambiguous morphology |

Two models are trained and compared:
1. **Baseline CNN** — custom 4-layer convolutional network trained from scratch
2. **ResNet-18** — ImageNet pretrained model fine-tuned using two-phase transfer learning

---

## Repository Structure

```
cpsc3810-galaxy-classifier/
├── galaxy_classifier.ipynb   # Main notebook — all code lives here
├── data/
│   ├── images/               # SDSS galaxy JPEG images
│   ├── labels.csv            # Object IDs and class labels
│   └── download.py           # Script used to originally fetch images from SDSS
├── outputs/                  # Auto-created: checkpoints, plots, metrics
└── README.md
```

All model training, evaluation, and visualisation is contained in
`galaxy_classifier.ipynb`. The `outputs/` directory is created automatically
when the notebook runs and will contain:

| File | Description |
|------|-------------|
| `best_baseline.pt` | Best baseline CNN checkpoint |
| `best_resnet.pt` | Best ResNet-18 checkpoint |
| `history_baseline.json` | Per-epoch training history for baseline |
| `history_resnet.json` | Per-epoch training history for ResNet |
| `results_summary.json` | Final test accuracy and F1 scores |
| `sample_images.png` | Grid of example images per class |
| `training_curves.png` | Loss and accuracy vs. epoch for both models |
| `confusion_matrices.png` | Raw and normalised confusion matrices |
| `per_class_f1_comparison.png` | Per-class F1 bar chart |
| `gradcam_grid.png` | Grad-CAM visualisations for ResNet-18 |

---

## How to Run (Google Colab)

### Step 1 — Enable GPU

Runtime → Change runtime type → T4 GPU → Save

### Step 2 — Clone the repository to Google Drive

The notebook expects the project to live at
`/content/drive/MyDrive/cpsc3810-galaxy-classifier`. In a Colab cell, run:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/abheekda1/cpsc3810-galaxy-classifier /content/drive/MyDrive/cpsc3810-galaxy-classifier
```

This only needs to be done once — the cloned repo persists in Drive across sessions.

### Step 3 — Open the notebook

In the Files panel on the left, navigate to:
```
drive/MyDrive/cpsc3810-galaxy-classifier/galaxy_classifier.ipynb
```
Double-click to open it, or go to File → Open notebook → Google Drive and select it.

### Step 4 — Run all cells

Click Runtime → Run all. The notebook will automatically:

1. Mount Google Drive
2. `cd` into the repository
3. Install dependencies
4. Verify GPU is available
5. Load `data/labels.csv` and `data/images/` from the repo
6. Train the baseline CNN
7. Train ResNet-18 in two phases 
8. Evaluate both models on the test set
9. Save all plots, metrics, and checkpoints to `outputs/`
10. Generate Grad-CAM visualisations

All outputs are written to the `outputs/` directory inside the cloned repo.
To save checkpoints and plots across sessions, copy them to Drive using the
cell at the end of Section 10:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs/ /content/drive/MyDrive/galaxy-outputs/
```

---

## About the Dataset

Images were downloaded from the SDSS SkyServer using `data/download.py`, which:

1. Reads the Galaxy Zoo 2 catalog (`gz2_hart16.csv`, available at https://data.galaxyzoo.org)
2. Assigns morphology labels based on weighted citizen vote fractions (≥ 60% agreement threshold)
3. Fetches 128×128 JPEG cutouts from the SDSS SkyServer for each galaxy
4. Saves the resulting `data/images/` folder and `data/labels.csv`

The images and labels are already included in the repository — you do not need
to run `download.py` to reproduce the results.

---

## Notebook Sections

| Section | Description |
|---------|-------------|
| 0. Setup | Mount Drive, clone repo, install deps, check GPU |
| 1. Hyperparameters | All training config in one place |
| 2. Imports | All required libraries |
| 3. Data | Load labels, visualise samples, define transforms and DataLoaders |
| 4. Models | BaselineCNN and ResNet-18 definitions |
| 5. Training loop | `train_one_epoch`, `evaluate`, `run_phase` functions |
| 6. Train baseline | Train custom CNN with SGD |
| 7. Train ResNet-18 | Phase A (frozen backbone) then Phase B (full fine-tune) |
| 8. Evaluation | Test metrics, confusion matrices, training curves, F1 comparison |
| 9. Grad-CAM | Interpretability visualisations (stretch goal) |
| 10. Results summary | Final accuracy and F1 table, saves `results_summary.json` |

---

## Key Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Image size | 128 × 128 |
| Batch size | 32 |
| Train / val / test split | 70% / 15% / 15% |
| Baseline optimizer | SGD (momentum=0.9) |
| ResNet optimizer | Adam |
| Head learning rate | 1e-3 |
| Backbone learning rate (Phase B) | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 7 epochs |
| Random seed | 42 |

---

## Expected Results

| Model | Test Accuracy | Weighted F1 |
|-------|--------------|-------------|
| Random baseline | 33.3% | — |
| Baseline CNN | ~83% | ~0.83 |
| ResNet-18 | ~94% | ~0.94 |