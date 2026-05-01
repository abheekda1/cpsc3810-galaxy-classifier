# Deep Space, Deeper Learning
## Galaxy Morphology Classifier — CPSC 381/581 Final Project

**Authors:** Jeet Parikh, Abheek Dhawan  
**Course:** CPSC 381/581: Machine Learning, Yale University

---

## Project Overview

A convolutional neural network pipeline that classifies galaxy images from the
Sloan Digital Sky Survey (SDSS) into morphological types using citizen-labeled
data from Galaxy Zoo 2.

**Morphology classes:**
| ID | Class | Description |
|----|-------|-------------|
| 0 | Smooth | Elliptical / lenticular — featureless, round |
| 1 | Disk | Spiral — clearly shows disk structure or arms |
| 2 | Irregular | Mergers, peculiar, or ambiguous morphology |

---

## Project Structure

```
galaxy-classifier/
├── data/
│   ├── __init__.py
│   ├── download.py        # Download SDSS images + build labels.csv
│   └── dataset.py         # PyTorch Dataset, transforms, DataLoaders
├── models/
│   ├── __init__.py
│   ├── baseline_cnn.py    # Custom 4-layer CNN (Phase 1 baseline)
│   └── resnet_transfer.py # ResNet-18 transfer learning (main model)
├── outputs/               # Checkpoints, metrics, plots (auto-created)
├── train.py               # Full training loop (Phase A + Phase B)
├── evaluate.py            # Test-set evaluation + plots
├── gradcam.py             # Grad-CAM interpretability (stretch goal)
├── app.py                 # Gradio web demo (stretch goal)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. GPU strongly recommended for training (CPU works but is slow).

### 2. Download the dataset

**Step A — Get the GZ2 catalog:**  
Download `gz2_hart16.csv` from https://data.galaxyzoo.org/ and place it in `data/`.

**Step B — Fetch SDSS images:**
```bash
python -m data.download
```

This will:
- Assign confident morphology labels (≥60 % voter agreement)
- Balance to ≤5,000 galaxies per class
- Download 128×128 JPEG cutouts from the SDSS SkyServer
- Save `data/labels.csv`

> ⚠️ Downloading ~15,000 images takes ~30–60 min depending on your connection.
> The script is resumable — already-downloaded images are skipped.

---

## Training

### Baseline CNN

```bash
python train.py --model baseline --epochs_a 40
```

### ResNet-18 (recommended)

Two-phase training:
- **Phase A** (10 epochs): Only the classification head is trained. The ImageNet
  backbone is frozen to avoid disrupting pretrained features before the head converges.
- **Phase B** (30 epochs): Full fine-tuning at a lower backbone LR (1e-4 vs 1e-3).

```bash
python train.py --model resnet --epochs_a 10 --epochs_b 30
```

**All hyperparameters:**
```
--batch_size   32       Mini-batch size
--epochs_a     10       Phase A epochs (head only)
--epochs_b     30       Phase B epochs (full fine-tune)
--head_lr      1e-3     Learning rate for FC head
--backbone_lr  1e-4     Learning rate for backbone (Phase B)
--weight_decay 1e-4     L2 regularisation
--patience     7        Early stopping patience (val loss)
--seed         42       Random seed
```

Checkpoints and training history are saved to `outputs/`.

---

## Evaluation

```bash
python evaluate.py --model resnet
```

Outputs saved to `outputs/`:
- `metrics_resnet.json` — accuracy, weighted F1, macro F1, per-class breakdown
- `confusion_matrix_resnet.png` — raw and normalised confusion matrix
- `training_curves_resnet.png` — loss and accuracy vs. epoch
- `per_class_f1_resnet.png` — bar chart of per-class F1 scores

---

## Grad-CAM (Stretch Goal)

Visualise which image regions drove a prediction:

```bash
python gradcam.py --model resnet --image_path data/images/12345.jpg
```

Saves a 3-panel figure (original | heatmap | overlay) to `outputs/gradcam_<id>.png`.

---

## Web Demo (Stretch Goal)

```bash
pip install gradio
python app.py
```

Opens a local web interface where you can upload any galaxy image and see
the prediction + Grad-CAM overlay in real time.

---

## Expected Results

| Model | Test Accuracy | Weighted F1 |
|-------|--------------|-------------|
| Random baseline | 33.3% | 0.333 |
| Baseline CNN | ~75–80% | ~0.75 |
| ResNet-18 (fine-tuned) | ~85–90% | ~0.87 |

---

## Reproducing Results

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download data (requires gz2_hart16.csv in data/)
python -m data.download

# 3. Train ResNet
python train.py --model resnet --epochs_a 10 --epochs_b 30 --seed 42

# 4. Evaluate
python evaluate.py --model resnet
```

All outputs will appear in `outputs/`.

---

## References

- Hart et al. (2016). Galaxy Zoo 2: detailed morphological classifications.
  *MNRAS*, 461(4), 3663–3682.
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
- SDSS SkyServer: https://skyserver.sdss.org
- Galaxy Zoo data release: https://data.galaxyzoo.org
