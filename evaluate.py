"""
evaluate.py
Load best checkpoint, run on test set, print metrics, save plots.

Usage:
    python evaluate.py --model resnet
    python evaluate.py --model baseline
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from data.dataset import build_dataloaders, CLASS_NAMES, NUM_CLASSES
from models.baseline_cnn import build_baseline
from models.resnet_transfer import build_resnet

DATA_DIR   = Path("data")
IMAGE_DIR  = DATA_DIR / "images"
LABEL_FILE = DATA_DIR / "labels.csv"
OUT_DIR    = Path("outputs")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",      choices=["baseline", "resnet"], default="resnet")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw counts", "Normalised (row = true class)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(title)

    plt.tight_layout()
    path = OUT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Confusion matrix saved → {path}")


def plot_training_history(model_name: str) -> None:
    history_path = OUT_DIR / f"history_{model_name}.json"
    if not history_path.exists():
        print(f"[eval] No training history found at {history_path} — skipping plot")
        return

    with open(history_path) as f:
        data = json.load(f)

    history = data["history"]
    epochs      = [r["epoch"]      for r in history]
    train_loss  = [r["train_loss"] for r in history]
    val_loss    = [r["val_loss"]   for r in history]
    train_acc   = [r["train_acc"]  for r in history]
    val_acc     = [r["val_acc"]    for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=13, fontweight="bold")

    ax1.plot(epochs, train_loss, label="Train loss",  marker="o", markersize=3)
    ax1.plot(epochs, val_loss,   label="Val loss",    marker="o", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss"); ax1.legend()

    ax2.plot(epochs, train_acc, label="Train acc", marker="o", markersize=3)
    ax2.plot(epochs, val_acc,   label="Val acc",   marker="o", markersize=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend()
    ax2.set_ylim(0, 1)

    # Draw vertical line at phase boundary if present
    phases = [r.get("phase", "A") for r in history]
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            for ax in (ax1, ax2):
                ax.axvline(x=epochs[i], color="gray", linestyle="--", alpha=0.6, label="Phase B start")

    plt.tight_layout()
    path = OUT_DIR / f"training_curves_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Training curves saved → {path}")


def plot_per_class_f1(y_true, y_pred, model_name: str) -> None:
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    f1s = [report[c]["f1-score"] for c in CLASS_NAMES]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, f1s, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-class F1 — {model_name}", fontweight="bold")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(y=1/NUM_CLASSES, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()
    plt.tight_layout()
    path = OUT_DIR / f"per_class_f1_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Per-class F1 saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    checkpoint = OUT_DIR / f"best_{args.model}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Run train.py first."
        )

    # Load data
    _, _, test_loader = build_dataloaders(
        label_file=LABEL_FILE,
        image_dir=IMAGE_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Load model
    if args.model == "resnet":
        model = build_resnet(num_classes=NUM_CLASSES, freeze_backbone=False, pretrained=False)
    else:
        model = build_baseline(num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"[eval] Loaded checkpoint: {checkpoint}")

    # Inference
    y_pred, y_true = get_predictions(model, test_loader)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'='*55}")
    print(f"  Test Results — {args.model}")
    print(f"{'='*55}")
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted F1      : {f1:.4f}")
    print(f"  Macro F1         : {f1_macro:.4f}")
    print(f"  Random baseline  : {1/NUM_CLASSES:.4f}  ({100/NUM_CLASSES:.1f}%)")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    # Save metrics to JSON
    metrics = {
        "model": args.model,
        "accuracy": round(acc, 4),
        "weighted_f1": round(f1, 4),
        "macro_f1": round(f1_macro, 4),
        "per_class": {
            c: {k: round(v, 4) for k, v in classification_report(
                y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
            )[c].items()}
            for c in CLASS_NAMES
        },
    }
    metrics_path = OUT_DIR / f"metrics_{args.model}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Metrics saved → {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, args.model)
    plot_training_history(args.model)
    plot_per_class_f1(y_true, y_pred, args.model)

    print("\n[eval] Done. All outputs in outputs/")


if __name__ == "__main__":
    main()
