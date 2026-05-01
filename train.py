"""
train.py
Full training loop with:
  - Phase A: train FC head only (frozen backbone)
  - Phase B: fine-tune end-to-end (unfrozen backbone, lower LR)
  - Early stopping on validation loss
  - Best-checkpoint saving
  - Learning-rate scheduling

Usage:
    python train.py --model resnet --epochs_a 10 --epochs_b 30
    python train.py --model baseline --epochs_a 40
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.dataset import build_dataloaders, CLASS_NAMES, NUM_CLASSES
from models.baseline_cnn import build_baseline
from models.resnet_transfer import build_resnet, unfreeze_backbone, get_param_groups, count_params

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
IMAGE_DIR  = DATA_DIR / "images"
LABEL_FILE = DATA_DIR / "labels.csv"
OUT_DIR    = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Config ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train galaxy morphology classifier")
    p.add_argument("--model",      choices=["baseline", "resnet"], default="resnet")
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--epochs_a",   type=int,   default=10,   help="Phase A epochs (head only)")
    p.add_argument("--epochs_b",   type=int,   default=30,   help="Phase B epochs (full fine-tune)")
    p.add_argument("--head_lr",    type=float, default=1e-3, help="Head learning rate")
    p.add_argument("--backbone_lr",type=float, default=1e-4, help="Backbone LR (Phase B)")
    p.add_argument("--weight_decay",type=float,default=1e-4)
    p.add_argument("--patience",   type=int,   default=7,    help="Early-stop patience (epochs)")
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


def run_phase(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    num_epochs: int,
    patience: int,
    checkpoint_path: Path,
    phase_name: str,
    history: list[dict],
) -> None:
    """Generic training phase loop with early stopping."""
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        row = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,     4),
        }
        history.append(row)

        print(
            f"[{phase_name}] Epoch {epoch:3d}/{num_epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            wait = 0
            print(f"            ↳ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"[{phase_name}] Early stopping after {epoch} epochs (patience={patience})")
                break


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[train] Device: {DEVICE}")
    print(f"[train] Model:  {args.model}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        label_file=LABEL_FILE,
        image_dir=IMAGE_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Model
    if args.model == "resnet":
        model = build_resnet(num_classes=NUM_CLASSES, freeze_backbone=True)
    else:
        model = build_baseline(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    count_params(model)

    criterion   = nn.CrossEntropyLoss()
    checkpoint  = OUT_DIR / f"best_{args.model}.pt"
    history: list[dict] = []

    # ── Phase A: head only ────────────────────────────────────────────────────
    if args.epochs_a > 0:
        optimizer_a = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        scheduler_a = ReduceLROnPlateau(optimizer_a, patience=3, factor=0.5, verbose=True)
        print(f"\n{'='*60}")
        print("Phase A — training head only")
        print(f"{'='*60}")
        run_phase(
            model, train_loader, val_loader,
            optimizer_a, scheduler_a, criterion,
            args.epochs_a, args.patience, checkpoint, "A", history,
        )

    # ── Phase B: full fine-tune (resnet only) ─────────────────────────────────
    if args.model == "resnet" and args.epochs_b > 0:
        # Load best Phase A weights before unfreezing
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        unfreeze_backbone(model)
        count_params(model)

        optimizer_b = torch.optim.Adam(
            get_param_groups(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr),
            weight_decay=args.weight_decay,
        )
        scheduler_b = ReduceLROnPlateau(optimizer_b, patience=3, factor=0.5, verbose=True)
        print(f"\n{'='*60}")
        print("Phase B — full fine-tuning")
        print(f"{'='*60}")
        run_phase(
            model, train_loader, val_loader,
            optimizer_b, scheduler_b, criterion,
            args.epochs_b, args.patience, checkpoint, "B", history,
        )

    # Save training history
    history_path = OUT_DIR / f"history_{args.model}.json"
    with open(history_path, "w") as f:
        json.dump({"config": vars(args), "history": history}, f, indent=2)
    print(f"\n[train] History saved → {history_path}")
    print(f"[train] Best checkpoint → {checkpoint}")
    print("\nRun  python evaluate.py  to generate metrics and plots.")


if __name__ == "__main__":
    main()
