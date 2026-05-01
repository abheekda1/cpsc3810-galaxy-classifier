"""
data/dataset.py
PyTorch Dataset + transform definitions for the Galaxy Zoo classifier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch
import numpy as np

# ── Label metadata ─────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Smooth", "Disk", "Irregular"]
NUM_CLASSES  = len(CLASS_NAMES)

# ── Image normalisation (ImageNet stats — works well for transfer learning) ────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline.

    Training augmentation rationale:
      - Random horizontal/vertical flip + 360° rotation: galaxies have no
        canonical orientation, so all rotations are equally valid.
      - ColorJitter: handles variation in SDSS photometric calibration.
      - No crop: galaxy morphology is global; cropping can remove key features.

    Args:
        split: One of 'train', 'val', 'test'.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test — deterministic
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class GalaxyDataset(Dataset):
    """
    Galaxy morphology dataset.

    Args:
        df:         DataFrame with columns ['objid', 'label'].
        image_dir:  Directory containing <objid>.jpg files.
        split:      'train', 'val', or 'test' (controls augmentation).
    """

    def __init__(self, df: pd.DataFrame, image_dir: Path, split: str = "train"):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = get_transforms(split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        path  = self.image_dir / f"{row['objid']}.jpg"
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Create a sampler that up-samples rare classes so each mini-batch
    is approximately class-balanced even if the raw dataset isn't.
    """
    class_counts = df["label"].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[df["label"].values]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(df),
        replacement=True,
    )


def build_dataloaders(
    label_file: Path,
    image_dir: Path,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load labels, stratified-split into train/val/test, return DataLoaders.

    Splitting is done before any augmentation is applied.

    Args:
        label_file:  Path to labels.csv produced by download.py.
        image_dir:   Directory with galaxy JPEG images.
        batch_size:  Mini-batch size for all loaders.
        val_frac:    Fraction of data held out for validation.
        test_frac:   Fraction of data held out for final test.
        num_workers: Parallel workers for DataLoader.
        seed:        Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(label_file)

    # Filter to images that exist on disk
    df = df[df["objid"].apply(lambda x: (image_dir / f"{x}.jpg").exists())]
    print(f"[dataset] {len(df):,} images found on disk")

    # Stratified split: train / val / test
    train_df, temp_df = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp_df["label"],
        random_state=seed,
    )

    print(f"[dataset] Split — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    train_ds = GalaxyDataset(train_df, image_dir, split="train")
    val_ds   = GalaxyDataset(val_df,   image_dir, split="val")
    test_ds  = GalaxyDataset(test_df,  image_dir, split="test")

    sampler = make_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
