"""
gradcam.py
Grad-CAM visualisation — stretch goal.
Shows which regions of a galaxy image drove the model's prediction.

Usage:
    python gradcam.py --model resnet --image_path data/images/12345.jpg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from data.dataset import CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from models.resnet_transfer import build_resnet

OUT_DIR = Path("outputs")
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Grad-CAM implementation ────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Hooks into the last convolutional layer of the network and computes
    a weighted combination of the feature maps, guided by the gradients
    flowing back from the target class score.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None):
        """
        Compute the Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) preprocessed image tensor.
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            cam:        (H, W) numpy array in [0, 1].
            pred_class: int — the predicted class index.
            pred_probs: (num_classes,) softmax probabilities.
        """
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE).requires_grad_(True)

        logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().detach().numpy()
        pred_class = int(np.argmax(probs))

        if target_class is None:
            target_class = pred_class

        # Backprop w.r.t. the target class score
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (h, w)
        cam = F.relu(torch.tensor(cam)).numpy()

        # Normalise and resize to input resolution
        if cam.max() > 0:
            cam = cam / cam.max()
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = np.array(
            Image.fromarray(cam).resize((w, h), resample=Image.BILINEAR)
        )
        return cam, pred_class, probs


# ── Visualisation ─────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation for display."""
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std  = np.array(IMAGENET_STD)[:, None, None]
    img = tensor.cpu().numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def visualise_gradcam(
    image_path: Path,
    model: torch.nn.Module,
    target_class: int | None = None,
) -> None:
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    input_t = preprocess(img_pil).unsqueeze(0)

    # Hook into ResNet's last conv block
    target_layer = model.layer4[-1]
    gcam = GradCAM(model, target_layer)
    cam, pred_class, probs = gcam.generate(input_t, target_class=target_class)

    img_np = denormalize(input_t.squeeze())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Grad-CAM  |  Predicted: {CLASS_NAMES[pred_class]} "
        f"({probs[pred_class]*100:.1f}%)",
        fontsize=13, fontweight="bold",
    )

    axes[0].imshow(img_np)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    overlay = img_np.copy()
    heatmap = plt.cm.jet(cam)[..., :3]
    blend   = 0.45 * overlay + 0.55 * heatmap
    axes[2].imshow(np.clip(blend, 0, 1))
    axes[2].set_title("Overlay (α=0.55)")
    axes[2].axis("off")

    # Probability bar
    fig.text(
        0.5, -0.02,
        "  ".join(f"{c}: {p*100:.1f}%" for c, p in zip(CLASS_NAMES, probs)),
        ha="center", fontsize=10, color="gray",
    )

    plt.tight_layout()
    out_path = OUT_DIR / f"gradcam_{Path(image_path).stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[gradcam] Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path",   required=True, type=Path)
    p.add_argument("--model",        choices=["resnet"], default="resnet")
    p.add_argument("--target_class", type=int, default=None,
                   help="Class index to explain (default: predicted class)")
    args = p.parse_args()

    checkpoint = OUT_DIR / f"best_{args.model}.pt"
    model = build_resnet(num_classes=NUM_CLASSES, freeze_backbone=False, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)

    visualise_gradcam(args.image_path, model, args.target_class)


if __name__ == "__main__":
    main()
