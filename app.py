"""
app.py
Gradio web demo — stretch goal.
Lets anyone upload a galaxy image and see the predicted morphology + Grad-CAM.

Install: pip install gradio
Run:     python app.py
"""

from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from data.dataset import CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from models.resnet_transfer import build_resnet
from gradcam import GradCAM, denormalize

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path("outputs/best_resnet.pt")

# ── Load model once ──────────────────────────────────────────────────────────
model = build_resnet(num_classes=NUM_CLASSES, freeze_backbone=False, pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model = model.to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def classify_galaxy(pil_image: Image.Image):
    """
    Main inference function called by Gradio.
    Returns: (label string, confidence dict, overlay numpy image)
    """
    if pil_image is None:
        return "No image provided", {}, None

    img_rgb = pil_image.convert("RGB")
    input_t = preprocess(img_rgb).unsqueeze(0)

    # Grad-CAM
    gcam = GradCAM(model, model.layer4[-1])
    cam, pred_class, probs = gcam.generate(input_t)

    label = f"{CLASS_NAMES[pred_class]}  ({probs[pred_class]*100:.1f}% confidence)"
    confidence = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    # Build overlay image
    img_np  = np.array(img_rgb.resize((128, 128))).astype(float) / 255.0
    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    return label, confidence, overlay


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Galaxy Morphology Classifier") as demo:
    gr.Markdown(
        """
        # 🔭 Galaxy Morphology Classifier
        Upload a galaxy image (e.g. from SDSS SkyServer) and the model will predict its
        morphological type and highlight the regions that influenced the decision.

        **Classes:** Smooth (elliptical/lenticular) · Disk (spiral) · Irregular/merger
        """
    )

    with gr.Row():
        inp = gr.Image(type="pil", label="Galaxy image")
        with gr.Column():
            label_out  = gr.Textbox(label="Prediction")
            conf_out   = gr.Label(label="Class probabilities", num_top_classes=3)
            overlay_out = gr.Image(type="numpy", label="Grad-CAM overlay")

    btn = gr.Button("Classify", variant="primary")
    btn.click(classify_galaxy, inputs=inp, outputs=[label_out, conf_out, overlay_out])

    gr.Examples(
        examples=[],  # Add example image paths here if available
        inputs=inp,
    )

if __name__ == "__main__":
    demo.launch(share=False)
