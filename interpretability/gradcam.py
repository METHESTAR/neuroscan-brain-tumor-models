"""Create a Grad-CAM visualization for a NeuroScan MRI classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from inference import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM heatmap for one MRI image and trained checkpoint."
    )
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "convnext_tiny.pth")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "gradcam" / "gradcam.png")
    parser.add_argument("--target-class", help="Class whose evidence to visualize; defaults to the model prediction.")
    parser.add_argument("--alpha", type=float, default=0.42, help="Heatmap overlay opacity (0 to 1).")
    return parser.parse_args()


def get_target_layer(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    if architecture == "convnext_tiny":
        return model.features[-1]
    if architecture == "resnet50":
        return model.layer4[-1]
    if architecture == "swin_tiny":
        # Torchvision Swin features are NHWC, unlike the convolutional models.
        return model.features[-1]
    raise ValueError(f"Unsupported architecture: {architecture}")


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"MRI image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, image_size = load_model(args.model, device)
    if args.target_class and args.target_class not in classes:
        raise ValueError(f"Unknown target class {args.target_class!r}; choose from {classes}")

    original = Image.open(args.image).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image_tensor = preprocess(original).unsqueeze(0).to(device)
    # Use the architecture recorded in the checkpoint because several torchvision
    # model classes have similar names.
    bundle = torch.load(args.model, map_location="cpu", weights_only=True)
    architecture = bundle["architecture"]
    target_layer = get_target_layer(model, architecture)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture_activation(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        probabilities = logits.softmax(dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        target_index = classes.index(args.target_class) if args.target_class else predicted_index
        logits[0, target_index].backward()
    finally:
        hook.remove()

    activation = activations[-1].detach()
    gradient = gradients[-1].detach()
    if architecture == "swin_tiny":
        activation = activation.permute(0, 3, 1, 2)
        gradient = gradient.permute(0, 3, 1, 2)
    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(original.height, original.width), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.cpu().numpy()
    cam_min, cam_max = float(cam.min()), float(cam.max())
    cam = (cam - cam_min) / max(cam_max - cam_min, 1e-8)

    predicted_class = classes[predicted_index]
    target_class = classes[target_index]
    true_label = args.image.parent.name.lower()
    if true_label not in classes:
        true_label = None
    confidence = float(probabilities[predicted_index].item())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Grad-CAM evidence map | NeuroScan AI", fontsize=15, fontweight="bold", y=0.96)
    fig.subplots_adjust(left=0.035, right=0.91, top=0.86, bottom=0.21, wspace=0.08)
    axes[0].imshow(original)
    axes[0].set_title("MRI input")
    axes[1].imshow(original)
    heat = axes[1].imshow(cam, cmap="jet", alpha=args.alpha, vmin=0, vmax=1)
    axes[1].set_title(f"Evidence for: {target_class}")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.03, label="Relative Grad-CAM intensity")
    fig.text(
        0.47,
        0.12,
        f"Prediction: {predicted_class} ({confidence:.1%})"
        + (f"  |  Folder label: {true_label}" if true_label else ""),
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.47,
        0.065,
        "Heatmap shows influential regions; it is not a tumor outline or clinical explanation.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    metadata = {
        "model": architecture,
        "checkpoint": args.model.name,
        "image_file": args.image.name,
        "folder_label": true_label,
        "predicted_class": predicted_class,
        "predicted_probability": confidence,
        "heatmap_target_class": target_class,
        "target_layer": "features[-1]" if architecture in {"convnext_tiny", "swin_tiny"} else "layer4[-1]",
        "device": str(device),
        "interpretation_note": "Grad-CAM is a coarse model-attention visualization, not a tumor segmentation or diagnostic result.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
