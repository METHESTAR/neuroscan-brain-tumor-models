from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def load_model(path: Path, device: torch.device):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    architecture = bundle["architecture"]
    classes = bundle["classes"]
    if architecture == "swin_tiny":
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, len(classes))
    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    return model, classes, int(bundle["image_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one MRI image with a NeuroScan model checkpoint.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, size = load_model(args.model, device)
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        probabilities = model(preprocess(image).unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()
    result = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    print(json.dumps({"device": str(device), "probabilities": result}, indent=2))


if __name__ == "__main__":
    main()
