from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import models, transforms


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def laplacian_variance(image: Image.Image) -> float:
    """A simple sharpness score; lower values generally indicate more blur."""
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(lap[1:-1, 1:-1].var())


def load_gate(checkpoint: str | Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model, state


def gate_tensor(image: Image.Image) -> torch.Tensor:
    pipeline = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    return pipeline(image.convert("RGB")).unsqueeze(0)


def score_brain_mri(model, image: Image.Image, device: torch.device) -> float:
    with torch.inference_mode():
        logits = model(gate_tensor(image).to(device))
        return float(torch.sigmoid(logits).item())


def assess_upload(image: Image.Image, model, state: dict, device: torch.device) -> dict:
    width, height = image.size
    if min(width, height) < 64:
        return {"accepted": False, "reason": "too_small", "sharpness": 0.0, "mri_score": 0.0}
    sharpness = laplacian_variance(image)
    blur_threshold = float(state["blur_threshold"])
    if sharpness < blur_threshold:
        return {
            "accepted": False,
            "reason": "blurry",
            "sharpness": sharpness,
            "mri_score": None,
        }
    mri_score = score_brain_mri(model, image, device)
    mri_threshold = float(state["mri_threshold"])
    accepted = mri_score >= mri_threshold
    return {
        "accepted": accepted,
        "reason": None if accepted else "not_brain_mri",
        "sharpness": sharpness,
        "mri_score": mri_score,
    }
