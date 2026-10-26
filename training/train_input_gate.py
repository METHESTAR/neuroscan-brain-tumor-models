from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MRI_ROOT = ROOT.parent / "data" / "prepared" / "swin_tiny_leakage_safe_v1"
DATA_ROOT = ROOT.parent / "data" / "input_gate"
OUT_PATH = ROOT / "models" / "input_gate_mobilenetv3_small.pth"
RESULT_PATH = ROOT / "results" / "input_gate" / "summary.json"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEDMNIST_BASE = "https://zenodo.org/records/10519652/files/"
NEGATIVES = {
    "organamnist": ("organamnist.npz", "68e3f8846a6bd62f0c9bf841c0d9eacc"),
}


def download(url: str, destination: Path, md5: str | None = None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {destination.name}…", flush=True)
        urllib.request.urlretrieve(url, destination)
    if md5:
        digest = hashlib.md5(destination.read_bytes()).hexdigest()
        if digest != md5:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {destination.name}: {digest}")


class ImageArray(Dataset):
    def __init__(self, images, labels, training=False):
        self.images, self.labels = images, labels
        augment = [transforms.Resize((224, 224))]
        if training:
            augment += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(7)]
        augment += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(augment)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if isinstance(image, Path):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image.convert("RGB")), torch.tensor(self.labels[index], dtype=torch.float32)


def get_mri_paths(split):
    paths = []
    for class_dir in sorted((MRI_ROOT / split).iterdir()):
        if class_dir.is_dir():
            paths.extend(p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    return paths


def read_medmnist(name, split, limit, seed):
    filename, md5 = NEGATIVES[name]
    path = DATA_ROOT / filename
    download(MEDMNIST_BASE + filename + "?download=1", path, md5)
    archive = np.load(path)
    x = archive[f"{split}_images"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(limit, len(x)), replace=False)
    return [Image.fromarray(x[i].squeeze()).convert("RGB") for i in indices]


def cap_paths(paths, limit, seed):
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    return paths[:limit]


def make_data(limit=4000, seed=42):
    # Equal-count source sampling limits dominance by one kind of non-MRI image.
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    mri_splits = {s: get_mri_paths(s) for s in ("train", "val", "test")}
    for split, target_x, target_y in (("train", train_x, train_y), ("val", val_x, val_y), ("test", test_x, test_y)):
        paths = cap_paths(mri_splits[split], limit, seed)
        target_x.extend(paths)
        target_y.extend([1.0] * len(paths))

    # OrganAMNIST is a compact public CT dataset. We deliberately use a focused
    # hard-negative set here; natural-photo rejection still needs broader input data.
    for source_index, source in enumerate(("organamnist",)):
        filename, md5 = NEGATIVES[source]
        path = DATA_ROOT / filename
        download(MEDMNIST_BASE + filename + "?download=1", path, md5)
        archive = np.load(path)
        packs = []
        for split, target_x, target_y in (("train", train_x, train_y), ("val", val_x, val_y), ("test", test_x, test_y)):
            imgs = archive[f"{split}_images"]
            rng = np.random.default_rng(seed + source_index + {"train": 0, "val": 11, "test": 22}[split])
            n = min(limit, len(imgs))
            indices = rng.choice(len(imgs), size=n, replace=False)
            samples = [Image.fromarray(imgs[i].squeeze()).convert("RGB") for i in indices]
            packs.append((target_x, target_y, samples))
        for target_x, target_y, samples in packs:
            target_x.extend(samples)
            target_y.extend([0.0] * len(samples))

    return (
        ImageArray(train_x, train_y, training=True),
        ImageArray(val_x, val_y),
        ImageArray(test_x, test_y),
    )


def evaluate(model, loader, device):
    model.eval()
    logits, labels = [], []
    loss_sum = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(1)
            loss_sum += float(criterion(out, y)) * len(y)
            logits.extend(out.cpu().tolist())
            labels.extend(y.cpu().tolist())
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.asarray(labels, dtype=int)
    pred = probs >= 0.5
    accuracy = float((pred == labels).mean())
    return loss_sum / len(labels), probs, labels, accuracy


def metrics_at(probs, labels, threshold):
    pred = probs >= threshold
    y = labels.astype(bool)
    return {
        "accuracy": float((pred == y).mean()),
        "brain_mri_sensitivity": float((pred[y]).mean()) if y.any() else 0.0,
        "non_mri_rejection_rate": float((~pred[~y]).mean()) if (~y).any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-source", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds, test_ds = make_data(args.max_per_source)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    best_val, best_epoch, stale = float("inf"), 0, 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(y)
        val_loss, val_probs, val_labels, val_acc = evaluate(model, val_loader, device)
        train_loss = total / len(train_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy_at_0_5": val_acc})
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_loss < best_val - 0.001:
            best_val, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict()}, OUT_PATH)
        else:
            stale += 1
            if stale >= args.patience:
                break

    # Set the MRI acceptance threshold to retain 99% of validation brain MRIs.
    best_state = torch.load(OUT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(best_state["model_state_dict"])
    val_loss, val_probs, val_labels, _ = evaluate(model, val_loader, device)
    mri_probs = val_probs[val_labels == 1]
    mri_threshold = float(np.quantile(mri_probs, 0.01))

    # Sharpness threshold preserves 99% of clean validation MRIs; report blur examples separately.
    from input_gate import laplacian_variance
    sharpness = [laplacian_variance(Image.open(p).convert("RGB")) for p in cap_paths(mri_splits_for_blur(), 1003, 42)]
    blur_threshold = float(np.quantile(sharpness, 0.01))
    torch.save({"model_state_dict": model.state_dict(), "mri_threshold": mri_threshold, "blur_threshold": blur_threshold}, OUT_PATH)

    test_loss, test_probs, test_labels, test_acc = evaluate(model, test_loader, device)
    # Synthetic Gaussian blur check on a small independent positive sample.
    blur_probe = []
    for path in cap_paths(mri_splits_for_blur(test=True), 200, 81):
        im = Image.open(path).convert("RGB")
        blur_probe.append({
            "clean": laplacian_variance(im),
            "gaussian_blur_radius_1_5": laplacian_variance(im.filter(ImageFilter.GaussianBlur(1.5))),
        })
    blur_detection_rates = {
        "gaussian_radius_1_5": sum(x["gaussian_blur_radius_1_5"] < blur_threshold for x in blur_probe) / max(1, len(blur_probe)),
        "gaussian_radius_2_0": sum(
            laplacian_variance(Image.open(path).convert("RGB").filter(ImageFilter.GaussianBlur(2.0))) < blur_threshold
            for path in cap_paths(mri_splits_for_blur(test=True), 200, 81)
        ) / max(1, len(blur_probe)),
        "gaussian_radius_3_0": sum(
            laplacian_variance(Image.open(path).convert("RGB").filter(ImageFilter.GaussianBlur(3.0))) < blur_threshold
            for path in cap_paths(mri_splits_for_blur(test=True), 200, 81)
        ) / max(1, len(blur_probe)),
    }

    state = {
        "architecture": "MobileNetV3-Small",
        "purpose": "brain MRI versus non-MRI input gate; blur check is a separate Laplacian-variance rule",
        "pretrained": "ImageNet-1K",
        "input_size": 224,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "mri_threshold": mri_threshold,
        "blur_threshold": blur_threshold,
        "validation_metrics_at_mri_threshold": metrics_at(val_probs, val_labels, mri_threshold),
        "test_metrics_at_mri_threshold": metrics_at(test_probs, test_labels, mri_threshold),
        "test_loss": test_loss,
        "test_accuracy_at_0_5": test_acc,
        "synthetic_blur_detection_rates": blur_detection_rates,
        "synthetic_blur_probe_count": len(blur_probe),
        "negative_training_source": "OrganAMNIST abdominal CT images (MedMNIST)",
        "source_counts": {
            "mri_train": sum(x == 1 for x in train_ds.labels),
            "mri_val": sum(x == 1 for x in val_ds.labels),
            "mri_test": sum(x == 1 for x in test_ds.labels),
            "non_mri_train": sum(x == 0 for x in train_ds.labels),
            "non_mri_val": sum(x == 0 for x in val_ds.labels),
            "non_mri_test": sum(x == 0 for x in test_ds.labels),
        },
        "history": history,
        "limitations": [
            "The held-out non-MRI evaluation uses OrganAMNIST abdominal CT images; it does not establish rejection of ordinary photographs or every other scan type.",
            "The blur check uses a sharpness heuristic and may reject intrinsically smooth images or accept some blurry ones.",
            "This is an input-quality gate for research software, not a clinical validation or diagnostic test.",
        ],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in state.items() if k not in {"history", "limitations"}}, indent=2), flush=True)


def mri_splits_for_blur(test=False):
    return get_mri_paths("test" if test else "val")


if __name__ == "__main__":
    main()
