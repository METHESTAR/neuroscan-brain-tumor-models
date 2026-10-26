"""Four-class brain MRI classification with pretrained Swin-Tiny or ResNet-50.

Expected training layout:

    dataset/
      train/<class name>/*.{png,jpg,...}
      val/<class name>/*.{png,jpg,...}   # optional; otherwise stratified from train/
      test/<class name>/*.{png,jpg,...}   # optional, evaluated once at the end

Training requires an explicit --train flag and stops if CUDA is unavailable.
If val/ is absent, validation is selected by a reproducible stratified split
from train/ without modifying the source folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, Swin_T_Weights


PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or train a four-class brain MRI classifier.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset root folder")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inspect-only", action="store_true", help="List classes and image counts; do not train")
    action.add_argument("--train", action="store_true", help="Train the selected architecture")
    parser.add_argument(
        "--architecture", choices=("swin_t", "resnet50", "convnext_tiny"), default="swin_t"
    )
    parser.add_argument("--output-dir", type=Path, help="Output folder (default: runs/<timestamp> inside this project)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs; early stopping usually ends sooner")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8, help="Stop after this many epochs without lower validation loss")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum validation-loss decrease counted as improvement")
    parser.add_argument("--validation-fraction", type=float, default=0.15, help="Fraction of train/ used for validation if val/ is absent")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume weights and epoch from a saved checkpoint")
    parser.add_argument("--num-workers", type=int, default=0, help="Windows-safe default is 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use ImageNet weights")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs and batch-size must be positive, and image-size must be at least 32")
    if args.learning_rate <= 0 or args.patience < 1 or args.num_workers < 0 or args.min_delta < 0:
        parser.error("learning-rate must be positive; patience must be positive; workers and min-delta cannot be negative")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("validation-fraction must be between 0 and 0.5")
    return args


def folder_dataset(folder: Path) -> datasets.ImageFolder:
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected a folder at: {folder}")
    return datasets.ImageFolder(folder)


def dataset_targets(dataset: Dataset) -> tuple[list[str], list[int]]:
    if isinstance(dataset, Subset):
        source = dataset.dataset
        if not isinstance(source, datasets.ImageFolder):
            raise TypeError("Expected a subset of an ImageFolder dataset")
        return source.classes, [source.targets[index] for index in dataset.indices]
    if not isinstance(dataset, datasets.ImageFolder):
        raise TypeError("Expected an ImageFolder dataset")
    return dataset.classes, dataset.targets


def count_by_class(dataset: Dataset) -> dict[str, int]:
    classes, targets = dataset_targets(dataset)
    counts = Counter(targets)
    return {name: counts.get(index, 0) for index, name in enumerate(classes)}


def print_dataset_summary(root: Path) -> None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    split_dirs = {name: root / name for name in SPLIT_NAMES if (root / name).is_dir()}
    print(f"Dataset root: {root}")
    if split_dirs:
        print("Recognized split folders:")
        class_reference: list[str] | None = None
        for split_name, folder in split_dirs.items():
            try:
                ds = folder_dataset(folder)
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"  {split_name}: could not read class folders ({exc})")
                continue
            if class_reference is None:
                class_reference = ds.classes
            counts = count_by_class(ds)
            print(f"  {split_name}: {len(ds):,} images; classes={ds.classes}")
            for class_name, count in counts.items():
                print(f"    {class_name}: {count:,}")
            if class_reference != ds.classes:
                print(f"    WARNING: class names/order differ from the first readable split: {class_reference}")
        if "train" not in split_dirs:
            print("Training needs a train/ folder. If val/ is absent, it will be stratified from train/.")
        return

    try:
        ds = folder_dataset(root)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not recognize the dataset. Use either root/<class>/* image files or "
            "root/train/<class>/* plus root/val/<class>/*."
        ) from exc
    counts = count_by_class(ds)
    print("No train/val/test split folders were found; this looks like a single class-folder dataset.")
    print(f"Images: {len(ds):,}; classes={ds.classes}")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count:,}")
    print("Training will not split these images automatically. Confirm patient-level grouping and split first.")


def load_split_datasets(
    root: Path, image_size: int, validation_fraction: float, seed: int
) -> tuple[dict[str, Dataset], list[str]]:
    root = root.expanduser().resolve()
    if not (root / "train").is_dir():
        raise ValueError("Training requires a train/ folder. The supplied test/ folder is never split or trained on.")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_folder = root / "train"
    train_source = datasets.ImageFolder(train_folder, transform=train_transform)
    classes = train_source.classes
    if (root / "val").is_dir():
        loaded: dict[str, Dataset] = {
            "train": train_source,
            "val": datasets.ImageFolder(root / "val", transform=eval_transform),
        }
    else:
        val_source = datasets.ImageFolder(train_folder, transform=eval_transform)
        indices = np.arange(len(train_source))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=train_source.targets,
        )
        loaded = {
            "train": Subset(train_source, train_indices.tolist()),
            "val": Subset(val_source, val_indices.tolist()),
        }
        print(
            f"Created an in-memory stratified validation split from train/: "
            f"{len(val_indices):,} validation images ({validation_fraction:.0%}); source files unchanged."
        )
    test_folder = root / "test"
    if test_folder.is_dir():
        loaded["test"] = datasets.ImageFolder(test_folder, transform=eval_transform)

    if len(classes) != 4:
        raise ValueError(f"Expected exactly four class folders, found {len(classes)}: {classes}")
    for split_name, ds in loaded.items():
        ds_classes, _ = dataset_targets(ds)
        if ds_classes != classes:
            raise ValueError(f"Class folders in {split_name}/ do not match train/: {ds_classes} vs {classes}")
        if len(ds) == 0:
            raise ValueError(f"The {split_name}/ folder contains no supported images")
        if any(value == 0 for value in count_by_class(ds).values()):
            raise ValueError(f"Every class must have at least one image in {split_name}/")
    return loaded, classes


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(ds: datasets.ImageFolder, batch_size: int, workers: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    use_amp = device.type == "cuda"
    amp_device = "cuda" if use_amp else "cpu"

    for images, labels in loader:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=amp_device, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.detach().item() * labels.size(0)
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    average_loss = total_loss / max(1, len(loader.dataset))
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return average_loss, float(accuracy), float(macro_f1), targets, predictions


def save_training_plot(history: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="macro F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def save_confusion_plot(labels: list[int], predictions: list[int], classes: list[str], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stopping instead of starting training on the CPU.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Training on {gpu_name} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB VRAM)")
    print("Patient-level split reminder: no patient should appear in more than one data split.")

    split_datasets, classes = load_split_datasets(
        args.data_dir, args.image_size, args.validation_fraction, args.seed
    )
    for split_name, ds in split_datasets.items():
        print(f"{split_name}: {len(ds):,} images; {count_by_class(ds)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (PROJECT_ROOT / "runs" / stamp)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pin_memory = True

    train_loader = make_loader(split_datasets["train"], args.batch_size, args.num_workers, True, pin_memory)
    val_loader = make_loader(split_datasets["val"], args.batch_size, args.num_workers, False, pin_memory)
    test_loader = (
        make_loader(split_datasets["test"], args.batch_size, args.num_workers, False, pin_memory)
        if "test" in split_datasets
        else None
    )

    train_counts = count_by_class(split_datasets["train"])
    class_weights = torch.tensor(
        [len(split_datasets["train"]) / (len(classes) * train_counts[name]) for name in classes],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.architecture == "resnet50":
        weights = None if args.no_pretrained else ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif args.architecture == "convnext_tiny":
        weights = None if args.no_pretrained else ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    else:
        weights = None if args.no_pretrained else Swin_T_Weights.DEFAULT
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, len(classes))
    model.to(device)
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_significant_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pt"

    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint.expanduser().resolve()
        resume = torch.load(resume_path, map_location=device, weights_only=True)
        if resume.get("architecture") != args.architecture or resume.get("classes") != classes:
            raise ValueError("Resume checkpoint architecture/classes do not match this run")
        model.load_state_dict(resume["model_state_dict"])
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        if "scaler_state_dict" in resume:
            scaler.load_state_dict(resume["scaler_state_dict"])
        best_val_loss = float(resume.get("val_loss", float("inf")))
        best_significant_val_loss = best_val_loss
        best_val_f1 = float(resume.get("val_macro_f1", 0.0))
        best_epoch = int(resume.get("epoch", 0))
        start_epoch = best_epoch + 1
        print(
            f"Resuming from {resume_path} at epoch {start_epoch}; "
            f"best saved validation loss is {best_val_loss:.4f}."
        )
        existing_metrics = output_dir / "metrics.csv"
        if existing_metrics.exists():
            with existing_metrics.open(newline="", encoding="utf-8") as stream:
                for old_row in csv.DictReader(stream):
                    history.append(
                        {
                            "epoch": int(old_row["epoch"]),
                            "train_loss": float(old_row["train_loss"]),
                            "train_accuracy": float(old_row["train_accuracy"]),
                            "train_macro_f1": float(old_row["train_macro_f1"]) if old_row["train_macro_f1"] else float("nan"),
                            "val_loss": float(old_row["val_loss"]),
                            "val_accuracy": float(old_row["val_accuracy"]),
                            "val_macro_f1": float(old_row["val_macro_f1"]),
                        }
                    )
    else:
        start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_accuracy, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_accuracy, val_f1, _, _ = run_epoch(model, val_loader, criterion, device, None, scaler)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}, macro-F1 {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "image_size": args.image_size,
                    "architecture": args.architecture,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                checkpoint_path,
            )

        if val_loss < best_significant_val_loss - args.min_delta:
            best_significant_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {args.patience} epochs without at least "
                    f"{args.min_delta:g} validation-loss improvement."
                )
                break

        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_training_plot(history, output_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary: dict[str, Any] = {
        "gpu": gpu_name,
        "classes": classes,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "validation_macro_f1_at_best_loss": best_val_f1,
        "architecture": args.architecture,
        "test": None,
    }
    if test_loader is not None:
        test_loss, test_accuracy, test_f1, test_labels, test_predictions = run_epoch(
            model, test_loader, criterion, device, None, scaler
        )
        report = classification_report(
            test_labels,
            test_predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
        summary["test"] = {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "macro_f1": test_f1,
            "classification_report": report,
        }
        with (output_dir / "test_report.json").open("w", encoding="utf-8") as stream:
            json.dump(summary["test"], stream, indent=2)
        save_confusion_plot(test_labels, test_predictions, classes, output_dir / "test_confusion_matrix.png")
        print(f"Test | loss {test_loss:.4f}, accuracy {test_accuracy:.4f}, macro-F1 {test_f1:.4f}")
    else:
        print("No test/ folder was supplied; held-out test evaluation was skipped.")

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Best checkpoint and metrics saved under: {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        if args.inspect_only:
            print_dataset_summary(args.data_dir)
        else:
            train(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
