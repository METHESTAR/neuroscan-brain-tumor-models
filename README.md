# NeuroScan AI: Brain Tumor MRI Classification

Four-class image classification experiments using the Mendeley Brain Tumor MRI Dataset, version 5 (Glioma, Meningioma, Pituitary, No Tumor). The dataset itself is **not included**; see the dataset DOI in `data_split_manifest.json`.

## Results

All three checkpoints were evaluated on the same 2,414-image held-out Test split. Accuracy and macro-F1 are percentages below; the full per-class precision, recall, F1, and support are in each model's `results/<model>/test_report.json`.

| Model | Best epoch | Best validation loss | Test accuracy | Test macro-F1 | Test loss |
|---|---:|---:|---:|---:|---:|
| Swin-Tiny | 12 | 0.1584 | 90.64% | 90.73% | 0.4121 |
| ResNet-50 | 4 | 0.1611 | 89.85% | 90.05% | 0.3364 |
| ConvNeXt-Tiny | 11 | 0.1265 | 93.16% | 93.15% | 0.2252 |

### Evaluation notes

- The prepared image-level split contains 4,712 Train, 1,003 Validation, and all 2,414 original Test images. Source dataset files were left unchanged.
- Train images matching Test or Validation by exact SHA-256 or pHash distance <= 4 were excluded from Train. The cross-split audit found zero exact or near matches at those thresholds.
- The dataset does not provide patient identifiers, so patient-level separation cannot be confirmed. Near-image filtering may conservatively exclude some distinct but visually similar images.
- The same Test set was evaluated for all three models. Treat the head-to-head comparison as exploratory; do not select/tune a final model based on these repeatedly viewed Test results. A new independent test set would be needed for a confirmatory comparison.
- This is a research classification experiment, not a clinically validated diagnostic system.

## Training setup

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)
- Pretrained weights: ImageNet
- Input size: 224 x 224
- Optimizer/loss: AdamW with class-weighted cross entropy
- Early stopping: validation loss; stop after 8 epochs without at least 0.001 improvement; 100-epoch cap
- Batch size: Swin-Tiny 8; ResNet-50 4; ConvNeXt-Tiny 8
- Best checkpoint selected by lowest validation loss

The `.pth` files in `models/` contain inference-ready model state dictionaries and class metadata; full optimizer/resume states are intentionally omitted to reduce file size. Git LFS is configured for these files.

## Files

- `models/`: best inference checkpoints
- `app.py`: local Streamlit image-upload interface (uses ConvNeXt-Tiny)
- `app-requirements.txt`: app and model dependencies
- `input_gate.py`: blur and brain-MRI input checks used before tumor classification
- `models/input_gate_mobilenetv3_small.pth`: trained MobileNetV3-Small input gate
- `training/train_input_gate.py`: reproducible input-gate training script; downloads the public OrganAMNIST negative examples automatically
- `results/input_gate/summary.json`: input-gate thresholds and held-out results
- `results/<model>/`: epoch metrics, test report, run summary, training curve, and confusion matrix
- `results/gradcam/`: example Grad-CAM visualization and prediction metadata
- `interpretability/`: reusable Grad-CAM script for all three checkpoints
- `training/`: training script and package versions
- `data_split_manifest.json`: dataset and duplicate-filtering notes

## Inference

Install the packages listed in `training/requirements.txt` in a compatible Python environment, then run:

```powershell
python inference.py --model models/convnext_tiny.pth --image path/to/mri.png
```

Supported model names: `swin_tiny`, `resnet50`, `convnext_tiny`.

## Streamlit app

The upload interface uses ConvNeXt-Tiny, which had the highest measured test accuracy of the three models in this experiment. Before tumor classification, it checks minimum image size, applies a Laplacian-variance blur screen, and uses a MobileNetV3-Small gate trained to separate brain MRI images from abdominal CT examples. It displays the predicted class and the probability scores for all four classes only when the input checks pass. By default, the app runs locally, keeps the uploaded image in memory, and does not write it to disk.

The gate was trained with early stopping on validation loss (5-epoch patience, 25-epoch cap). Its MRI acceptance threshold retains about 99% of Validation brain MRIs; held-out metrics are recorded in `results/input_gate/summary.json`. The blur threshold was calibrated to preserve 99% of clear Validation MRIs. It rejected 58.5% of 200 synthetic MRI images blurred at Gaussian radius 1.5, 94.5% at radius 2, and 97.5% at radius 3. See the results file for full metrics and training history.

These are narrow research checks. The non-MRI evaluation set is abdominal CT, so rejection of natural photos, X-rays, or every other MRI type has not been established. The blur heuristic can miss mild blur and may reject naturally smooth images. MedMNIST is CC BY 4.0; see the [official MedMNIST repository](https://github.com/MedMNIST/MedMNIST) for attribution and dataset details. The input gate is not clinically validated.

From the repository folder, install the app dependencies and launch it:

```powershell
python -m pip install -r app-requirements.txt
python -m streamlit run app.py
```

The browser page opened by Streamlit lets you choose a JPG, PNG, or other supported MRI image. The app falls back to CPU if CUDA is unavailable.

## Grad-CAM visualization

The example below uses ConvNeXt-Tiny on a glioma image from the prepared Validation split (not the held-out Test split). The model predicted `glioma` with 99.85% probability.

![ConvNeXt-Tiny Grad-CAM example for a validation glioma MRI](results/gradcam/convnext_tiny_validation_glioma.png)

Generate a Grad-CAM figure for your own MRI image with any of the three checkpoints:

```powershell
python interpretability/gradcam.py --model models/convnext_tiny.pth --image path/to/mri.png --output results/gradcam/my_mri_gradcam.png
```

Use `--target-class glioma` (or `meningioma`, `pituitary`, `notumor`) to visualize evidence for a chosen class instead of the model's predicted class. The adjacent JSON file records the prediction and heatmap target. Grad-CAM is a coarse map of image regions influencing a class score; it is not a tumor boundary, segmentation, or clinically validated explanation.
