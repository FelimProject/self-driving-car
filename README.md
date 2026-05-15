# Self-Driving Car — Semantic Segmentation
 
**Author:** Felim  
**Algorithm:** FCN-ResNet50 for Multi-Class Road Scene Segmentation  
**Dataset:** `cityt` Road Scene Dataset (local)
 
---
 
## Overview
 
This project implements a semantic segmentation pipeline for autonomous driving scenes using a pretrained FCN-ResNet50 backbone. The model is trained to classify each pixel of a road scene image into one of 12 semantic classes. Hyperparameter tuning is performed with Optuna, and the final model is applied to a video stream to produce a segmented output video.
 
---
 
## Segmentation Classes
 
| ID | Class |
|----|-------|
| 0 | Sky |
| 1 | Building |
| 2 | Pole |
| 3 | Road |
| 4 | Pavement |
| 5 | Tree |
| 6 | SignSymbol |
| 7 | Fence |
| 8 | Car |
| 9 | Pedestrian |
| 10 | Bicyclist |
| 11 | Unlabelled |
 
---
 
## Project Structure
 
```
self_driving_car/
├── Dataset/
│   ├── images_prepped_train/          # Training images
│   ├── images_prepped_test/           # Test/validation images
│   ├── annotations_prepped_train/     # Training segmentation masks
│   └── annotations_prepped_test/      # Test/validation masks
├── best_model.pth                     # Saved best model weights
├── nD_6.mp4                           # Input video for inference
├── result.mp4                         # Raw output video
├── resultfinal.mp4                    # Final H.264-encoded output video
└── self_driving_car_v2.ipynb          # Main notebook
```
 
---
 
## Requirements
 
Install dependencies before running:
 
```bash
pip install torchmetrics optuna
```
 
Core libraries used:
 
- `torch`, `torchvision` — model and training
- `albumentations` — image augmentation
- `accelerate` — multi-device training abstraction
- `torchmetrics` — Dice score and Mean IoU metrics
- `optuna` — hyperparameter tuning
- `plotly` — tuning result visualization
- `opencv-python` — video inference
- `PIL`, `numpy`, `matplotlib` — data loading and visualization
---
 
## Notebook Walkthrough
 
### 1. Configurations and Preparations
 
- Mounts Google Drive and sets the dataset root path
- Imports all required dependencies
- Previews 5 training image–mask pairs
- Defines the 12 segmentation class labels
- Visualizes a sample annotation with a color legend
### 2. Data Preparation and Cleaning
 
- Scans training masks to determine the number of classes (`NUM_CLASSES = 12`)
- Defines augmentation pipelines:
  - **Train:** resize to 256×256, horizontal flip, brightness/contrast jitter, shift-scale-rotate, ImageNet normalization
  - **Val:** resize to 256×256, ImageNet normalization only
- Implements `CityDataset` — a custom `Dataset` that matches image and mask filenames and applies transforms
- Creates `DataLoader` instances (batch size = 4, 2 workers)
- Verifies batch shapes, dtypes, and class distribution
### 3. Model Training
 
- Loads pretrained `fcn_resnet50` and replaces the final classifier and auxiliary classifier heads with `Conv2d(in, 12, 1)` layers
- Loss function: `CrossEntropyLoss`
- Metrics: `DiceScore` (macro) and `MeanIoU` via `torchmetrics`
- Uses HuggingFace `Accelerator` for device-agnostic training
- Implements:
  - `train_one_epoch` — forward pass, backward pass, metric accumulation
  - `validate` — no-grad evaluation loop
  - `fit` — full training loop with early stopping (patience = 5) and best model checkpointing
### 4. Hyperparameter Tuning
 
- Uses Optuna with `TPESampler` and `MedianPruner`
- Search space:
  - `learning_rate`: log-uniform in [1e-5, 1e-2]
  - `optimizer`: Adam, AdamW, or SGD
  - `momentum`: [0.85, 0.95]
  - `beta1`: [0.90, 0.99]
  - `beta2`: [0.99, 0.999]
  - `weight_decay`: log-uniform in [1e-6, 1e-2]
- Runs 5 trials, optimizing for validation Dice score
- Plots optimization history and parameter importances with Plotly
### 5. Inference on Video
 
- Loads best model weights from `best_model.pth`
- Assigns a random RGB color per class (class 0 = black background)
- Reads `nD_6.mp4` frame by frame using OpenCV
- Runs segmentation per frame and overlays the color mask
- Writes raw output to `result.mp4`
- Re-encodes to H.264 with `ffmpeg` → `resultfinal.mp4`
- Displays the final video inline in the notebook via base64 HTML embed
---
 
## Training Configuration (defaults)
 
| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| Batch size | 4 |
| Epochs | 20 |
| Early stopping patience | 5 |
| Loss | CrossEntropyLoss |
| Metrics | Dice (macro), Mean IoU |
| Device | CUDA if available, else CPU |
 
---
 
## Outputs
 
| File | Description |
|------|-------------|
| `best_model.pth` | Model checkpoint with best validation Dice score |
| `result.mp4` | Raw segmented video output |
| `resultfinal.mp4` | Final video re-encoded with H.264 (libx264, crf=28) |
 
---
 
## Notes
 
- The notebook is designed for Google Colab with Google Drive mounted at `/content/drive/MyDrive/`.
- Adjust `PATH` in Section 1 if running locally.
- The `CityDataset` class uses filename intersection to safely handle any missing pairs between images and masks.
- Mask values are clamped to `[0, NUM_CLASSES - 1]` during training to prevent index errors from out-of-range labels.
