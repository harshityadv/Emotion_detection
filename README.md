<div align="center">

# Multi-Model Facial Expression Recognition (FER)
### Attention-Guided Feature Fusion on FER2013

[![Python](https://img.shields.io/badge/Python-3.9%E2%80%933.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.20-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-FER2013-8B5CF6?style=for-the-badge)](https://www.kaggle.com/datasets/msambare/fer2013)

*A comprehensive benchmark of Traditional ML, Transfer Learning, and a novel Custom CNN architecture for real-time facial emotion recognition.*

</div>

---

## Table of Contents

- [Overview](#-overview)
- [The Novelty — MMEF + CBAM](#-the-novelty--mmef--cbam)
- [Features](#-features)
- [Models](#-model)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Advanced Training Techniques](#-advanced-training-techniques)
- [Results](#-results)
- [Contributing](#-contributing)

---

##  Overview

This repository provides a **unified, end-to-end pipeline** for Facial Expression Recognition (FER) on the **FER2013** dataset (7 emotion classes: *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*). The project spans the full ML spectrum — from classical Scikit-Learn baselines to cutting-edge deep learning — enabling rigorous, apples-to-apples comparisons across paradigms.

**Key Goals:**
-  Benchmark state-of-the-art transfer learning models (ResNet-50, EfficientNet-B2, MobileNetV3, ViT-Tiny) fine-tuned for the FER domain.
-  Introduce a novel **Custom CNN** architecture that fuses micro-, mid-, and macro-scale facial features using **MMEF** and **CBAM** — purpose-built for subtle expression recognition.
-  Deploy a **real-time webcam inference** pipeline using **MediaPipe Face Detection** for production-grade performance.

---

## The Novelty — MMEF + CBAM

> The flagship model of this repository is the **`CustomCNN_MMEF`** — a purpose-built architecture for the unique challenges of facial micro-expression recognition.

### The Problem: Why Standard CNNs Struggle with Micro-Expressions

Standard CNN architectures progressively downsample feature maps through successive pooling layers. By the final layer, the network retains **only coarse, macro-level semantic information** — the precise, subtle spatial details that define fleeting micro-expressions (e.g., a slight lip twitch for *disgust*, a brief eyebrow raise for *surprise*) are lost during this downsampling cascade.

### The Solution: Multi-Scale Micro-Expression Fusion (MMEF)

MMEF directly addresses this by **preserving and fusing intermediate feature maps** across three distinct scales:

| Scale | Block | Feature Map | Captures |
|-------|-------|-------------|----------|
| Micro | Block 2 | `(B, 64, 12, 12)` — `f2` | Fine-grained textures, wrinkles, subtle muscle contractions |
| Mid | Block 3 | `(B, 128, 6, 6)` — `f3` | Intermediate facial structures, eye regions, lip shapes |
| Macro | Block 4 | `(B, 256, 3, 3)` — `f4` | High-level semantic features, global face layout |

Instead of discarding `f2` and `f3`, MMEF **aligns them to a common spatial resolution (3×3)** via Adaptive Average Pooling, then concatenates all three scales into a rich 448-channel fused tensor: `cat[f2, f3, f4] → (B, 448, 3, 3)`.

### The Fusion Bottleneck

A lightweight **1×1 convolution** then projects this 448-channel tensor back to 256 channels. This is the core of the MMEF novelty — the network **learns, end-to-end, the optimal blend** of micro-detail, mid-scale structure, and macro-semantic context for each emotion class, rather than being forced to rely solely on the deepest, most abstract representation.

```
Input (B, 3, 48, 48)
       |
  Block1+CBAM --> (B, 32, 24, 24)
       |
  Block2+CBAM --> (B, 64, 12, 12) ----------------------------------> f2 (Micro)
       |                                                                    |
  Block3+CBAM --> (B, 128, 6, 6) -----------------------------------> f3 (Mid)
       |                                                                    |
  Block4+CBAM --> (B, 256, 3, 3) -----------------------------------> f4 (Macro)
                                                                           |
                                              AdaptivePool(3x3) on f2, f3
                                                                           |
                                    cat[f2_aligned, f3_aligned, f4] --> (B, 448, 3, 3)
                                                                           |
                                         FusionBottleneck (1x1 Conv) --> (B, 256, 3, 3)
                                                                           |
                                                          Classifier --> (B, 7)
```

### CBAM: Convolutional Block Attention Module

Applied **after every convolutional block**, CBAM acts as a learned filter that amplifies discriminative facial regions and suppresses background noise. It operates sequentially in two stages:

1. **Channel Attention (CAM):** *"What to attend to"* — Uses both GlobalAvgPool and GlobalMaxPool to generate a channel descriptor. A shared 2-layer MLP with a bottleneck produces a per-channel gating weight `Mc in [0,1]^C`, re-weighting which feature types (edges, textures, etc.) are important.

2. **Spatial Attention (SAM):** *"Where to attend to"* — Channel-wise average and max pooling create two spatial maps, concatenated and passed through a 7×7 convolution to produce a spatial gate `Ms in [0,1]^(H×W)`, highlighting which pixels in the face are most discriminative.

This dual-attention mechanism ensures the model focuses on the right features in the right locations — critical for parsing the subtle, spatially localised signals of micro-expressions.

### Grad-CAM: Visualising What the Model Attends To

The webcam inference pipeline integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** for real-time interpretability. Grad-CAM computes the gradient of the predicted class score with respect to the activations of a target convolutional layer, then uses the gradient-weighted sum as a spatial saliency map. Regions with high saliency are where the model "looked" to make its prediction.

In practice this produces a heatmap — rendered with the JET colormap and blended directly onto the face crop — that highlights which facial sub-regions (e.g., corners of the mouth for *happy*, furrowed brows for *angry*) drove the classification decision. This is especially valuable for debugging MMEF: saliency maps confirm whether the model is exploiting fine-grained micro-scale cues (from `f2`) or relying too heavily on coarse macro-scale features (from `f4`).

Grad-CAM is supported for all CNN-based models in the pipeline. The target layer is selected per architecture:

| Model | Grad-CAM Target Layer |
|-------|-----------------------|
| Custom CNN (MMEF + CBAM) | `fusion_bottleneck[-1]` — the 1x1 projection output |
| ResNet-50 | `model.layer4[-1]` — final bottleneck block |
| EfficientNet-B2 | `model.features[-1]` — last feature stage |
| MobileNetV3-Small | `model.features[-1]` — last inverted residual |
| Mini-Xception | `blocks[-1].sep_block[-2]` — final separable conv |
| ViT-Tiny / ANN | Not supported (attention-based / fully-connected) |

To reduce overhead during live inference, Grad-CAM is recomputed every `N` frames (configurable via `GRADCAM_EVERY_N`, default: 3), with the cached heatmap displayed on intermediate frames.

### Rolling Queue: Temporal Smoothing for Stable Real-Time Predictions

Single-frame emotion predictions are inherently noisy — minor pose changes, lighting shifts, or motion blur can cause the predicted label to flicker between classes across consecutive frames. The webcam pipeline implements a multi-layer temporal smoothing stack using Python's `collections.deque` as the underlying rolling queue:

1. **Probability Rolling Average** — A per-face `deque(maxlen=SMOOTH_WINDOW)` (default: 7 frames) accumulates raw softmax probability vectors. The mean of this queue is taken as the smoothed probability distribution before argmax, suppressing transient spikes from individual noisy frames.

2. **Label Lock (Streak Counter)** — A new emotion label must be the argmax of the smoothed distribution for `LOCK_FRAMES` (default: 4) consecutive frames before it replaces the currently displayed label. This prevents single-frame outliers from triggering a label switch.

3. **Time Lock (Minimum Display Duration)** — Once a label is accepted and displayed, it is held on screen for a minimum of `MIN_DISPLAY_MS` milliseconds (default: 600 ms), regardless of what the model predicts in the interim. This produces a visually stable, human-readable output even at high frame rates.

4. **Bounding Box Smoothing** — Face bounding box coordinates are averaged over a `deque(maxlen=BOX_SMOOTH)` (default: 5 frames) to eliminate the jittery box movement that MediaPipe can produce with slight head motion.

5. **Temperature Scaling** — Softmax logits are divided by a temperature parameter `T > 1` (default: 1.5) before computing probabilities. This flattens the distribution, making individual class peaks less dominant and producing smoother probability trajectories across frames.

All five mechanisms operate independently per detected face index, allowing multiple faces in the frame to each have their own smoothing state.

---

##  Features

- **Unified, Modular Pipeline** — `config.py` serves as a single source of truth. Adding a new model is as simple as registering it in `MODEL_REGISTRY`.
- **7-Model Comparison** — Train and evaluate Deep Learning (ANN, CustomCNN, Mini-Xception, MobileNetV3, EfficientNet-B2, ResNet-50, ViT-Tiny) in a single command.
- **Novel Architecture** — Custom CNN with MMEF + CBAM, designed from first principles for the FER domain.
- **Class Imbalance Handling** — Inverse-frequency class weights computed automatically from the training data distribution.
- **Focal Loss Ready** — Class-weighted Cross-Entropy loss with label smoothing (easily extendable to Focal Loss).
- **Differential Learning Rates** — Pretrained backbones use `lr × 0.1`; custom classification heads use the full `lr` — applied uniformly across all transfer learning models via name-based parameter group dispatch.
- **Cosine Annealing LR Scheduler** — Smooth, cyclical learning rate decay for better convergence.
- **Early Stopping** — Configurable patience (default: 12 epochs) with best-checkpoint saving.
- **Real-Time Inference** — Live webcam script using `MediaPipe Face Detection` for fast, accurate face cropping and per-frame emotion prediction.
- **Rich Evaluation Reports** — Per-class classification reports, confusion matrices (Seaborn heatmaps), and training history curves (Matplotlib).
- **Dynamic DataLoaders** — Automatically applies appropriate augmentation, normalization, and image resizing (48×48 or 224×224) based on the model type.

---

## Models

| # | Model | Type | Input Size | Params (approx.) | Notes |
|---|-------|------|------------|-------------------|-------|
| 1 | **Custom CNN (MMEF + CBAM)** | DL — Custom | 48×48 | ~2M | Flagship model |
| 2 | ANN / MLP | DL — Custom | 48×48 | ~10M | Fully connected baseline |
| 3 | Mini-Xception | DL — Custom | 48×48 | ~1M | Depthwise separable + residual |
| 4 | MobileNetV3-Small | DL — Pretrained | 224×224 | ~2.5M | ImageNet fine-tuned, mobile-optimised |
| 5 | EfficientNet-B2 | DL — Pretrained | 224×224 | ~9.1M | Compound-scaled, higher capacity than B0 |
| 6 | ResNet-50 | DL — Pretrained | 224×224 | ~25.6M | Bottleneck residual blocks |
| 7 | ViT-Tiny | DL — Pretrained | 224×224 | ~5.7M | Vision Transformer, via `timm` |

---

## Installation

### Prerequisites

> **Version Constraints — Please read carefully:**
> - **Python:** `>= 3.9` and `<= 3.11` (required for MediaPipe compatibility)
> - **MediaPipe:** `== 0.10.20` (strict pin — other versions may break the webcam inference pipeline)

### Step 1: Clone the Repository

```bash
git clone https://github.com/harshityadv/Emotion_detection.git
cd Emotion_detection
```

### Step 2: Create a Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

### Step 3: Install Core Dependencies

```bash
# Install PyTorch first (visit https://pytorch.org/get-started/locally/ for CUDA-specific builds)
# Example: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only:
pip install torch torchvision
```

### Step 4: Install Remaining Dependencies

```bash
pip install scikit-learn opencv-python matplotlib seaborn timm Pillow numpy

# Install MediaPipe at the EXACT pinned version
pip install mediapipe==0.10.20
```

### Step 5: Verify Installation

```python
python -c "import torch; import mediapipe; import timm; print('All dependencies OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Dataset Preparation

This project uses the **FER2013** dataset. Download it from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).

The dataset must be organised into **class-named subdirectories** compatible with `torchvision.datasets.ImageFolder`:

```
Emotion_detection/
├── train/
│   ├── angry/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

> **Note:** The paths to `train/` and `test/` are configured centrally in `config.py` via `TRAIN_DIR` and `TEST_DIR`, both derived automatically from the project root — no hardcoded paths needed.

The `data_loader.py` module handles:
- **15% validation split** from the training set (stratified)
- **Data augmentation** for training (random horizontal flips, rotations, color jitter)
- **Automatic class weight computation** using inverse-frequency weighting for imbalanced classes
- **Separate 48×48 and 224×224 DataLoader pipelines** for small and large models respectively

---

## Usage

### 1. Train All Models (Local)

```bash
# Train every model registered in MODEL_REGISTRY
python train_all_models.py
```

### 2. Train a Specific Model

```bash
# Train only the Custom CNN
python train_all_models.py CustomCNN_MMEF

# Train multiple models selectively
python train_all_models.py ResNet50 EfficientNetB2

# Available model names:
# CustomCNN_MMEF | ANN | MiniXception | MobileNetV3
# EfficientNetB2 | ResNet50 | ViTTiny
```

### 3. Evaluate Trained Models on the Test Set

```bash
# Evaluate all models with checkpoints in ./checkpoints/
python evaluate_models.py
```

This generates:
- Per-class **Classification Report** (Precision, Recall, F1)
- **Confusion Matrix** heatmaps saved to `results/`
- **Training History** plots (loss & accuracy curves) from `results/*_history.json`

### 4. Real-Time Webcam Inference

```bash
# Run live emotion detection using your webcam
python run_webcam_mediapipe.py
```

The webcam script:
1. Captures frames from your default webcam.
2. Uses **MediaPipe Face Detection** to locate and crop faces in real time.
3. Preprocesses each face crop and passes it through the selected PyTorch model.
4. Overlays the predicted emotion label and confidence score directly on the video feed.

> **Tip:** Press `Q` to quit the webcam window.


## Project Structure

```
Emotion_detection/
|
+-- config.py                   # Central config: paths, hyperparameters, model registry
+-- models.py                   # All model architectures (CBAM, MMEF, pretrained wrappers)
+-- data_loader.py              # DataLoaders, augmentation, class weights
+-- train_all_models.py         # Main training orchestrator (DL)
+-- evaluate_models.py          # Test-set evaluation, classification reports, confusion matrices
+-- run_webcam_mediapipe.py     # Real-time webcam inference with MediaPipe + Grad-CAM
+-- face_detection_mediapipe.py # MediaPipe face detection helper utilities
+-- requirements.txt            # Python package dependencies
|
+-- train/                      # FER2013 training images (class-named subdirs)
|   +-- angry/
|   +-- disgust/
|   +-- fear/
|   +-- happy/
|   +-- neutral/
|   +-- sad/
|   +-- surprise/
|
+-- test/                       # FER2013 test images (same structure as train/)
|
+-- checkpoints/                # Saved model weights (.pth)
|   +-- CustomCNN_MMEF_best.pth
|   +-- ResNet50_best.pth
|   +-- ...
|
+-- results/                    # Evaluation outputs
    +-- evaluation_results.json
    +-- *_confusion_matrix.png
    +-- *_history.json
    +-- model_comparison.png
    +-- training_results.json
```

---

## Advanced Training Techniques

### Focal Loss with Label Smoothing & Class Weights

FER2013 is significantly imbalanced (*Happy* has ~8,000 samples vs *Disgust* with ~500). The training pipeline addresses this on multiple fronts simultaneously:

```python
# Inverse-frequency class weights (computed automatically in data_loader.py)
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),   # up-weights rare classes (disgust, fear)
    label_smoothing=0.1,               # prevents overconfident predictions
)
```

### Differential Learning Rates

A critical technique for transfer learning — pretrained backbone layers use a **10× smaller learning rate** to preserve valuable ImageNet representations, while custom classification heads train at the full rate:

```python
# Automatically applied to ALL models via name-based parameter dispatch
param_groups = [
    {'params': backbone_params, 'lr': base_lr * 0.1},  # e.g., 3e-5
    {'params': head_params,     'lr': base_lr},         # e.g., 3e-4
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
```

### Cosine Annealing LR Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

This produces a smooth, cyclic learning rate decay rather than abrupt step drops, consistently yielding better final accuracy on image classification tasks.

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DL_LR` | `3e-4` | Base (head) learning rate |
| `DL_WEIGHT_DECAY` | `1e-4` | AdamW regularisation |
| `DL_LABEL_SMOOTHING` | `0.10` | Cross-entropy label smoothing |
| `DL_BATCH_SIZE` | `64` | Batch size for all DL models |
| `DL_EPOCHS` | `50` | Maximum training epochs |
| `EARLY_STOP_PATIENCE` | `12` | Epochs without improvement before stopping |
| `VAL_SPLIT` | `0.15` | Fraction of train data for validation |

---

## Results

Benchmark results on the FER2013 test set (7,178 samples). All metrics are weighted averages across the seven emotion classes. Confusion matrices and per-class breakdowns are available in `results/`.

| Model | Test Accuracy | Precision (W) | Recall (W) | F1-Score (W) | Inference (ms/sample) |
|-------|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN (MMEF + CBAM)** | **59.15%** | **59.18%** | **59.15%** | **57.91%** | 1.19 |
| ResNet-50 | 66.79% | 66.81% | 66.79% | 66.71% | 66.72 |
| EfficientNet-B2 | 65.78% | 65.79% | 65.78% | 65.57% | 27.06 |
| MobileNetV3-Small | 63.07% | 63.47% | 63.07% | 62.94% | 2.33 |
| ViT-Tiny | 61.23% | 60.98% | 61.23% | 60.84% | 23.06 |
| Mini-Xception | 50.70% | 52.44% | 50.70% | 49.90% | 8.39 |
| ANN / MLP | 30.12% | 38.03% | 30.12% | 32.27% | 0.13 |

> Precision (W), Recall (W) and F1-Score (W) are weighted averages. The Custom CNN (MMEF + CBAM) achieves competitive accuracy with the lowest inference latency among all models except ANN, making it most suitable for the real-time webcam deployment scenario.

> Confusion matrices and training curves are saved to the `results/` directory after running `python evaluate_models.py`.

---

## Contributing

Contributions are welcome! To add a new model:

1. Implement the model class in `models.py`.
2. Add a `build_model()` branch in the factory function.
3. Register it in `MODEL_REGISTRY` in `config.py` with the correct `type` (`dl_small`, `dl_large`, or `dl_mlp`).
4. Train with: `python train_all_models.py <YourModelName>`

---

<div align="center">

Built for the FER2013 research community.

*If this repository helped your research or coursework, please consider giving it a star.*

</div>
