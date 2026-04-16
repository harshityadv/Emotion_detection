# ═══════════════════════════════════════════════════════════════════════════
# config.py — Central configuration for Multi-Model Emotion Recognition
# ═══════════════════════════════════════════════════════════════════════════
import os
import torch

# This gets the directory where config.py is located
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR   = os.path.join(PROJECT_DIR, "train")
TEST_DIR    = os.path.join(PROJECT_DIR, "test")
CKPT_DIR    = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Emotion labels (FER2013 standard order via ImageFolder) ───────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

# ── Hyperparameters — Deep Learning ───────────────────────────────────────
DL_BATCH_SIZE      = 64
DL_EPOCHS          = 50          # Full training — college GPU available
DL_LR              = 3e-4
DL_WEIGHT_DECAY    = 1e-4
DL_LABEL_SMOOTHING = 0.1
VAL_SPLIT          = 0.15
EARLY_STOP_PATIENCE = 12         # Stop if val acc doesn't improve for 12 epochs

# ── Image sizes ───────────────────────────────────────────────────────────
IMG_SIZE_SMALL = 48              # For  Mini-Xception (native FER size)
IMG_SIZE_LARGE = 224             # For pretrained models (MobileNet, ResNet, EfficientNet, ViT)

# ── Model registry — all models to train and compare ─────────────────────
#  type: 'dl_small'  = PyTorch deep learning, 48×48 input
#  type: 'dl_large'  = PyTorch deep learning, 224×224 input (pretrained)
#  type: 'dl_mlp'    = PyTorch MLP (ANN), uses flattened 48×48 features
MODEL_REGISTRY = {
    'CustomCNN_MMEF': {
        'type': 'dl_small', 
        'display_name': 'Custom CNN (MMEF + CBAM)',
        'img_size': IMG_SIZE_SMALL,
    },
    'ANN': {
        'type': 'dl_mlp',
        'display_name': 'Artificial Neural Network (ANN/MLP)',
        'img_size': IMG_SIZE_SMALL,
    },
    'MobileNetV3': {
        'type': 'dl_large',
        'display_name': 'MobileNetV3-Small',
        'img_size': IMG_SIZE_LARGE,
    },

    # ── NEW: EfficientNetB0 replaced by EfficientNetB2 ────────────────────
    # B2 has a larger input resolution (260px native, resized to 224 here)
    # and more channels, giving ~+1–2% accuracy over B0 on image tasks.
    'EfficientNetB2': {
        'type': 'dl_large',
        'display_name': 'EfficientNet-B2',
        'img_size': IMG_SIZE_LARGE,
    },

    'MiniXception': {
        'type': 'dl_small',
        'display_name': 'Mini-Xception',
        'img_size': IMG_SIZE_SMALL,
    },
    'ViTTiny': {
        'type': 'dl_large',
        'display_name': 'Vision Transformer (ViT-Tiny)',
        'img_size': IMG_SIZE_LARGE,
    },

    # ── NEW: ResNet50 ─────────────────────────────────────────────────────
    # Deeper than ResNet18 (50 layers vs 18). Uses bottleneck blocks, giving
    # ~25M parameters and stronger feature representations for FER.
    'ResNet50': {
        'type': 'dl_large',
        'display_name': 'ResNet-50',
        'img_size': IMG_SIZE_LARGE,
    },
}

# ── Emotion colors for visualization ──────────────────────────────────────
EMOTION_COLORS = {
    'angry':    (0,   0,   220),
    'disgust':  (0,   140, 255),
    'fear':     (180, 0,   180),
    'happy':    (0,   220, 0),
    'neutral':  (180, 180, 180),
    'sad':      (220, 80,  0),
    'surprise': (0,   220, 220),
}
