# ═══════════════════════════════════════════════════════════════════════════
# models.py — All 8 model architectures for emotion recognition
# ═══════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import NUM_CLASSES


# ═══════════════════════════════════════════════════════════════════════════
# 1. TRADITIONAL ML MODELS (sklearn-based)
# ═══════════════════════════════════════════════════════════════════════════

def get_svm_model():
    """SVM with RBF kernel — strong baseline for image classification."""
    from sklearn.svm import SVC
    return SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        decision_function_shape='ovr',
        random_state=42,
        verbose=True,
    )


def get_knn_model():
    """K-Nearest Neighbors classifier."""
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        n_jobs=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. ANN (Multi-Layer Perceptron) — PyTorch for GPU training
# ═══════════════════════════════════════════════════════════════════════════

class ANN(nn.Module):
    """
    Simple feed-forward neural network (MLP).
    Input: flattened 48x48x3 = 6912 features (or 48x48x1 = 2304 if grayscale).
    Uses 3-channel input to stay consistent with transforms.
    """
    def __init__(self, input_dim=48*48*3, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
# 3. CUSTOM CNN — designed for 48x48 grayscale emotion recognition
# ═══════════════════════════════════════════════════════════════════════════

class CustomCNN(nn.Module):
    """
    Lightweight CNN for 48x48 input (3-channel grayscale replicated).
    4 conv blocks → Global Average Pool → FC head.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 48x48x3 → 24x24x32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2: 24x24x32 → 12x12x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3: 12x12x64 → 6x6x128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 4: 6x6x128 → 3x3x256
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MINI-XCEPTION — depthwise separable convolutions + residual connections
#    Based on the architecture from Arriaga et al. (2017)
# ═══════════════════════════════════════════════════════════════════════════

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    """Residual block with depthwise separable convolutions."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.sep_block = nn.Sequential(
            SeparableConv2d(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            SeparableConv2d(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.sep_block(x) + self.residual(x)


class MiniXception(nn.Module):
    """
    Mini-Xception for facial expression recognition.
    Input: 48x48x3 (grayscale replicated to 3 channels).
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Middle flow — 4 residual Xception blocks
        self.blocks = nn.Sequential(
            XceptionBlock(64, 128),    # 48→24
            XceptionBlock(128, 256),   # 24→12
            XceptionBlock(256, 512),   # 12→6
            XceptionBlock(512, 1024),  # 6→3
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MobileNetV3-Small — pretrained, fine-tuned
# ═══════════════════════════════════════════════════════════════════════════

class EmotionMobileNetV3(nn.Module):
    """MobileNetV3-Small pretrained on ImageNet, fine-tuned for FER."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Freeze early layers, unfreeze last 3 inverted residual blocks
        for param in base.parameters():
            param.requires_grad = False
        # Unfreeze features[9:] (last few blocks) and classifier
        for param in list(base.features[9:].parameters()):
            param.requires_grad = True

        # Replace classifier
        in_features = base.classifier[-1].in_features
        base.classifier = nn.Sequential(
            nn.Linear(576, 512), nn.BatchNorm1d(512), nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════
# 6. EfficientNet-B0 — pretrained, fine-tuned
# ═══════════════════════════════════════════════════════════════════════════

class EmotionEfficientNetB0(nn.Module):
    """EfficientNet-B0 pretrained on ImageNet, fine-tuned for FER."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze early layers
        for param in base.parameters():
            param.requires_grad = False
        # Unfreeze last 2 blocks of features
        for param in list(base.features[6:].parameters()):
            param.requires_grad = True

        # Replace classifier
        in_features = base.classifier[-1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, num_classes),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Vision Transformer (ViT-Tiny) — via timm library
# ═══════════════════════════════════════════════════════════════════════════

class EmotionViTTiny(nn.Module):
    """ViT-Tiny pretrained, fine-tuned for emotion recognition."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        import timm
        self.model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=True,
            num_classes=0,  # remove default head
            drop_rate=dropout,
        )
        embed_dim = self.model.embed_dim  # typically 192 for ViT-Tiny
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        # Freeze early transformer blocks, fine-tune last 4 + head
        for param in self.model.parameters():
            param.requires_grad = False
        for block in self.model.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.model.norm.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model.forward_features(x)
        # ViT returns [B, num_patches+1, embed_dim]; take CLS token
        cls_token = features[:, 0]
        return self.head(cls_token)


# ═══════════════════════════════════════════════════════════════════════════
# Factory function — get any model by name
# ═══════════════════════════════════════════════════════════════════════════

def build_model(model_name, num_classes=NUM_CLASSES):
    """Build a PyTorch model by name. Returns the model (not moved to device)."""
    if model_name == 'ANN':
        return ANN(input_dim=48*48*3, num_classes=num_classes)
    elif model_name == 'CustomCNN':
        return CustomCNN(num_classes=num_classes)
    elif model_name == 'MiniXception':
        return MiniXception(num_classes=num_classes)
    elif model_name == 'MobileNetV3':
        return EmotionMobileNetV3(num_classes=num_classes)
    elif model_name == 'EfficientNetB0':
        return EmotionEfficientNetB0(num_classes=num_classes)
    elif model_name == 'ViTTiny':
        return EmotionViTTiny(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown DL model: {model_name}")
