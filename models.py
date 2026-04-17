# ═══════════════════════════════════════════════════════════════════════════
# models.py — All model architectures for FER2013 emotion recognition
#
# CHANGES vs previous version:
#   • EfficientNetB0  →  EfficientNetB2  (higher accuracy, ~+1–2%)
#   • Added EmotionResNet50             (deeper ResNet with bottleneck blocks)
#   • build_model() factory updated for both changes
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
    Input: flattened 48×48×3 = 6912 features.
    Uses 3-channel input to stay consistent with data_loader transforms.
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
# 3. CustomCNN+CBAM+MMEF
# ═══════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).

    Mechanism
    ---------
    Given a feature map F ∈ R^(C×H×W):
      1. Apply both GlobalAvgPool and GlobalMaxPool over H,W → two (C×1×1) vectors.
      2. Feed each through a shared 2-layer MLP with bottleneck ratio `r`
         (reduces C → C//r → C, capturing inter-channel relationships).
      3. Sum the MLP outputs and apply Sigmoid → channel attention map Mc ∈ [0,1]^C.
      4. Multiply: F' = Mc ⊗ F  (broadcast over H, W).

    Parameters
    ----------
    in_channels : number of input feature channels (C).
    reduction   : bottleneck ratio (default 16 → C//16 hidden units).
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        # Clamp hidden dim so very small channel counts don't produce 0
        hidden = max(in_channels // reduction, 4)

        # Shared MLP — applied to both avg-pooled and max-pooled descriptors
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape

        # Global Average Pooling: (B, C, H, W) → (B, C)
        avg = x.flatten(2).mean(dim=2)           # spatial avg
        # Global Max Pooling:     (B, C, H, W) → (B, C)
        mx  = x.flatten(2).max(dim=2).values     # spatial max

        # Shared MLP on each descriptor
        avg_out = self.shared_mlp(avg)            # (B, C)
        max_out = self.shared_mlp(mx)             # (B, C)

        # Combine, gate with Sigmoid, reshape for broadcast
        scale = torch.sigmoid(avg_out + max_out)  # (B, C)
        scale = scale.view(B, C, 1, 1)            # (B, C, 1, 1)
        return x * scale                          # channel-scaled feature map


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).

    Mechanism
    ---------
    Given a channel-attended feature map F' ∈ R^(C×H×W):
      1. Compute channel-wise AvgPool and MaxPool → two (1×H×W) maps.
      2. Concatenate along channel dim → (2×H×W).
      3. Apply a single 7×7 conv → (1×H×W).
      4. Sigmoid → spatial attention map Ms ∈ [0,1]^(H×W).
      5. Multiply: F'' = Ms ⊗ F'  (broadcast over C).

    Parameters
    ----------
    kernel_size : size of the spatial conv kernel (7 recommended in paper).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        # 2-channel input (avg-pool + max-pool descriptor maps), 1 output channel
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise Average and Max pooling → (B, 1, H, W) each
        avg = x.mean(dim=1, keepdim=True)           # (B, 1, H, W)
        mx  = x.max(dim=1, keepdim=True).values     # (B, 1, H, W)

        # Concatenate along channel axis → (B, 2, H, W)
        cat = torch.cat([avg, mx], dim=1)

        # Spatial gate
        scale = torch.sigmoid(self.conv(cat))        # (B, 1, H, W)
        return x * scale                             # spatially-scaled feature map


class CBAM(nn.Module):
    """
    Full Convolutional Block Attention Module.

    Sequentially applies ChannelAttention then SpatialAttention to the
    input feature map.  The output shape is IDENTICAL to the input shape,
    making CBAM a drop-in module after any conv block.

    Parameter count per CBAM instance:
      - CAM: 2 × (C × (C//r))  ≈ 2 × 256 × 16 = 8 192  (for C=256, r=16)
      - SAM: 2 × 7 × 7 × 1    = 98
    Total overhead is negligible compared to the conv layers themselves.

    Parameters
    ----------
    in_channels : number of feature channels (C).
    reduction   : channel reduction ratio for CAM (default 16).
    kernel_size : spatial conv kernel size for SAM (default 7).
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)   # WHAT to attend to
        x = self.spatial_attention(x)   # WHERE to attend
        return x



class CustomCNNMMEF(nn.Module):
    """
    CustomCNN with CBAM + Multi-Scale Micro-Expression Fusion (MMEF).
    Combines two complementary architectural novelties:

      1. CBAM (per-block attention):
         After each conv block a Channel + Spatial gate filters out
         background activations and highlights discriminative facial regions.

      2. MMEF (cross-scale feature fusion):
         Instead of discarding intermediate feature maps through successive
         pooling, MMEF saves the CBAM-purified outputs of Block 2 (f2, 12×12)
         and Block 3 (f3, 6×6), aligns them to 3×3 via AdaptiveAvgPool, and
         concatenates them with Block 4 (f4, 3×3).  A lightweight 1×1 conv
         Fusion Bottleneck then learns the optimal blend of micro-scale detail
         (f2), mid-scale structure (f3), and macro-semantic context (f4).

    Feature flow:
      Block1+CBAM : (B,  3, 48, 48) → (B,  32, 24, 24)
      Block2+CBAM : (B, 32, 24, 24) → (B,  64, 12, 12)  ← f2 (micro)
      Block3+CBAM : (B, 64, 12, 12) → (B, 128,  6,  6)  ← f3 (mid)
      Block4+CBAM : (B,128,  6,  6) → (B, 256,  3,  3)  ← f4 (macro)

      AdaptivePool  f2 → (B,  64, 3, 3)
      AdaptivePool  f3 → (B, 128, 3, 3)
      cat[f2,f3,f4]   → (B, 448, 3, 3)
      FusionBottleneck → (B, 256, 3, 3)   [1×1 conv + BN + ReLU]
      Classifier       → (B, num_classes)
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        # ── Convolutional blocks ───────────────────────────────────────────

        # Block 1: (B, 3, 48, 48) → (B, 32, 24, 24)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.cbam1 = CBAM(in_channels=32, reduction=4)    # small channel → r=4

        # Block 2: (B, 32, 24, 24) → (B, 64, 12, 12)  ← FUSION SOURCE: micro detail
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.cbam2 = CBAM(in_channels=64, reduction=8)

        # Block 3: (B, 64, 12, 12) → (B, 128, 6, 6)   ← FUSION SOURCE: mid-scale
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.cbam3 = CBAM(in_channels=128, reduction=16)

        # Block 4: (B, 128, 6, 6) → (B, 256, 3, 3)    ← FUSION SOURCE: macro semantic
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.cbam4 = CBAM(in_channels=256, reduction=16)

        # ── Fusion Bottleneck (core MMEF novelty) ─────────────────────────
        # 1×1 conv projects fused 448-ch tensor → 256 ch.
        # This learned projection lets the network decide how much
        # micro/mid/macro context to use per class — end-to-end.
        self.fusion_bottleneck = nn.Sequential(
            nn.Conv2d(64 + 128 + 256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                            # (B, 256, 1, 1)
            nn.Flatten(),                                       # (B, 256)
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),                        # (B, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x  = self.cbam1(self.block1(x))           # (B,  32, 24, 24)

        # Stage 2 — preserve for micro-detail fusion
        f2 = self.cbam2(self.block2(x))            # (B,  64, 12, 12)

        # Stage 3 — preserve for mid-scale fusion
        f3 = self.cbam3(self.block3(f2))           # (B, 128,  6,  6)

        # Stage 4 — deep semantic features
        f4 = self.cbam4(self.block4(f3))           # (B, 256,  3,  3)

        # MMEF: align early maps to 3×3 and fuse
        f2_aligned = F.adaptive_avg_pool2d(f2, (3, 3))   # (B,  64, 3, 3)
        f3_aligned = F.adaptive_avg_pool2d(f3, (3, 3))   # (B, 128, 3, 3)

        fused = torch.cat([f2_aligned, f3_aligned, f4], dim=1)  # (B, 448, 3, 3)
        fused = self.fusion_bottleneck(fused)                    # (B, 256, 3, 3)

        return self.classifier(fused)                            # (B, num_classes)





# ═══════════════════════════════════════════════════════════════════════════
# 4. MINI-XCEPTION — depthwise separable convolutions + residual connections
#    Based on the architecture from Arriaga et al. (2017)
# ═══════════════════════════════════════════════════════════════════════════

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: fewer params, same receptive field."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        # Step 1 — filter each channel independently (depthwise)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size,
                                   padding=padding, groups=in_ch, bias=False)
        # Step 2 — mix channels with 1×1 convolution (pointwise)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    """Residual block with depthwise separable convolutions."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Skip connection: 1×1 conv to match channel dims, stride 2 to halve spatial
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
    Input: 48×48×3 (grayscale replicated to 3 channels).
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        # Middle flow — 4 residual Xception blocks (each halves spatial dims)
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

        # Freeze all early layers to preserve low-level feature detectors
        for param in base.parameters():
            param.requires_grad = False

        # Unfreeze the last few inverted-residual blocks for domain adaptation
        for param in list(base.features[9:].parameters()):
            param.requires_grad = True

        # Replace the stock classifier with a custom head for NUM_CLASSES
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
# 6. EfficientNet-B2 — pretrained, fine-tuned
#    UPGRADED FROM B0:
#      • Native resolution: 260px (vs 224px for B0)  — both resized to 224 here
#      • Compound scaling: wider + deeper than B0
#      • ~9.1M params vs ~5.3M for B0 — better capacity for 7-class FER
# ═══════════════════════════════════════════════════════════════════════════

class EmotionEfficientNetB2(nn.Module):
    """EfficientNet-B2 pretrained on ImageNet, fine-tuned for FER2013."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

        # Freeze all layers; we selectively unfreeze below
        for param in base.parameters():
            param.requires_grad = False

        # Unfreeze the last 3 feature blocks (index 6, 7, 8) for adaptation.
        # B2 has 9 feature blocks (features[0..8]); early blocks detect edges/
        # textures that are broadly useful, while later blocks encode semantics.
        for param in list(base.features[6:].parameters()):
            param.requires_grad = True

        # in_features for B2 classifier is 1408 (vs 1280 for B0)
        in_features = base.classifier[-1].in_features  # 1408
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
# 7. ResNet-50 
#    Architecture: 50-layer ResNet using bottleneck blocks (1×1 → 3×3 → 1×1).
#    ~25.6M parameters. Stronger feature extractor than ResNet-18 (11M params).
#    Strategy: freeze early layers (conv1, layer1, layer2), fine-tune layer3+
# ═══════════════════════════════════════════════════════════════════════════

class EmotionResNet50(nn.Module):
    """ResNet-50 pretrained on ImageNet, fine-tuned for FER2013."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # ── Freeze backbone stages ─────────────────────────────────────────
        # conv1 and layer1–layer2 learn generic edges/textures; keep frozen.
        for param in base.parameters():
            param.requires_grad = False

        # Unfreeze layer3 and layer4 — these encode high-level semantics
        # (facial features) that need to adapt from ImageNet → FER domain.
        for param in base.layer3.parameters():
            param.requires_grad = True
        for param in base.layer4.parameters():
            param.requires_grad = True

        # ── Custom classification head ─────────────────────────────────────
        # ResNet-50's final fc has in_features = 2048 (bottleneck output).
        in_features = base.fc.in_features  # 2048
        base.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Vision Transformer (ViT-Tiny) — via timm library
# ═══════════════════════════════════════════════════════════════════════════

class EmotionViTTiny(nn.Module):
    """ViT-Tiny pretrained, fine-tuned for emotion recognition."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        import timm
        self.model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=True,
            num_classes=0,    # remove default head; we add our own below
            drop_rate=dropout,
        )
        embed_dim = self.model.embed_dim  # 192 for ViT-Tiny

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Freeze all transformer blocks; unfreeze last 4 + norm layer
        for param in self.model.parameters():
            param.requires_grad = False
        for block in self.model.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.model.norm.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model.forward_features(x)
        # ViT returns [B, num_patches+1, embed_dim]; index 0 = CLS token
        cls_token = features[:, 0]
        return self.head(cls_token)


# ═══════════════════════════════════════════════════════════════════════════
# Factory function — get any model by name
# ═══════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build a PyTorch model by name string.
    Returns the model on CPU (caller is responsible for .to(device)).
    """
    if model_name == 'ANN':
        return ANN(input_dim=48 * 48 * 3, num_classes=num_classes)
    elif model_name == 'MiniXception':
        return MiniXception(num_classes=num_classes)
    elif model_name == 'CustomCNN_MMEF':   
        return CustomCNNMMEF(num_classes=num_classes)
    elif model_name == 'MobileNetV3':
        return EmotionMobileNetV3(num_classes=num_classes)

    # ── CHANGED: EfficientNetB0 → EfficientNetB2 ──────────────────────────
    elif model_name == 'EfficientNetB2':
        return EmotionEfficientNetB2(num_classes=num_classes)

    # ── NEW: ResNet-50 ────────────────────────────────────────────────────
    elif model_name == 'ResNet50':
        return EmotionResNet50(num_classes=num_classes)

    elif model_name == 'ViTTiny':
        return EmotionViTTiny(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Valid names: ANN, MiniXception, MobileNetV3, "
            f"EfficientNetB2, ResNet50, ViTTiny"
        )
