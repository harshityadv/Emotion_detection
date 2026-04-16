# ═══════════════════════════════════════════════════════════════════════════
# data_loader.py — Unified data loading for Deep Learning and ML models
#
# CHANGES vs previous version:
#   • Class-weight computation clarified with inline math explanation.
#   • Returned class_weights are kept on CPU here; callers must do
#     .to(DEVICE) before passing to nn.CrossEntropyLoss — this keeps
#     data_loader.py device-agnostic and easier to test.
# ═══════════════════════════════════════════════════════════════════════════
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from config import (
    TRAIN_DIR, TEST_DIR, VAL_SPLIT, DL_BATCH_SIZE,
    IMG_SIZE_SMALL, IMG_SIZE_LARGE, DEVICE, NUM_CLASSES
)


# ── Transforms ────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int) -> transforms.Compose:
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),          # grayscale → fake RGB
        transforms.Resize((img_size + 8, img_size + 8)),      # slight upscale for crop
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int) -> transforms.Compose:
    """Validation / test transforms — no augmentation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_inference_transform(img_size: int) -> transforms.Compose:
    """Inference transform for real-time detection (input is already RGB)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── DataLoaders for Deep Learning ─────────────────────────────────────────

def get_dl_dataloaders(
    img_size:    int,
    batch_size:  int = DL_BATCH_SIZE,
    num_workers: int = 4,
):
    """
    Build (train_loader, val_loader, test_loader, class_weights, class_names).

    Train/val come from TRAIN_DIR (split by VAL_SPLIT), test from TEST_DIR.

    Class-weight formula
    ─────────────────────
    FER2013 is imbalanced (e.g., 'disgust' has ~436 images vs 'happy' ~8,989).
    We compute inverse-frequency weights so the loss penalises mistakes on
    rare classes more heavily:

        raw_weight[c]  = 1 / count[c]
        weight[c]      = (raw_weight[c] / Σ raw_weight) * num_classes

    Multiplying by num_classes re-scales weights so they average to 1.0,
    which keeps the overall loss magnitude comparable to unweighted CE.

    IMPORTANT: class_weights are returned on CPU.  The caller must do
    class_weights.to(DEVICE) before passing to nn.CrossEntropyLoss.
    """
    # ── Full training dataset (with augmentation) ────────────────────────
    full_train = datasets.ImageFolder(root=TRAIN_DIR,
                                      transform=get_train_transforms(img_size))
    class_names = full_train.classes
    print(f"[DataLoader] Classes: {class_names}")

    # ── Count samples per class ──────────────────────────────────────────
    class_counts = torch.zeros(len(class_names))
    for _, label in full_train.samples:
        class_counts[label] += 1
    for cls, count in zip(class_names, class_counts):
        print(f"  {cls:10s}: {int(count)} images")

    # ── Compute normalised inverse-frequency class weights (CPU tensor) ──
    # Step 1: raw inverse frequency
    raw_weights = 1.0 / class_counts                         # shape: (C,)
    # Step 2: normalise so weights sum to num_classes (avg weight = 1.0)
    class_weights = (raw_weights / raw_weights.sum()) * len(class_names)
    # NOTE: intentionally kept on CPU — callers move to DEVICE.
    print(f"  Class weights (CPU): {[f'{w:.3f}' for w in class_weights]}")

    # ── Train / Val split (reproducible with fixed seed) ─────────────────
    total      = len(full_train)
    val_size   = int(total * VAL_SPLIT)
    train_size = total - val_size
    generator  = torch.Generator().manual_seed(42)

    train_indices, val_indices = random_split(
        range(total), [train_size, val_size], generator=generator
    )

    # Val subset uses val transforms (no augmentation)
    val_dataset_full = datasets.ImageFolder(root=TRAIN_DIR,
                                            transform=get_val_transforms(img_size))
    train_subset = Subset(full_train,        train_indices.indices)
    val_subset   = Subset(val_dataset_full,  val_indices.indices)

    print(f"  Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── Test set (separate directory) ────────────────────────────────────
    test_dataset = datasets.ImageFolder(root=TEST_DIR,
                                        transform=get_val_transforms(img_size))
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    print(f"  Test:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_weights, class_names


# ── Feature extraction for traditional ML (SVM, KNN) ─────────────────────

def extract_features_for_ml(n_components: int = 150):
    """
    Load images as flattened grayscale vectors, apply PCA reduction.
    Returns X_train, y_train, X_test, y_test, pca_model.
    """
    print("[ML Features] Loading images for traditional ML models...")

    # Native FER2013 size: 48×48 grayscale
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE_SMALL, IMG_SIZE_SMALL)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=tf)
    test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=tf)

    def dataset_to_numpy(dataset):
        X, y = [], []
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        for imgs, labels in loader:
            X.append(imgs.reshape(imgs.size(0), -1).numpy())  # flatten HWC → vector
            y.append(labels.numpy())
        return np.vstack(X), np.concatenate(y)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test,  y_test  = dataset_to_numpy(test_dataset)
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # PCA reduces 48*48=2304 dims → n_components for faster SVM/KNN training
    print(f"  Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained_var:.4f}")

    return X_train_pca, y_train, X_test_pca, y_test, pca
