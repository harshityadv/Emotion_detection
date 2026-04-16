
import os
import sys
import time
import json
import pickle
import numpy as np
import torch
import torch.nn as nn

from config import (
    DEVICE, DL_EPOCHS, DL_LR, DL_WEIGHT_DECAY, DL_LABEL_SMOOTHING,
    DL_BATCH_SIZE, EARLY_STOP_PATIENCE, MODEL_REGISTRY, CKPT_DIR,
    RESULTS_DIR, NUM_CLASSES, IMG_SIZE_SMALL, IMG_SIZE_LARGE, EMOTIONS,
)
from data_loader import get_dl_dataloaders, extract_features_for_ml
from models import build_model, get_svm_model, get_knn_model


# ═══════════════════════════════════════════════════════════════════════════
# Differential learning-rate helpers
# ═══════════════════════════════════════════════════════════════════════════

# Keywords that identify the *head* (newly initialised, high LR) vs the
# *backbone* (pretrained, low LR).  These cover every model in the registry:
#   ANN          → 'net'
#   MiniXception → 'entry'/'blocks' (backbone), 'classifier' (head)
#   MobileNetV3  → model.features (backbone), model.classifier (head)
#   EfficientNet → model.features (backbone), model.classifier (head)
#   ResNet50     → model.layer* (backbone),   model.fc (head)
#   ViTTiny      → model.blocks (backbone),   head (head)
_HEAD_KEYWORDS = ('classifier', 'fc', 'head', 'net')


def build_param_groups(model: nn.Module, base_lr: float) -> list:
    """
    Split model parameters into two groups:

        backbone_params  →  lr = base_lr * 0.1
        head_params      →  lr = base_lr

    The logic is architecture-agnostic: any parameter whose fully-qualified
    name contains a HEAD_KEYWORD is placed in head_params; everything else
    that requires a gradient goes into backbone_params.

    This mirrors the approach used in the Grad-CAM notebook where the
    ResNet backbone (layer3/layer4) used 3e-5 and the FC head used 3e-4.

    Args:
        model:   any nn.Module from models.py
        base_lr: the "head" learning rate (e.g., DL_LR = 3e-4)

    Returns:
        list of two dicts suitable for torch.optim.AdamW / Adam.
    """
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Frozen parameter — skip entirely (not in any group)
            continue

        # Check if the parameter belongs to the classification head
        is_head = any(kw in name for kw in _HEAD_KEYWORDS)

        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"  [LR groups] backbone={len(backbone_params)} params @ lr={base_lr * 0.1:.2e}"
          f"  |  head={len(head_params)} params @ lr={base_lr:.2e}")

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': base_lr * 0.1})
    if head_params:
        param_groups.append({'params': head_params, 'lr': base_lr})

    # Safety: if all params ended up in one bucket, fall back to a single group
    if not param_groups:
        raise RuntimeError(f"No trainable parameters found in model!")

    return param_groups


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a loader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════
# Main deep-learning training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_dl_model(
    model_name:   str,
    model:        nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,   # CPU tensor from data_loader
    epochs: int  = DL_EPOCHS,
    lr:     float = DL_LR,
):
    """
    Full deep-learning training loop with:

      1. Class-weighted CrossEntropyLoss with label smoothing
         ─ class_weights (from data_loader) are moved to DEVICE here so
           the caller doesn't need to remember to do it.

      2. Differential learning rates (via build_param_groups)
         ─ Pretrained backbone layers: lr * 0.1
         ─ Newly initialised head layers: lr
         ─ Applies cleanly to ALL architectures through name-based dispatch.

      3. Cosine-annealing learning-rate schedule

      4. Early stopping (patience = EARLY_STOP_PATIENCE epochs)

    Args:
        class_weights: 1-D CPU tensor, length == NUM_CLASSES.
                       Produced by get_dl_dataloaders().

    Returns:
        (best_val_acc, elapsed_seconds, history_dict)
    """
    # ── Move model to device ──────────────────────────────────────────────
    model = model.to(DEVICE)
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | Base LR: {lr}")
    print(f"{'='*70}")

    # ── Loss function ─────────────────────────────────────────────────────
    # Move class_weights to DEVICE before passing to CrossEntropyLoss.
    # data_loader returns them on CPU to stay device-agnostic; we convert here.
    weights_on_device = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=weights_on_device,
        label_smoothing=DL_LABEL_SMOOTHING,
    )
    print(f"  Loss: CrossEntropyLoss  "
          f"(label_smoothing={DL_LABEL_SMOOTHING}, weights on {DEVICE})")

    # ── Differential learning-rate optimiser ──────────────────────────────
    # build_param_groups() inspects parameter names to separate backbone
    # from head, then assigns lr*0.1 and lr respectively.
    param_groups = build_param_groups(model, base_lr=lr)
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=DL_WEIGHT_DECAY)

    # CosineAnnealingLR smoothly reduces LR from the initial value down to
    # eta_min over `epochs` steps — better than StepLR for FER training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Training state ────────────────────────────────────────────────────
    best_val_acc    = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
    }
    ckpt_path  = os.path.join(CKPT_DIR, f'{model_name}_best.pth')
    start_time = time.time()

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f'  Epoch [{epoch:02d}/{epochs}]  '
              f'Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  |  '
              f'Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}', end='')

        # Save checkpoint whenever validation accuracy improves
        if v_acc > best_val_acc:
            best_val_acc     = v_acc
            patience_counter = 0
            torch.save({
                'model_name':       model_name,
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          v_acc,
                'classes':          EMOTIONS,
            }, ckpt_path)
            print(f'  ✓ saved (best={best_val_acc:.4f})')
        else:
            patience_counter += 1
            print(f'  (patience {patience_counter}/{EARLY_STOP_PATIENCE})')

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  ⚠  Early stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed / 60:.1f} min  |  "
          f"Best Val Acc: {best_val_acc:.4f}")

    # Save full training history for later plotting / analysis
    history_path = os.path.join(RESULTS_DIR, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    return best_val_acc, elapsed, history


# ═══════════════════════════════════════════════════════════════════════════
# Traditional ML training (sklearn)
# ═══════════════════════════════════════════════════════════════════════════

def train_ml_model(model_name, model, X_train, y_train, X_test, y_test):
    """Train a sklearn model and save it to disk as a pickle."""
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    print(f"  Train Acc: {train_acc:.4f}  |  Test Acc: {test_acc:.4f}")
    print(f"  Training time: {elapsed / 60:.1f} min")

    ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_model.pkl')
    with open(ckpt_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {ckpt_path}")

    return test_acc, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Models in registry: {list(MODEL_REGISTRY.keys())}")

    # Optionally train a single model: python train_all_models.py ResNet50
    if len(sys.argv) > 1:
        selected_models = sys.argv[1:]
        for m in selected_models:
            if m not in MODEL_REGISTRY:
                print(f"Unknown model: '{m}'.  Available: {list(MODEL_REGISTRY.keys())}")
                return
    else:
        selected_models = list(MODEL_REGISTRY.keys())

    results = {}

    # Cache dataloaders so we don't reload for each model of the same size
    dl_small_data = None   # 48×48  DataLoaders + class_weights
    dl_large_data = None   # 224×224 DataLoaders + class_weights
    ml_data       = None   # sklearn PCA features

    for model_name in selected_models:
        cfg   = MODEL_REGISTRY[model_name]
        mtype = cfg['type']

        try:
            if mtype == 'ml':
                # ── Traditional ML ────────────────────────────────────────
                if ml_data is None:
                    ml_data = extract_features_for_ml(n_components=150)
                X_train, y_train, X_test, y_test, pca = ml_data

                model = get_svm_model() if model_name == 'SVM' else get_knn_model()
                acc, elapsed = train_ml_model(model_name, model,
                                              X_train, y_train, X_test, y_test)
                results[model_name] = {
                    'test_acc': acc,
                    'train_time_min': elapsed / 60,
                    'type': 'ml',
                }
                # Save PCA transformer alongside the model
                with open(os.path.join(CKPT_DIR, 'pca_model.pkl'), 'wb') as f:
                    pickle.dump(pca, f)

            elif mtype in ('dl_small', 'dl_mlp'):
                # ── Small PyTorch model (48×48) ───────────────────────────
                if dl_small_data is None:
                    print(f"\n[DataLoader] Building 48×48 loaders ...")
                    dl_small_data = get_dl_dataloaders(IMG_SIZE_SMALL,
                                                       batch_size=DL_BATCH_SIZE)
                train_loader, val_loader, _, class_weights, _ = dl_small_data

                model = build_model(model_name, NUM_CLASSES)
                # class_weights are on CPU from data_loader; train_dl_model
                # moves them to DEVICE internally before building the loss.
                val_acc, elapsed, hist = train_dl_model(
                    model_name, model, train_loader, val_loader, class_weights,
                    epochs=DL_EPOCHS, lr=DL_LR,
                )
                results[model_name] = {
                    'best_val_acc':  val_acc,
                    'train_time_min': elapsed / 60,
                    'type': mtype,
                }

            elif mtype == 'dl_large':
                # ── Large PyTorch model (224×224) — pretrained backbone ───
                if dl_large_data is None:
                    print(f"\n[DataLoader] Building 224×224 loaders ...")
                    dl_large_data = get_dl_dataloaders(IMG_SIZE_LARGE,
                                                       batch_size=DL_BATCH_SIZE)
                train_loader, val_loader, _, class_weights, _ = dl_large_data

                model = build_model(model_name, NUM_CLASSES)
                val_acc, elapsed, hist = train_dl_model(
                    model_name, model, train_loader, val_loader, class_weights,
                    epochs=DL_EPOCHS, lr=DL_LR,
                )
                results[model_name] = {
                    'best_val_acc':  val_acc,
                    'train_time_min': elapsed / 60,
                    'type': mtype,
                }

        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback; traceback.print_exc()
            results[model_name] = {'error': str(e)}

    # ── Persist overall training results ─────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'Val/Test Acc':>12} {'Time (min)':>12}")
    print(f"  {'─'*48}")
    for name, res in results.items():
        if 'error' in res:
            print(f"  {name:<22} {'ERROR':>12} {'N/A':>12}")
        else:
            acc = res.get('best_val_acc', res.get('test_acc', 0))
            print(f"  {name:<22} {acc:>12.4f} {res['train_time_min']:>12.1f}")
    print(f"  {'─'*48}")
    print(f"  Results saved → {results_path}")


if __name__ == '__main__':
    main()
