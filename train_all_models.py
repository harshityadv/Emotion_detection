# ═══════════════════════════════════════════════════════════════════════════
# train_all_models.py — Master training script for all 8 models
# ═══════════════════════════════════════════════════════════════════════════
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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def train_dl_model(model_name, model, train_loader, val_loader,
                   class_weights, epochs=DL_EPOCHS, lr=DL_LR):
    """
    Full deep learning training loop with:
    - Class-weighted cross-entropy with label smoothing
    - Cosine annealing LR
    - Differential LR for pretrained vs head parameters
    - Early stopping
    """
    model = model.to(DEVICE)
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | LR: {lr}")
    print(f"{'='*70}")

    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=DL_LABEL_SMOOTHING)

    # Separate parameters: pretrained backbone (lower LR) vs head (higher LR)
    pretrained_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name or 'head' in name or 'fc' in name or 'net' in name:
                head_params.append(param)
            else:
                pretrained_params.append(param)

    if pretrained_params:
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': lr * 0.1},
            {'params': head_params,       'lr': lr},
        ], weight_decay=DL_WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                       weight_decay=DL_WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_best.pth')
    start_time = time.time()

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

        if v_acc > best_val_acc:
            best_val_acc = v_acc
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
            print(f"  ⚠ Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed/60:.1f} min | Best Val Acc: {best_val_acc:.4f}")

    # Save training history
    with open(os.path.join(RESULTS_DIR, f'{model_name}_history.json'), 'w') as f:
        json.dump(history, f)

    return best_val_acc, elapsed, history


def train_ml_model(model_name, model, X_train, y_train, X_test, y_test):
    """Train a sklearn model and save it."""
    print(f"\n{'='*70}")
    print(f"  Training: {model_name}")
    print(f"{'='*70}")

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)
    print(f"  Train Acc: {train_acc:.4f}  |  Test Acc: {test_acc:.4f}")
    print(f"  Training time: {elapsed/60:.1f} min")

    # Save model
    ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_model.pkl')
    with open(ckpt_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {ckpt_path}")

    return test_acc, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Main training orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Models to train: {list(MODEL_REGISTRY.keys())}")

    # Allow training a specific model via command-line argument
    if len(sys.argv) > 1:
        selected_models = sys.argv[1:]
        for m in selected_models:
            if m not in MODEL_REGISTRY:
                print(f"Unknown model: {m}. Available: {list(MODEL_REGISTRY.keys())}")
                return
    else:
        selected_models = list(MODEL_REGISTRY.keys())

    results = {}

    # ── Preload data for different model types ────────────────────────────
    dl_small_data = None  # 48x48 DataLoaders
    dl_large_data = None  # 224x224 DataLoaders
    ml_data       = None  # sklearn features

    for model_name in selected_models:
        cfg = MODEL_REGISTRY[model_name]
        mtype = cfg['type']

        try:
            if mtype == 'ml':
                # Traditional ML model
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
                # Also save PCA transformer
                with open(os.path.join(CKPT_DIR, 'pca_model.pkl'), 'wb') as f:
                    pickle.dump(pca, f)

            elif mtype in ('dl_small', 'dl_mlp'):
                # PyTorch model with 48x48 input
                if dl_small_data is None:
                    dl_small_data = get_dl_dataloaders(IMG_SIZE_SMALL,
                                                       batch_size=DL_BATCH_SIZE)
                train_loader, val_loader, test_loader, class_weights, _ = dl_small_data

                model = build_model(model_name, NUM_CLASSES)
                val_acc, elapsed, hist = train_dl_model(
                    model_name, model, train_loader, val_loader, class_weights,
                    epochs=DL_EPOCHS, lr=DL_LR
                )
                results[model_name] = {
                    'best_val_acc': val_acc,
                    'train_time_min': elapsed / 60,
                    'type': mtype,
                }

            elif mtype == 'dl_large':
                # PyTorch model with 224x224 input (pretrained)
                if dl_large_data is None:
                    dl_large_data = get_dl_dataloaders(IMG_SIZE_LARGE,
                                                       batch_size=DL_BATCH_SIZE)
                train_loader, val_loader, test_loader, class_weights, _ = dl_large_data

                model = build_model(model_name, NUM_CLASSES)
                val_acc, elapsed, hist = train_dl_model(
                    model_name, model, train_loader, val_loader, class_weights,
                    epochs=DL_EPOCHS, lr=DL_LR
                )
                results[model_name] = {
                    'best_val_acc': val_acc,
                    'train_time_min': elapsed / 60,
                    'type': mtype,
                }

        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback; traceback.print_exc()
            results[model_name] = {'error': str(e)}

    # ── Save overall results ──────────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"  All results saved to {results_path}")
    print(f"{'='*70}")

    # Print summary
    print(f"\n{'Model':<20} {'Val/Test Acc':>12} {'Time (min)':>12}")
    print('-' * 50)
    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<20} {'ERROR':>12} {'N/A':>12}")
        else:
            acc = res.get('best_val_acc', res.get('test_acc', 0))
            print(f"{name:<20} {acc:>12.4f} {res['train_time_min']:>12.1f}")


if __name__ == '__main__':
    main()
