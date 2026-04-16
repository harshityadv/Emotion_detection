# ═══════════════════════════════════════════════════════════════════════════
# evaluate_models.py — Evaluate all trained models and generate comparisons
# ═══════════════════════════════════════════════════════════════════════════
import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from config import (
    DEVICE, CKPT_DIR, RESULTS_DIR, TEST_DIR, MODEL_REGISTRY,
    NUM_CLASSES, EMOTIONS, IMG_SIZE_SMALL, IMG_SIZE_LARGE, DL_BATCH_SIZE,
)
from data_loader import get_val_transforms, extract_features_for_ml
from models import build_model
from torchvision import datasets
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_dl_model(model, test_loader, device):
    """Evaluate a PyTorch model. Returns (y_true, y_pred, inference_time_per_image)."""
    model.eval()
    y_true, y_pred = [], []
    total_time = 0
    n_samples = 0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        start = time.time()
        outputs = model(imgs)
        total_time += time.time() - start
        preds = outputs.argmax(1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        n_samples += len(labels)

    avg_time_ms = (total_time / n_samples) * 1000  # ms per image
    return np.array(y_true), np.array(y_pred), avg_time_ms


def evaluate_ml_model(model, X_test, y_test):
    """Evaluate a sklearn model. Returns (y_true, y_pred, inference_time_per_image)."""
    start = time.time()
    y_pred = model.predict(X_test)
    total_time = time.time() - start
    avg_time_ms = (total_time / len(y_test)) * 1000
    return y_test, y_pred, avg_time_ms


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {model_name}')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=150)
    plt.close(fig)


def plot_comparison_bar(results_dict, save_dir):
    """Plot comparison bar charts for all models."""
    names = list(results_dict.keys())
    accuracies = [results_dict[n]['accuracy'] for n in names]
    f1_scores  = [results_dict[n]['f1_weighted'] for n in names]
    times_ms   = [results_dict[n]['inference_time_ms'] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Accuracy
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = axes[0].barh(names, accuracies, color=colors)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].set_xlim(0, 1)
    for bar, val in zip(bars, accuracies):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=9)

    # F1-Score
    bars = axes[1].barh(names, f1_scores, color=colors)
    axes[1].set_xlabel('Weighted F1-Score')
    axes[1].set_title('Weighted F1-Score Comparison')
    axes[1].set_xlim(0, 1)
    for bar, val in zip(bars, f1_scores):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=9)

    # Inference time
    bars = axes[2].barh(names, times_ms, color=colors)
    axes[2].set_xlabel('Inference Time (ms/image)')
    axes[2].set_title('Inference Speed Comparison')
    for bar, val in zip(bars, times_ms):
        axes[2].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{val:.2f}ms', va='center', fontsize=9)

    fig.suptitle('Model Performance Comparison on FER2013 Test Set', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Comparison chart saved.")


def plot_per_class_accuracy(results_dict, save_dir):
    """Plot per-class accuracy for all models side by side."""
    names = list(results_dict.keys())
    n_models = len(names)
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(NUM_CLASSES)
    width = 0.8 / n_models

    for i, name in enumerate(names):
        per_class = results_dict[name].get('per_class_accuracy', [0]*NUM_CLASSES)
        ax.bar(x + i * width, per_class, width, label=name, alpha=0.85)

    ax.set_xticks(x + width * n_models / 2)
    ax.set_xticklabels(EMOTIONS, rotation=45)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=150)
    plt.close(fig)
    print(f"  ✓ Per-class accuracy chart saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    all_results = {}

    # Load ML features if needed
    ml_data = None

    for model_name, cfg in MODEL_REGISTRY.items():
        mtype = cfg['type']
        display = cfg['display_name']
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {display}")
        print(f"{'─'*60}")

        try:
            if mtype == 'ml':
                # Load sklearn model
                ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_model.pkl')
                if not os.path.exists(ckpt_path):
                    print(f"  ⚠ Checkpoint not found: {ckpt_path}, skipping.")
                    continue
                with open(ckpt_path, 'rb') as f:
                    model = pickle.load(f)

                if ml_data is None:
                    ml_data = extract_features_for_ml(n_components=150)
                _, _, X_test, y_test, _ = ml_data

                y_true, y_pred, infer_ms = evaluate_ml_model(model, X_test, y_test)

            else:
                # Load PyTorch model
                ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_best.pth')
                if not os.path.exists(ckpt_path):
                    print(f"  ⚠ Checkpoint not found: {ckpt_path}, skipping.")
                    continue

                img_size = cfg.get('img_size', IMG_SIZE_LARGE)
                model = build_model(model_name, NUM_CLASSES)
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(DEVICE)

                test_dataset = datasets.ImageFolder(
                    root=TEST_DIR, transform=get_val_transforms(img_size)
                )
                test_loader = DataLoader(test_dataset, batch_size=DL_BATCH_SIZE,
                                         shuffle=False, num_workers=4, pin_memory=True)

                y_true, y_pred, infer_ms = evaluate_dl_model(model, test_loader, DEVICE)

            # Compute metrics
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average='weighted')
            report = classification_report(y_true, y_pred, target_names=EMOTIONS,
                                           output_dict=True)

            # Per-class accuracy
            cm = confusion_matrix(y_true, y_pred)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)

            print(f"  Accuracy:      {acc:.4f}")
            print(f"  F1 (weighted): {f1:.4f}")
            print(f"  Inference:     {infer_ms:.2f} ms/image")
            print(f"  Per-class:     {dict(zip(EMOTIONS, [f'{a:.3f}' for a in per_class_acc]))}")

            all_results[model_name] = {
                'display_name': display,
                'accuracy': float(acc),
                'f1_weighted': float(f1),
                'inference_time_ms': float(infer_ms),
                'per_class_accuracy': per_class_acc.tolist(),
                'classification_report': report,
            }

            # Confusion matrix plot
            plot_confusion_matrix(y_true, y_pred, EMOTIONS, model_name, RESULTS_DIR)

        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            import traceback; traceback.print_exc()

    # ── Generate comparison plots ─────────────────────────────────────────
    if len(all_results) >= 2:
        plot_comparison_bar(all_results, RESULTS_DIR)
        plot_per_class_accuracy(all_results, RESULTS_DIR)

    # ── Save full results ─────────────────────────────────────────────────
    eval_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — Test Set Performance")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1-Score':>10} {'Speed(ms)':>10}")
    print(f"  {'─'*55}")
    # Sort by accuracy descending
    sorted_names = sorted(all_results.keys(), key=lambda n: all_results[n]['accuracy'], reverse=True)
    for name in sorted_names:
        r = all_results[name]
        print(f"  {r['display_name']:<22} {r['accuracy']:>10.4f} "
              f"{r['f1_weighted']:>10.4f} {r['inference_time_ms']:>10.2f}")
    print(f"  {'─'*55}")
    print(f"  Results saved to: {eval_path}")


if __name__ == '__main__':
    main()
