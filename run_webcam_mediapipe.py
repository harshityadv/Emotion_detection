# ═══════════════════════════════════════════════════════════════════════════
# run_webcam_mediapipe.py — Real-time emotion detection with MediaPipe
# Replaces MTCNN with Google MediaPipe for faster face detection
# Supports any of the 8 trained models for emotion prediction
#
# FIXES vs previous version:
#   1. Added full smoothing pipeline (prob averaging, label lock,
#      time lock, box smoothing, temperature scaling) — ported from notebook
#   2. Fixed Grad-CAM: added None guard for first-frame crash
#   3. Fixed Grad-CAM: inference transform now includes Grayscale step
#      (model was trained on grayscale — sending real RGB gave garbage heatmaps)
#   4. Grad-CAM only enabled for CNN-based models (not ViT/ANN/MLP)
# ═══════════════════════════════════════════════════════════════════════════
import sys
import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from collections import deque
import torch.nn.functional as F
from torchvision import transforms

from config import DEVICE, CKPT_DIR, EMOTIONS, EMOTION_COLORS, NUM_CLASSES
from face_detection_mediapipe import MediaPipeFaceDetector
from models import build_model


# ═══════════════════════════════════════════════════════════════════════════
# Inference transform
# FIX: must include Grayscale() — model was trained on grayscale→RGB images.
# The original get_inference_transform() in data_loader.py skips this step,
# so we define the correct one here explicitly.
# ═══════════════════════════════════════════════════════════════════════════

def get_correct_inference_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # ← critical: match training
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Grad-CAM — fixed version
# ═══════════════════════════════════════════════════════════════════════════

class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # FIX: guard against None on first call (before any backward has run)
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_gradcam(face_bgr, heatmap, alpha=0.45):
    """Blend Grad-CAM heatmap onto face crop."""
    h, w = face_bgr.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    heat8 = np.uint8(255 * heat)
    color = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
    return cv2.addWeighted(face_bgr, 1 - alpha, color, alpha, 0)


def get_gradcam_layer(model, model_name):
    """
    Return the target conv layer for Grad-CAM, or None if not supported.
    Only CNN-based models support Grad-CAM (not ViT, ANN, SVM, KNN).
    """
    if model_name == 'EfficientNetB0':
        return model.model.features[-1]
    elif model_name == 'MobileNetV3':
        return model.model.features[-1]
    elif model_name == 'CustomCNN':
        return model.features[-3]       # last Conv2d before AdaptiveAvgPool
    elif model_name == 'MiniXception':
        return model.blocks[-1].sep_block[-2]  # last BN before pool
    else:
        return None  # ANN, ViT, SVM, KNN — not applicable


# ═══════════════════════════════════════════════════════════════════════════
# Smoothing helpers — ported from notebook Cell 8
# ═══════════════════════════════════════════════════════════════════════════

def smooth_probs(face_idx, raw_probs, prob_buffers, smooth_window):
    """Maintain a rolling buffer of prob arrays and return their mean."""
    if face_idx not in prob_buffers:
        prob_buffers[face_idx] = deque(maxlen=smooth_window)
    prob_buffers[face_idx].append(raw_probs)
    return np.mean(prob_buffers[face_idx], axis=0)


def locked_label(face_idx, raw_label, label_state, lock_frames):
    """Prevent flip-flopping: new emotion must win lock_frames frames in a row."""
    state = label_state.setdefault(face_idx, {
        'current': raw_label, 'candidate': raw_label, 'streak': 0
    })
    if raw_label == state['candidate']:
        state['streak'] += 1
    else:
        state['candidate'] = raw_label
        state['streak'] = 1
    if state['streak'] >= lock_frames:
        state['current'] = raw_label
    return state['current']


def time_locked_label(face_idx, raw_label, display_state, min_display_ms):
    """Once an emotion is shown, keep it displayed for at least min_display_ms."""
    now = time.time()
    state = display_state.setdefault(face_idx, {'label': raw_label, 'until': 0.0})
    if now >= state['until']:
        state['label'] = raw_label
        state['until'] = now + min_display_ms / 1000.0
    return state['label']


def smooth_box(face_idx, x1, y1, x2, y2, box_buffers, box_smooth):
    """Average bounding box coords over box_smooth frames to reduce jitter."""
    if face_idx not in box_buffers:
        box_buffers[face_idx] = deque(maxlen=box_smooth)
    box_buffers[face_idx].append((x1, y1, x2, y2))
    return tuple(int(np.mean([b[i] for b in box_buffers[face_idx]])) for i in range(4))


# ═══════════════════════════════════════════════════════════════════════════
# Prediction with temperature scaling
# ═══════════════════════════════════════════════════════════════════════════

def predict_emotion(model, face_bgr, infer_tf, device, temperature=1.0):
    """
    Predict emotion for a face crop (BGR) with optional temperature scaling.
    temperature > 1 softens the distribution (less spiky), good for smoothing.
    Returns (label, confidence, probs_array).
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    tensor = infer_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits / temperature, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return EMOTIONS[idx], float(probs[idx]), probs, tensor


# ═══════════════════════════════════════════════════════════════════════════
# Drawing function
# ═══════════════════════════════════════════════════════════════════════════

def draw_results(frame, x1, y1, x2, y2, label, confidence, probs):
    """Draw bounding box, label, and probability bars."""
    color = EMOTION_COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f'{label.upper()} {confidence * 100:.0f}%'
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Probability bars
    bar_x, bar_y, bar_w, bar_gap = x2 + 10, y1, 90, 14
    if bar_x + bar_w + 55 < frame.shape[1]:
        for i, (emo, prob) in enumerate(zip(EMOTIONS, probs)):
            by = bar_y + i * bar_gap
            filled = int(prob * bar_w)
            emo_color = EMOTION_COLORS.get(emo, (200, 200, 200))
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + 10), (60, 60, 60), -1)
            if filled > 0:
                cv2.rectangle(frame, (bar_x, by), (bar_x + filled, by + 10), emo_color, -1)
            cv2.putText(frame, f'{emo[:3]} {prob * 100:.0f}%',
                        (bar_x + bar_w + 4, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Main webcam loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Config ────────────────────────────────────────────────────────────
    # Smoothing parameters (tweak as needed)
    SMOOTH_WINDOW  = 7      # Average probabilities over last N frames
    LOCK_FRAMES    = 4      # Emotion must win N consecutive frames to switch
    BOX_SMOOTH     = 5      # Average bounding box coords over last N frames
    TEMPERATURE    = 1.5    # Softmax temperature (>1 = softer, <1 = sharper)
    MIN_DISPLAY_MS = 600    # Hold displayed emotion for at least this many ms

    # Grad-CAM options
    GRADCAM_OVERLAY = True  # Set False to disable heatmap overlay
    GRADCAM_EVERY_N = 3     # Recompute Grad-CAM every N frames (saves FPS)
    GRADCAM_ALPHA   = 0.40  # Heatmap blend strength

    # ── Model selection ───────────────────────────────────────────────────
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'EfficientNetB0'
    from config import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")
        return

    cfg = MODEL_REGISTRY[model_name]
    img_size = cfg.get('img_size', 224)
    use_device = torch.device('cpu')  # CPU for real-time webcam inference

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, f'{model_name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("   Run train_all_models.py first.")
        return

    print(f"Loading model: {cfg['display_name']}...")
    model = build_model(model_name, NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=use_device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(use_device)
    model.eval()
    print(f"✓ Model loaded | Val Acc: {ckpt['val_acc']:.4f}")

    # ── Inference transform (with Grayscale fix) ──────────────────────────
    infer_tf = get_correct_inference_transform(img_size)

    # ── Grad-CAM setup (CNN models only) ─────────────────────────────────
    gradcam = None
    target_layer = get_gradcam_layer(model, model_name)
    if GRADCAM_OVERLAY and target_layer is not None:
        gradcam = GradCAM(model, target_layer)
        print(f"✓ Grad-CAM enabled on {model_name}")
    elif GRADCAM_OVERLAY:
        print(f"ℹ Grad-CAM not supported for {model_name} — overlay disabled")

    # ── Per-face smoothing state ──────────────────────────────────────────
    _prob_buffers  = {}   # face_idx → deque of prob arrays
    _box_buffers   = {}   # face_idx → deque of (x1, y1, x2, y2)
    _label_state   = {}   # face_idx → {current, candidate, streak}
    _display_state = {}   # face_idx → {label, until}
    _gc_cache      = {}   # face_idx → cached Grad-CAM overlay (BGR)

    # ── MediaPipe face detector ───────────────────────────────────────────
    detector = MediaPipeFaceDetector(model_selection=0, min_detection_confidence=0.5)

    # ── Webcam ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    gc_label     = 'GradCAM:ON' if (gradcam is not None) else 'GradCAM:OFF'
    smooth_label = f'W{SMOOTH_WINDOW}/L{LOCK_FRAMES}/T{TEMPERATURE}'
    print(f"▶ Starting live webcam feed with {cfg['display_name']}")
    print(f"  {gc_label}  Smooth:{smooth_label}")
    print("  Press 'q' to quit | 's' to save snapshot")

    fps_display = 0.0
    t_prev      = time.time()
    frame_no    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        t_now        = time.time()
        fps_display  = 0.9 * fps_display + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev       = t_now

        # ── Detect faces with MediaPipe ───────────────────────────────────
        faces = detector.detect_faces(frame)

        for face_idx, (x1r, y1r, x2r, y2r, det_conf) in enumerate(faces):
            # Smooth bounding box to reduce jitter
            x1, y1, x2, y2 = smooth_box(
                face_idx, x1r, y1r, x2r, y2r, _box_buffers, BOX_SMOOTH
            )

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # ── Predict with temperature scaling ─────────────────────────
            _, _, raw_probs, tensor = predict_emotion(
                model, face_crop, infer_tf, use_device, TEMPERATURE
            )

            # ── Smooth probabilities over rolling window ──────────────────
            smoothed_probs = smooth_probs(
                face_idx, raw_probs, _prob_buffers, SMOOTH_WINDOW
            )

            # ── Derive label from smoothed probs ─────────────────────────
            raw_label  = EMOTIONS[int(np.argmax(smoothed_probs))]
            confidence = float(smoothed_probs.max())

            # ── Label lock: must win LOCK_FRAMES consecutive frames ───────
            locked = locked_label(face_idx, raw_label, _label_state, LOCK_FRAMES)

            # ── Time lock: hold displayed emotion for MIN_DISPLAY_MS ──────
            display_label = time_locked_label(
                face_idx, locked, _display_state, MIN_DISPLAY_MS
            )

            # ── Grad-CAM overlay (optional, CNN models only) ──────────────
            if gradcam is not None:
                compute_gc = (frame_no % GRADCAM_EVERY_N == 0) or (face_idx not in _gc_cache)
                if compute_gc:
                    face_rgb_gc = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_gc      = Image.fromarray(face_rgb_gc)
                    tensor_gc   = infer_tf(pil_gc).unsqueeze(0).to(use_device)
                    with torch.enable_grad():
                        heatmap = gradcam.generate(tensor_gc)
                    _gc_cache[face_idx] = overlay_gradcam(face_crop, heatmap, alpha=GRADCAM_ALPHA)

                if face_idx in _gc_cache:
                    gc_resized = cv2.resize(_gc_cache[face_idx], (x2 - x1, y2 - y1))
                    frame[y1:y2, x1:x2] = gc_resized

            # ── Draw results using smoothed values ────────────────────────
            frame = draw_results(
                frame, x1, y1, x2, y2, display_label, confidence, smoothed_probs
            )

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(
            frame,
            f'FPS: {fps_display:.1f} | {model_name} | {gc_label} | Smooth:{smooth_label}',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2
        )
        cv2.putText(
            frame, 'MediaPipe  [Q=quit | S=snapshot]',
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )

        cv2.imshow('Emotion Detection (MediaPipe)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snap = f'snapshot_{frame_no}.jpg'
            cv2.imwrite(snap, frame)
            print(f'📸 Snapshot saved: {snap}')

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"✓ Webcam closed. Processed {frame_no} frames.")


if __name__ == '__main__':
    main()