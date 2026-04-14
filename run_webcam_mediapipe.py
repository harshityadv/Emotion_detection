# ═══════════════════════════════════════════════════════════════════════════
# run_webcam_mediapipe.py — Real-time emotion detection with MediaPipe
# Replaces MTCNN with Google MediaPipe for faster face detection
# Supports any of the 8 trained models for emotion prediction
# ═══════════════════════════════════════════════════════════════════════════
import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

from config import DEVICE, CKPT_DIR, EMOTIONS, EMOTION_COLORS, NUM_CLASSES
from face_detection_mediapipe import MediaPipeFaceDetector
from data_loader import get_inference_transform
from models import build_model


# ═══════════════════════════════════════════════════════════════════════════
# Grad-CAM for CNN-based models (optional visualization)
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


# ═══════════════════════════════════════════════════════════════════════════
# Prediction function
# ═══════════════════════════════════════════════════════════════════════════

def predict_emotion(model, face_bgr, infer_tf, device):
    """Predict emotion for a face crop (BGR)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb).convert('L').convert('RGB')  # grayscale→RGB
    tensor = infer_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return EMOTIONS[idx], float(probs[idx]), probs


# ═══════════════════════════════════════════════════════════════════════════
# Drawing function
# ═══════════════════════════════════════════════════════════════════════════

def draw_results(frame, x1, y1, x2, y2, label, confidence, probs):
    """Draw bounding box, label, and probability bars."""
    color = EMOTION_COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f'{label.upper()} {confidence*100:.0f}%'
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
            cv2.putText(frame, f'{emo[:3]} {prob*100:.0f}%',
                        (bar_x + bar_w + 4, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Main webcam loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Select model (default: EfficientNetB0-large, or pass via CLI)
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'EfficientNetB0'
    from config import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")
        return

    cfg = MODEL_REGISTRY[model_name]
    img_size = cfg.get('img_size', 224)
    use_device = torch.device('cpu')  # CPU for real-time webcam inference

    # Load model
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

    # Inference transform
    infer_tf = get_inference_transform(img_size)

    # MediaPipe face detector
    detector = MediaPipeFaceDetector(model_selection=0, min_detection_confidence=0.5)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print(f"▶ Starting live webcam feed with {cfg['display_name']}")
    print("  Press 'q' to quit, 'm' to cycle models")

    fps_counter = 0
    fps_time = cv2.getTickCount()
    fps_display = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces with MediaPipe
        faces = detector.detect_faces(frame)

        for (x1, y1, x2, y2, det_conf) in faces:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Predict emotion
            label, conf, probs = predict_emotion(model, face_crop, infer_tf, use_device)

            # Draw results
            draw_results(frame, x1, y1, x2, y2, label, conf, probs)

        # FPS counter
        fps_counter += 1
        elapsed = (cv2.getTickCount() - fps_time) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_time = cv2.getTickCount()

        cv2.putText(frame, f'FPS: {fps_display:.1f} | Model: {model_name}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'MediaPipe Face Detection',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Emotion Detection (MediaPipe)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✓ Webcam closed.")


if __name__ == '__main__':
    main()
