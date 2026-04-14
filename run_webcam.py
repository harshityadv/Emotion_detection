import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── 1. Setup & Paths ──────────────────────────────────────────────────────
DEVICE = torch.device('cpu')  # Using CPU for local laptop inference
CKPT_PATH = r"C:\Users\Dell\DAC-204Project\best_model2.pth"

# ── 2. Model Definition ───────────────────────────────────────────────────
class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout/2),
            nn.Linear(256, num_classes)
        )
        self.model = base
    def forward(self, x): 
        return self.model(x)

# ── 3. Transforms ─────────────────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── 4. Load Weights ───────────────────────────────────────────────────────
print("Loading model weights...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
emotions = ckpt['classes']
model = EmotionResNet(num_classes=len(emotions)).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✓ Model loaded successfully! Best Val Acc: {ckpt['val_acc']:.4f}")

# ── 5. Setup Webcam & Face Detector ───────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) # 0 targets your default laptop webcam

print("▶ Starting live webcam feed. Press 'q' on your keyboard to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Crop and process the face
        face_crop = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb).convert('RGB')
        tensor = infer_tf(pil_img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            label = emotions[torch.argmax(outputs).item()]

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Live Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()