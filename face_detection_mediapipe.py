# ═══════════════════════════════════════════════════════════════════════════
# face_detection_mediapipe.py — MediaPipe-based face detection
# Replaces MTCNN / Haar Cascade for faster real-time CPU/GPU processing
# ═══════════════════════════════════════════════════════════════════════════
import cv2
import mediapipe as mp


class MediaPipeFaceDetector:
    """
    Face detector using Google's MediaPipe.
    Advantages over MTCNN:
      - Runs at 200+ FPS on CPU (vs ~15-30 FPS for MTCNN)
      - No PyTorch dependency for detection
      - Optimised for real-time applications by Google
      - Works well on both CPU and GPU
    """

    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        """
        Args:
            model_selection: 0 = short-range (<2m, best for webcam),
                             1 = full-range (<5m)
            min_detection_confidence: minimum confidence threshold [0, 1]
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )
        print(f'✓ MediaPipe face detector loaded '
              f'(model={model_selection}, conf={min_detection_confidence})')

    def detect_faces(self, frame):
        """
        Detect faces in a BGR frame (OpenCV format).

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            list of (x1, y1, x2, y2, confidence) tuples in pixel coordinates.
            Coordinates are clamped to frame boundaries.
        """
        h, w, _ = frame.shape

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]

                # Convert relative coordinates to absolute pixel coordinates
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))

                # Add margin (10%) for better emotion recognition
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)

                # Skip tiny detections
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    faces.append((x1, y1, x2, y2, confidence))

        return faces

    def close(self):
        """Release resources."""
        self.detector.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass
