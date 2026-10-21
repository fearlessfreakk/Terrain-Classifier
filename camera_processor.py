import av
import cv2
import time
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from utils import preprocess_frame
from logger import log_prediction

# Must match the 4 output classes of terrain_classifier.h5
# Order must match the alphabetical folder order from the Kaggle dataset
CLASS_NAMES = ["Grassy", "Marshy", "Rocky", "Sandy"]

# Safety alerts for specific terrain types
ALERTS = {
    "Marshy": "⚠️ Unsafe Terrain - Marshy Ground",
    "Rocky":  "⚠️ Rough Surface - Rocky Terrain",
    "Sandy":  "⚠️ Low Traction - Sandy Ground"
}

class TerrainProcessor(VideoProcessorBase):
    def __init__(self, model, session_id):
        self.model      = model
        self.session_id = session_id
        self.prev_time  = time.time()
        self.prev_terrain = None
        self.last_change  = None

    def recv(self, frame):
        # Convert incoming WebRTC frame to BGR numpy array
        img = frame.to_ndarray(format="bgr24")

        # ── Inference ──
        processed  = preprocess_frame(img)
        preds      = self.model.predict(processed, verbose=0)
        class_id   = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_id]) * 100
        terrain    = CLASS_NAMES[class_id]

        # Show "Unknown" if confidence is too low
        display_label = terrain if confidence >= 60 else "Unknown"

        # ── FPS ──
        curr_time      = time.time()
        elapsed        = curr_time - self.prev_time
        fps            = int(1 / elapsed) if elapsed > 0 else 0
        self.prev_time = curr_time

        # ── Terrain change detection ──
        if self.prev_terrain and self.prev_terrain != display_label:
            self.last_change = f"Changed: {self.prev_terrain} -> {display_label}"
        self.prev_terrain = display_label

        # ── Log ──
        log_prediction(display_label, confidence, fps, self.session_id)

        # ── Overlay text on frame ──
        cv2.putText(img, f"Terrain: {display_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(img, f"Confidence: {confidence:.1f}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(img, f"FPS: {fps}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if self.last_change:
            cv2.putText(img, self.last_change, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        alert = ALERTS.get(display_label)
        if alert:
            cv2.putText(img, alert, (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── Return as av.VideoFrame (required by streamlit-webrtc) ──
        return av.VideoFrame.from_ndarray(img, format="bgr24")
