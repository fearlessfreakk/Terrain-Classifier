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
        self.model        = model
        self.session_id   = session_id
        self.prev_time    = time.time()
        self.prev_terrain = None
        self.last_change  = None
        self.last_log_time = 0.0

    def recv(self, frame):
        # Convert incoming WebRTC frame to BGR numpy array
        img = frame.to_ndarray(format="bgr24")

        # ── Inference ──
        processed  = preprocess_frame(img)
        preds      = self.model.predict(processed, verbose=0)
        
        # Robust probability conversion (supports both softmax and raw logits)
        raw_preds = preds[0]
        if np.max(raw_preds) > 1.0 or np.min(raw_preds) < 0.0 or not np.isclose(np.sum(raw_preds), 1.0, atol=0.1):
            exp_preds = np.exp(raw_preds - np.max(raw_preds))
            probabilities = exp_preds / np.sum(exp_preds)
        else:
            probabilities = raw_preds

        class_id   = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id]) * 100
        terrain    = CLASS_NAMES[class_id]

        # Show "Unknown" if confidence is too low
        display_label = terrain if confidence >= 50 else "Unknown"

        # ── FPS ──
        curr_time      = time.time()
        elapsed        = curr_time - self.prev_time
        fps            = int(1 / elapsed) if elapsed > 0 else 0
        self.prev_time = curr_time

        # ── Terrain change detection ──
        terrain_changed = False
        if self.prev_terrain and self.prev_terrain != display_label:
            self.last_change = f"Changed: {self.prev_terrain} -> {display_label}"
            terrain_changed = True
        self.prev_terrain = display_label

        # ── Log (throttled to 1s intervals or on change to prevent I/O lag) ──
        if (curr_time - self.last_log_time >= 1.0) or terrain_changed:
            log_prediction(display_label, confidence, fps, self.session_id)
            self.last_log_time = curr_time

        # ── High-contrast overlay card ──
        h, w, _ = img.shape
        card_w  = min(w - 20, 440)
        card_h  = 205
        
        # Dark overlay card background
        cv2.rectangle(img, (15, 15), (15 + card_w, 15 + card_h), (15, 23, 42), -1)
        cv2.rectangle(img, (15, 15), (15 + card_w, 15 + card_h), (51, 65, 85), 2)

        color_map = {
            "Grassy":  (74, 222, 128),  # Vibrant Green
            "Marshy":  (0, 140, 255),   # Amber/Orange
            "Rocky":   (200, 130, 250), # Lavender/Purple
            "Sandy":   (60, 210, 250),  # Yellow/Gold
            "Unknown": (148, 163, 184) # Muted Slate
        }
        text_color = color_map.get(display_label, (74, 222, 128))

        cv2.putText(img, f"Terrain: {display_label}", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, text_color, 2)

        cv2.putText(img, f"Confidence: {confidence:.1f}%", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (241, 245, 249), 2)

        cv2.putText(img, f"FPS: {fps}", (30, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 163, 184), 2)

        if self.last_change:
            cv2.putText(img, self.last_change, (30, 168),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

        alert = ALERTS.get(display_label)
        if alert:
            cv2.putText(img, alert, (30, 202),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)

        # ── Return as av.VideoFrame (required by streamlit-webrtc) ──
        return av.VideoFrame.from_ndarray(img, format="bgr24")
