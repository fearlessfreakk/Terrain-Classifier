import cv2
import numpy as np

# Must match the model's input shape (150x150)
IMG_HEIGHT = 150
IMG_WIDTH  = 150

def preprocess_frame(frame):
    """Resize and normalize a BGR frame for model inference."""
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0).astype(np.float32)
    return frame
