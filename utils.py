import cv2
import numpy as np
from PIL import Image

# Must match the model's input shape (150x150)
IMG_HEIGHT = 150
IMG_WIDTH  = 150

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize and normalize a BGR frame (from OpenCV/WebRTC) for model inference."""
    frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame_norm = frame_resized.astype(np.float32) / 255.0
    return np.expand_dims(frame_norm, axis=0)

def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize a PIL Image (RGB from camera_input or uploader) for model inference."""
    img_rgb = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
