# 🛰️ Geospatial Surface Recognition System

Real-time terrain classification using a custom CNN, streamed live via mobile camera through a Streamlit web app.

> **KIIT Deemed to be University** — B.Tech Computer Science Project (2025–26)  
> Guide: Prof. Dr. Ajit Kumar Pasayat

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit app
├── camera_processor.py     # WebRTC video processor (inference + overlay)
├── logger.py               # CSV prediction logger
├── qr_generator.py         # QR code generator for mobile access
├── session_manager.py      # Session ID management
├── utils.py                # Frame preprocessing
├── terrain_classifier.py   # Model training & evaluation script
├── requirements.txt
├── model/
│   └── terrain_classifier.h5   # Place your trained model here
└── logs/
    └── terrain_log.csv         # Auto-generated prediction log
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | 3-block CNN (Conv2D + LeakyReLU + MaxPooling) |
| Input Shape | 150 × 150 × 3 (RGB) |
| Classes | Grassy, Marshy, Rocky, Sandy |
| Loss | SparseCategoricalCrossentropy |
| Optimizer | Adam |
| Epochs | 20 |
| Validation Accuracy | ~89% |
| Dataset | [Kaggle - terrain-recognition](https://www.kaggle.com/datasets/atharv1610/terrain-recognition) |

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model
```
model/
└── terrain_classifier.h5
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open on mobile
- Make sure your phone and laptop are on the **same Wi-Fi**
- Scan the QR code shown in the app, or open the URL manually
- Allow camera access when prompted

---

## 📊 Features

- **Live camera inference** via WebRTC (works on mobile too)
- **QR code** for instant mobile access
- **Session logging** — every prediction saved to CSV
- **Terrain change detection** — alerts when terrain switches
- **Safety warnings** for Marshy, Rocky, Sandy terrain
- **Analytics dashboard** — prediction history, charts, confidence stats

---

## 🔧 Training the Model

To retrain from scratch:
```bash
python terrain_classifier.py
```

Dataset folder structure:
```
dataset/
├── Grassy/
├── Marshy/
├── Rocky/
└── Sandy/
```

---

## ⚠️ Notes

- The `.h5` model file (~450MB) is not included due to GitHub's 100MB file size limit
- Use [Git LFS](https://git-lfs.github.com/) or host on [Hugging Face](https://huggingface.co/) to share weights
- WebRTC requires HTTPS in production — for local use, `localhost` works fine

---

## 👥 Team

| Name | Roll No |
|---|---|
| Supreet Kumar Patel | |
| Prakhar Patel | |
| Amrutha Jampala | |
| Abhiamrit Veera | |
| Rashu Shankar | |
| S Priyanshu Nayak | |
