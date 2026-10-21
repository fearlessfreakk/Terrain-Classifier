# 🛰️ Geospatial Surface Recognition System

Real-time terrain classification using a custom CNN, streamed live via mobile phone browser through a Streamlit web application.

> **KIIT Deemed to be University** — B.Tech Computer Science Project (2025–26)  
> **Guide:** Prof. Dr. Ajit Kumar Pasayat

---

## 📁 Project Structure

```
├── .streamlit/
│   └── config.toml             # Streamlit server & custom theme configuration
├── app.py                      # Main Streamlit web application
├── camera_processor.py         # WebRTC video processor (inference + overlay)
├── logger.py                   # CSV prediction logger
├── qr_generator.py             # QR code generator for mobile access
├── session_manager.py          # Session ID management
├── utils.py                    # Frame & image preprocessing
├── terrain_classifier.py       # Model training & evaluation script
├── packages.txt                # Linux dependencies for Streamlit Cloud
├── requirements.txt            # Python dependencies
├── model/
│   └── terrain_classifier.h5   # Trained CNN model (~17.5 MB)
└── logs/
    └── terrain_log.csv         # Auto-generated prediction logs
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | 3-block CNN (Conv2D + LeakyReLU + MaxPooling) + Dense Head |
| Input Shape | 150 × 150 × 3 (RGB) |
| Output Classes | 4: **Grassy**, **Marshy**, **Rocky**, **Sandy** |
| Loss Function | SparseCategoricalCrossentropy |
| Optimizer | Adam |
| Epochs | 20 |
| Validation Accuracy | ~89% |
| Dataset | [Kaggle - terrain-recognition](https://www.kaggle.com/datasets/atharv1610/terrain-recognition) |

---

## 📱 Mobile Phone Browser Support

When you scan the QR code displayed on the PC screen:

1. **Streamlit Cloud Deployment (Recommended)**:
   - Streamlit Cloud serves over **HTTPS** (`https://<app-name>.streamlit.app`).
   - Mobile browsers (Chrome on Android, Safari on iOS) automatically allow camera permissions on HTTPS.
   - The stream automatically requests the **rear/environment camera** (`facingMode: "environment"`).

2. **Mobile Snapshot & Upload Mode (`st.camera_input`)**:
   - Tab 2 provides native mobile camera snapshots.
   - Works on all mobile devices even over local Wi-Fi without SSL certificate headaches.

3. **Local Wi-Fi Testing**:
   - Modern mobile browsers require HTTPS for WebRTC video feeds (`navigator.mediaDevices.getUserMedia`).
   - To use live WebRTC over local Wi-Fi, you can expose the local port with a secure tunnel:
     ```bash
     npx localtunnel --port 8501
     # or
     ngrok http 8501
     ```
   - Paste the resulting `https://...` link into the **⚙️ QR Code URL Settings** in the app to update the QR code.

---

## 🚀 1-Click Streamlit Cloud Deployment

This repository is pre-configured and deployment-ready:

1. **Commit and push** all files (including `model/terrain_classifier.h5`):
   ```bash
   git add .
   git commit -m "Deploy terrain classifier"
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New App"**.
4. Select your repository (`Terrain-Classifier`), branch (`main`), and set **Main file path** to `app.py`.
5. Click **"Deploy!"**.
6. Open the deployed HTTPS URL on your PC or mobile, scan the QR code, and start real-time classification!

---

## 💻 Local Setup & Execution

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the application
```bash
streamlit run app.py
```

### 3. Open on Mobile
- Point your mobile camera at the on-screen QR code.
- Allow camera access when prompted.
- Select the rear environment camera to classify ground terrain in real-time.

---

## 📊 Key Features

- **WebRTC Live Streaming**: Low-latency video feed with STUN server connectivity across mobile and desktop.
- **Rear Camera Selection**: Automatically prioritizes the environment/back camera on mobile phones.
- **Native Mobile Snapshot Mode**: Point, shoot, and classify terrain with class probabilities and confidence gauges.
- **Dynamic QR Code**: Automatically detects deployment host/HTTPS and updates the QR code.
- **Safety Advisories**: Visual warning badges for hazardous terrains (Marshy, Rocky, Sandy).
- **Session Logging & Analytics**: Records prediction timestamps, FPS, and confidence with CSV export and real-time distribution charts.

---

## 🔧 Training the Model

To retrain the model from scratch:
```bash
python terrain_classifier.py
```

Expected dataset folder structure:
```
dataset/
├── Grassy/
├── Marshy/
├── Rocky/
└── Sandy/
```