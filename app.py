import streamlit as st
import tensorflow as tf
import socket
import os
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from camera_processor import TerrainProcessor
from qr_generator import generate_qr
from session_manager import create_session

st.set_page_config(page_title="Intelligent Terrain Classification", layout="wide")

# ── Load model (cached so it only loads once) ──
@st.cache_resource
def load_model():
    model_path = "model/terrain_classifier.h5"
    if not os.path.exists(model_path):
        st.error(
            f"Model not found at '{model_path}'.\n\n"
            "Place your terrain_classifier.h5 file inside a folder called 'model/'."
        )
        st.stop()
    return tf.keras.models.load_model(model_path)

model   = load_model()
session = create_session()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()
    return ip


st.title("🛰️ Intelligent Terrain Classification System")

tab1, tab2, tab3 = st.tabs([
    "📊 History & Analytics",
    "📷 Live Camera (QR Enabled)",
    "ℹ️ Project Overview"
])

# ──────────────────────────────────────────────
# TAB 1: HISTORY & ANALYTICS
# ──────────────────────────────────────────────
with tab1:
    st.subheader("📊 Previous Sessions & Analytics")

    log_path = "logs/terrain_log.csv"

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        st.dataframe(df.tail(200), use_container_width=True)

        st.markdown("### 📈 Terrain Distribution")
        st.bar_chart(df["Terrain"].value_counts())

        st.markdown("### 📊 System Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Total Predictions", len(df))
        col2.metric(
            "Average Confidence (%)",
            round(df["Confidence"].astype(float).mean(), 2)
        )

        if st.button("🗑️ Clear Log"):
            os.remove(log_path)
            st.success("Log cleared. Refresh the page.")
    else:
        st.info("No previous data available. Start the live camera to generate logs.")


# ──────────────────────────────────────────────
# TAB 2: LIVE CAMERA
# ──────────────────────────────────────────────
with tab2:
    st.subheader("📷 Live Terrain Classification")

    ip       = get_local_ip()
    port     = 8501
    app_url  = f"http://{ip}:{port}"

    st.markdown("### 📱 Scan QR Code to Open on Mobile")
    qr_img = generate_qr(app_url)
    st.image(qr_img, width=240)
    st.caption(f"Or open manually: {app_url}")

    st.info(
        "• Laptop and mobile must be on the **same Wi-Fi**\n"
        "• Allow camera access when prompted\n"
        "• Classification starts automatically"
    )

    webrtc_streamer(
        key="terrain-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: TerrainProcessor(model, session["id"]),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


# ──────────────────────────────────────────────
# TAB 3: PROJECT OVERVIEW
# ──────────────────────────────────────────────
with tab3:
    st.subheader("ℹ️ Project Overview")
    st.markdown("""
    **Geospatial Surface Recognition System**

    **Objective:**  
    Real-time terrain classification using a custom-trained CNN with live mobile camera access.

    **Model Details:**
    - Architecture: 3-block CNN (Conv2D + LeakyReLU + MaxPooling) + Dense head
    - Input: 150 × 150 RGB images
    - Classes: Grassy, Marshy, Rocky, Sandy
    - Validation Accuracy: ~89%
    - Dataset: [Kaggle - terrain-recognition](https://www.kaggle.com/datasets/atharv1610/terrain-recognition)

    **Key Features:**
    - Live camera inference via WebRTC
    - QR-based mobile streaming
    - Session-wise prediction logging
    - Terrain change detection
    - Safety alerts for hazardous terrain

    **Applications:**  
    Autonomous vehicles, robotics, surveillance, disaster response, environmental monitoring.

    ---
    **KIIT Deemed to be University** | B.Tech Computer Science | 2025–26  
    Guide: Prof. Dr. Ajit Kumar Pasayat
    """)
