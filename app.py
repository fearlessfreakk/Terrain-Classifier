import streamlit as st
import tensorflow as tf
import numpy as np
import socket
import os
import time
import pandas as pd
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from camera_processor import TerrainProcessor, CLASS_NAMES, ALERTS
from qr_generator import generate_qr
from session_manager import create_session
from utils import preprocess_pil_image
from logger import log_prediction

# Page configuration
st.set_page_config(
    page_title="Intelligent Terrain Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────
# STUN / ICE SERVERS FOR WEBRTC (CRUCIAL FOR MOBILE & CLOUD)
# ──────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
})

# Rear / back camera preference on mobile phones with fallback
MEDIA_STREAM_CONSTRAINTS = {
    "video": {
        "facingMode": {"ideal": "environment"},
        "width": {"ideal": 640},
        "height": {"ideal": 480},
    },
    "audio": False
}

# ──────────────────────────────────────────────
# MODEL LOADING WITH CACHE & ERROR RESILIENCE
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model/terrain_classifier.h5"
    if not os.path.exists(model_path):
        st.error(
            f"⚠️ Model file not found at '{model_path}'.\n\n"
            "Please ensure 'model/terrain_classifier.h5' exists in the repository."
        )
        st.stop()
    
    # Check if the file is an un-pulled Git LFS pointer (< 2KB)
    if os.path.getsize(model_path) < 2048:
        st.error(
            "⚠️ The model file appears to be a Git LFS pointer rather than the binary model.\n\n"
            "Please push the full 17.5 MB 'model/terrain_classifier.h5' to GitHub without LFS pointer substitution."
        )
        st.stop()

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load Keras model: {e}")
        st.stop()

model = load_model()

# Initialize or retain session in Streamlit state
if "session_info" not in st.session_state:
    st.session_state.session_info = create_session()
session = st.session_state.session_info

# ──────────────────────────────────────────────
# NETWORK / URL RESOLUTION
# ──────────────────────────────────────────────
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

def detect_public_url():
    """Detect if running on Streamlit Community Cloud / reverse proxy or local machine."""
    # 1. Environment variable override (e.g. deployed container)
    env_url = os.environ.get("STREAMLIT_SERVER_BASE_URL", "").strip()
    if env_url:
        return env_url, "Custom Environment URL"

    # 2. Check HTTP request headers from Streamlit context
    try:
        headers = getattr(st.context, "headers", {})
        host = headers.get("host", "")
        proto = headers.get("x-forwarded-proto", "http")

        # Streamlit Cloud deployment detection
        if host and "streamlit.app" in host:
            return f"https://{host}", "Streamlit Cloud (HTTPS)"

        if host and not (host.startswith("localhost") or host.startswith("127.0.0.1")):
            return f"{proto}://{host}", f"Proxy Host ({proto})"
    except Exception:
        pass

    # 3. Local network IP fallback
    local_ip = get_local_ip()
    return f"http://{local_ip}:8501", "Local Wi-Fi Network"

detected_url, url_source = detect_public_url()

# ──────────────────────────────────────────────
# HEADER & APP BRANDING
# ──────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🛰️ Intelligent Terrain Classification System")
    st.caption("AI-powered surface recognition with live mobile camera streaming & WebRTC")

with col_status:
    st.markdown(
        f"""
        <div style="background-color: #1e293b; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155; margin-top: 10px;">
            <div style="font-size: 12px; color: #94a3b8;">ACTIVE SESSION</div>
            <div style="font-size: 15px; font-weight: 600; color: #38bdf8;">#{session['id']}</div>
            <div style="font-size: 11px; color: #10b981;">● Model Active (4 Classes)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# TABS NAVIGATION
# ──────────────────────────────────────────────
tab_live, tab_mobile, tab_analytics, tab_overview = st.tabs([
    "📷 Live Camera (QR & WebRTC)",
    "📱 Mobile Camera Snap & Upload",
    "📊 History & Analytics",
    "ℹ️ Project & Deployment Guide"
])

# ──────────────────────────────────────────────
# TAB 1: LIVE WEBRTC CAMERA & QR CODE
# ──────────────────────────────────────────────
with tab_live:
    st.subheader("📷 Real-Time Terrain Stream")

    col_qr, col_stream = st.columns([1, 2])

    with col_qr:
        st.markdown("### 📱 Mobile Phone Access")
        st.markdown("Scan this QR code with your mobile camera to open the application:")

        # URL override expander for ngrok / custom domain
        with st.expander("⚙️ QR Code URL Settings", expanded=False):
            custom_url = st.text_input(
                "Public or Tunnel URL (optional):",
                value="",
                placeholder="e.g. https://xyz.ngrok-free.app or your Streamlit Cloud link",
                help="If running locally without HTTPS, enter your ngrok/localtunnel URL here to test mobile camera WebRTC."
            ).strip()

        active_url = custom_url if custom_url else detected_url
        qr_img = generate_qr(active_url)
        st.image(qr_img, width=220)

        st.markdown(f"**Target URL:** [`{active_url}`]({active_url})")
        st.caption(f"Detected mode: **{url_source}**")

        st.markdown("---")
        st.markdown("#### 💡 Mobile Browser Tips")
        st.markdown(
            """
            - **On Streamlit Cloud**: Camera works immediately over HTTPS!
            - **On Local Wi-Fi (`http://...`)**: Modern mobile browsers (Chrome/Safari) block WebRTC camera access on insecure HTTP origins.
              - **Option A**: Use **'📱 Mobile Camera Snap'** tab (works on any phone without HTTPS!).
              - **Option B**: Deploy to Streamlit Cloud for automatic HTTPS.
              - **Option C**: Run a tunnel: `npx localtunnel --port 8501` or `ngrok http 8501`.
            """
        )

    with col_stream:
        st.markdown("### 🎥 Live Video Classification")
        st.caption("Point camera at ground surface. The AI will classify terrain and overlay real-time stats.")

        webrtc_streamer(
            key="terrain-live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            video_processor_factory=lambda: TerrainProcessor(model, session["id"]),
            async_processing=True,
        )

# ──────────────────────────────────────────────
# TAB 2: MOBILE CAMERA SNAP & UPLOAD MODE
# ──────────────────────────────────────────────
with tab_mobile:
    st.subheader("📱 Mobile Camera Snapshot & Image Analyzer")
    st.markdown(
        "Take a quick photo using your phone's native camera or upload an image. "
        "This mode works smoothly on all mobile devices and bypasses browser WebRTC streaming restrictions."
    )

    mode = st.radio(
        "Choose Input Method:",
        ["📸 Take Photo with Camera", "📁 Upload Image File"],
        horizontal=True
    )

    input_image = None

    if mode == "📸 Take Photo with Camera":
        camera_photo = st.camera_input("Point camera at terrain and take snapshot")
        if camera_photo is not None:
            input_image = Image.open(camera_photo)
    else:
        uploaded_file = st.file_uploader(
            "Choose a terrain image (JPG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)

    if input_image is not None:
        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(input_image, caption="Captured Terrain Image", use_container_width=True)

        with col_pred:
            # Preprocess and run inference
            start_t = time.time()
            tensor = preprocess_pil_image(input_image)
            raw_preds = model.predict(tensor, verbose=0)[0]
            elapsed_ms = (time.time() - start_t) * 1000

            # Convert logits to probabilities if needed
            if np.max(raw_preds) > 1.0 or np.min(raw_preds) < 0.0 or not np.isclose(np.sum(raw_preds), 1.0, atol=0.1):
                exp_preds = np.exp(raw_preds - np.max(raw_preds))
                probs = exp_preds / np.sum(exp_preds)
            else:
                probs = raw_preds

            top_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(probs[top_idx]) * 100

            # Log prediction to CSV
            fps_dummy = int(1000 / max(elapsed_ms, 1))
            log_prediction(predicted_class, confidence, fps_dummy, session["id"])

            # Color scheme for terrain
            badge_colors = {
                "Grassy": "#22c55e",
                "Marshy": "#f97316",
                "Rocky":  "#a855f7",
                "Sandy":  "#eab308"
            }
            theme_color = badge_colors.get(predicted_class, "#38bdf8")

            st.markdown(
                f"""
                <div style="background-color: #1e293b; border-left: 6px solid {theme_color}; padding: 18px; border-radius: 8px; margin-bottom: 16px;">
                    <div style="font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Classification</div>
                    <div style="font-size: 32px; font-weight: 700; color: {theme_color}; margin-top: 4px;">{predicted_class}</div>
                    <div style="font-size: 16px; color: #f8fafc; margin-top: 4px;">Confidence: <b>{confidence:.2f}%</b> &nbsp;|&nbsp; Latency: <b>{elapsed_ms:.1f} ms</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Safety advisory
            if predicted_class in ALERTS:
                st.warning(ALERTS[predicted_class])
            else:
                st.success("✅ Safe Traversable Terrain - Nominal Traction")

            st.markdown("#### 📊 Class Probabilities Breakdown")
            for i, name in enumerate(CLASS_NAMES):
                prob_pct = float(probs[i]) * 100
                st.write(f"**{name}**: {prob_pct:.1f}%")
                st.progress(float(probs[i]))

# ──────────────────────────────────────────────
# TAB 3: HISTORY & ANALYTICS
# ──────────────────────────────────────────────
with tab_analytics:
    st.subheader("📊 Session History & Analytics")

    log_path = "logs/terrain_log.csv"

    if os.path.exists(log_path):
        try:
            df = pd.read_csv(log_path)
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Predictions", len(df))
            
            avg_conf = round(pd.to_numeric(df["Confidence"], errors="coerce").mean(), 2)
            m_col2.metric("Average Confidence", f"{avg_conf}%" if not np.isnan(avg_conf) else "N/A")

            top_terrain = df["Terrain"].mode()[0] if not df["Terrain"].empty else "N/A"
            m_col3.metric("Dominant Terrain", top_terrain)

            st.markdown("### 📈 Terrain Distribution")
            st.bar_chart(df["Terrain"].value_counts())

            st.markdown("### 📋 Prediction Log (Latest 200 Entries)")
            st.dataframe(df.tail(200), use_container_width=True)

            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"terrain_logs_{session['id']}.csv",
                    mime="text/csv",
                )
            with btn_col2:
                if st.button("🗑️ Clear Log Records"):
                    try:
                        os.remove(log_path)
                        st.success("Log cleared successfully. Refreshing...")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing log: {e}")
        else:
            st.info("Log file is currently empty. Run the live camera or snapshot mode to generate data.")
    else:
        st.info("No prediction data logged yet. Capture a photo or start the live camera to record data.")

# ──────────────────────────────────────────────
# TAB 4: PROJECT OVERVIEW & DEPLOYMENT GUIDE
# ──────────────────────────────────────────────
with tab_overview:
    st.subheader("ℹ️ Project Overview & Deployment")

    st.markdown("""
    ### 🛰️ Geospatial Surface Recognition System
    Real-time terrain classification using deep learning (CNN) with live mobile streaming via WebRTC.

    ---
    ### 🚀 Deployment Ready on Streamlit Cloud (1-Click)
    This repository is fully configured for deployment on **Streamlit Community Cloud**:
    1. Push your repository to **GitHub**:
       ```bash
       git add .
       git commit -m "Deployment ready for Streamlit Cloud"
       git push origin main
       ```
    2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
    3. Click **'New App'**, choose this repository, and set `app.py` as the main file path.
    4. Click **Deploy!**
    
    *Because Streamlit Cloud serves your app via automatic **HTTPS**, scanning the QR code from any mobile phone browser (iOS Safari, Android Chrome) will immediately grant camera permissions without local Wi-Fi certificate restrictions!*

    ---
    ### 📱 Mobile Phone Connectivity Details
    - **Back Camera Priority**: Mobile camera stream automatically requests the rear environment camera (`facingMode: {"ideal": "environment"}`).
    - **WebRTC STUN / ICE Integration**: Configured with Google public STUN servers for reliable NAT/firewall traversal across mobile cellular and Wi-Fi networks.
    - **Dual Input Strategy**: Includes both continuous WebRTC live streaming and native mobile snapshot capture (`st.camera_input`).

    ---
    ### 🧠 Model Specifications
    - **Architecture**: 3-block Convolutional Neural Network (Conv2D + LeakyReLU + MaxPooling2D) + Dense 512 + Dropout (0.5)
    - **Input Dimensions**: 150 × 150 × 3 RGB
    - **Target Classes (4)**: Grassy, Marshy, Rocky, Sandy
    - **Validation Accuracy**: ~89% over 20 epochs
    - **Dataset**: [Kaggle Terrain Recognition](https://www.kaggle.com/datasets/atharv1610/terrain-recognition)

    ---
    **KIIT Deemed to be University** | B.Tech Computer Science | 2025–26  
    **Guide**: Prof. Dr. Ajit Kumar Pasayat
    """)
