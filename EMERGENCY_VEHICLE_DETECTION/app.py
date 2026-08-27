import cv2
import time
import pandas as pd
import streamlit as st
import numpy as np
from ultralytics import YOLOWorld
from logger import init_logger, log_detection

st.set_page_config(page_title="Smart City Traffic System", page_icon="🚦", layout="wide")

st.markdown("""
<style>
/* Glassmorphism sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 25, 0.85);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
/* Sidebar text visibility */
[data-testid="stSidebar"], [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}
/* Main background */
.stApp {
    background-color: #0B0F19;
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}
/* Cards */
.stMetric {
    background: linear-gradient(145deg, #1A2035, #111526);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #2A3245;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
/* Titles */
h1 {
    color: #00FFAA;
    text-shadow: 0 0 10px rgba(0, 255, 170, 0.3);
}
h3 {
    color: #94A3B8;
}
/* Dataframe styling */
[data-testid="stDataFrame"] {
    background-color: #111526;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #2A3245;
}
/* Traffic Signal */
.signal-box {
    display: inline-flex;
    background: #111526;
    padding: 10px;
    border-radius: 20px;
    border: 2px solid #2A3245;
    gap: 15px;
}
.light {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background-color: #222;
    transition: all 0.3s ease;
}
.red-on { background-color: #FF3B30; box-shadow: 0 0 20px #FF3B30; }
.green-on { background-color: #34C759; box-shadow: 0 0 20px #34C759; }
</style>
""", unsafe_allow_html=True)

st.title("🚦 Smart City Emergency Vehicle Detection")
st.markdown("### Next-Gen Zero-Shot Object Detection (YOLO-World)")

init_logger()

# Sidebar for configuration
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=60)
st.sidebar.title("System Controls")
stream_source = st.sidebar.radio("Video Source", ["Webcam", "RTSP / IP Camera"])

rtsp_url = ""
if stream_source == "RTSP / IP Camera":
    rtsp_url = st.sidebar.text_input("Enter RTSP URL", "rtsp://...")

conf_threshold = st.sidebar.slider("Detection Confidence", 0.05, 1.0, 0.15, 0.05)
st.sidebar.markdown("---")
start_button = st.sidebar.button("🚀 INITIATE SYSTEM", use_container_width=True, type="primary")
stop_button = st.sidebar.button("⏹️ SHUTDOWN", use_container_width=True)

# Load YOLO-World model
@st.cache_resource
def load_model():
    # yolov8s-world.pt or yolov8s-worldv2.pt will be downloaded automatically
    model = YOLOWorld('yolov8s-worldv2.pt')
    # Set custom classes - YOLO-World can detect anything you prompt it with!
    # Using more descriptive prompts helps the zero-shot model understand what we are looking for.
    model.set_classes(["ambulance", "police car", "fire engine", "car", "van", "motorcycle"])
    return model

model = load_model()

# Custom color mapping for visual aesthetics
COLORS = {
    "ambulance": (0, 0, 255),       # Red
    "fire engine": (0, 0, 255),     # Red
    "police car": (255, 0, 0),      # Blue
    "car": (0, 255, 128),           # Green
    "motorcycle": (255, 255, 0)     # Cyan
}

# Layout
col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown("#### 📺 Live Surveillance Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("#### 📊 Real-Time Analytics")
    signal_placeholder = st.empty()
    fps_metric, emergency_metric = st.columns(2)
    with fps_metric:
        fps_display = st.empty()
    with emergency_metric:
        emerg_display = st.empty()
        
    st.markdown("#### 📜 Recent Detections")
    log_placeholder = st.empty()

def update_signal(emergency_detected):
    if emergency_detected:
        html = """
        <div class="signal-box">
            <div class="light"></div>
            <div class="light green-on"></div>
        </div>
        <span style='margin-left: 15px; color:#34C759; font-weight:bold;'>PRIORITY OVERRIDE</span>
        """
    else:
        html = """
        <div class="signal-box">
            <div class="light red-on"></div>
            <div class="light"></div>
        </div>
        <span style='margin-left: 15px; color:#FF3B30; font-weight:bold;'>NORMAL TRAFFIC</span>
        """
    signal_placeholder.markdown(html, unsafe_allow_html=True)

update_signal(False)
fps_display.metric("FPS", "0.0")
emerg_display.metric("Emergencies", "0")

if start_button and not stop_button:
    source = 0 if stream_source == "Webcam" else rtsp_url
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        st.sidebar.error("Error: Could not open video stream.")
    else:
        st.sidebar.success("✅ System Active")
        
        frame_count = 0
        start_time = time.time()
        emergency_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (800, 600))
            
            # YOLO-World Inference
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            emergency_in_frame = False
            
            for box in result.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Model returns classes in the order we set them
                if cls_idx < len(model.names):
                    class_name = model.names[cls_idx]
                else:
                    continue
                    
                # Treat 'van' as ambulance for this specific context since Indian ambulances look like vans
                display_name = "ambulance" if class_name == "van" else class_name
                    
                is_emergency = display_name in ["ambulance", "fire engine", "police car"]
                if is_emergency:
                    emergency_in_frame = True
                    emergency_count += 1
                
                # Aesthetics
                color = COLORS.get(display_name, (255, 255, 255))
                thickness = 3 if is_emergency else 2
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Neon glow effect for emergency vehicles
                if is_emergency:
                    # Draw a semi-transparent box for "glow"
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                label = f"{display_name.upper()} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                log_detection(display_name, conf)
            
            # Update Dashboard state
            update_signal(emergency_in_frame)
            
            # Update metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            fps_display.metric("FPS", f"{fps:.1f}")
            emerg_display.metric("Emergencies", emergency_count)
            
            # Logs
            if frame_count % 5 == 0:  # Update log display every 5 frames to save resources
                try:
                    logs_df = pd.read_csv("data_log.csv").tail(8)
                    log_placeholder.dataframe(logs_df, use_container_width=True, hide_index=True)
                except:
                    pass
            
            # Render video
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            if stop_button:
                break
                
        cap.release()
        update_signal(False)
        st.sidebar.warning("System Shutdown.")
