"""
Crowd Density Estimator — Streamlit Dashboard
Real-time crowd detection on webcam or RTSP streams with zone-based density analysis.
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from collections import deque
from detector import CrowdDetector

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crowd Density Estimator",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished dark dashboard look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Global ---- */
    .stApp { background: #f8f9fa; color: #1e1e1e; }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e5e5;
    }
    h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif; color: #111827 !important; }
    p, span, div { color: #374151; }

    /* ---- Metric cards ---- */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border-color: #3b82f6; }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #16a34a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
        font-weight: 600;
    }

    /* ---- Zone grid ---- */
    .zone-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #e5e5e5;
        transition: transform 0.15s;
    }
    .zone-card:hover { transform: scale(1.04); }
    .zone-low      { background: rgba(34,197,94,0.08);  color: #15803d; border-color: #22c55e40; }
    .zone-medium   { background: rgba(234,179,8,0.08); color: #a16207; border-color: #eab30840; }
    .zone-high     { background: rgba(249,115,22,0.08); color: #c2410c; border-color: #f9731640; }
    .zone-critical { background: rgba(239,68,68,0.08);  color: #b91c1c; border-color: #ef444440; }

    /* ---- Alert rows ---- */
    .alert-row {
        background: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #7f1d1d;
        border-top: 1px solid #fee2e2;
        border-right: 1px solid #fee2e2;
        border-bottom: 1px solid #fee2e2;
    }

    /* ---- Status pill ---- */
    .status-live {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        animation: pulse-live 1.5s infinite;
    }
    @keyframes pulse-live {
        0%, 100% { background: rgba(63,185,80,0.25); color: #3fb950; }
        50%      { background: rgba(63,185,80,0.45); color: #56d364; }
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "streaming": False,
        "detector": None,
        "cap": None,
        "history": deque(maxlen=300),
        "alerts": deque(maxlen=50),
        "peak_count": 0,
        "frame_count": 0,
        "fps": 0.0,
        "last_zone_data": {"total": 0, "zones": []},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 👥 Crowd Density Estimator")
    st.markdown("---")

    # Source selection
    st.markdown("### 📹 Video Source")
    source_type = st.radio("Source", ["Webcam", "RTSP Stream"], horizontal=True, label_visibility="collapsed")

    if source_type == "Webcam":
        cam_index = st.selectbox("Camera Index", [0, 1, 2], index=0)
        source = cam_index
    else:
        source = st.text_input("RTSP URL", placeholder="rtsp://username:password@ip:port/stream")

    st.markdown("---")

    # Detection settings
    st.markdown("### ⚙️ Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    col_r, col_c = st.columns(2)
    with col_r:
        grid_rows = st.number_input("Grid Rows", 1, 8, 3)
    with col_c:
        grid_cols = st.number_input("Grid Cols", 1, 8, 3)

    st.markdown("---")

    # Zone thresholds
    st.markdown("### 🚦 Zone Thresholds")
    thresh_low = st.slider("Low ≤", 1, 20, 3)
    thresh_med = st.slider("Medium ≤", 1, 30, 6)
    thresh_high = st.slider("High ≤", 1, 50, 10)

    st.markdown("---")

    # Controls
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    with col_stop:
        stop_btn = st.button("⏹ Stop", use_container_width=True)


# ---------------------------------------------------------------------------
# Start / Stop logic
# ---------------------------------------------------------------------------
def _start_stream(source, confidence, grid_rows, grid_cols, thresh_low, thresh_med, thresh_high):
    """Open video capture and initialize detector."""
    # Init detector
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    det = CrowdDetector(config_path)
    det.update_confidence(confidence)
    det.update_grid(grid_rows, grid_cols)
    det.update_thresholds(thresh_low, thresh_med, thresh_high)

    # Open capture
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error(f"❌ Cannot open source: {source}")
        return False

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    st.session_state.detector = det
    st.session_state.cap = cap
    st.session_state.streaming = True
    st.session_state.history = deque(maxlen=300)
    st.session_state.alerts = deque(maxlen=50)
    st.session_state.peak_count = 0
    st.session_state.frame_count = 0
    return True


def _stop_stream():
    """Release capture and reset state."""
    st.session_state.streaming = False
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None


if start_btn:
    if not source and source_type == "RTSP Stream":
        st.sidebar.error("Please enter an RTSP URL.")
    else:
        _start_stream(source, confidence, grid_rows, grid_cols, thresh_low, thresh_med, thresh_high)

if stop_btn:
    _stop_stream()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.markdown("# 👥 Crowd Density Monitor")

# Status bar
if st.session_state.streaming:
    src_label = f"Webcam {source}" if isinstance(source, int) else source
    st.markdown(f'<span class="status-live">● LIVE</span> &nbsp; Monitoring: **{src_label}**', unsafe_allow_html=True)
else:
    st.info("Select a video source and click **▶ Start** in the sidebar to begin monitoring.")

# Metrics placeholder
metrics_placeholder = st.empty()


# Video + zones layout
video_col, data_col = st.columns([3, 1])

with video_col:
    video_placeholder = st.empty()

with data_col:
    st.markdown("#### 🗺️ Zone Density")
    zone_placeholder = st.empty()
    st.markdown("#### 🚨 Recent Alerts")
    alert_placeholder = st.empty()

# History chart and log placeholder
history_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------
if st.session_state.streaming and st.session_state.cap is not None:
    det = st.session_state.detector
    cap = st.session_state.cap

    # Apply live settings
    det.update_confidence(confidence)
    det.update_grid(grid_rows, grid_cols)
    det.update_thresholds(thresh_low, thresh_med, thresh_high)

    frame_times = deque(maxlen=30)

    while st.session_state.streaming:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            # For RTSP, retry
            time.sleep(0.1)
            ret, frame = cap.read()
            if not ret or frame is None:
                video_placeholder.warning("⚠️ Lost video feed. Retrying...")
                time.sleep(1)
                continue

        # Run detection
        annotated, zone_data = det.detect(frame)
        st.session_state.last_zone_data = zone_data
        st.session_state.frame_count += 1

        total = zone_data["total"]
        if total > st.session_state.peak_count:
            st.session_state.peak_count = total

        # History
        st.session_state.history.append({"time": datetime.now().strftime("%H:%M:%S"), "count": total})

        # Collect alerts
        for z in zone_data["zones"]:
            if z["level"] == "Critical":
                st.session_state.alerts.appendleft(
                    f'🔴 {datetime.now().strftime("%H:%M:%S")} — **{z["id"]}** critical ({z["count"]} people)'
                )

        # FPS
        frame_times.append(time.time() - t0)
        if len(frame_times) > 1:
            st.session_state.fps = 1.0 / (sum(frame_times) / len(frame_times))

        # --- Render video ---
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB", use_container_width=True)

        # --- Render zone grid ---
        zone_html = ""
        cols_per_row = det.grid_cols
        zones = zone_data["zones"]
        for i, z in enumerate(zones):
            css_class = f"zone-{z['level'].lower()}"
            zone_html += f'<div class="zone-card {css_class}">{z["id"]}<br>{z["count"]} • {z["level"]}</div>'
            if (i + 1) % cols_per_row == 0:
                zone_html += "<div style='clear:both'></div>"

        grid_css = f"""
        <div style="display:grid; grid-template-columns: repeat({cols_per_row}, 1fr); gap:6px;">
            {zone_html}
        </div>
        """
        zone_placeholder.markdown(grid_css, unsafe_allow_html=True)

        # --- Render alerts ---
        alerts_list = list(st.session_state.alerts)[:10]
        if alerts_list:
            alert_html = "".join(f'<div class="alert-row">{a}</div>' for a in alerts_list)
            alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
        else:
            alert_placeholder.markdown("_No alerts yet._")

        # --- Render metrics ---
        active_alerts = sum(1 for z in zone_data["zones"] if z["level"] == "Critical")
        metrics_html = f"""
        <div style="display:flex; justify-content:space-between; gap:10px; margin-bottom: 20px;">
            <div class="metric-card" style="flex:1;"><div class="metric-value">{{total}}</div><div class="metric-label">People Detected</div></div>
            <div class="metric-card" style="flex:1;"><div class="metric-value">{{st.session_state.peak_count}}</div><div class="metric-label">Peak Count</div></div>
            <div class="metric-card" style="flex:1;"><div class="metric-value">{{active_alerts}}</div><div class="metric-label">Active Alerts</div></div>
            <div class="metric-card" style="flex:1;"><div class="metric-value">{{st.session_state.fps:.1f}}</div><div class="metric-label">FPS</div></div>
        </div>
        """
        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)

        # --- Render history chart & log ---
        if len(st.session_state.history) > 5:
            import pandas as pd
            df = pd.DataFrame(list(st.session_state.history))
            
            with history_placeholder.container():
                hc1, hc2 = st.columns([4, 1])
                with hc1:
                    st.line_chart(df.set_index("time")["count"], height=180)
                with hc2:
                    st.markdown("##### Recent Counts")
                    recent_hist = list(st.session_state.history)[-8:]
                    recent_hist.reverse()
                    log_text = "<br>".join([f"`{item['time']}` &nbsp; **{item['count']}**" for item in recent_hist])
                    st.markdown(f"<div style='font-size:0.9em; height:180px; overflow-y:auto;'>{log_text}</div>", unsafe_allow_html=True)

        # Small sleep to avoid pegging CPU and allow Streamlit rerun checks
        time.sleep(0.03)
