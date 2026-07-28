"""
Streamlit dashboard for real-time ALPR — supports webcam and IP/RTSP camera streams.

Run with:
    streamlit run dashboard.py

Shows a live annotated feed alongside a running table of every detected plate
(timestamp, plate text, confidence), all persisted to a CSV log file on disk.
"""

import time

import cv2
import pandas as pd
import streamlit as st

from alpr_core import DEFAULT_LOG_PATH, PlateLogger, build_alpr, open_camera, run_alpr_on_frame

st.set_page_config(page_title="Real-Time ALPR Dashboard", layout="wide")
st.title("🚘 Real-Time License Plate Recognition")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    # ── Video source ──────────────────────────────────────────────────────
    st.subheader("📹 Video Source")
    source_mode = st.radio(
        "Source type",
        options=["Webcam", "IP / RTSP Camera"],
        index=0,
        horizontal=True,
    )

    if source_mode == "Webcam":
        camera_index = st.number_input(
            "Camera index", min_value=0, max_value=10, value=0, step=1,
            help="0 = default webcam, 1 = second camera, etc."
        )
        video_source = camera_index  # int
    else:
        stream_url = st.text_input(
            "Stream URL",
            value="",
            placeholder="rtsp://user:pass@192.168.1.10:554/stream  or  http://ip:port/video",
            help="Enter an RTSP or HTTP stream URL from your IP camera.",
        )
        video_source = stream_url.strip()  # str

    st.divider()

    # ── ALPR settings ─────────────────────────────────────────────────────
    st.subheader("⚙️ ALPR Settings")
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)
    use_perspective = st.checkbox("Perspective correction (steep angles)", value=True)
    use_vehicle_crop = st.checkbox(
        "Vehicle-aware crop (isolate the matching car, not the whole frame)", value=True
    )
    log_path = st.text_input("Log file", value=DEFAULT_LOG_PATH)

    st.divider()
    run = st.toggle("▶️ Start stream", value=False, key="run_camera")

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ALPR models (first run downloads ONNX weights)...")
def get_alpr(use_perspective_correction: bool):
    return build_alpr(use_perspective_correction=use_perspective_correction)


@st.cache_resource
def get_logger(path: str, use_vehicle_crop: bool):
    return PlateLogger(log_path=path, use_vehicle_crop=use_vehicle_crop)


alpr = get_alpr(use_perspective)
logger = get_logger(log_path, use_vehicle_crop)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
col_video, col_log = st.columns([2, 1])
with col_video:
    st.subheader("Live feed")
    frame_slot = st.empty()
    fps_slot = st.empty()
with col_log:
    st.subheader("Detected plates")
    table_slot = st.empty()

if "cap" not in st.session_state:
    st.session_state.cap = None


def render_log_table():
    rows = logger.read_all()
    if not rows:
        table_slot.info("No plates detected yet.")
        return
    df = pd.DataFrame(rows[::-1])  # most recent first
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    for col in ["timestamp", "plate_text", "vehicle_image_path", "crop_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    table_slot.dataframe(df, use_container_width=True, hide_index=True)

    captured = [r for r in rows if r.get("vehicle_image_path")]
    if captured:
        latest = captured[-1]
        crop_type = latest.get("crop_type", "")
        crop_note = (
            "isolated to matched vehicle"
            if crop_type == "vehicle"
            else "full frame — no vehicle matched"
            if crop_type == "full_frame"
            else ""
        )
        caption = f"Latest vehicle capture — plate {latest['plate_text']} (conf {latest['confidence']})"
        if crop_note:
            caption += f" · {crop_note}"
        st.caption(caption)
        st.image(latest["vehicle_image_path"], use_container_width=True)


# ---------------------------------------------------------------------------
# Stream validation helper
# ---------------------------------------------------------------------------
def _source_ready() -> bool:
    """Return True if the currently selected source is usable."""
    if source_mode == "IP / RTSP Camera":
        if not video_source:
            st.warning("⚠️ Enter a stream URL in the sidebar before starting.")
            return False
        if not (video_source.startswith("rtsp://") or video_source.startswith("http://") or video_source.startswith("https://")):
            st.warning("⚠️ URL should start with rtsp://, http://, or https://")
            return False
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if run:
    if not _source_ready():
        st.stop()

    if st.session_state.cap is None:
        try:
            st.session_state.cap = open_camera(video_source)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    cap = st.session_state.cap
    prev_time = time.time()
    fps = 0.0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to grab frame — check the camera/stream and try again.")
            break

        annotated_frame, detections = run_alpr_on_frame(alpr, frame, min_conf=min_conf)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        logged_any = False
        for d in detections:
            if logger.log_if_new(d.plate_text, d.confidence, frame=frame, plate_bbox=d.bounding_box):
                logged_any = True

        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(rgb_frame, channels="RGB", use_container_width=True)
        fps_slot.caption(f"FPS: {fps:.1f}")

        if logged_any:
            render_log_table()

        run = st.session_state.run_camera
        time.sleep(0.01)
else:
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    frame_slot.info("Stream stopped. Toggle '▶️ Start stream' in the sidebar to begin.")
    render_log_table()
