#!/usr/bin/env python3
"""
speed_dashboard.py — Streamlit dashboard for vehicle speed estimation.

Run:
    streamlit run speed_dashboard.py

Supports: webcam (index 0-10), RTSP streams, HTTP streams.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from src.speed_estimator import SpeedEstimator, DEFAULT_ROI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Argus — Vehicle Speed Estimation",
    page_icon="🏎️",
    layout="wide",
)

MAX_LOG = 300


# ── Singleton state ───────────────────────────────────────────────────────────
@st.cache_resource
def get_shared_state() -> dict:
    return {
        "lock":      threading.Lock(),
        "thread":    None,
        "running":   False,
        "frame_rgb": None,
        "fps":       0.0,
        "vehicles":  0,
        "records":   [],           # latest frame's speed records
        "log":       deque(maxlen=MAX_LOG),
        "peak_speed": 0.0,
        "total_vehicles_seen": 0,
        "error":     None,
    }


# ── Inference worker ──────────────────────────────────────────────────────────
def _worker(
    video_source: int | str,
    conf: float,
    roi_points: np.ndarray,
    target_w: float,
    target_h: float,
    state: dict,
) -> None:
    lock = state["lock"]

    try:
        estimator = SpeedEstimator(
            model_path="models/yolo11n.pt",
            roi=roi_points,
            target_width=target_w,
            target_height=target_h,
            conf=conf,
        )
    except Exception as exc:
        with lock:
            state["error"]   = str(exc)
            state["running"] = False
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        with lock:
            state["error"]   = f"Cannot open source: {video_source}"
            state["running"] = False
        return

    # Detect actual FPS from the source
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps and source_fps > 0:
        estimator.update_fps(source_fps)
    else:
        estimator.update_fps(30.0)

    fps_counter, fps_start, fps_val = 0, time.time(), 0.0

    while True:
        with lock:
            if not state["running"]:
                break

        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        annotated, records = estimator.process_frame(raw)

        fps_counter += 1
        if fps_counter >= 8:
            fps_val     = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with lock:
            state["frame_rgb"] = frame_rgb
            state["fps"]       = fps_val
            state["vehicles"]  = len(records)
            state["records"]   = records

            ts = datetime.now().strftime("%H:%M:%S")
            for r in records:
                if r["speed_kmh"] > 5:  # only log moving vehicles
                    state["log"].appendleft({
                        "time":      ts,
                        "id":        r["tracker_id"],
                        "class":     r["class_name"],
                        "speed":     r["speed_kmh"],
                    })
                    state["total_vehicles_seen"] += 1
                if r["speed_kmh"] > state["peak_speed"]:
                    state["peak_speed"] = r["speed_kmh"]

    cap.release()
    with lock:
        state["running"] = False


# ── UI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{background:#0d0f14;color:#e8eaf0}
    [data-testid="stSidebar"]{background:#13161e}
    .card{background:#1a1e2a;border:1px solid #2a2f3e;border-radius:12px;
          padding:14px 20px;text-align:center;margin-bottom:8px}
    .val{font-size:2rem;font-weight:700}
    .lbl{font-size:.78rem;color:#8891a8}
    .log-row{font-family:monospace;font-size:.82rem;background:#1a1e2a;
             border-radius:6px;padding:5px 12px;margin-bottom:3px}
    .badge{border-radius:4px;padding:2px 8px;font-size:.72rem;font-weight:700}
    .stButton>button{background:#2563eb;color:#fff;border:none;
                     border-radius:8px;padding:8px 22px;font-weight:600}
    </style>
    """, unsafe_allow_html=True)

    state = get_shared_state()
    lock  = state["lock"]

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏎️ Speed Estimation")
        st.caption("Argus — Vehicle Speed Monitor")
        st.divider()

        source_mode = st.radio("Source", ["Webcam", "IP / RTSP Camera"])
        if source_mode == "Webcam":
            cam_idx = st.number_input("Camera Index", 0, 10, 0)
            video_source = int(cam_idx)
        else:
            video_source = st.text_input(
                "Stream URL", value="rtsp://",
                help="RTSP or HTTP stream URL",
            )

        conf_val = st.slider("Confidence", 0.10, 0.80, 0.40, 0.05)
        st.divider()

        st.markdown("**Road Calibration (metres)**")
        target_w = st.number_input("Road Width", 1.0, 100.0, 12.0, 1.0)
        target_h = st.number_input("Measurement Distance", 5.0, 500.0, 50.0, 5.0)
        st.divider()

        c1, c2 = st.columns(2)
        start_clicked = c1.button("▶ Start", width="stretch")
        stop_clicked  = c2.button("⏹ Stop",  width="stretch")

        if start_clicked:
            with lock:
                already = state["running"]
                alive   = state["thread"] is not None and state["thread"].is_alive()
            if not already and not alive:
                with lock:
                    state.update({
                        "running": True, "frame_rgb": None, "fps": 0.0,
                        "vehicles": 0, "records": [],
                        "log": deque(maxlen=MAX_LOG),
                        "peak_speed": 0.0, "total_vehicles_seen": 0,
                        "error": None,
                    })
                t = threading.Thread(
                    target=_worker,
                    args=(video_source, float(conf_val),
                          DEFAULT_ROI.copy(), float(target_w), float(target_h),
                          state),
                    daemon=True,
                )
                t.start()
                with lock:
                    state["thread"] = t

        if stop_clicked:
            with lock:
                state["running"] = False

        with lock:
            running = state["running"]
            err     = state["error"]
            alive   = state["thread"] is not None and state["thread"].is_alive()

        if err:
            st.error(f"⛔ {err}")
        elif running and alive:
            st.success("🟢 Live")
        elif not running and alive:
            st.warning("🟡 Stopping…")
        else:
            st.warning("🔴 Stopped")

        st.divider()
        st.markdown("**Legend**")
        st.markdown("""
- 🟡 Cyan traces — vehicle paths
- 📊 Speed labels — km/h per vehicle
- 🔷 ROI polygon — measurement zone
        """)

    # ── Read state ────────────────────────────────────────────────────────
    with lock:
        frame_rgb  = state["frame_rgb"]
        fps        = state["fps"]
        vehicles   = state["vehicles"]
        records    = list(state["records"])
        log_items  = list(state["log"])
        peak       = state["peak_speed"]
        total_seen = state["total_vehicles_seen"]
        running    = state["running"]

    # ── Main content ──────────────────────────────────────────────────────
    st.markdown("## 🏎️ Live Speed Monitor")

    # Metrics row
    cols = st.columns(5)
    metrics = [
        ("FPS",              f"{fps:.1f}",         "#00E6FF"),
        ("Vehicles Now",     vehicles,              "#00E6FF"),
        ("Peak Speed",       f"{peak:.0f} km/h",   "#FF3333" if peak > 80 else "#32FF32"),
        ("Total Tracked",    total_seen,            "#FF8C00"),
        ("Avg Speed",
         f"{sum(r['speed_kmh'] for r in records) / max(len(records), 1):.0f} km/h" if records else "—",
         "#c8c8c8"),
    ]
    for col, (lbl, val, color) in zip(cols, metrics):
        col.markdown(
            f"<div class='card'><div class='val' style='color:{color}'>{val}</div>"
            f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    vid_col, log_col = st.columns([3, 2])

    with vid_col:
        ph = st.empty()
        if frame_rgb is not None:
            ph.image(frame_rgb, channels="RGB", width="stretch")
        else:
            ph.markdown(
                "<div style='height:420px;background:#1a1e2a;border-radius:12px;"
                "display:flex;align-items:center;justify-content:center;"
                "color:#4a5568;font-size:1.1rem'>▶ Press Start in the sidebar</div>",
                unsafe_allow_html=True)

    with log_col:
        st.markdown("#### 📋 Speed Log")

        # Current frame speed table
        if records:
            html = ""
            for r in sorted(records, key=lambda x: -x["speed_kmh"]):
                spd = r["speed_kmh"]
                color = "#FF3333" if spd > 80 else "#FF8C00" if spd > 50 else "#32FF32"
                html += (
                    f"<div class='log-row'>"
                    f"<span class='badge' style='background:#1a1e2a;color:#00E6FF;"
                    f"border:1px solid #00E6FF'>#{r['tracker_id']}</span>  "
                    f"<span style='color:#aaa'>{r['class_name']}</span>  "
                    f"<span style='color:{color};font-weight:700;font-size:1rem'>"
                    f"{spd:.0f} km/h</span>"
                    f"</div>"
                )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#444;margin-top:20px'>No vehicles in ROI…</div>",
                        unsafe_allow_html=True)

        # Historical log
        st.markdown("#### 📜 History")
        hist_html = ""
        for entry in log_items[:40]:
            spd   = entry["speed"]
            color = "#FF3333" if spd > 80 else "#FF8C00" if spd > 50 else "#32FF32"
            hist_html += (
                f"<div class='log-row'>"
                f"<span style='color:#555'>{entry['time']}</span>  "
                f"<span style='color:#aaa'>#{entry['id']} {entry['class']}</span>  "
                f"<span style='color:{color};font-weight:600'>{spd:.0f} km/h</span>"
                f"</div>"
            )
        st.markdown(hist_html or "<div style='color:#444'>No records yet…</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Session Summary")
    t1, t2, t3 = st.columns(3)
    t1.metric("🚗 Total Vehicles", total_seen)
    t2.metric("🏎️ Peak Speed",    f"{peak:.0f} km/h")
    t3.metric("📐 ROI Size",      f"{12}m × {50}m")

    if running:
        time.sleep(0.15)
        st.rerun()


if __name__ == "__main__":
    main()
