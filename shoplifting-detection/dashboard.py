#!/usr/bin/env python3
"""
dashboard.py — Streamlit dashboard for Shoplifting detection.

Run:
    streamlit run dashboard.py
"""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import streamlit as st

from src.shoplifting_detector import ShopliftingDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Argus — Shoplifting Detection",
    page_icon="🛒",
    layout="wide",
)

MAX_LOG = 300


@st.cache_resource
def get_shared_state() -> dict:
    return {
        "lock":       threading.Lock(),
        "thread":     None,
        "running":    False,
        "frame_rgb":  None,
        "fps":        0.0,
        "log":        deque(maxlen=MAX_LOG),
        "total_count": 0,
        "status":     "SAFE",
        "error":      None,
    }


def _worker(video_source: int | str, conf: float, state: dict) -> None:
    lock = state["lock"]

    try:
        detector = ShopliftingDetector(conf=conf)
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

    fps_counter, fps_start, fps_val = 0, time.time(), 0.0

    while True:
        with lock:
            if not state["running"]:
                break

        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        annotated, events = detector.process_frame(raw)

        fps_counter += 1
        if fps_counter >= 8:
            fps_val     = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with lock:
            state["frame_rgb"]  = frame_rgb
            state["fps"]        = fps_val
            
            if events:
                state["status"] = "SHOPLIFTING DETECTED"
                for e in events:
                    state["log"].appendleft(e)
                    state["total_count"] += 1
            else:
                state["status"] = "SAFE"

    cap.release()
    with lock:
        state["running"] = False


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

    with st.sidebar:
        st.markdown("## 🛒 Shoplifting Monitor")
        st.caption("Argus — Retail Security")
        st.divider()

        source_mode = st.radio("Source", ["Webcam", "IP / RTSP Camera"])
        if source_mode == "Webcam":
            cam_idx = st.number_input("Camera Index", 0, 10, 0)
            video_source = int(cam_idx)
        else:
            video_source = st.text_input("Stream URL", value="rtsp://",
                                         help="RTSP or HTTP stream URL")

        conf_val = st.slider("Confidence", 0.10, 0.90, 0.50, 0.05)
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
                        "log": deque(maxlen=MAX_LOG),
                        "total_count": 0,
                        "status": "SAFE",
                        "error": None,
                    })
                t = threading.Thread(
                    target=_worker,
                    args=(video_source, float(conf_val), state),
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

    # ── Read state ────────────────────────────────────────────────────────
    with lock:
        frame_rgb  = state["frame_rgb"]
        fps        = state["fps"]
        log_items  = list(state["log"])
        total_cnt  = state["total_count"]
        status     = state["status"]
        running    = state["running"]

    st.markdown("## 🛒 Shoplifting & Retail Monitor")

    cols = st.columns(3)
    metrics = [
        ("FPS",           f"{fps:.1f}",  "#00E6FF"),
        ("Status",        status,        "#FF3333" if status == "SHOPLIFTING DETECTED" else "#32FF32"),
        ("Total Incidents", total_cnt,   "#FF3333"),
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
        st.markdown("#### 📋 Incident Log")
        html = ""
        for entry in log_items[:50]:
            fg, bg = "#FF3333", "#300"
            html += (
                f"<div class='log-row'>"
                f"<span style='color:#555'>{entry['time']}</span>  "
                f"<span class='badge' style='background:{bg};color:{fg};"
                f"border:1px solid {fg}'>ALERT</span>  "
                f"<span style='color:#ccc'>{entry['type']} ({entry['conf']:.2f})</span>  "
                f"</div>"
            )
        st.markdown(html or "<div style='color:#444;margin-top:20px'>No incidents detected…</div>",
                    unsafe_allow_html=True)

    if running:
        time.sleep(0.15)
        st.rerun()


if __name__ == "__main__":
    main()
