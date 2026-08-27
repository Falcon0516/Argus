#!/usr/bin/env python3
"""
dashboard.py — Argus Streamlit dashboard.
Run: streamlit run dashboard.py

Fix: uses @st.cache_resource to create a singleton state dict that
survives Streamlit reruns (module-level dicts are reset on every rerun).
"""
from __future__ import annotations
import threading
import time
from collections import deque

import cv2
import streamlit as st

from src.detector          import ArgusDetector, PERSON_CLASS, MOTORCYCLE_CLASS
from src.association       import associate_riders, box_to_xyxy, upper_body_crop, crop_box
from src.violation_engine  import ViolationEngine, classify_helmet_result
from src.visualizer        import draw_frame, draw_hud

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Argus — Traffic Violation Detection",
    page_icon="🚦", layout="wide",
)

INFER_SIZE = 640
MAX_LOG    = 300


# ── Singleton shared state (survives Streamlit reruns) ────────────────────────
@st.cache_resource
def get_shared_state() -> dict:
    """Created once per Streamlit server process; never re-initialized on rerun."""
    return {
        "lock":      threading.Lock(),
        "thread":    None,
        "running":   False,
        "frame_rgb": None,
        "fps":       0.0,
        "counts":    {"moto": 0, "riders": 0},
        "violations": [],
        "log":       deque(maxlen=MAX_LOG),
        "totals":    {"triple": 0, "no_helmet": 0, "plates": 0},
        "error":     None,
    }


# ── Inference worker ──────────────────────────────────────────────────────────
def _inference_worker(video_source: int | str, conf: float, state: dict) -> None:
    lock = state["lock"]

    try:
        detector = ArgusDetector(conf=conf)
        engine   = ViolationEngine()
    except Exception as exc:
        with lock:
            state["error"]   = str(exc)
            state["running"] = False
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        with lock:
            state["error"]   = f"Cannot open camera {video_source}"
            state["running"] = False
        return

    fps_counter, fps_start, fps_val = 0, time.time(), 0.0

    while True:
        # Check stop signal
        with lock:
            if not state["running"]:
                break

        ret, raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.resize(raw, (INFER_SIZE, INFER_SIZE))

        # Stage 1 — base detection
        base_result  = detector.detect_base(frame)
        person_boxes: list[tuple] = []
        moto_boxes:   list[tuple] = []
        if base_result.boxes is not None:
            for box in base_result.boxes:
                cls = int(box.cls[0])
                b   = box_to_xyxy(box)
                if cls == PERSON_CLASS:
                    person_boxes.append(b)
                elif cls == MOTORCYCLE_CLASS:
                    moto_boxes.append(b)

        # Stage 2 — associate riders to motorcycles
        groups = associate_riders(person_boxes, moto_boxes)

        # Stage 3a — helmet check (only for riders)
        helmet_statuses: dict[int, dict] = {}
        all_rider_idxs = {p for riders in groups.values() for p in riders}
        for p_idx in all_rider_idxs:
            crop     = upper_body_crop(frame, person_boxes[p_idx])
            h_result = detector.detect_helmet(crop)
            helmet_statuses[p_idx] = classify_helmet_result(h_result)

        # Stage 3b — plate detection (only within motorcycle crop)
        plate_detections: dict[int, list] = {}
        plate_texts:      dict[int, str]  = {}
        for m_idx, box in enumerate(moto_boxes):
            mc_crop  = crop_box(frame, box, pad_frac=0.05)
            p_result = detector.detect_plate(mc_crop)
            plates   = []
            if p_result and p_result.boxes is not None:
                for pb in p_result.boxes:
                    cls_name = p_result.names.get(int(pb.cls[0]), "plate")
                    plates.append({"label": cls_name, "conf": float(pb.conf[0])})
            plate_detections[m_idx] = plates
            if plates:
                plate_texts[m_idx] = f"{plates[0]['label']} {plates[0]['conf']:.2f}"

        # Stage 4 — evaluate violations
        violations = engine.evaluate(groups, helmet_statuses, plate_detections)

        # Draw
        draw_frame(frame, person_boxes, moto_boxes, groups,
                   helmet_statuses, violations, plate_texts)

        fps_counter += 1
        if fps_counter >= 8:
            fps_val     = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        counts = {"moto": len(moto_boxes), "riders": len(all_rider_idxs)}
        draw_hud(frame, fps_val, counts, len(violations))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write to shared state
        with lock:
            state["frame_rgb"]  = frame_rgb
            state["fps"]        = fps_val
            state["counts"]     = counts
            state["violations"] = violations
            for v in violations:
                state["log"].appendleft({
                    "time":  v.timestamp,
                    "type":  v.violation_type,
                    "label": v.log_line(),
                })
                if v.violation_type == "TRIPLE_RIDING":
                    state["totals"]["triple"]    += 1
                elif v.violation_type == "NO_HELMET":
                    state["totals"]["no_helmet"] += 1
                elif v.violation_type == "PLATE":
                    state["totals"]["plates"]    += 1

    cap.release()
    with lock:
        state["running"] = False


# ── Streamlit UI ──────────────────────────────────────────────────────────────
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
        st.markdown("## 🚦 Argus")
        st.caption("Smart Traffic Violation Detection")
        st.divider()
        source_mode = st.radio("Source", ["Webcam", "IP / RTSP Camera"])
        if source_mode == "Webcam":
            cam_idx = st.number_input("Camera Index", 0, 10, 0)
            video_source = int(cam_idx)
        else:
            video_source = st.text_input("Stream URL", value="rtsp://", help="Enter an RTSP or HTTP stream URL from your IP camera.")
            
        conf_val = st.slider("Confidence", 0.10, 0.80, 0.28, 0.02)
        st.divider()

        c1, c2 = st.columns(2)
        start_clicked = c1.button("▶ Start", width="stretch")
        stop_clicked  = c2.button("⏹ Stop",  width="stretch")

        # Start
        if start_clicked:
            with lock:
                already_running = state["running"]
                thread_alive    = state["thread"] is not None and state["thread"].is_alive()
            if not already_running and not thread_alive:
                with lock:
                    # Reset state for new session
                    state["running"]    = True
                    state["frame_rgb"]  = None
                    state["fps"]        = 0.0
                    state["counts"]     = {"moto": 0, "riders": 0}
                    state["violations"] = []
                    state["log"]        = deque(maxlen=MAX_LOG)
                    state["totals"]     = {"triple": 0, "no_helmet": 0, "plates": 0}
                    state["error"]      = None
                t = threading.Thread(
                    target=_inference_worker,
                    args=(video_source, float(conf_val), state),
                    daemon=True,
                )
                t.start()
                with lock:
                    state["thread"] = t

        # Stop
        if stop_clicked:
            with lock:
                state["running"] = False

        # Status indicator
        with lock:
            running = state["running"]
            err     = state["error"]
            talive  = state["thread"] is not None and state["thread"].is_alive()

        if err:
            st.error(f"⛔ {err}")
        elif running and talive:
            st.success("🟢 Live")
        elif not running and talive:
            st.warning("🟡 Stopping…")
        else:
            st.warning("🔴 Stopped")

        st.divider()
        st.markdown("**Legend**")
        st.markdown("""
- 🔵 Cyan — Motorcycle
- 🟢 Green — Compliant rider  
- 🔴 Red — Violation (no helmet)
- 🟡 Gold — License plate
- 🚨 Banner — Triple riding / No helmet
        """)

    # ── Read shared state ─────────────────────────────────────────────────
    with lock:
        frame_rgb  = state["frame_rgb"]
        fps        = state["fps"]
        counts     = dict(state["counts"])
        violations = list(state["violations"])
        log_items  = list(state["log"])
        totals     = dict(state["totals"])
        running    = state["running"]

    # ── Main content ──────────────────────────────────────────────────────
    st.markdown("## 🎥 Live Feed")

    cols = st.columns(5)
    metrics = [
        ("FPS",             f"{fps:.1f}",                "#00E6FF"),
        ("Motorcycles",     counts.get("moto", 0),       "#00E6FF"),
        ("Riders",          counts.get("riders", 0),     "#c8c8c8"),
        ("Active Violations", len(violations),             "#FF3333" if violations else "#32FF32"),
        ("Session Alerts",  totals["triple"] + totals["no_helmet"], "#FF8C00"),
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
        st.markdown("#### 📋 Detection Log")
        TYPE_STYLE = {
            "TRIPLE_RIDING": ("#FF3333", "#300"),
            "NO_HELMET":     ("#FF8C00", "#200"),
            "PLATE":         ("#FFD700", "#210"),
        }
        html = ""
        for entry in log_items[:60]:
            fg, bg = TYPE_STYLE.get(entry["type"], ("#aaa", "#111"))
            html += (
                f"<div class='log-row'>"
                f"<span style='color:#555'>{entry['time']}</span>  "
                f"<span class='badge' style='background:{bg};color:{fg};"
                f"border:1px solid {fg}'>{entry['type']}</span>  "
                f"<span style='color:#ccc'>{entry['label'][:55]}</span>"
                f"</div>"
            )
        st.markdown(html or "<div style='color:#444;margin-top:20px'>No violations yet…</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Session Summary")
    t1, t2, t3 = st.columns(3)
    t1.metric("🚨 Triple Riding",    totals["triple"])
    t2.metric("⛑ No Helmet",         totals["no_helmet"])
    t3.metric("🔢 Plates Detected",  totals["plates"])

    # Auto-refresh while inference is live
    if running:
        time.sleep(0.12)
        st.rerun()


if __name__ == "__main__":
    main()
