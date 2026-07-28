"""
Streamlit dashboard: build a watchlist of named people (each from their own reference
photo + a manually selected box), then find any of them live in your webcam or IP/RTSP
camera feed using whole-body appearance — a proper person re-ID embedding (OSNet, via
BoxMOT) blended with a clothing-color histogram and body aspect ratio. CPU-only.

Run:
    streamlit run app.py
"""

import colorsys
import threading
import time
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Model loading (cached across reruns)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_detector() -> YOLO:
    # Nano model: smallest/fastest YOLO variant, good for CPU. Auto-downloads on first run.
    return YOLO("yolov8n.pt")


@st.cache_resource(show_spinner="Downloading OSNet ReID weights (first run only)...")
def load_reid_backend():
    """
    Proper person re-identification embedder via BoxMOT.
    osnet_x0_25_msmt17 is the smallest OSNet variant (trained on MSMT17), auto-downloaded
    by BoxMOT the first time it's used — no gated/manual Google-Drive checkpoint fetching.
    """
    from boxmot.reid.core.reid import ReID

    return ReID(Path("osnet_x0_25_msmt17.pt"), device="cpu", half=False)


# --------------------------------------------------------------------------
# Feature extraction & similarity
# --------------------------------------------------------------------------

def extract_deep_features(reid_backend, img_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Batch-extract OSNet embeddings for N boxes in one full frame (single forward pass)."""
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    feats = reid_backend(img_bgr, boxes=boxes_xyxy)
    feats = np.asarray(feats, dtype=np.float32)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats


def extract_aux_features(crop_bgr: np.ndarray) -> dict:
    """Clothing-color histogram + body aspect ratio, computed straight from the crop."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    hist = hist.flatten().astype(np.float32)

    h, w = crop_bgr.shape[:2]
    aspect = h / max(w, 1)

    return {"hist": hist, "aspect": aspect}


def combined_similarity(ref_deep, ref_aux, cur_deep, cur_aux, w_deep, w_hist, w_ratio):
    deep_sim = float(np.dot(ref_deep, cur_deep))  # both L2-normalized -> cosine similarity
    hist_sim = float(cv2.compareHist(ref_aux["hist"], cur_aux["hist"], cv2.HISTCMP_INTERSECT))
    ratio_diff = abs(ref_aux["aspect"] - cur_aux["aspect"]) / max(ref_aux["aspect"], cur_aux["aspect"], 1e-6)
    ratio_sim = max(0.0, 1.0 - ratio_diff)
    score = w_deep * deep_sim + w_hist * hist_sim + w_ratio * ratio_sim
    return score


def match_against_watchlist(people: dict, cur_deep, cur_aux, w_deep, w_hist, w_ratio):
    """Score one detected person against every name in the watchlist; return the best match."""
    best_name, best_score = None, -1.0
    for name, info in people.items():
        score = combined_similarity(info["deep"], info["aux"], cur_deep, cur_aux, w_deep, w_hist, w_ratio)
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def name_to_color(name: str) -> tuple[int, int, int]:
    """Deterministic, visually distinct BGR color per name (stable for the whole session)."""
    hue = (hash(name) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# --------------------------------------------------------------------------
# Thread-safe shared state between the Streamlit main thread and the
# streamlit-webrtc video callback thread.
# --------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector: YOLO | None = None
        self.reid_backend = None
        self.people: dict = {}     # name -> {"deep": ndarray, "aux": dict}
        self.threshold: float = 0.55
        self.w_deep: float = 0.7
        self.w_hist: float = 0.2
        self.w_ratio: float = 0.1
        self.det_conf: float = 0.4
        self.frame_width: int = 640
        self.process_every: int = 2
        self.last_fps: float = 0.0
        self.log: list[dict] = []
        self.log_cooldown_s: float = 3.0
        self._last_log_time: dict = {}   # name -> last logged timestamp

    def snapshot(self):
        with self.lock:
            return (
                self.detector, self.reid_backend, dict(self.people),
                self.threshold, self.w_deep, self.w_hist, self.w_ratio,
                self.det_conf, self.frame_width, self.process_every,
            )

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def report_fps(self, fps: float):
        with self.lock:
            self.last_fps = fps

    def read_fps(self) -> float:
        with self.lock:
            return self.last_fps

    def maybe_log(self, name: str, score: float):
        with self.lock:
            now = time.time()
            last = self._last_log_time.get(name, 0.0)
            if now - last > self.log_cooldown_s:
                self._last_log_time[name] = now
                self.log.append({
                    "_ts": now, "time": datetime.now().strftime("%H:%M:%S"),
                    "name": name, "score": round(score, 3),
                })
                if len(self.log) > 100:
                    self.log.pop(0)

    def read_log(self):
        with self.lock:
            return list(reversed(self.log[-30:]))

    def clear_log(self):
        with self.lock:
            self.log = []
            self._last_log_time = {}


if "shared" not in st.session_state:
    st.session_state.shared = SharedState()

shared: SharedState = st.session_state.shared


# --------------------------------------------------------------------------
# Video processing callback (runs on a background thread) — used by webrtc
# --------------------------------------------------------------------------

class PersonMatchProcessor:
    def __init__(self, state: SharedState):
        self.state = state
        self._frame_idx = 0
        self._last_results = []
        self._last_time = time.time()
        self._fps_smooth = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        (detector, reid_backend, people, threshold,
         w_deep, w_hist, w_ratio, det_conf, frame_width, process_every) = self.state.snapshot()

        h, w = img.shape[:2]
        if w > frame_width:
            scale = frame_width / w
            img = cv2.resize(img, (frame_width, int(h * scale)))

        self._frame_idx += 1

        if detector is not None and reid_backend is not None and people:
            if self._frame_idx % max(process_every, 1) == 0:
                results = detector.predict(img, classes=[0], conf=det_conf, verbose=False)
                boxes_raw = results[0].boxes

                clean_boxes = []
                for box in boxes_raw:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
                    if x2 - x1 < 10 or y2 - y1 < 20:
                        continue
                    clean_boxes.append((x1, y1, x2, y2))

                new_results = []
                if clean_boxes:
                    boxes_arr = np.array(clean_boxes, dtype=np.float32)
                    deep_feats = extract_deep_features(reid_backend, img, boxes_arr)
                    for (x1, y1, x2, y2), cur_deep in zip(clean_boxes, deep_feats):
                        crop = img[y1:y2, x1:x2]
                        cur_aux = extract_aux_features(crop)
                        best_name, best_score = match_against_watchlist(
                            people, cur_deep, cur_aux, w_deep, w_hist, w_ratio
                        )
                        is_match = best_score >= threshold
                        new_results.append(((x1, y1, x2, y2), best_name, best_score, is_match))
                        if is_match:
                            self.state.maybe_log(best_name, best_score)
                self._last_results = new_results

            for (x1, y1, x2, y2), name, score, is_match in self._last_results:
                if is_match:
                    color = name_to_color(name)
                    label = f"{name} {score:.2f}"
                else:
                    color = (0, 0, 220)
                    label = f"person {score:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        elif detector is not None:
            cv2.putText(img, "Add a person to the watchlist to start tracking", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        now = time.time()
        inst_fps = 1.0 / max(now - self._last_time, 1e-6)
        self._fps_smooth = inst_fps if self._fps_smooth == 0 else (0.9 * self._fps_smooth + 0.1 * inst_fps)
        self._last_time = now
        self.state.report_fps(self._fps_smooth)
        cv2.putText(img, f"FPS: {self._fps_smooth:.1f}", (10, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# Person inference helper (shared by both webcam and IP-cam modes)
# --------------------------------------------------------------------------

def process_frame_for_persons(img: np.ndarray, detector, reid_backend, people: dict,
                               threshold: float, w_deep: float, w_hist: float, w_ratio: float,
                               det_conf: float, frame_width: int,
                               shared_state: SharedState) -> np.ndarray:
    """Run YOLO + ReID on one BGR frame and return the annotated copy."""
    h, w = img.shape[:2]
    if w > frame_width:
        scale = frame_width / w
        img = cv2.resize(img, (frame_width, int(h * scale)))

    if detector is not None and reid_backend is not None and people:
        results = detector.predict(img, classes=[0], conf=det_conf, verbose=False)
        boxes_raw = results[0].boxes

        clean_boxes = []
        for box in boxes_raw:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
            if x2 - x1 < 10 or y2 - y1 < 20:
                continue
            clean_boxes.append((x1, y1, x2, y2))

        if clean_boxes:
            boxes_arr = np.array(clean_boxes, dtype=np.float32)
            deep_feats = extract_deep_features(reid_backend, img, boxes_arr)
            for (x1, y1, x2, y2), cur_deep in zip(clean_boxes, deep_feats):
                crop = img[y1:y2, x1:x2]
                cur_aux = extract_aux_features(crop)
                best_name, best_score = match_against_watchlist(
                    people, cur_deep, cur_aux, w_deep, w_hist, w_ratio
                )
                is_match = best_score >= threshold
                if is_match:
                    color = name_to_color(best_name)
                    label = f"{best_name} {best_score:.2f}"
                    shared_state.maybe_log(best_name, best_score)
                else:
                    color = (0, 0, 220)
                    label = f"person {best_score:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    elif detector is not None:
        cv2.putText(img, "Add a person to the watchlist to start tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return img


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Live Person Tracker", page_icon="🧍", layout="wide")
st.title("🧍 Live Person Tracker")
st.caption(
    "Add one or more named people to your watchlist — each from their own reference photo and "
    "a manually selected box — then find any of them live in your webcam or IP camera feed using "
    "whole-body appearance. Works at a distance where faces aren't resolvable. Runs 100% on CPU."
)

if "people" not in st.session_state:
    st.session_state.people = {}       # name -> {"deep":..., "aux":..., "thumb": np.ndarray}
if "form_version" not in st.session_state:
    st.session_state.form_version = 0  # bump to reset the name/upload widgets after "Add person"

with st.sidebar:
    st.header("Settings")

    # ── Video source ──────────────────────────────────────────────────────
    st.subheader("📹 Video Source")
    source_mode = st.radio(
        "Source type",
        options=["Webcam (WebRTC)", "IP / RTSP Camera"],
        index=0,
        horizontal=False,
        help="WebRTC streams directly from your browser. IP/RTSP connects to a network camera via OpenCV.",
    )

    if source_mode == "IP / RTSP Camera":
        stream_url = st.text_input(
            "Stream URL",
            value="",
            placeholder="rtsp://user:pass@192.168.1.10:554/stream  or  http://ip:port/video",
        )
        ip_run = st.toggle("▶️ Start IP stream", value=False, key="ip_run")

    st.divider()

    # ── Detection / matching settings ─────────────────────────────────────
    st.subheader("⚙️ Settings")
    det_conf = st.slider("Detector confidence", 0.1, 0.9, 0.4, 0.05,
                          help="YOLO person-detection confidence threshold.")
    frame_width = st.select_slider("Frame width (processing)", options=[320, 480, 640, 800, 960], value=640)
    process_every = st.slider("Run detection every N frames", 1, 5, 2,
                               help="Skip frames to boost FPS; boxes hold between skipped frames.")
    threshold = st.slider("Match threshold (combined score)", 0.20, 0.90, 0.55, 0.01)

    st.divider()
    st.caption("Feature weights (should roughly sum to 1.0)")
    w_deep = st.slider("ReID embedding weight", 0.0, 1.0, 0.7, 0.05,
                        help="OSNet person re-identification embedding — the strongest signal.")
    w_hist = st.slider("Clothing color weight", 0.0, 1.0, 0.2, 0.05)
    w_ratio = st.slider("Height/width ratio weight", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    fps_placeholder = st.empty()

shared.update(
    threshold=threshold, w_deep=w_deep, w_hist=w_hist, w_ratio=w_ratio,
    det_conf=det_conf, frame_width=frame_width, process_every=process_every,
)

detector = load_detector()
reid_backend = load_reid_backend()
shared.update(detector=detector, reid_backend=reid_backend, people=dict(st.session_state.people))

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("1. Add people to the watchlist")

    fv = st.session_state.form_version
    name_input = st.text_input("Name", key=f"name_input_{fv}", placeholder="e.g. Alex")
    uploaded = st.file_uploader(
        "Reference photo", type=["jpg", "jpeg", "png"], key=f"uploader_{fv}"
    )

    add_status = st.empty()

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        orig_w, orig_h = pil_img.size
        display_w = 480
        scale = display_w / orig_w
        display_h = int(orig_h * scale)
        resized = pil_img.resize((display_w, display_h))

        st.caption("Use the sliders to box the person, name them above, then click 'Add person'.")

        default_x = (int(orig_w * 0.30), int(orig_w * 0.70))
        default_y = (int(orig_h * 0.10), int(orig_h * 0.95))
        x1, x2 = st.slider("Left ↔ right", 0, orig_w, default_x, key=f"box_x_{fv}")
        y1, y2 = st.slider("Top ↔ bottom", 0, orig_h, default_y, key=f"box_y_{fv}")

        preview = resized.copy()
        draw = ImageDraw.Draw(preview)
        draw.rectangle(
            [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
            outline=(0, 200, 0), width=3,
        )
        st.image(preview, use_container_width=True)

        add_clicked = st.button(
            "Add person", type="primary", disabled=not name_input.strip()
        )

        if add_clicked:
            clean_name = name_input.strip()
            if x2 - x1 < 10 or y2 - y1 < 20:
                add_status.error("That box is too small — widen the sliders around the person.")
            elif clean_name in st.session_state.people:
                add_status.error(f"'{clean_name}' is already on the watchlist — remove it first to replace.")
            else:
                full_rgb = np.array(pil_img)
                full_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
                box_arr = np.array([[x1, y1, x2, y2]], dtype=np.float32)

                person_deep = extract_deep_features(reid_backend, full_bgr, box_arr)[0]
                person_aux = extract_aux_features(full_bgr[y1:y2, x1:x2])
                thumb = full_rgb[y1:y2, x1:x2]

                st.session_state.people[clean_name] = {
                    "deep": person_deep, "aux": person_aux, "thumb": thumb,
                }
                shared.update(people=dict(st.session_state.people))
                st.session_state.form_version += 1  # resets name/upload/sliders for the next person
                st.rerun()
    else:
        add_status.info("Upload a photo, box the person with the sliders, name them, then click 'Add person'.")

    st.divider()
    st.subheader("Watchlist")

    if st.session_state.people:
        for name in list(st.session_state.people.keys()):
            info = st.session_state.people[name]
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                st.image(info["thumb"], width=48)
            with c2:
                color = name_to_color(name)
                st.markdown(
                    f"<span style='color: rgb({color[2]},{color[1]},{color[0]})'>\u25CF</span> **{name}**",
                    unsafe_allow_html=True,
                )
            with c3:
                if st.button("Remove", key=f"remove_{name}"):
                    del st.session_state.people[name]
                    shared.update(people=dict(st.session_state.people))
                    st.rerun()

        if st.button("Clear watchlist"):
            st.session_state.people = {}
            shared.update(people={})
            shared.clear_log()
            st.rerun()
    else:
        st.caption("No one added yet — add at least one person above to start matching.")

    st.divider()
    st.subheader("Detection log")
    lcol1, lcol2 = st.columns(2)
    with lcol1:
        refresh_log = st.button("Refresh log")
    with lcol2:
        if st.button("Clear log"):
            shared.clear_log()

    log_entries = shared.read_log()
    if log_entries:
        st.table([{"Time": e["time"], "Name": e["name"], "Score": e["score"]} for e in log_entries])
    else:
        st.caption("No matches logged yet. Click 'Refresh log' to check for new detections.")

# --------------------------------------------------------------------------
# Right column — live feed (Webcam via WebRTC  OR  IP/RTSP via OpenCV loop)
# --------------------------------------------------------------------------

with col2:
    if source_mode == "Webcam (WebRTC)":
        st.subheader("2. Live webcam feed")
        ctx = webrtc_streamer(
            key="person-match",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=lambda: PersonMatchProcessor(shared),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.state.playing:
            fps_placeholder.metric("Live FPS", f"{shared.read_fps():.1f}")
        else:
            fps_placeholder.info("Click START above to begin the webcam feed.")

    else:
        # ── IP / RTSP Camera mode ──────────────────────────────────────────
        st.subheader("2. IP / RTSP Camera feed")

        if "ip_cap" not in st.session_state:
            st.session_state.ip_cap = None

        frame_slot = st.empty()
        fps_slot   = st.empty()

        if ip_run:
            url = stream_url.strip() if "stream_url" in dir() else ""
            if not url:
                st.warning("⚠️ Enter a stream URL in the sidebar first.")
                st.stop()

            if st.session_state.ip_cap is None:
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    st.error(f"❌ Could not open stream: {url}")
                    st.stop()
                st.session_state.ip_cap = cap

            cap = st.session_state.ip_cap
            prev_t = time.time()
            fps_smooth = 0.0
            frame_idx = 0

            (det_snap, reid_snap, people_snap, thresh_snap,
             wd_snap, wh_snap, wr_snap, dc_snap, fw_snap, pe_snap) = shared.snapshot()

            while st.session_state.get("ip_run", False):
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read frame from stream. Check the URL and network.")
                    break

                frame_idx += 1
                if frame_idx % max(pe_snap, 1) == 0:
                    annotated = process_frame_for_persons(
                        frame.copy(), det_snap, reid_snap, people_snap,
                        thresh_snap, wd_snap, wh_snap, wr_snap,
                        dc_snap, fw_snap, shared
                    )
                    # refresh snapshot periodically so sidebar tweaks take effect
                    if frame_idx % 30 == 0:
                        (det_snap, reid_snap, people_snap, thresh_snap,
                         wd_snap, wh_snap, wr_snap, dc_snap, fw_snap, pe_snap) = shared.snapshot()

                now = time.time()
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(now - prev_t, 1e-6))
                prev_t = now

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_slot.image(rgb, channels="RGB", use_container_width=True)
                fps_slot.caption(f"FPS: {fps_smooth:.1f}")
                fps_placeholder.metric("Live FPS", f"{fps_smooth:.1f}")
                time.sleep(0.01)
        else:
            if st.session_state.ip_cap is not None:
                st.session_state.ip_cap.release()
                st.session_state.ip_cap = None
            frame_slot.info("Stream stopped. Toggle '▶️ Start IP stream' in the sidebar to begin.")

st.divider()
with st.expander("Tips & limitations"):
    st.markdown(
        """
- **How matching works**: each detected person is compared against **every name on your
  watchlist** using a weighted blend of an OSNet person re-identification embedding, an HSV
  clothing-color histogram, and body height/width aspect ratio. Whichever watchlisted person
  scores highest is shown as the label, if their score clears the match threshold — otherwise
  the box is drawn red and labeled generically as "person".
- **Each name gets a stable, distinct color** in both the watchlist panel and the live overlay,
  so multiple simultaneous matches are easy to tell apart at a glance.
- **IP / RTSP Camera**: enter the full stream URL (e.g. `rtsp://admin:pass@192.168.1.10:554/stream`
  or `http://192.168.1.10:8080/video`) and toggle **Start IP stream** in the sidebar. OpenCV
  reads the stream directly; no browser WebRTC relay is needed.
- **This is still short/medium-term, same-outfit re-identification** — not face recognition.
- **Distance**: detection (YOLO finding a person at all) is still the bottleneck at very long range.
- **Speed vs. accuracy**: raise "Run detection every N frames" and lower frame width for more FPS.
- **Detection log**: click "Refresh log" to pull the latest entries.
- Everything runs locally — no images or video leave your machine.
        """
    )
