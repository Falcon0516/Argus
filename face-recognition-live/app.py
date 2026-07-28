"""
Streamlit dashboard: drop reference photos of one or more people, then find each
of them live in your webcam feed. CPU-only inference via InsightFace + ONNX Runtime.

Run:
    streamlit run app.py
"""

import os
import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --------------------------------------------------------------------------
# Model loading (cached across reruns)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_face_app(model_pack: str, det_size: int) -> FaceAnalysis:
    app = FaceAnalysis(name=model_pack, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# A small, visually distinct palette. "Unknown" (no match) is always drawn separately in red.
PALETTE = [
    (0, 200, 0),      # green
    (255, 191, 0),    # deep sky blue-ish
    (0, 165, 255),    # orange
    (255, 0, 170),    # magenta
    (0, 220, 220),    # yellow-ish
    (200, 0, 0),      # blue
    (180, 105, 255),  # pink
    (0, 128, 128),    # olive/teal
]


def color_for_label(label: str, all_labels: list) -> tuple:
    idx = sorted(all_labels).index(label) % len(PALETTE)
    return PALETTE[idx]


# --------------------------------------------------------------------------
# Thread-safe shared state between the Streamlit main thread and the
# streamlit-webrtc video callback thread.
# --------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.face_app: FaceAnalysis | None = None
        self.ref_db: dict[str, np.ndarray] = {}   # label -> embedding
        self.threshold: float = 0.55
        self.frame_width: int = 640
        self.process_every: int = 1
        self.last_fps: float = 0.0

    def snapshot(self):
        with self.lock:
            return (
                self.face_app,
                dict(self.ref_db),
                self.threshold,
                self.frame_width,
                self.process_every,
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


if "shared" not in st.session_state:
    st.session_state.shared = SharedState()

shared: SharedState = st.session_state.shared


# --------------------------------------------------------------------------
# Video processing callback (runs on a background thread)
# --------------------------------------------------------------------------

class FaceMatchProcessor:
    def __init__(self, state: SharedState):
        self.state = state
        self._frame_idx = 0
        self._last_results = []
        self._last_time = time.time()
        self._fps_smooth = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        face_app, ref_db, threshold, frame_width, process_every = self.state.snapshot()
        labels = list(ref_db.keys())

        # Downscale for speed, keep aspect ratio.
        h, w = img.shape[:2]
        if w > frame_width:
            scale = frame_width / w
            img = cv2.resize(img, (frame_width, int(h * scale)))

        self._frame_idx += 1

        if face_app is not None and ref_db:
            if self._frame_idx % max(process_every, 1) == 0:
                faces = face_app.get(img)
                results = []
                for face in faces:
                    best_label, best_sim = None, -1.0
                    for label, emb in ref_db.items():
                        sim = cosine_similarity(emb, face.normed_embedding)
                        if sim > best_sim:
                            best_label, best_sim = label, sim
                    is_match = best_sim >= threshold
                    results.append((face.bbox.astype(int), best_label, best_sim, is_match))
                self._last_results = results

            for bbox, best_label, sim, is_match in self._last_results:
                x1, y1, x2, y2 = bbox
                if is_match:
                    color = color_for_label(best_label, labels)
                    label_text = f"{best_label} {sim:.2f}"
                else:
                    color = (0, 0, 220)  # red
                    label_text = f"Unknown {sim:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label_text, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        elif face_app is not None:
            cv2.putText(img, "Upload at least one reference photo to start matching", (10, 30),
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
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Live Face Match", page_icon="🔍", layout="wide")
st.title("🔍 Live Face Match")
st.caption("Drop photos of one or more people, then find each of them live in your webcam feed. Runs 100% on CPU, nothing leaves your machine.")

with st.sidebar:
    st.header("Settings")
    model_pack = st.selectbox(
        "Model pack",
        options=["buffalo_sc", "buffalo_s", "buffalo_l"],
        index=0,
        help="buffalo_sc = fastest on CPU. buffalo_l = most accurate, slowest.",
    )
    det_size = st.select_slider(
        "Detector size", options=[256, 320, 416, 640], value=320,
        help="Lower = faster, may miss small/far faces. Higher = more accurate, slower.",
    )
    frame_width = st.select_slider(
        "Frame width (processing)", options=[320, 480, 640, 800, 960], value=640,
        help="Frames are downscaled to this width before detection. Biggest FPS lever.",
    )
    process_every = st.slider(
        "Run detection every N frames", min_value=1, max_value=5, value=1,
        help="Skip frames to boost perceived FPS; boxes hold between skipped frames.",
    )
    threshold = st.slider(
        "Match threshold (cosine similarity)", min_value=0.20, max_value=0.80, value=0.55, step=0.01,
        help="Raise if you get false positives, lower if real matches are missed. Applies to everyone in the gallery.",
    )

    st.divider()
    fps_placeholder = st.empty()

shared.update(threshold=threshold, frame_width=frame_width, process_every=process_every)

if "ref_people" not in st.session_state:
    st.session_state.ref_people = {}  # file_key -> {"label": str, "embedding": np.ndarray, "preview": np.ndarray}

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("1. Reference photos (one or more people)")
    uploaded_files = st.file_uploader(
        "Drop a clear, front-facing photo per person",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    face_app = load_face_app(model_pack, det_size)
    shared.update(face_app=face_app)

    ref_status = st.empty()

    if uploaded_files:
        current_keys = set()
        for f in uploaded_files:
            file_key = f"{f.name}_{f.size}"
            current_keys.add(file_key)

            if file_key not in st.session_state.ref_people:
                with st.spinner(f"Processing {f.name}..."):
                    file_bytes = np.frombuffer(f.getvalue(), np.uint8)
                    ref_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    faces = face_app.get(ref_img)
                    face = largest_face(faces)

                    if face is None:
                        st.session_state.ref_people[file_key] = {
                            "label": os.path.splitext(f.name)[0],
                            "embedding": None,
                            "preview": None,
                            "error": True,
                        }
                    else:
                        x1, y1, x2, y2 = face.bbox.astype(int)
                        preview = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 0), 3)
                        st.session_state.ref_people[file_key] = {
                            "label": os.path.splitext(f.name)[0],
                            "embedding": face.normed_embedding,
                            "preview": preview,
                            "error": False,
                        }

        # Drop entries for files the user has removed from the uploader.
        for stale_key in [k for k in st.session_state.ref_people if k not in current_keys]:
            del st.session_state.ref_people[stale_key]

        # Render a small gallery with an editable name per person.
        for file_key, info in st.session_state.ref_people.items():
            if info["error"]:
                st.error(f"No face detected in **{info['label']}**'s photo — try a clearer, front-facing image.")
                continue
            with st.container(border=True):
                gcol1, gcol2 = st.columns([1, 2])
                with gcol1:
                    st.image(info["preview"], use_container_width=True)
                with gcol2:
                    new_label = st.text_input("Name", value=info["label"], key=f"label_input_{file_key}")
                    st.session_state.ref_people[file_key]["label"] = new_label

        # Build the shared reference DB from valid entries, keyed by (possibly edited) name.
        ref_db = {}
        for info in st.session_state.ref_people.values():
            if not info["error"] and info["embedding"] is not None:
                ref_db[info["label"]] = info["embedding"]

        # Warn about duplicate names (last one wins in the dict, so flag it).
        labels_list = [info["label"] for info in st.session_state.ref_people.values() if not info["error"]]
        dupes = {l for l in labels_list if labels_list.count(l) > 1}
        if dupes:
            st.warning(f"Duplicate names will overwrite each other: {', '.join(dupes)}. Give each person a unique name.")

        shared.update(ref_db=ref_db)

        if ref_db:
            ref_status.success(f"{len(ref_db)} {'person' if len(ref_db) == 1 else 'people'} loaded — matching is live.")
        else:
            ref_status.info("Upload at least one photo with a detectable face to start matching.")
    else:
        st.session_state.ref_people = {}
        shared.update(ref_db={})
        ref_status.info("Upload one or more photos to start matching against the live feed.")

with col2:
    st.subheader("2. Live webcam feed")
    ctx = webrtc_streamer(
        key="face-match",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=lambda: FaceMatchProcessor(shared),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        fps_placeholder.metric("Live FPS", f"{shared.read_fps():.1f}")
    else:
        fps_placeholder.info("Click START above to begin the webcam feed.")

st.divider()
with st.expander("Tips"):
    st.markdown(
        """
- **Multiple people**: upload one photo per person. Each detected face in the live feed is
  matched against everyone in your gallery and labeled with whoever scores highest — as long
  as that score clears the threshold. Otherwise it's labeled "Unknown".
- **Names**: edit the name field under each thumbnail — it's what shows up as the on-screen label.
  Give everyone a unique name; duplicates will overwrite each other.
- **Reference photos**: single face, well-lit, front-facing works best for each person.
- **Threshold**: start at 0.55. Raise it if people get mismatched with each other, lower it if a
  real match isn't being picked up. The same threshold applies to everyone in the gallery.
- **Speed vs. accuracy**: `buffalo_sc` + smaller detector size + smaller frame width = higher FPS.
  `buffalo_l` + larger sizes = better accuracy on small/angled faces, but slower on CPU. More
  people in the gallery adds a small amount of comparison overhead per detected face, but this
  is negligible compared to detection cost.
- Everything runs locally in this process — no images or video leave your machine.
        """
    )
