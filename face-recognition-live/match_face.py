"""
Real-time face matching against a reference photo, CPU-only.

Usage:
    python match_face.py --reference path/to/person.jpg
    python match_face.py --reference path/to/person.jpg --camera 0 --threshold 0.55
    python match_face.py --reference path/to/person.jpg --video path/to/file.mp4

Controls:
    q  - quit
    s  - save a screenshot of the current frame to ./captures/
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def load_face_app(model_pack: str, det_size: int) -> FaceAnalysis:
    app = FaceAnalysis(name=model_pack, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def get_reference_embedding(app: FaceAnalysis, reference_path: str) -> np.ndarray:
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    img = cv2.imread(reference_path)
    if img is None:
        raise ValueError(f"Could not read image (unsupported format?): {reference_path}")

    faces = app.get(img)
    if not faces:
        raise ValueError(
            "No face detected in the reference image. "
            "Use a clear, front-facing, well-lit photo."
        )

    # If multiple faces are found in the reference photo, use the largest one.
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def main():
    parser = argparse.ArgumentParser(description="Real-time face matching (CPU) against a reference photo.")
    parser.add_argument("--reference", required=True, help="Path to the reference photo of the person to find.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--video", default=None, help="Path to a video file instead of a live webcam.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity match threshold (default: 0.55).")
    parser.add_argument("--model-pack", default="buffalo_sc", choices=["buffalo_sc", "buffalo_s", "buffalo_l"],
                         help="InsightFace model pack. buffalo_sc = fastest on CPU, buffalo_l = most accurate/slowest.")
    parser.add_argument("--det-size", type=int, default=320, help="Detector input size (smaller = faster). Default: 320.")
    parser.add_argument("--frame-width", type=int, default=640, help="Resize incoming frames to this width before processing.")
    parser.add_argument("--process-every", type=int, default=1,
                         help="Run detection every N frames (skip frames to boost FPS). Default: 1 (every frame).")
    args = parser.parse_args()

    print(f"[setup] Loading InsightFace model pack '{args.model_pack}' on CPU...")
    app = load_face_app(args.model_pack, args.det_size)

    print(f"[setup] Building reference embedding from: {args.reference}")
    ref_embedding = get_reference_embedding(app, args.reference)
    print("[setup] Reference embedding ready.")

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    os.makedirs("captures", exist_ok=True)

    fps_smooth = 0.0
    last_time = time.time()
    frame_idx = 0
    last_results = []  # cached (bbox, sim, is_match) between skipped frames

    print("[run] Press 'q' to quit, 's' to save a screenshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[run] End of stream or camera read failed.")
            break

        frame = resize_keep_aspect(frame, args.frame_width)
        frame_idx += 1

        if frame_idx % args.process_every == 0:
            faces = app.get(frame)
            last_results = []
            for face in faces:
                sim = cosine_similarity(ref_embedding, face.normed_embedding)
                is_match = sim >= args.threshold
                last_results.append((face.bbox.astype(int), sim, is_match))

        for bbox, sim, is_match in last_results:
            x1, y1, x2, y2 = bbox
            color = (0, 200, 0) if is_match else (0, 0, 220)
            label = f"{'MATCH' if is_match else 'no match'} {sim:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        now = time.time()
        inst_fps = 1.0 / max(now - last_time, 1e-6)
        fps_smooth = inst_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst_fps)
        last_time = now
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face match (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = os.path.join("captures", f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(fname, frame)
            print(f"[run] Saved screenshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
