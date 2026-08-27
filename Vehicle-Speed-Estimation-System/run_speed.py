#!/usr/bin/env python3
"""
run_speed.py — OpenCV live speed estimation entry point.

Usage:
    python run_speed.py                                     # default webcam
    python run_speed.py --cam 1                             # second camera
    python run_speed.py --cam "rtsp://192.168.1.10:554/s"   # RTSP stream
    python run_speed.py --conf 0.35                         # lower confidence
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from src.speed_estimator import SpeedEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOW_TITLE = "Argus — Vehicle Speed Estimation  [q] quit"


def run(source: int | str = 0, conf: float = 0.40) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source %s", str(source))
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info("Source %s opened (%dx%d @ %.1f FPS)", str(source), w, h, source_fps)

    estimator = SpeedEstimator(
        model_path="models/yolo11n.pt",
        conf=conf,
        fps=source_fps,
    )

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 540)

    fps_counter, fps_start, fps_val = 0, time.time(), 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            annotated, records = estimator.process_frame(frame)

            # Overlay FPS
            fps_counter += 1
            if fps_counter >= 8:
                fps_val     = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start   = time.time()

            cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Vehicles: {len(records)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            for r in records:
                if r["speed_kmh"] > 5:
                    log.info("Vehicle #%d (%s): %d km/h",
                             r["tracker_id"], r["class_name"], int(r["speed_kmh"]))

            cv2.imshow(WINDOW_TITLE, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Argus vehicle speed estimation")
    p.add_argument("--cam",  type=str,   default="0",
                   help="Camera index (0, 1) or RTSP/HTTP URL")
    p.add_argument("--conf", type=float, default=0.40)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cam_val = int(args.cam) if args.cam.isdigit() else args.cam
    run(source=cam_val, conf=args.conf)
