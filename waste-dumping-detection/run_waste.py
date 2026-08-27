#!/usr/bin/env python3
"""
run_waste.py — OpenCV live waste and litter detection.

Usage:
    python run_waste.py                                     # default webcam
    python run_waste.py --cam 1                             # second camera
    python run_waste.py --cam "rtsp://192.168.1.10:554/s"   # RTSP stream
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import cv2

from src.waste_detector import WasteDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOW_TITLE = "Argus — Waste & Litter Detection  [q] quit"


def run(source: int | str = 0, conf: float = 0.25) -> None:
    detector = WasteDetector(conf=conf)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source %s", str(source))
        sys.exit(1)
    
    log.info("Source %s opened (%dx%d)", str(source),
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 540)

    fps_counter, fps_start, fps_val = 0, time.time(), 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            annotated, events = detector.process_frame(frame)

            for e in events:
                log.info("🗑️ Detected: %s (conf: %.2f)", e["type"], e["conf"])

            fps_counter += 1
            if fps_counter >= 8:
                fps_val = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

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
    p = argparse.ArgumentParser(description="Argus waste and litter detection")
    p.add_argument("--cam",  type=str,   default="0",
                   help="Camera index or RTSP/HTTP URL")
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cam_val = int(args.cam) if args.cam.isdigit() else args.cam
    run(source=cam_val, conf=args.conf)
