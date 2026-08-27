#!/usr/bin/env python3
"""
run_webcam.py — Argus live webcam entry point.

Usage:
    python run_webcam.py
    python run_webcam.py --cam 1 --conf 0.28
"""
from __future__ import annotations
import argparse
import logging
import sys
import time

import cv2
import numpy as np

from src.detector        import ArgusDetector, PERSON_CLASS, MOTORCYCLE_CLASS
from src.association     import associate_riders, box_to_xyxy, upper_body_crop, crop_box
from src.violation_engine import ViolationEngine, classify_helmet_result
from src.visualizer      import draw_frame, draw_hud

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

WINDOW_TITLE = "Argus — Traffic Violation Detection  [q] quit"
INFER_SIZE   = 640


def run(cam_index: int = 0, conf: float = 0.28) -> None:
    detector = ArgusDetector(conf=conf)
    engine   = ViolationEngine()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        log.error("Cannot open camera %d", cam_index)
        sys.exit(1)
    log.info("Camera %d opened (%dx%d)", cam_index,
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 540)

    fps_counter, fps_start, fps_val = 0, time.time(), 0.0

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                continue

            frame = cv2.resize(raw, (INFER_SIZE, INFER_SIZE))

            # ── Stage 1: Base detection ───────────────────────────────────
            base_result = detector.detect_base(frame)
            boxes_all   = base_result.boxes

            person_boxes: list[tuple] = []
            moto_boxes:   list[tuple] = []

            if boxes_all is not None:
                for box in boxes_all:
                    cls = int(box.cls[0])
                    b   = box_to_xyxy(box)
                    if cls == PERSON_CLASS:
                        person_boxes.append(b)
                    elif cls == MOTORCYCLE_CLASS:
                        moto_boxes.append(b)

            # ── Stage 2: Associate riders to motorcycles ──────────────────
            groups = associate_riders(person_boxes, moto_boxes)

            # ── Stage 3a: Helmet check per rider ─────────────────────────
            helmet_statuses: dict[int, dict] = {}
            all_rider_idxs = set(p for riders in groups.values() for p in riders)

            for p_idx in all_rider_idxs:
                crop = upper_body_crop(frame, person_boxes[p_idx])
                h_result = detector.detect_helmet(crop)
                helmet_statuses[p_idx] = classify_helmet_result(h_result)

            # ── Stage 3b: Plate detection per motorcycle ──────────────────
            plate_detections: dict[int, list] = {}
            plate_texts:      dict[int, str]  = {}

            for m_idx, box in enumerate(moto_boxes):
                mc_crop = crop_box(frame, box, pad_frac=0.05)
                p_result = detector.detect_plate(mc_crop)
                plates = []
                if p_result and p_result.boxes is not None:
                    for pb in p_result.boxes:
                        cls_name = p_result.names.get(int(pb.cls[0]), "plate")
                        plates.append({"label": cls_name, "conf": float(pb.conf[0])})
                plate_detections[m_idx] = plates
                if plates:
                    plate_texts[m_idx] = f"{plates[0]['label']} {plates[0]['conf']:.2f}"

            # ── Stage 4: Evaluate violations ──────────────────────────────
            violations = engine.evaluate(groups, helmet_statuses, plate_detections)

            for v in violations:
                log.info(v.log_line())

            # ── Draw ──────────────────────────────────────────────────────
            draw_frame(frame, person_boxes, moto_boxes, groups,
                       helmet_statuses, violations, plate_texts)

            counts = {
                "moto":   len(moto_boxes),
                "riders": len(all_rider_idxs),
            }

            fps_counter += 1
            if fps_counter >= 8:
                fps_val     = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start   = time.time()

            draw_hud(frame, fps_val, counts, len(violations))
            cv2.imshow(WINDOW_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Argus live webcam")
    p.add_argument("--cam",  type=int,   default=0)
    p.add_argument("--conf", type=float, default=0.28)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(cam_index=args.cam, conf=args.conf)
