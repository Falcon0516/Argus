"""
Real-time License Plate Recognition from a webcam stream (OpenCV window UI).
Cross-platform (Windows / Linux / macOS) using fast-alpr (ONNX Runtime).

Every detected plate is automatically logged to a CSV file (default:
plate_log.csv) with timestamp, plate text, and confidence.

Usage:
    python main.py                      # default webcam (index 0)
    python main.py --camera 1           # pick a different camera
    python main.py --log plates.csv     # custom log file path
    python main.py --no-perspective     # disable perspective correction
"""

import argparse
import time

import cv2

from alpr_core import DEFAULT_LOG_PATH, PlateLogger, build_alpr, open_camera, run_alpr_on_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time ALPR on a webcam stream")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.5,
        help="Minimum OCR confidence to log/print a plate (0-1)",
    )
    parser.add_argument("--log", type=str, default=DEFAULT_LOG_PATH, help="CSV file to log detected plates to")
    parser.add_argument("--no-window", action="store_true", help="Run headless (no display window)")
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Disable perspective correction for angled plates",
    )
    parser.add_argument(
        "--no-vehicle-crop",
        action="store_true",
        help=(
            "Disable vehicle-aware cropping: captured images will be the full "
            "frame instead of just the vehicle nearest the detected plate"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading ALPR models (first run downloads ONNX weights)...")
    alpr = build_alpr(use_perspective_correction=not args.no_perspective)
    logger = PlateLogger(log_path=args.log, use_vehicle_crop=not args.no_vehicle_crop)
    print(f"Logging detections to: {args.log}")
    if not args.no_vehicle_crop:
        print("Vehicle-aware cropping enabled (first capture downloads YOLOv8 weights).")

    cap = open_camera(args.camera)

    prev_time = time.time()
    fps = 0.0

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, exiting.")
                break

            annotated_frame, detections = run_alpr_on_frame(alpr, frame, min_conf=args.min_conf)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            for d in detections:
                was_new = logger.log_if_new(
                    d.plate_text, d.confidence, frame=frame, plate_bbox=d.bounding_box
                )
                if was_new:
                    print(f"[{d.timestamp}] Plate: {d.plate_text}  (conf: {d.confidence:.2f})")

            if not args.no_window:
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("ALPR - press q to quit", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
