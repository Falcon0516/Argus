import argparse
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

from pose_core import PoseProcessor
from anomaly_detector import AnomalyDetector

def main():
    parser = argparse.ArgumentParser(description="Argus Live Pose Anomaly Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0) or path to video file")
    parser.add_argument("--conf", type=float, default=0.35, help="Pose detection confidence threshold")
    parser.add_argument("--save-alerts", action="store_true", default=True, help="Save alert frames to disk")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"\n========================================================")
    print(f"       PROJECT ARGUS - LIVE POSE ANOMALY ENGINE         ")
    print(f"========================================================")
    print(f"[*] Loading YOLOv8-Pose Model (yolov8n-pose.pt)...")
    
    # Auto-downloads ~6.5MB weights on first run
    model = YOLO("yolov8n-pose.pt")
    
    pose_proc = PoseProcessor(conf_threshold=args.conf)
    detector = AnomalyDetector(sequence_len=30)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source '{source}'")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    os.makedirs("alerts", exist_ok=True)
    last_alert_time = 0
    prev_time = time.time()

    print(f"[*] Stream active. Press 'q' to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Persistent Multi-Person Tracking via ByteTrack
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        anomaly_flagged = False
        active_event_type = ""

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            keypoints_data = r.keypoints

            if boxes is not None and boxes.id is not None and keypoints_data is not None:
                track_ids = boxes.id.int().cpu().tolist()
                bboxes = boxes.xyxy.cpu().numpy()
                kpts = keypoints_data.data.cpu().numpy() # Shape (N, 17, 3)

                for track_id, bbox, raw_kpt in zip(track_ids, bboxes, kpts):
                    norm_kpt = pose_proc.normalize_keypoints(raw_kpt, bbox)
                    event = detector.update(track_id, norm_kpt, bbox, raw_kpt)

                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 255, 0) # Green for Normal

                    if event["status"] == "FALL_DETECTED":
                        color = (0, 0, 255) # Red
                        anomaly_flagged = True
                        active_event_type = "FALL DETECTED"
                    elif event["status"] == "FIGHT_STANCE":
                        color = (0, 165, 255) # Orange
                        anomaly_flagged = True
                        active_event_type = "FIGHT / VIOLENCE DETECTED"
                    elif event["status"] == "SUSPICIOUS_GAIT":
                        color = (255, 255, 0) # Cyan
                        anomaly_flagged = True
                        active_event_type = "SUSPICIOUS GAIT"

                    # Render skeleton and bounding box
                    pose_proc.draw_skeleton(frame, raw_kpt, color=color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Text tag
                    label = f"ID:{track_id} | {event['status']}"
                    if event["confidence"] > 0:
                        label += f" ({int(event['confidence']*100)}%)"

                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw Argus HUD Header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (10, 15, 25), -1)
        cv2.putText(frame, f"PROJECT ARGUS :: POSE ANOMALY ENGINE", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 245), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw Critical Anomaly Alert Bar
        if anomaly_flagged:
            cv2.rectangle(frame, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
            alert_text = f"CRITICAL ALERT: {active_event_type} | LOGGING TO ARGUS MESH"
            cv2.putText(frame, alert_text, (25, frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            if args.save_alerts and (curr_time - last_alert_time > 3.0):
                snapshot_path = f"alerts/alert_{int(curr_time)}.jpg"
                cv2.imwrite(snapshot_path, frame)
                print(f"[!] Alert snapshot saved: {snapshot_path}")
                last_alert_time = curr_time

        cv2.imshow("Project Argus - Pose Anomaly Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Pipeline closed gracefully.")

if __name__ == "__main__":
    main()