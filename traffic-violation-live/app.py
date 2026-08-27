import cv2
import json
import torch
import os
import numpy as np
from sort import Sort
from traffic_core import TrafficViolationEngine

def run_traffic_monitor(video_src="traffic_test.mp4", config_path="configs/intersection_cam01.json"):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize YOLOv5 model (loads PyTorch / DirectML)
    print("[INFO] Loading vehicle detection model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # Filter for vehicles only (car, motorcycle, bus, truck)
    model.classes = [2, 3, 5, 7]
    model.conf = 0.40

    # Initialize SORT tracker & Argus violation engine
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    engine = TrafficViolationEngine(config)
    os.makedirs("alerts", exist_ok=True)

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_src}")
        return

    print("[INFO] Starting traffic monitoring pipeline. Press 'q' to exit.")
    incident_logs = []
    frame_idx = 0

    # Demo toggle: simulates red light state
    is_red_light = True 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame skip for performance optimization (process every 2nd frame)
        if frame_idx % 2 == 0:
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()

            # Format for SORT: [x1, y1, x2, y2, score]
            dets_for_sort = detections[:, :5] if len(detections) > 0 else np.empty((0, 5))
            tracked_objects = tracker.update(dets_for_sort)

            # Evaluate violations
            violations = engine.process_tracks(tracked_objects, is_red_light=is_red_light)

            # Draw calibration stop-line
            p1 = tuple(config["stop_line"]["pt1"])
            p2 = tuple(config["stop_line"]["pt2"])
            line_color = (0, 0, 255) if is_red_light else (0, 255, 0)
            cv2.line(frame, p1, p2, line_color, 3)

            # Process violations and save evidence
            for v in violations:
                incident_logs.append(v)
                x1, y1, x2, y2 = v["bbox"]
                
                # Save snapshot evidence
                snapshot_file = f"alerts/{v['type']}_trk{v['track_id']}_f{frame_idx}.jpg"
                cv2.imwrite(snapshot_file, frame)
                print(f"🚨 [VIOLATION DETECTED] {v['type']} | Track ID: {v['track_id']} -> Saved: {snapshot_file}")

            # Draw bounding boxes & tracking IDs
            for trk in tracked_objects:
                x1, y1, x2, y2, trk_id = map(int, trk)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"ID:{trk_id}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

            cv2.imshow("Project Argus - Traffic Violation Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    with open("traffic_violation_report.json", "w") as f:
        json.dump(incident_logs, f, indent=4)
    print(f"\n[INFO] Run finished. {len(incident_logs)} violations logged to 'traffic_violation_report.json'")

if __name__ == "__main__":
    # Provide a traffic clip or 0 for webcam
    run_traffic_monitor(video_src=0)