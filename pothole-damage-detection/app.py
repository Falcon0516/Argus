import cv2
import os
import json
from datetime import datetime
from pothole_core import PotholeSegmentor

def run_pipeline(video_path="sample_video.mp4", sample_interval_sec=1):
    detector = PotholeSegmentor(model_path="model/best.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = int(fps * sample_interval_sec)
    frame_count = 0
    inspection_records = []

    print("[INFO] Starting Road Damage Assessment Pipeline...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            annotated_frame, detections = detector.process_frame(frame)

            for item in detections:
                record = {
                    "damage_type": item["damage_type"],
                    "severity": item["severity"],
                    "area_m2": item["area_m2"],
                    "confidence": item["confidence"],
                    "timestamp": timestamp,
                    "camera_id": "PATROL_UNIT_04",
                    "gps_lat": 12.9716,  # Replace with actual NMEA/GPS feed if integrated
                    "gps_lon": 77.5946
                }
                inspection_records.append(record)

                # Save snapshot if damage is severe
                if item["severity"] == "Severe":
                    snapshot_path = f"alerts/severe_pothole_f{frame_count}.jpg"
                    cv2.imwrite(snapshot_path, annotated_frame)
                    print(f"  [ALERT] Severe damage recorded -> {snapshot_path}")

            cv2.imshow("Project Argus - Road Damage Assessment", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save summary report to JSON
    with open("road_damage_report.json", "w") as f:
        json.dump(inspection_records, f, indent=4)

    print(f"\n[INFO] Inspection Complete. {len(inspection_records)} incidents logged to 'road_damage_report.json'")

if __name__ == "__main__":
    run_pipeline("sample_video.mp4", sample_interval_sec=0.5)