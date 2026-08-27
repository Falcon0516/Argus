"""
fire_detector.py
================
Fire and Smoke Detection engine using YOLOv8.

Adapted from: luminous0219/fire-and-smoke-detection-yolov8
"""
from __future__ import annotations

from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# Classes from the custom trained model
# 0: fire
# 1: smoke

CLASS_COLORS = {
    0: (0, 0, 255),    # Red for Fire
    1: (150, 150, 150) # Grey for Smoke
}

CLASS_NAMES = {
    0: "FIRE",
    1: "SMOKE"
}


class FireDetector:
    def __init__(self, model_path: str = "models/fire_smoke.pt", conf: float = 0.35) -> None:
        self.model = YOLO(model_path)
        self.conf = conf

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run inference on the frame.
        Returns:
            annotated — Frame with bounding boxes drawn.
            events    — List of detections for logging.
        """
        results = self.model(frame, verbose=False, conf=self.conf)
        
        annotated = frame.copy()
        events = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                cls_name = CLASS_NAMES.get(cls_id, "UNKNOWN")
                color = CLASS_COLORS.get(cls_id, (0, 255, 0))
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # Draw label
                label = f"⚠ {cls_name} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                events.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": cls_name,
                    "conf": round(conf_val, 2),
                    "bbox": (x1, y1, x2, y2)
                })
                
        return annotated, events
