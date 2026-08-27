"""
waste_detector.py
=================
Waste and litter detection engine using YOLOv8.

Model trained on the TACO dataset (Trash Annotations in Context).
Contains 60 classes of litter (bottles, wrappers, cans, cartons, etc.).
"""
from __future__ import annotations

import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO


class WasteDetector:
    def __init__(self, model_path: str = "models/waste_detector.pt", conf: float = 0.35) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        results = self.model(frame, verbose=False, conf=self.conf)
        
        annotated = frame.copy()
        events = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                cls_name = self.names.get(cls_id, "Litter")
                
                # Draw box (Orange for waste/litter)
                color = (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{cls_name} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                events.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": cls_name,
                    "conf": round(conf_val, 2),
                    "bbox": (x1, y1, x2, y2)
                })
                
        return annotated, events
