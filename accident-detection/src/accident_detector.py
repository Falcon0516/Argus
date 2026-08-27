"""
accident_detector.py
====================
Real-time Traffic Accident Detection engine using YOLOv8 heuristics.

Adapted from: 772003pranav/Accident-Detection
"""
from __future__ import annotations

from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class AccidentDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.40) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.speed_threshold = 5.0
        self.prolonged_collision_frames = 4
        self.min_collision_distance = 80
        
        # Keep track of prolonged collisions across frames
        self.prolonged_collision_count = 0
        self.last_boxes = []

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        # Run YOLO with tracking enabled
        results = self.model(frame, verbose=False, conf=self.conf)
        
        annotated = frame.copy()
        events = []
        accident_detected = False
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # We filter for vehicles (COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck)
            vehicle_classes = [2, 3, 5, 7]
            
            valid_boxes = []
            for i, box in enumerate(boxes):
                cls_id = int(classes[i])
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Draw ALL boxes in gray to debug what YOLO sees
                if cls_id not in vehicle_classes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (150, 150, 150), 1)
                    cv2.putText(annotated, f"cls:{cls_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
                else:
                    valid_boxes.append(box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            collision_detected = False
            for i in range(len(valid_boxes)):
                for j in range(i + 1, len(valid_boxes)):
                    box1 = valid_boxes[i]
                    box2 = valid_boxes[j]
                    
                    iou = calculate_iou(box1, box2)
                    if iou > 0.05:  # Lowered from 0.15
                        collision_detected = True
                    
                    dist = ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5
                    if dist < self.min_collision_distance:
                        collision_detected = True

            if collision_detected:
                self.prolonged_collision_count += 1
                if self.prolonged_collision_count >= self.prolonged_collision_frames:
                    accident_detected = True
            else:
                self.prolonged_collision_count = max(0, self.prolonged_collision_count - 2)

            self.last_boxes = valid_boxes

        if accident_detected:
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w-1, h-1), (0, 0, 255), 10)
            
            label = "⚠ ACCIDENT DETECTED!"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
            cv2.rectangle(annotated, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 255), -1)
            cv2.putText(annotated, label, (20, 10 + th + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            
            events.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "Accident",
                "conf": 1.0,
            })
            
        return annotated, events
