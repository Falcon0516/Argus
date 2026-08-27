"""
Crowd Density Detection Engine
Uses YOLOv8 to detect people and compute zone-based density levels.
"""

import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Create a file logger with a standard format."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Color palette for zone levels
# ---------------------------------------------------------------------------

ZONE_COLORS = {
    "Low":      (76, 175, 80),    # green  (BGR)
    "Medium":   (0, 215, 255),    # yellow
    "High":     (0, 140, 255),    # orange
    "Critical": (60, 60, 220),    # red
}

BOX_COLOR      = (255, 0, 200)   # magenta for person boxes
TEXT_COLOR      = (255, 255, 255)
GRID_LINE_COLOR = (180, 180, 180)


# ---------------------------------------------------------------------------
# CrowdDetector
# ---------------------------------------------------------------------------

class CrowdDetector:
    """YOLOv8-based crowd density detector with grid-zone analysis."""

    def __init__(self, config_path: str = "config.json"):
        base = os.path.dirname(os.path.abspath(config_path)) or "."

        # Load config
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        # Resolve model path
        model_path = self.cfg.get("model_path", "yolov8s.pt")
        if not os.path.isabs(model_path):
            model_path = os.path.join(base, model_path)
        self.model = YOLO(model_path)

        # Try GPU
        try:
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.half = False
        except Exception:
            self.device = "cpu"
            self.half = False

        # Detection settings
        self.confidence = float(self.cfg.get("confidence", 0.45))
        self.grid_rows = int(self.cfg.get("grid", {}).get("rows", 3))
        self.grid_cols = int(self.cfg.get("grid", {}).get("cols", 3))
        self.thresholds = self.cfg.get("thresholds", {"low": 3, "medium": 6, "high": 10})

        # Loggers
        log_dir = os.path.join(base, self.cfg.get("log_dir", "logs"))
        self.det_logger = setup_logger("detections", os.path.join(log_dir, "detections.log"))
        self.alert_logger = setup_logger("alerts", os.path.join(log_dir, "alerts.log"))

        # Track which zones are currently in alert (to avoid spamming)
        self._alerted_zones: set = set()

        # Warm up model
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=640, classes=[0], conf=0.5, verbose=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> tuple:
        """
        Run detection on a single frame.

        Returns:
            annotated_frame (np.ndarray): frame with overlays drawn
            zone_data (dict): {"total": int, "zones": [{"id", "count", "level"}, ...]}
        """
        if frame is None:
            return None, {"total": 0, "zones": []}

        # Run YOLOv8
        results = self.model.predict(
            frame,
            imgsz=640,
            conf=self.confidence,
            classes=[0],  # person only
            device=self.device,
            half=self.half,
            verbose=False,
        )

        # Extract person boxes
        boxes = []
        for r in results:
            for box in getattr(r, "boxes", []) or []:
                try:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    boxes.append((x1, y1, x2, y2, conf))
                except Exception:
                    continue

        # Assign boxes to grid zones
        h, w = frame.shape[:2]
        zone_h = h // self.grid_rows
        zone_w = w // self.grid_cols
        zone_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        for (x1, y1, x2, y2, _) in boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            row = min(max(cy // zone_h, 0), self.grid_rows - 1)
            col = min(max(cx // zone_w, 0), self.grid_cols - 1)
            zone_counts[row][col] += 1

        # Build zone data
        total = len(boxes)
        zone_data = {"total": total, "zones": []}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                zone_id = f"Z{r * self.grid_cols + c + 1}"
                count = int(zone_counts[r][c])
                level = self._classify(count)
                zone_data["zones"].append({"id": zone_id, "count": count, "level": level})

                # Alert logging
                if level == "Critical" and zone_id not in self._alerted_zones:
                    self._alerted_zones.add(zone_id)
                    self.alert_logger.info(f"ALERT | {zone_id} | count={count} | level=CRITICAL")
                elif level != "Critical" and zone_id in self._alerted_zones:
                    self._alerted_zones.discard(zone_id)

        # Detection log (one line per frame)
        self.det_logger.info(f"total={total} | zones={json.dumps({z['id']: z['count'] for z in zone_data['zones']})}")

        # Draw overlays
        annotated = self._draw(frame, boxes, zone_data, zone_h, zone_w)
        return annotated, zone_data

    def update_confidence(self, value: float):
        self.confidence = max(0.05, min(float(value), 0.95))

    def update_grid(self, rows: int, cols: int):
        self.grid_rows = max(1, rows)
        self.grid_cols = max(1, cols)

    def update_thresholds(self, low: int, medium: int, high: int):
        self.thresholds = {"low": low, "medium": medium, "high": high}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify(self, count: int) -> str:
        if count <= self.thresholds["low"]:
            return "Low"
        elif count <= self.thresholds["medium"]:
            return "Medium"
        elif count <= self.thresholds["high"]:
            return "High"
        return "Critical"

    def _draw(self, frame: np.ndarray, boxes: list, zone_data: dict, zone_h: int, zone_w: int) -> np.ndarray:
        """Draw grid, zone labels, person bounding boxes."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Semi-transparent zone overlays
        overlay = out.copy()
        for z in zone_data["zones"]:
            idx = int(z["id"][1:]) - 1
            r, c = divmod(idx, self.grid_cols)
            x0, y0 = c * zone_w, r * zone_h
            x1, y1 = x0 + zone_w, y0 + zone_h
            color = ZONE_COLORS.get(z["level"], (100, 100, 100))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        # Grid lines
        for r in range(1, self.grid_rows):
            y = r * zone_h
            cv2.line(out, (0, y), (w, y), GRID_LINE_COLOR, 1)
        for c in range(1, self.grid_cols):
            x = c * zone_w
            cv2.line(out, (x, 0), (x, h), GRID_LINE_COLOR, 1)

        # Zone labels
        for z in zone_data["zones"]:
            idx = int(z["id"][1:]) - 1
            r, c = divmod(idx, self.grid_cols)
            x0, y0 = c * zone_w, r * zone_h
            color = ZONE_COLORS.get(z["level"], (200, 200, 200))
            label = f'{z["id"]}: {z["level"]} ({z["count"]})'
            cv2.putText(out, label, (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Person bounding boxes
        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(out, f"{conf:.0%}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1, cv2.LINE_AA)

        # Total count banner
        cv2.putText(out, f"People: {zone_data['total']}", (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
        return out
