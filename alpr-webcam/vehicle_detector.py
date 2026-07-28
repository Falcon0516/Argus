"""
Vehicle detection (car / motorcycle / bus / truck) used to map a detected
license plate to the *specific vehicle it belongs to*, rather than saving
the whole camera frame when the scene contains multiple vehicles.

Uses a COCO-pretrained YOLO model (via ultralytics) filtered down to
vehicle classes only. This is a separate, general-purpose object detector
from the plate detector fast-alpr uses internally - fast-alpr's YOLO model
is trained only to find plates, not cars/trucks, so a second small model
is needed for this.

Import is lazy (inside VehicleDetector.__init__) so nothing here has any
cost - or even requires ultralytics to be installed - if vehicle-aware
cropping is disabled.
"""

import logging

logger = logging.getLogger(__name__)

# COCO class ids for vehicle-like objects.
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Small + fast; auto-downloads on first use (like fast-alpr's own weights).
DEFAULT_VEHICLE_MODEL = "yolov8n.pt"


class VehicleDetector:
    """Thin wrapper around an Ultralytics YOLO model, filtered to vehicle
    classes only."""

    def __init__(self, model_name: str = DEFAULT_VEHICLE_MODEL, conf: float = 0.35):
        from ultralytics import YOLO  # imported here so this stays optional

        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame):
        """Returns a list of (x1, y1, x2, y2, class_name, conf) boxes for
        every vehicle detected in `frame`. Never raises - a bad frame just
        yields no boxes."""
        try:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
        except Exception as e:
            logger.warning("Vehicle detection error, skipping: %s", e)
            return []

        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, VEHICLE_CLASS_IDS[cls_id], conf))
        return boxes


def _bbox_center(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box[:4]
    return x1 <= x <= x2 and y1 <= y <= y2


def find_vehicle_for_plate(plate_bbox, vehicle_boxes):
    """Given a plate bounding box and a list of vehicle boxes from the same
    frame, returns the single vehicle box the plate actually belongs to
    (not just "a" vehicle in frame):

      1. If the plate's center falls inside one or more vehicle boxes, the
         *smallest-area* containing box wins. Area is the tie-breaker
         because a plate can only ever physically sit on one vehicle, and
         when boxes overlap in 2D (a car behind another car, etc.) the
         nearer/smaller box in the image is the one the plate is actually
         mounted on.
      2. If the plate center doesn't fall inside any vehicle box (common
         when the plate detector's box is a little outside a tight vehicle
         box), fall back to the vehicle box whose center is nearest to the
         plate's center.

    Returns None if `vehicle_boxes` is empty - caller should fall back to
    saving the full frame in that case.
    """
    if not vehicle_boxes:
        return None

    plate_center = _bbox_center(plate_bbox)

    containing = [b for b in vehicle_boxes if _point_in_box(plate_center, b)]
    if containing:
        def area(b):
            x1, y1, x2, y2 = b[:4]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return min(containing, key=area)

    def dist(b):
        cx, cy = _bbox_center(b)
        return (cx - plate_center[0]) ** 2 + (cy - plate_center[1]) ** 2

    return min(vehicle_boxes, key=dist)


def crop_vehicle(frame, vehicle_box, padding_ratio: float = 0.08):
    """Crops `frame` to `vehicle_box` with a small padding margin so the
    saved image isn't a razor-tight bounding box. Clamps to frame bounds.
    Returns the full frame unchanged if the box is degenerate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = vehicle_box[:4]
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = box_w * padding_ratio, box_h * padding_ratio

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]
