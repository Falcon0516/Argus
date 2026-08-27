"""
speed_estimator.py
==================
Vehicle speed estimation using perspective transformation.

Adapted from: kemalkilicaslan/Vehicle-Speed-Estimation-System
Key idea:
  1. Define a trapezoidal ROI on the road (in pixel coords)
  2. Map it to a rectangle of known real-world dimensions (meters)
  3. Track vehicle centre-points across frames
  4. Compute speed = distance_in_meters / time_in_seconds * 3.6 → km/h
"""
from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# ── Default calibration (can be overridden per-session) ───────────────────────
DEFAULT_ROI = np.array([
    [750,  350],   # top-left
    [1150, 350],   # top-right
    [1870, 1079],  # bottom-right
    [50,   1079],  # bottom-left
], dtype=np.int32)

DEFAULT_TARGET_WIDTH  = 12   # metres — road width
DEFAULT_TARGET_HEIGHT = 50   # metres — measurement distance

# COCO vehicle class IDs
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck


class ViewTransformer:
    """Pixel ↔ real-world coordinate mapper via perspective transform."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32),
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.array([]).reshape(0, 2)
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


class SpeedEstimator:
    """
    End-to-end speed estimation pipeline.
    Designed for frame-by-frame invocation (works with both video files and live streams).
    """

    def __init__(
        self,
        model_path: str = "models/yolo11n.pt",
        roi: np.ndarray | None = None,
        target_width: float = DEFAULT_TARGET_WIDTH,
        target_height: float = DEFAULT_TARGET_HEIGHT,
        conf: float = 0.40,
        iou: float = 0.7,
        fps: float = 30.0,
        trace_length: int = 10,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.fps = fps

        # ROI
        self.roi = roi if roi is not None else DEFAULT_ROI.copy()
        target_rect = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ])
        self.view_transformer = ViewTransformer(self.roi, target_rect)
        self.polygon_zone = sv.PolygonZone(polygon=self.roi)

        # Tracking
        self.tracker  = sv.ByteTrack()
        self.smoother = sv.DetectionsSmoother()

        # Per-vehicle coordinate history  {tracker_id: deque([y1, y2, ...])}
        coord_buf_len = max(int(fps), 10)
        self.coordinates: dict[int, deque] = defaultdict(lambda: deque(maxlen=coord_buf_len))

        # Annotators
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=trace_length,
            color=sv.Color.from_bgr_tuple((0, 200, 255)),   # cyan traces
        )
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=2,
            text_scale=0.7,
            text_position=sv.Position.TOP_CENTER,
            color=sv.Color.from_bgr_tuple((0, 200, 255)),
        )

    # ── public API ────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run detection + tracking + speed estimation on a single frame.

        Returns:
            annotated_frame  — the frame with traces, speed labels, and ROI drawn
            speed_records    — list of dicts: {tracker_id, class_name, speed_kmh, bbox}
        """
        # Detect
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter to vehicles + confidence
        mask_cls  = np.isin(detections.class_id, VEHICLE_CLASSES)
        mask_conf = detections.confidence > self.conf
        detections = detections[mask_cls & mask_conf]

        # ROI filter
        roi_mask   = self.polygon_zone.trigger(detections)
        detections = detections[roi_mask]

        # Track + smooth
        detections = self.tracker.update_with_detections(detections)
        detections = detections.with_nms(threshold=self.iou)
        detections = self.smoother.update_with_detections(detections)

        # Centre points → perspective transform
        centres = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
        if len(centres) > 0:
            transformed = self.view_transformer.transform_points(centres).astype(int)
            for tid, (_, y) in zip(detections.tracker_id, transformed):
                self.coordinates[tid].append(y)

        # Compute speeds
        labels: list[str] = []
        records: list[dict] = []
        class_names = results.names  # {0: 'person', 2: 'car', ...}

        for i, tid in enumerate(detections.tracker_id):
            buf = self.coordinates[tid]
            if len(buf) >= 2:
                dist = abs(buf[-1] - buf[0])
                t    = len(buf) / self.fps
                speed = dist / t * 3.6  # m/s → km/h
            else:
                speed = 0.0

            cls_id   = int(detections.class_id[i])
            cls_name = class_names.get(cls_id, str(cls_id))
            bbox     = detections.xyxy[i].tolist()

            labels.append(f"{int(speed)} km/h")
            records.append({
                "tracker_id": int(tid),
                "class_name": cls_name,
                "speed_kmh":  round(speed, 1),
                "bbox":       bbox,
            })

        # Annotate
        annotated = frame.copy()

        # Draw ROI polygon
        cv2.polylines(annotated, [self.roi], True, (0, 255, 255), 2, cv2.LINE_AA)

        annotated = self.trace_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        return annotated, records

    def update_fps(self, fps: float) -> None:
        """Call this if stream FPS changes (e.g. after opening camera)."""
        self.fps = fps
        buf_len  = max(int(fps), 10)
        self.coordinates = defaultdict(lambda: deque(maxlen=buf_len))
