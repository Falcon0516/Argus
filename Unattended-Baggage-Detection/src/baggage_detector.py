"""
baggage_detector.py
===================
Unattended baggage detection engine.

Replaces the original Deep SORT + TensorFlow pipeline with
ultralytics YOLO ByteTrack — zero extra dependencies.

Logic:
  1. YOLO detects persons (cls 0) and bags (cls 24=backpack, 26=handbag, 28=suitcase)
  2. ByteTrack assigns persistent IDs to both
  3. Each bag is assigned to its nearest person (owner) on first appearance
  4. If the owner moves beyond `distance_threshold` pixels from the bag → UNATTENDED
  5. If the owner returns within threshold → ATTENDED again
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs
PERSON_CLASS = 0
BAG_CLASSES  = [24, 26, 28]   # backpack, handbag, suitcase

BAG_CLASS_NAMES = {24: "backpack", 26: "handbag", 28: "suitcase"}


@dataclass
class TrackedBag:
    bag_id: int
    owner_id: int | None = None
    bbox: tuple = (0, 0, 0, 0)
    is_unattended: bool = False
    unattended_since: str | None = None
    cls_name: str = "bag"


def _centre(box: tuple) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _distance(a: tuple, b: tuple) -> float:
    cx1, cy1 = _centre(a)
    cx2, cy2 = _centre(b)
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)


class BaggageDetector:
    """Frame-by-frame unattended baggage detection."""

    def __init__(
        self,
        model_path: str = "models/yolo11n.pt",
        conf: float = 0.35,
        distance_threshold: float = 150.0,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.distance_threshold = distance_threshold

        # Persistent state across frames
        self.bag_registry: dict[int, TrackedBag] = {}   # {bag_track_id: TrackedBag}

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Returns:
            annotated — frame with boxes + labels drawn
            events    — list of dicts for the log
        """
        # Run YOLO with ByteTrack
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            verbose=False, conf=self.conf,
        )

        person_tracks: list[dict] = []   # {id, bbox}
        bag_tracks:    list[dict] = []   # {id, bbox, cls}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                bbox = tuple(map(int, box.xyxy[0].tolist()))

                if cls_id == PERSON_CLASS:
                    person_tracks.append({"id": track_id, "bbox": bbox})
                elif cls_id in BAG_CLASSES:
                    bag_tracks.append({
                        "id": track_id, "bbox": bbox,
                        "cls": BAG_CLASS_NAMES.get(cls_id, "bag"),
                    })

        # Build person lookup {id: bbox}
        person_lookup = {p["id"]: p["bbox"] for p in person_tracks}

        # Process each bag
        events: list[dict] = []
        active_bag_ids = set()

        for bt in bag_tracks:
            bid  = bt["id"]
            bbox = bt["bbox"]
            active_bag_ids.add(bid)

            # Get or create tracked bag
            if bid not in self.bag_registry:
                self.bag_registry[bid] = TrackedBag(
                    bag_id=bid, bbox=bbox, cls_name=bt["cls"]
                )

            tb = self.bag_registry[bid]
            tb.bbox = bbox
            tb.cls_name = bt["cls"]

            # ── Owner assignment ──────────────────────────────────────
            if tb.owner_id is None:
                # Assign nearest person as owner
                min_dist = float("inf")
                nearest  = None
                for pid, pbox in person_lookup.items():
                    d = _distance(bbox, pbox)
                    if d < min_dist:
                        min_dist = d
                        nearest  = pid
                if nearest is not None and min_dist < self.distance_threshold:
                    tb.owner_id = nearest

            # ── Check unattended status ───────────────────────────────
            if tb.owner_id is not None and tb.owner_id in person_lookup:
                dist = _distance(bbox, person_lookup[tb.owner_id])
                if dist > self.distance_threshold:
                    if not tb.is_unattended:
                        tb.is_unattended = True
                        tb.unattended_since = datetime.now().strftime("%H:%M:%S")
                        events.append({
                            "time":  tb.unattended_since,
                            "type":  "UNATTENDED",
                            "bag_id": bid,
                            "cls":    tb.cls_name,
                            "owner":  tb.owner_id,
                            "dist":   round(dist, 0),
                        })
                else:
                    if tb.is_unattended:
                        events.append({
                            "time":  datetime.now().strftime("%H:%M:%S"),
                            "type":  "RECOVERED",
                            "bag_id": bid,
                            "cls":    tb.cls_name,
                            "owner":  tb.owner_id,
                            "dist":   round(dist, 0),
                        })
                    tb.is_unattended = False
                    tb.unattended_since = None
            elif tb.owner_id is not None:
                # Owner not visible → unattended
                if not tb.is_unattended:
                    tb.is_unattended = True
                    tb.unattended_since = datetime.now().strftime("%H:%M:%S")
                    events.append({
                        "time":  tb.unattended_since,
                        "type":  "UNATTENDED",
                        "bag_id": bid,
                        "cls":    tb.cls_name,
                        "owner":  tb.owner_id,
                        "dist":   -1,
                    })

        # ── Clean up stale bags ───────────────────────────────────────
        stale = [k for k in self.bag_registry if k not in active_bag_ids]
        for k in stale:
            del self.bag_registry[k]

        # ── Draw annotations ─────────────────────────────────────────
        annotated = frame.copy()

        # Draw persons (thin grey)
        for p in person_tracks:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(annotated, f"person #{p['id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # Draw bags (green = attended, red = unattended)
        for bt in bag_tracks:
            bid  = bt["id"]
            bbox = bt["bbox"]
            tb   = self.bag_registry.get(bid)
            if tb and tb.is_unattended:
                color = (0, 0, 255)      # RED
                label = f"⚠ UNATTENDED {tb.cls_name} #{bid}"
                thick = 3
            else:
                color = (0, 220, 0)      # GREEN
                owner_txt = f" → owner #{tb.owner_id}" if tb and tb.owner_id else ""
                label = f"{bt['cls']} #{bid}{owner_txt}"
                thick = 2

            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw line to owner
            if tb and tb.owner_id is not None and tb.owner_id in person_lookup:
                bag_c  = _centre(bbox)
                own_c  = _centre(person_lookup[tb.owner_id])
                line_c = (0, 0, 255) if tb.is_unattended else (0, 200, 0)
                cv2.line(annotated,
                         (int(bag_c[0]), int(bag_c[1])),
                         (int(own_c[0]), int(own_c[1])),
                         line_c, 1, cv2.LINE_AA)

        return annotated, events
