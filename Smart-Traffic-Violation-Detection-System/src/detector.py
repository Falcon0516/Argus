"""
detector.py
Stage 1 — Load models and run base detection on full frame.
Returns raw YOLO results; filtering/association is done downstream.
"""
from __future__ import annotations
import numpy as np
from ultralytics import YOLO

BASE_MODEL_PATH   = "models/yolo11n.pt"
HELMET_MODEL_PATH = "models/helmetdetectionmodel.mlpackage"
PLATE_MODEL_PATH  = "license_plate_detector.mlpackage"

# COCO class IDs we care about
PERSON_CLASS     = 0
BICYCLE_CLASS    = 1
MOTORCYCLE_CLASS = 3

# Helmet model class names (confirmed: 0=head, 1=helmet, 2=person)
HELMET_CLS  = 1   # wearing helmet
HEAD_CLS    = 0   # bare head = no helmet


class ArgusDetector:
    """Loads all models once; exposes detect() per frame."""

    def __init__(self, conf: float = 0.30):
        self.conf = conf
        self.base   = YOLO(BASE_MODEL_PATH)
        self.helmet = YOLO(HELMET_MODEL_PATH)
        self.plate  = YOLO(PLATE_MODEL_PATH)

    # ── Stage 1: Base detection ───────────────────────────────────────────
    def detect_base(self, frame: np.ndarray, persist: bool = True):
        """Run ByteTrack on full frame; return Results object."""
        return self.base.track(
            frame, persist=persist,
            tracker="bytetrack.yaml",
            verbose=False, conf=self.conf,
        )[0]

    # ── Stage 3a: Helmet check on crop ───────────────────────────────────
    def detect_helmet(self, crop: np.ndarray):
        """Run helmet model on a cropped rider region."""
        if crop.size == 0:
            return None
        return self.helmet.predict(crop, verbose=False, conf=0.20)[0]

    # ── Stage 3b: Plate detection on motorcycle crop ──────────────────────
    def detect_plate(self, crop: np.ndarray):
        """Run plate detector on a cropped motorcycle region."""
        if crop.size == 0:
            return None
        return self.plate.predict(crop, verbose=False, conf=self.conf)[0]
