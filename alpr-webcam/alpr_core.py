"""
Shared ALPR setup and logging used by both main.py (OpenCV window) and
dashboard.py (Streamlit dashboard), so the two entry points stay in sync.
"""

import csv
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import onnxruntime as ort

from fast_alpr import ALPR  # noqa: E402

from perspective import correct_perspective  # noqa: E402
from vehicle_detector import VehicleDetector, crop_vehicle, find_vehicle_for_plate  # noqa: E402

# --- Fix a real CoreML/ONNX Runtime bug, not just silence its noise ------
# fast-alpr's YOLOv9 plate detector uses an end-to-end NMS op. On a frame
# with zero plate detections, that op's output is a zero-element dynamic-
# shape tensor, which Apple's CoreML execution provider cannot handle:
# https://github.com/microsoft/onnxruntime/issues/20372
# open_image_models catches the resulting exception internally and returns
# "no detections" for that frame - it never crashes - but on any Mac with
# CoreML available it fires on essentially every plate-free frame, flooding
# stderr with ERROR-level engine logs on every single one. build_alpr()
# below excludes CoreML for the *detector* only (OCR keeps it, since OCR
# doesn't run NMS and isn't subject to this) - that avoids the bug's root
# cause instead of just hiding its output.
#
# open_image_models' own inference module hardcodes its logger to INFO on
# import (see its yolo_v9/inference.py) - that overrides anything set on it
# beforehand, so the two lines below have to run after the `from fast_alpr
# import ALPR` line above, not before, or they get clobbered.
ort.set_default_logger_severity(4)  # 4 = FATAL only; hides ORT's own ERROR-level engine spam
logging.getLogger("open_image_models.detection.core.yolo_v9.inference").setLevel(logging.ERROR)

DEFAULT_DETECTOR = "yolo-v9-t-384-license-plate-end2end"
DEFAULT_OCR = "cct-xs-v2-global-model"
DEFAULT_LOG_PATH = "plate_log.csv"
DEFAULT_CAPTURE_DIR = "captures"
DEFAULT_CAPTURE_CONF_THRESHOLD = 0.99  # save a vehicle image at/above this confidence

# CoreML can't handle the detector's empty-detection NMS output shape (see
# note above) - excluded for the detector only, in _get_detector_providers().
_DETECTOR_EXCLUDED_PROVIDERS = ("CoreMLExecutionProvider",)


def _get_detector_providers() -> list[str] | None:
    """ONNX Runtime providers for the plate *detector* specifically, with
    CoreMLExecutionProvider excluded (see module-level note above for why).
    OCR is unaffected and keeps using whatever providers ONNX Runtime finds
    by default, including CoreML, since it isn't subject to this bug."""
    available = ort.get_available_providers()
    filtered = [p for p in available if p not in _DETECTOR_EXCLUDED_PROVIDERS]
    return filtered or None  # None -> let ORT fall back to its own default


def normalize_confidence(conf) -> float:
    """OCR confidence can come back as a single float OR a list of
    per-character confidences depending on model/version. Normalize either
    shape down to one float so comparisons never crash."""
    if conf is None:
        return 0.0
    if isinstance(conf, (list, tuple)):
        conf = [c for c in conf if c is not None]
        return float(sum(conf) / len(conf)) if conf else 0.0
    return float(conf)


class _PerspectiveCorrectingOCR:
    """Wraps an existing fast-alpr OCR object: warps the cropped plate to a
    straight-on rectangle before handing it to the real OCR model."""

    def __init__(self, inner_ocr):
        self.inner_ocr = inner_ocr

    def predict(self, cropped_plate):
        corrected = correct_perspective(cropped_plate)
        return self.inner_ocr.predict(corrected)


def build_alpr(
    detector_model: str = DEFAULT_DETECTOR,
    ocr_model: str = DEFAULT_OCR,
    use_perspective_correction: bool = True,
) -> ALPR:
    """Create an ALPR instance, optionally wrapping its OCR step with
    perspective correction for better reads on steep-angle plates.

    The detector is built with CoreML excluded from its providers (see the
    module-level note above _get_detector_providers) to avoid a real ONNX
    Runtime/CoreML crash-on-empty-detections bug, not just to quiet it.
    """
    alpr = ALPR(
        detector_model=detector_model,
        ocr_model=ocr_model,
        detector_providers=_get_detector_providers(),
    )
    if use_perspective_correction:
        alpr.ocr = _PerspectiveCorrectingOCR(alpr.ocr)
    return alpr


@dataclass
class Detection:
    timestamp: str
    plate_text: str
    confidence: float
    bounding_box: tuple = field(default=None)


class PlateLogger:
    """Logs every detected plate (timestamp, text, confidence) to a CSV
    file, with a short debounce so a stationary plate isn't re-logged every
    single frame. Thread-safe so it can be shared with a Streamlit app.

    When a plate's confidence is at or above `capture_conf_threshold`, an
    image is also saved to `capture_dir` and the CSV row records its path -
    this is what maps a plate reading to a specific vehicle photo.

    If `use_vehicle_crop` is True (default), a separate vehicle detector
    (car/truck/bus/motorcycle, via YOLO) is run on that frame and the saved
    image is cropped to *just the vehicle nearest the plate* - not the raw
    frame - so multi-car scenes don't capture every vehicle mixed together
    under one plate's row. `crop_type` in the CSV records how the saved
    image was produced:
      - "vehicle"    : successfully isolated to the matched vehicle's box
      - "full_frame" : no vehicle box could be matched, fell back to the
                       whole frame (e.g. plate detected but the vehicle
                       detector missed it, or vehicle cropping is disabled)
      - ""           : no image was captured at all (below confidence
                       threshold)
    """

    EXPECTED_HEADER = ["timestamp", "plate_text", "confidence", "vehicle_image_path", "crop_type"]

    def __init__(
        self,
        log_path: str = DEFAULT_LOG_PATH,
        debounce_seconds: float = 3.0,
        capture_dir: str = DEFAULT_CAPTURE_DIR,
        capture_conf_threshold: float = DEFAULT_CAPTURE_CONF_THRESHOLD,
        use_vehicle_crop: bool = True,
        vehicle_conf_threshold: float = 0.35,
    ):
        self.log_path = Path(log_path)
        self.debounce_seconds = debounce_seconds
        self.capture_dir = Path(capture_dir)
        self.capture_conf_threshold = capture_conf_threshold
        self.use_vehicle_crop = use_vehicle_crop
        self._vehicle_conf_threshold = vehicle_conf_threshold
        self._vehicle_detector = None  # lazily built on first capture - see _get_vehicle_detector
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _get_vehicle_detector(self):
        """Builds the vehicle detector on first use only (avoids paying the
        model-load cost when no plate has hit the capture threshold yet, or
        when vehicle cropping is disabled). Falls back gracefully to
        full-frame capture if ultralytics/weights aren't available."""
        if not self.use_vehicle_crop:
            return None
        if self._vehicle_detector is None:
            try:
                self._vehicle_detector = VehicleDetector(conf=self._vehicle_conf_threshold)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Could not load vehicle detector, falling back to full-frame capture: %s", e
                )
                self.use_vehicle_crop = False
                return None
        return self._vehicle_detector

    def _ensure_header(self):
        """Creates the log file with the current header if missing, or
        migrates it in place if an older/mismatched schema is found (e.g. a
        3-column log from before vehicle_image_path was added). This keeps
        every row padded to a consistent column count so downstream CSV/
        DataFrame parsing never desyncs."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(self.EXPECTED_HEADER)
            return

        with open(self.log_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
            rows = list(reader)

        needs_migration = existing_header != self.EXPECTED_HEADER
        if not needs_migration:
            return

        n = len(self.EXPECTED_HEADER)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.EXPECTED_HEADER)
            for row in rows:
                padded = (row + [""] * n)[:n]
                writer.writerow(padded)

    def _safe_filename(self, plate_text: str, timestamp: str) -> str:
        safe_plate = "".join(c for c in plate_text if c.isalnum()) or "UNKNOWN"
        safe_ts = timestamp.replace(":", "-")
        return f"{safe_ts}_{safe_plate}.jpg"

    def log_if_new(self, plate_text: str, confidence: float, frame=None, plate_bbox=None) -> bool:
        """Logs the plate if it hasn't been seen within the debounce window.
        If `frame` is provided and confidence >= capture_conf_threshold, also
        saves an image and records its path + how it was cropped.

        When `plate_bbox` is also provided and vehicle cropping is enabled,
        the saved image is cropped to just the vehicle nearest that plate
        (see vehicle_detector.find_vehicle_for_plate) instead of the full
        frame - this is what keeps multi-vehicle scenes from mapping a
        plate to every car in shot. Without a usable bbox, or if no vehicle
        can be matched, it falls back to saving the whole frame.

        Returns True if it was actually logged (i.e. is a "new" reading)."""
        now = time.time()
        with self._lock:
            last_seen = self._last_seen.get(plate_text, 0)
            if now - last_seen < self.debounce_seconds:
                return False
            self._last_seen[plate_text] = now

        timestamp = datetime.now().isoformat(timespec="seconds")

        image_path = ""
        crop_type = ""
        if frame is not None and confidence >= self.capture_conf_threshold:
            import cv2

            image_to_save = frame
            crop_type = "full_frame"

            detector = self._get_vehicle_detector() if plate_bbox is not None else None
            if detector is not None:
                try:
                    vehicle_boxes = detector.detect(frame)
                    matched = find_vehicle_for_plate(plate_bbox, vehicle_boxes)
                    if matched is not None:
                        image_to_save = crop_vehicle(frame, matched)
                        crop_type = "vehicle"
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "Vehicle-aware crop failed, saving full frame instead: %s", e
                    )

            filename = self._safe_filename(plate_text, timestamp)
            full_path = self.capture_dir / filename
            cv2.imwrite(str(full_path), image_to_save)
            image_path = str(full_path)

        with self._lock:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, plate_text, f"{confidence:.3f}", image_path, crop_type])
        return True

    def read_all(self):
        """Returns all logged rows as a list of dicts (for dashboard display),
        guaranteed to have exactly the expected columns with no None/NaN
        values - safe to hand straight to a DataFrame."""
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                clean = {k: (row.get(k) or "") for k in self.EXPECTED_HEADER}
                rows.append(clean)
            return rows


def run_alpr_on_frame(alpr: ALPR, frame, min_conf: float = 0.5):
    """
    Runs ALPR on a single frame and returns (annotated_frame, detections)
    where detections is a list of Detection objects that passed min_conf.
    Never raises - a bad frame just yields no detections.
    """
    try:
        drawn = alpr.draw_predictions(frame)
        annotated_frame = drawn.image
        results = drawn.results
    except Exception as e:
        logging.getLogger(__name__).warning("Inference error on frame, skipping: %s", e)
        return frame, []

    detections = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for r in results:
        text = r.ocr.text if r.ocr else None
        conf = normalize_confidence(r.ocr.confidence if r.ocr else None)
        if not text or conf < min_conf:
            continue
        bbox = getattr(getattr(r, "detection", None), "bounding_box", None)
        detections.append(Detection(timestamp=timestamp, plate_text=text, confidence=conf, bounding_box=bbox))

    return annotated_frame, detections


def open_camera(index: int):
    """Open a webcam in a way that works across Windows/Linux/macOS."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap
