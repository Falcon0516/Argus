# Real-Time ALPR Webcam Project

Real-time License Plate Recognition from a live webcam stream, built on
[fast-alpr](https://github.com/ankandrew/fast-alpr) (YOLOv9 detector + ONNX Runtime OCR).
Runs identically on Windows, Linux, and macOS — no native compilation needed.

## Files

| File                 | Purpose                                                          |
|----------------------|-------------------------------------------------------------------|
| `alpr_core.py`       | Shared setup: model loading, perspective correction hook, logger  |
| `perspective.py`     | Standalone perspective-correction (deskew) for angled plate crops |
| `vehicle_detector.py`| YOLO vehicle detection + plate-to-vehicle matching/cropping       |
| `main.py`            | OpenCV preview window UI                                          |
| `dashboard.py`       | Streamlit web dashboard UI                                        |
| `plate_log.csv`      | Auto-created log of every detected plate (timestamp/text/confidence/image/crop type) |

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you have an NVIDIA GPU and want faster inference, install the GPU ONNX backend instead:

   ```bash
   pip install fast-alpr[onnx-gpu] opencv-python streamlit pandas
   ```

## Run — OpenCV window

```bash
python main.py
```

| Flag                | Description                                          |
|----------------------|-------------------------------------------------------|
| `--camera N`         | Camera index (default `0`)                             |
| `--min-conf`         | Minimum OCR confidence to report a plate (0-1)         |
| `--log FILE`         | CSV file to log detected plates to (default `plate_log.csv`) |
| `--no-window`        | Run headless, no preview window (useful on servers)    |
| `--no-perspective`   | Disable the perspective-correction step                |
| `--no-vehicle-crop`  | Disable vehicle-aware cropping (capture full frame instead) |

Press `q` to quit.

## Run — Streamlit dashboard

```bash
streamlit run dashboard.py
```

Opens a browser tab with:
- A live annotated webcam feed
- A running table of every detected plate (timestamp, plate text, confidence)
- Sidebar controls for camera index, confidence threshold, perspective
  correction toggle, vehicle-aware crop toggle, and log file path

Toggle "Start camera" in the sidebar to begin/stop the stream.

## How perspective correction works

The plate detector returns an axis-aligned bounding box. For a plate viewed
from a steep angle, that box is a skewed crop, which hurts OCR. Before OCR
runs, `perspective.py`:

1. Finds edges in the cropped plate region (Canny + contour detection)
2. Looks for a 4-point quadrilateral that plausibly matches the plate shape
3. Warps that quadrilateral to a flat, straight-on rectangle via
   `cv2.getPerspectiveTransform` / `cv2.warpPerspective`
4. Falls back to the original crop untouched if no clean quadrilateral is
   found (so near-frontal plates are unaffected)

This is wired in via `fast-alpr`'s custom-OCR hook (`alpr.ocr`), so it works
by wrapping whatever OCR model is already loaded — no changes to the
detection step needed. Disable it with `--no-perspective` (CLI) or the
sidebar checkbox (dashboard) if you want to compare with/without.

## Logging

Every plate that passes the confidence threshold is automatically logged to
CSV (`plate_log.csv` by default) with columns: `timestamp`, `plate_text`,
`confidence`, `vehicle_image_path`, `crop_type`. A plate already logged
within the last 3 seconds isn't re-logged (debounce), so a parked/stationary
vehicle doesn't spam the log.

## Mapping a plate to a vehicle photo (vehicle-aware cropping)

When a plate's OCR confidence is at or above `DEFAULT_CAPTURE_CONF_THRESHOLD`
(0.99 by default, in `alpr_core.py`), an image gets saved to the `captures/`
folder as a `.jpg`, named `<timestamp>_<plate>.jpg`, and its path is written
into the `vehicle_image_path` column of that plate's CSV row.

That image is no longer just whatever the camera saw in the whole frame. A
separate vehicle detector (`vehicle_detector.py` — YOLOv8n pretrained on
COCO, filtered to the car/truck/bus/motorcycle classes) runs on the same
frame, and the saved image is cropped to just the vehicle that plate
actually belongs to, using this matching logic: first, every vehicle box in
the frame is found; if the plate's center falls inside one or more of them,
the smallest (nearest) containing box is picked, which is what handles
multi-car scenes correctly instead of capturing every vehicle mixed
together; if no box contains the plate (the plate detector's box can sit
slightly outside a tight vehicle box), it falls back to the vehicle box
whose center is nearest the plate's center; and if no vehicle at all can be
matched, the full frame is saved instead.

The `crop_type` column records which of these happened for each row:
`vehicle` means it was successfully isolated to the matched vehicle,
`full_frame` means it fell back to the whole frame, and empty means no
image was captured (below the confidence threshold).

This is on by default; disable it with `--no-vehicle-crop` (CLI) or the
sidebar checkbox (dashboard) to always save the full frame instead. Vehicle
detection only runs on frames that actually cross the capture confidence
threshold — not every frame — so it doesn't cost FPS on the live preview.

**Known limitation:** matching is purely 2D/geometric (box containment,
then nearest-center); it doesn't reason about depth or occlusion. Two
vehicles parked flush against each other with plates close together can
still occasionally be mismatched, since a small/near vehicle box can
contain a plate that actually belongs to a vehicle just behind it. The
current approach handles the common cases well (a car in an open driveway
or road view, or a couple of cars spaced apart in frame) — a
stereo/depth cue would be the next step for tighter accuracy in dense
parking scenarios.

To change the capture confidence threshold:

```python
logger = PlateLogger(log_path="plate_log.csv", capture_conf_threshold=0.95)
```

The Streamlit dashboard also shows the most recently captured vehicle photo,
along with whether it was isolated to the matched vehicle or fell back to
the full frame, directly under the detections table.

## Notes

- First run downloads the ONNX model weights automatically (needs internet
  once). The first vehicle-matched capture additionally downloads the
  small YOLOv8n COCO weights (`yolov8n.pt`, a few MB) the same way.
- Camera opening uses `cv2.CAP_ANY` so OpenCV picks the correct native
  backend per OS (AVFoundation on macOS, V4L2 on Linux, DirectShow/MSMF on
  Windows).
- Existing `plate_log.csv` files from before vehicle-aware cropping was
  added migrate automatically the next time `PlateLogger` starts — no
  manual steps needed, old rows just get an empty `crop_type`.
- On macOS, the plate detector explicitly excludes ONNX Runtime's CoreML
  execution provider (OCR still uses it). CoreML can't handle the
  detector's end-to-end NMS output on frames with zero plate detections
  (a known ONNX Runtime issue, microsoft/onnxruntime#20372), which
  otherwise floods the terminal with `[E:onnxruntime:...]` errors on every
  plate-free frame. Detection still runs at real-time speed on CPU.
- Want a Flask version instead of Streamlit (e.g. for a lighter-weight REST
  API + custom frontend)? The `alpr_core.py` module is UI-agnostic, so a
  Flask app can reuse `build_alpr()` / `run_alpr_on_frame()` / `PlateLogger`
  directly — happy to add that if useful.
