# Vehicle Speed Estimation System

Real-time vehicle speed estimation using YOLO + ByteTrack + perspective transformation.

Adapted from [kemalkilicaslan/Vehicle-Speed-Estimation-System](https://github.com/kemalkilicaslan/Vehicle-Speed-Estimation-System) with live stream support.

## How It Works

```
Live Frame
    │
    ▼
[YOLO11n] Detect vehicles (car, motorcycle, bus, truck)
    │
    ▼
[PolygonZone] Filter to ROI (trapezoidal measurement zone)
    │
    ▼
[ByteTrack] Track vehicles across frames
    │
    ▼
[ViewTransformer] Perspective transform → real-world coordinates
    │
    ▼
[Speed = Δdistance / Δtime × 3.6] → km/h per vehicle
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# OpenCV live window (webcam)
python run_speed.py

# OpenCV with RTSP stream
python run_speed.py --cam "rtsp://192.168.1.10:554/stream"

# Streamlit dashboard
streamlit run dashboard.py
```

## Calibration

The default ROI is tuned for a specific overhead camera angle. For accurate speed readings with your camera:

1. Measure the real-world road width and distance in your camera view
2. Adjust **Road Width** and **Measurement Distance** in the dashboard sidebar
3. For the ROI polygon coordinates, edit `src/speed_estimator.py` → `DEFAULT_ROI`

## Project Structure

```
├── src/
│   └── speed_estimator.py   # Core engine: ViewTransformer + SpeedEstimator
├── run_speed.py             # OpenCV entry point
├── dashboard.py             # Streamlit dashboard
├── models/                  # YOLO model (auto-downloaded on first run)
└── requirements.txt
```