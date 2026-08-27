# Argus — Smart Traffic Violation Detection System

A real-time traffic violation detection system for Apple Silicon Macs using YOLO11 and CoreML.

## Detects
- 🚨 **Triple Riding** — flags motorcycles with ≥ 3 riders using spatial IoU association
- ⛑ **No Helmet** — per-rider helmet status checked only on motorcycle rider crops (eliminates false positives)
- 🔢 **License Plates** — detected within motorcycle bounding regions only

## Architecture

```
Full Frame
    │
    ▼
[Stage 1] YOLO11n — detect persons + motorcycles
    │
    ▼
[Stage 2] Spatial Association — group riders per motorcycle (vertical+horizontal IoU)
    │
    ├── TRIPLE RIDING if riders ≥ 3
    │
    ▼
[Stage 3a] Helmet model — run on upper-body crop of each rider only
    │
    ▼
[Stage 3b] Plate model — run on motorcycle crop only
    │
    ▼
[Stage 4] Violation Engine — emit structured violation events
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install ultralytics coremltools streamlit
```

## Models Required

Place the following in `models/`:
- `yolo11n.pt` — auto-downloaded by ultralytics on first run
- `helmetdetectionmodel.mlpackage` — classes: `head` (no helmet), `helmet`, `person`

Place at project root:
- `license_plate_detector.mlpackage`

To convert PyTorch models to CoreML:
```bash
python convert_to_coreml.py
```

## Run

```bash
# OpenCV live window
python run_webcam.py --conf 0.25

# Streamlit dashboard (http://localhost:8501)
streamlit run dashboard.py
```

## Project Structure

```
├── src/
│   ├── detector.py          # Stage 1 — model loading + base detection
│   ├── association.py       # Stage 2 — person↔motorcycle IoU grouping
│   ├── violation_engine.py  # Stage 3 — triple riding + helmet rules
│   └── visualizer.py        # Stage 4 — vivid bbox drawing + banners
├── run_webcam.py            # OpenCV entry point
├── dashboard.py             # Streamlit dashboard
└── convert_to_coreml.py     # One-time CoreML export script
```
