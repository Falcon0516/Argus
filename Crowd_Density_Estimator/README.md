# Crowd Density Estimator

Real-time crowd density detection using YOLOv8 with a Streamlit dashboard. Supports webcam and RTSP streams.

## Features

- **YOLOv8 Person Detection** — Detects people in real-time with configurable confidence threshold
- **Zone-Based Density Grid** — Divides the frame into a configurable grid and classifies each zone (Low / Medium / High / Critical)
- **Webcam & RTSP Support** — Works with local cameras and network IP/RTSP streams
- **Live Dashboard** — Streamlit UI with video feed, zone map, alerts, and history chart
- **File-Based Logging** — All detections and alerts logged to `logs/detections.log` and `logs/alerts.log`

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

## Project Structure

```
├── app.py              # Streamlit dashboard
├── detector.py         # YOLOv8 detection engine
├── config.json         # Configuration
├── requirements.txt    # Dependencies
├── yolov8s.pt          # YOLO model weights
└── logs/               # Auto-created
    ├── detections.log  # Frame-by-frame detection counts
    └── alerts.log      # Critical zone alerts
```

## Configuration

Edit `config.json` to change defaults:

| Key | Description | Default |
|-----|-------------|---------|
| `model_path` | Path to YOLO weights | `yolov8s.pt` |
| `confidence` | Detection confidence | `0.45` |
| `grid.rows/cols` | Zone grid dimensions | `3×3` |
| `thresholds.low/medium/high` | Zone level boundaries | `3 / 6 / 10` |
| `log_dir` | Directory for log files | `logs` |

All settings can also be adjusted live from the Streamlit sidebar.
