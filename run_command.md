# Argus Dashboard Commands

This file contains the commands to instantly launch the interactive **Streamlit Dashboard** for every AI model integrated into the Argus system.

To run any model, simply copy and paste its corresponding block into your terminal. All projects are self-contained and run within their own isolated virtual environments (`.venv`).

---

## 1. Traffic Accident Detection
```bash
cd /Users/falcon/Downloads/claude/Argus/accident-detection
.venv/bin/streamlit run dashboard.py
```

## 2. Shoplifting Detection
```bash
cd /Users/falcon/Downloads/claude/Argus/shoplifting-detection
.venv/bin/streamlit run dashboard.py
```

## 3. Realtime Weapon Detection
```bash
cd /Users/falcon/Downloads/claude/Argus/realtime-weapon-detection
.venv/bin/streamlit run dashboard.py
```

## 4. Vandalism / Graffiti / Spitting Detection (Vandal Vision)
```bash
cd /Users/falcon/Downloads/claude/Argus/vandal-vision
.venv/bin/streamlit run dashboard.py
```

## 5. Fire and Smoke Detection
```bash
cd /Users/falcon/Downloads/claude/Argus/fire-and-smoke-detection
.venv/bin/streamlit run dashboard.py
```

## 6. Waste Dumping / Litter Detection
```bash
cd /Users/falcon/Downloads/claude/Argus/waste-dumping-detection
.venv/bin/streamlit run dashboard.py
```

## 7. Smart Traffic Violation Detection (Helmets, Seatbelts, Tripling)
```bash
cd /Users/falcon/Downloads/claude/Argus/Smart-Traffic-Violation-Detection-System
.venv/bin/streamlit run dashboard.py
```

## 8. Unattended Baggage Detection
```bash
cd /Users/falcon/Downloads/claude/Argus/Unattended-Baggage-Detection
.venv/bin/streamlit run dashboard.py
```

## 9. Automatic License Plate Recognition (ALPR)
```bash
cd /Users/falcon/Downloads/claude/Argus/alpr-webcam
.venv/bin/streamlit run dashboard.py
```

## 10. Vehicle Speed Estimation
```bash
cd /Users/falcon/Downloads/claude/Argus/Vehicle-Speed-Estimation-System
.venv/bin/streamlit run dashboard.py
```

---
### Legacy / App.py Projects

For some of the earliest modules built into Argus, the dashboard entry point is named `app.py` instead of `dashboard.py`.

**11. Face Recognition**
```bash
cd /Users/falcon/Downloads/claude/Argus/face-recognition-live
.venv/bin/streamlit run app.py
```

**12. Person Tracking**
```bash
cd /Users/falcon/Downloads/claude/Argus/person-tracking-live
.venv/bin/streamlit run app.py
```

**13. Pose Anomaly Detection**
```bash
cd /Users/falcon/Downloads/claude/Argus/pose-anomaly-live
.venv/bin/streamlit run app.py
```

**14. Pothole Damage Detection**
```bash
cd /Users/falcon/Downloads/claude/Argus/pothole-damage-detection
.venv/bin/streamlit run app.py
```

**15. Crowd Density Estimator**
```bash
cd /Users/falcon/Downloads/claude/Argus/Crowd_Density_Estimator
.venv/bin/streamlit run app.py
```

**16. Emergency Vehicle Detection (Ambulance/Firetruck)**
```bash
cd /Users/falcon/Downloads/claude/Argus/EMERGENCY_VEHICLE_DETECTION
.venv/bin/streamlit run app.py
```

---
### Command Line Utilities (No GUI)

**17. Person Re-ID Search** (Search for a specific person across multiple video files)
```bash
cd /Users/falcon/Downloads/claude/Argus/person-reid-search
.venv/bin/python reid_search.py --video1 path/to/cam1.mp4 --video2 path/to/cam2.mp4
```
