# Live Person Tracker (whole-body, CPU, cross-platform)

Upload a reference photo, select a box around the person, and find them live in your
webcam feed — using whole-body appearance rather than a face. Useful when the camera is
far away and faces aren't resolvable.

Matching is a weighted blend of:
- a **real person re-identification embedding** (OSNet, via [BoxMOT](https://github.com/mikel-brostrom/boxmot) —
  a model actually trained to match people across cameras, not a generic ImageNet classifier),
- an **HSV clothing-color histogram** (what they're wearing),
- and a **height/width aspect ratio** comparison (body proportions).

All CPU-only, via YOLOv8n (person detection) + OSNet (appearance re-ID) + OpenCV color
histograms. Runs on Windows, Linux, and macOS.

## 1. Setup

```bash
cd person-tracking-live
python3 -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

**Requires Python 3.10–3.13** (BoxMOT's supported range).

First run downloads: the YOLOv8n weights (~6MB, from Ultralytics) and the OSNet ReID
weights (`osnet_x0_25_msmt17`, a few MB, auto-fetched by BoxMOT) — both are reliable,
no manual/gated Google-Drive downloads.

## 2. Run

```bash
streamlit run app.py
```

Opens `http://localhost:8501`. Allow webcam access when your browser asks. Then:

1. Upload a photo containing the person you want to track.
2. Use the **left↔right** and **top↔bottom** sliders to box them — the green rectangle
   updates live on the preview.
3. Click **"Use this box as reference"**.
4. Click **Start** under the live feed.
5. Green box + "MATCH" = that person is in frame. Red = a detected person who doesn't match.

Check the **Detection log** panel (click "Refresh log") for a timestamped history of matches.

## Tuning

| Setting | What it does |
|---|---|
| Detector confidence | How eager YOLO is to call something a "person" — lower catches more, but more false detections |
| Frame width | Downscale before processing — biggest FPS lever |
| Run detection every N frames | Skip frames for more FPS, at the cost of laggier box updates |
| Match threshold | Combined score cutoff for a "match" |
| ReID / color / ratio weights | How much each feature contributes to the match score — tune per scenario (e.g. raise color weight if the target's outfit is very distinctive) |

## What changed from the first version

- **Reference box selection** no longer depends on `streamlit-drawable-canvas` (that
  package's `image_to_url` call broke on current Streamlit). It's now two range sliders
  with a live rectangle preview — same result, no fragile dependency.
- **Matching backend** swapped from a generic ImageNet-pretrained MobileNetV3 to
  **OSNet**, a model actually trained for person re-identification (Market1501/MSMT17),
  loaded through BoxMOT's `ReidAutoBackend` — which auto-downloads the weights, sidestepping
  the gated Google-Drive checkpoints that made plain `torchreid` painful to install.
  Detections in each frame are now embedded in a single batched forward pass instead of
  one-by-one, which is also faster.

## Important limitations

- **Not face recognition.** This is still short/medium-term, same-outfit re-identification.
  Every public person-reID model (OSNet included) assumes the person doesn't change clothes;
  if they do, or lighting shifts drastically, matching degrades. It's meant for tracking
  someone through a session (minutes to hours), not "find this person any day, any outfit."
- **Detection is still the bottleneck at long range.** If YOLO can't detect a person at all
  (too small/far in frame), no amount of appearance matching will help — get closer or raise
  the camera's zoom/resolution.
- Everything runs locally — no images or video leave your machine.
