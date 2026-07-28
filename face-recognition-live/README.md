# Real-Time Face Match (CPU, cross-platform)

Drop in a reference photo of a person, and find them live in a webcam feed (or a video
file). Runs on CPU via ONNX Runtime — works on Windows, Linux, and macOS (including
Apple Silicon).

Built on [InsightFace](https://github.com/deepinsight/insightface) (SCRFD detector + ArcFace embeddings).

Two ways to run it:
- **`app.py`** — a Streamlit web dashboard: drag-and-drop a photo in the browser, click Start, see live matches.
- **`match_face.py`** — a plain CLI/OpenCV-window version.

## 1. Setup

```bash
cd face-recognition-live
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

The first run will auto-download the model pack (a few MB to ~300MB depending on which
pack you choose) into `~/.insightface/models/`.

## 2a. Run the web dashboard (recommended)

```bash
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501`. Your browser will ask for webcam
permission — allow it. Then:

1. Drop a reference photo in the left panel.
2. Click **Start** under the live feed on the right.
3. Green box + "MATCH" = that person is in frame. Red = not a match.

All settings (model pack, detector size, frame width, frame skipping, match threshold)
are adjustable live from the sidebar — handy for tuning speed vs. accuracy for your CPU.

Everything runs locally in this process — no images or video are sent anywhere.

## 2b. Run the CLI version

```bash
python match_face.py --reference person.jpg
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Webcam device index |
| `--video` | none | Use a video file instead of the webcam |
| `--threshold` | `0.55` | Cosine similarity cutoff for a "match" (raise for stricter matching) |
| `--model-pack` | `buffalo_sc` | `buffalo_sc` (fastest), `buffalo_s`, or `buffalo_l` (most accurate, slowest) |
| `--det-size` | `320` | Detector input resolution — lower is faster, less accurate on small/far faces |
| `--frame-width` | `640` | Frames are downscaled to this width before processing (biggest FPS lever) |
| `--process-every` | `1` | Only run detection every N frames; boxes hold between skipped frames |

Press **q** to quit, **s** to save a screenshot (saved to `./captures/`).

## Tuning tips

- **Threshold**: 0.55 is a reasonable starting point for `buffalo_sc`/`buffalo_s`. If you get
  false positives, raise it (e.g. 0.6–0.65). If it's missing real matches, lower it slightly.
- **Reference photo**: use a clear, front-facing, well-lit, single-face photo for best results.
- **FPS**: the two biggest levers are frame width (smaller = faster) and detector size
  (smaller = faster but worse at detecting small/distant faces). "Run detection every N frames"
  roughly multiplies perceived FPS at the cost of slightly laggier box updates.
- **Multiple people**: the app scores every detected face in the frame against the
  reference — it will box everyone but only mark the matching person in green.

## Notes on licensing

InsightFace's code is MIT-licensed, but the pretrained model weights are for
non-commercial research use. For commercial deployment, review InsightFace's model
licensing terms or train/fine-tune your own weights.
