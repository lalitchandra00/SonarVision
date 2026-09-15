<div align="center">

# SonarVision

**End-to-end side-scan sonar object detection**

Cleaner input · YOLOv8s detector · 4 input types · deployable REST API

</div>

SonarVision detects submerged objects — pipes, wrecks, cylinders, ghost nets, aircraft and humans — in side-scan sonar imagery. Because raw sonar frames are dominated by speckle, salt-and-pepper and Gaussian noise, the project couples a sonar-specific denoising stage directly to the detector: **the exact same preprocessing used during training is applied again at inference time**, so the model always sees clean, contrast-stable input.

The system ships as a complete, reproducible ML pipeline:

- **Jupyter notebooks** for noise filtering, batch dataset preparation, training, and all prediction modes;
- **Trained model artifacts** — `best.pt` (Ultralytics) and `best.onnx` (runtime-optimized);

detection.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [The Noise-Filtering Pipeline](#the-noise-filtering-pipeline)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Model Performance](#model-performance)
- [Prediction Modes](#prediction-modes)
- [REST API](#rest-api)
- [Deployment](#deployment)
- [Getting Started](#getting-started)
- [Project Layout](#project-layout)
- [Known Issues](#known-issues)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Architecture

```
┌──────────────────────────  OFFLINE (pre-training)  ──────────────────────────┐
│                                                                               │
│  Raw side-scan      Median blur → Bilateral → light CLAHE      Filtered      │
│  sonar images  ───▶  (speckle / salt-and-pepper / Gaussian) ─▶  dataset      │
│  (.jpeg/.png)              noise_filtering.ipynb                              │
└────────────────────────────────────────────┬──────────────────────────────────┘
                                             │  YOLO-format labels (original zip)
                                             ▼
┌────────────────────────────  TRAINING (Google Colab)  ──────────────────────┐
│                                                                               │
│  YOLOv8s  ←  data.yaml (auto-derived class names)  ←  filtered images + txt  │
│  imgsz=640 · batch 80→16 fallback · AMP · cos_lr · close_mosaic=10           │
│  per-epoch checkpoints → Google Drive (auto-resume)                           │
│  ──▶  backend/best/best.pt   +   best.onnx   +   training_summary.txt        │
└────────────────────────────────────────────────────────────┬──────────────────┘
                                                             │
┌────────────────────────────  INFERENCE  ────────────────────▼─────────────────┐
│                                                                               │
│  Image  POST /predict/image     Video  POST /predict/video                    │
│  Webcam POST /predict/realtime  Log    POST /predict/log  (.xtf / .jsf)       │
│                                                                               │
│        filter_noise() → preprocess_for_model() → YOLOv8 ONNX → NMS → boxes    │
│                                                                               │
│  ──▶ JSON: Cloudinary-hosted image URLs + structured detections               │
└────────────────────────────────────────────────────────────────────────────┘
```

A single source of truth — `filter_noise()` / `preprocess_for_model()` in `backend/noise_filtering.ipynb` — is loaded by every consumer (training prep notebooks, prediction notebooks, CLI scripts and the FastAPI service), guaranteeing there is **no preprocessing mismatch** between training and inference. Boxed prediction images are saved locally and uploaded to Cloudinary; the JSON carries their secure URLs.

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Sonar-aware preprocessing** | Median blur → bilateral filter → light CLAHE, applied identically in training and inference |
| **Detector** | YOLOv8s trained for 6 submarine-object classes |
| **Runtime-optimized inference** | Pure `onnxruntime` (`YOLOOnnx`), ~150 MB RSS vs ~1.5 GB for ultralytics+torch |
| **Cloudinary-hosted results** | Boxed prediction images uploaded to Cloudinary → secure URLs in JSON (no heavy base64 payloads) |
| **4 inference modes** | Single image, video, live webcam, and raw `.xtf`/`.jsf` survey logs |
| **Geotagged log detections** | Every shipwreck/target box mapped to ping-row, interpolated lat/lon/time → CSV + GeoJSON |
| **Tile stitching** | Large logs are sliced into 1024×1024 overlapping tiles and duplicate detections merged across seams |
| **Deployable service** | FastAPI + `render.yaml` blueprint — runs on Render's free tier (512 MB) |

---

## Repository Structure

```
SonarVision/
├── api/
│   └── predict_api.py            # FastAPI app — 4 endpoints, ONNX inference
├── backend/
│   ├── noise_filtering.ipynb     # filter_noise + preprocess_for_model (source of truth)
│   ├── model.ipynb               # YOLOv8s training on Colab with auto-resume
│   ├── sonar_ingest.py           # .xtf parsing → waterfalls → tiles → merged boxes
│   ├── best/
│   │   ├── best.pt               # Ultralytics weights (YOLOv8s, imgsz 640)
│   │   ├── best.onnx             # ONNX export — used by the API at runtime
│   │   └── training_summary.txt  # classes, best epoch, mAP scores
│   ├── runs/dataset_yolov8/      # metrics: results.csv, curves, confusion matrix, weights
│   └── predictions/
│       ├── image_prediction.ipynb    # single-image detection
│       ├── video_prediction.ipynb    # per-frame video detection
│       ├── realtime_prediction.ipynb # live webcam detection
│       └── log_prediction.ipynb      # .xtf logs → geotagged detections (CSV/GeoJSON/video)
├── input/
│   ├── image/                    # test images (test1–test10.*)
│   ├── video/                    # test video ("Side Scan Sonar.mp4")
│   ├── realtime/                 # captured webcam frames
│   ├── logs/xtf_files/           # drop .xtf survey logs here
│   └── log_images/<survey>/      # extracted waterfall strips, tiles, manifest, nav
├── output/ (inside .gitignore)
│   ├── predictions/              # annotated image / video / realtime frames
│   ├── log_prediction/           # annotated tiles, geotag/*.csv|*.geojson, review video
│   └── noise_filter/             # noise-filtered copies of every input
├── requirements.txt              # API runtime deps (headless OpenCV + onnxruntime)
├── render.yaml                   # Render Blueprint deployment config
├── .env                          # Cloudinary credentials (git-ignored, local only)
└── .gitignore
```

---

## The Noise-Filtering Pipeline

Side-scan sonar is corrupted by three noise types, each addressed with a dedicated stage in `filter_noise()`:

| Step | Operation | Parameters | Targets |
|------|-----------|------------|---------|
| 1 | Median blur | `ksize = 5` | Speckle / salt-and-pepper |
| 2 | Bilateral filter | `d = 9, σ_color = 50, σ_space = 50` | Residual Gaussian noise (edge-preserving) |
| 3 | Light CLAHE | `clipLimit = 1.2, tileGridSize = 16×16` | Gentle local contrast, no texture boosting |

Cleanup runs on a single grayscale channel (sonar is effectively monochrome — faster and numerically stable), then the result is merged back to a 3-channel BGR image.

### Key functions (`backend/noise_filtering.ipynb`)

- **`filter_noise(image, ...)`** — pure denoising; accepts an ndarray or file path, returns uint8 BGR 3-channel (optionally `float32` in `[0,1]` via `to_float`).
- **`preprocess_for_model(image, target_w=1024, target_h=1024, ...)`** — denoise **and** resize to a model-ready tensor (the prediction-time entry point used everywhere).
- **`filter_archive_to(archive, subset, out_dir)`** — batch-filters every image of a given zip subset straight from memory.

> A full-range global stretch (`stretch=True`) is **off by default** — the current imagery is already fairly clean, and aggressive stretching only re-amplifies faint seabed texture into visible grain.

### Loading the filter at inference time

Notebooks and the API load the functions directly from the notebook file:

```python
def load_noise_filter_from_nb(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = {"__name__": "noise_filtering_mod", "cv2": cv2, "np": np, "Path": Path}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def filter_noise" in source or "def preprocess_for_model" in source:
            exec(source, ns)
    return ns
```

This guarantees training prep and every inference path use identical code.

---

## Dataset

Training data (original images + YOLO labels):
- [lalitchandra00/sonar_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_dataset)
- [lalitchandra00/sonar_filtered_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_filtered_dataset)

The class names are derived from the label files themselves (robust to any class ordering) and written to `data.yaml` at runtime.

### Classes detected

| ID | Class |
|----|-------|
| 0 | `pipe` |
| 1 | `shipwrecks` |
| 2 | `cylinder` |
| 3 | `ghostnet` |
| 4 | `plane` |
| 5 | `human` |

---

## Training the Model

Training runs in **`backend/model.ipynb`** on Google Colab with automatic resume — every finished epoch copies `last.pt` + a `state.json` to Google Drive, so a killed session simply re-runs the notebook and continues from the last epoch.

### Configuration (`model.ipynb` CONFIG cell)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_SIZE` | `s` | YOLOv8s — good T4 balance |
| `EPOCHS` | 100 | Early-stopping patience usually converges earlier |
| `PATIENCE` | 30 | Stop if val mAP50-95 plateaus |
| `IMGSZ` | 640 | Input size (source images are 1280) |
| `BATCH` | 80 | Auto-fallback 80 → 40 → 20 → 16 on CUDA OOM |
| `WORKERS` | 2 | |
| `SEED` | 42 | Reproducibility |
| `cos_lr` | True | Cosine annealing schedule |
| `close_mosaic` | 10 | Disable mosaic in the last 10 epochs |
| `amp` | True | Mixed precision |
| `RESUME` | True | Auto-resume from `training_checkpoints/last.pt` |




### Exported artifacts

- `backend/best/best.pt` — Ultralytics weights.
- `backend/best/best.onnx` — ONNX export (same `imgsz`); **this is what the API loads**.
- `backend/best/training_summary.txt` — class list, best epoch, mAP scores.

---

## Model Performance

Recorded in `backend/best/training_summary.txt` (also visible in `runs/dataset_yolov8/results.png`):

| Metric | Value |
|--------|-------|
| Model | YOLOv8s (`imgsz=640`) |
| Best epoch | 88 |
| Val mAP50 | **0.8506** |
| Val mAP50-95 | **0.6661** |
| Total epochs | 100 |
| Batch | 80 |

---

## Prediction Modes

All notebooks auto-install any missing dependency (`ultralytics`, `pyxtf`) and load the noise filter + model the same way**.

### Image prediction — `backend/predictions/image_prediction.ipynb`

Reads `input/image/*`, runs `filter_noise → preprocess_for_model → YOLO`, draws boxes that pass their  confidence threshold which is 0.70, and saves annotated output.

### Video prediction — `backend/predictions/video_prediction.ipynb`

Frame-by-frame: noise-filter → predict. A cleaned copy of the whole video goes to `output/noise_filter/.../videos/`; **only frames with detections** are saved as boxed PNGs to `output/predictions/video_prediction/`.

### Realtime prediction — `backend/predictions/realtime_prediction.ipynb`

Live webcam detection for now (using webcam as drone proxy) with an inline side-by-side view (raw left / model view right). Frames are throttled to ~1 FPS (can be adjusted accordng to user), raw and noise-filtered copies are archived, detection frames are saved, and the loop stops after `RUN_SECONDS` or `stop_loop = True`.

### Log prediction — `backend/predictions/log_prediction.ipynb` (`.xtf` surveys) (work in progress for this file type, 70% - 80% is completed)

The end-to-end sonar-log workflow:

1. **Extract** — `pyxtf` parses the `.xtf`; sonar pings are sorted by time, stacked per channel (port/starboard), normalised to 16-bit and rendered to 8-bit waterfall strips (dB + percentile stretch).
2. **Tile** — strips are sliced into 1024×1024 tiles (128 px overlap).
3. **Filter & predict** — every tile passes through the same `preprocess_for_model` then the detector.
4. **Merge** — duplicate detections spanning tile seams are merged by IoU + row-closeness (`merge_detections`, `IOU_MERGE=0.35`).
5. **Visualise** — annotated tiles + boxed strip overviews.
6. **Geotag** — each box is mapped to a ping row; lat/lon/time are interpolated from the file's navigation packets → `detections.csv` and `detections.geojson`.
7. **Review video** — an optional MP4 of the survey.


---

## REST API

`api/predict_api.py` exposes every inference mode as a single FastAPI service. The model runs through a pure `onnxruntime` inference shim (`YOLOOnnx`) — no `torch`/`ultralytics` at runtime, keeping the resident set around **~150 MB** (vs ~1.5 GB) so it fits comfortably in a free-tier container.

Responses are JSON: structured detection metadata plus **already-drawn images** (vivid per-class bounding boxes with a black-backed `"class conf%"` label) uploaded to **Cloudinary**. The JSON carries their secure `https://res.cloudinary.com/...` URLs instead of base64 payloads, so responses stay small even for videos and long sonar logs.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | App info, classes, default confidence, links to `/docs` |
| `GET`  | `/health` | Always 200; reports `model_loaded` / `model_error` |
| `GET`  | `/ready` | Readiness probe — 503 until the model has warmed up |
| `POST` | [`/predict/image`](https://sonarvision.onrender.com/docs#/default/predict_image_predict_image_post) | Upload one image → Cloudinary URL of the boxed image + detections |
| `POST` | [`/predict/video`](https://sonarvision.onrender.com/docs#/default/predict_video_predict_video_post) | Upload a video → Cloudinary URLs of all boxed frames (1 frame / 3 s) + per-frame detections |
| `POST` | [`/predict/realtime`](https://sonarvision.onrender.com/docs#/default/predict_realtime_predict_realtime_post) | Upload a webcam image (using webcam as drone proxy) frame → instant detection (one JSON per frame, frontend capture loop) |
| `POST` | [`/predict/log`](https://sonarvision.onrender.com/docs#/default/predict_log_predict_log_post) | Upload `.xtf` / `.jsf` → Cloudinary URLs of boxed overviews + tiles + merged, geotagged detections |

All prediction endpoints accept the uploaded file as multipart field `file` and an optional `conf` query parameter (`default 0.70`, range `0.01–1.0`). The live service is hosted at **https://sonarvision.onrender.com** — you can test every endpoint interactively from its `#/default/...` page in [the Swagger UI](https://sonarvision.onrender.com/docs).



### Example response (image)

```json
{
  "success": true,
  "width": 1280,
  "height": 1280,
  "conf_threshold": 0.7,
  "elapsed_ms": 7452.7,
  "detections": [
    {
      "class_id": 2,
      "class": "cylinder",
      "confidence": 0.7265,
      "bbox": {
        "x1": 978.63,
        "y1": 561.81,
        "x2": 1016.71,
        "y2": 607.58
      }
    }
  ],
  "annotated_image_url": "https://res.cloudinary.com/heavycloud/image/upload/v1789460332/SonarVision/image/prediction_3c63662e.png"
}

```


### Cloudinary setup

Boxed prediction images are uploaded to your Cloudinary account and their secure URLs are returned in the JSON. Credentials are read from the environment (automatically loaded from `<repo>/.env`, which is git-ignored):

```bash
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
# optional: parallel upload workers for video/log batches
SONARVISION_UPLOAD_WORKERS=8
```

Alternatively a single `CLOUDINARY_URL` (`cloudinary://api_key:api_secret@cloud_name`) can be used instead of the three fields.

- Images land in sub-folders mirroring the input type: `SonarVision/image`, `SonarVision/realtime`, `SonarVision/video`, `SonarVision/logs` — with unique per-request IDs (no overwrites).
- Video and log images are uploaded **in parallel** (`SONARVISION_UPLOAD_WORKERS`, default 8) so all URLs are available in a single response.
- If credentials are missing or an upload fails, the corresponding URL is `null` and the prediction still succeeds.
- The boxed images are also saved locally under `output/predictions/` as an archive.

### Run the API locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn api.predict_api:app --host 0.0.0.0 --port 8000
# or
python api/predict_api.py
```

Interactive Swagger UI: <https://sonarvision.onrender.com/docs>

---

## Deployment

The repo includes **`render.yaml`** — a Render Blueprint that provisions a Python web service. To deploy:

1. Push this repository to GitHub.
2. In the [Render dashboard](https://dashboard.render.com), choose **New → Blueprint** and select the repository.
3. Render installs the root `requirements.txt`, sets `PYTHON_VERSION=3.11.9`, and starts:
   ```bash
   python -m uvicorn api.predict_api:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120
   ```
4. The health probe hits `/health` (always 200); the service is ready once `/ready` returns 200.
5. Add the **Cloudinary env vars** in the Render dashboard (Settings → Environment): `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` (or single `CLOUDINARY_URL`), plus optional `SONARVISION_UPLOAD_WORKERS`. Without them the API still runs — image URLs are simply `null`.

Free-tier deployment notes baked into the repo:

- **Headless OpenCV** (`opencv-python-headless`) — no `libGL.so.1` dependency on Linux.
- **`MPLBACKEND=Agg`** wherever matplotlib is imported — no display server required.
- **`YOLO_OFFLINE=1` / `ULTRALYTICS_OFFLINE=1`** — no telemetry / update checks at startup.
- **Background model warmup** — the server accepts connections immediately and loads the ONNX model off the critical path; a warmup failure surfaces as a `503` with the error in `/health` instead of crashing the worker.
- **Read-only filesystem handling** — annotated outputs fall back to `/tmp` when `output/` is not writable.

---


## Project Layout

```
input/
  image/                        test images (test1.jpeg ... test10.jpg)
  video/                        test video (Side Scan Sonar.mp4)
  realtime/                     captured webcam frames
  logs/xtf_files/               .xtf survey logs for log_prediction
  log_images/<survey>/          extracted strips, tiles, manifest.json, nav.npz

```


---

## Reproducibility

| What | Where |
|------|-------|
| Training dataset | [lalitchandra00/sonar_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_dataset) + [lalitchandra00/sonar_filtered_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_filtered_dataset) |
| Dataset preparation | `backend/noise_filtering.ipynb` → `filter_archive_to()` (denoise + match labels) |
| Trained weights | `backend/best/best.pt`, `backend/best/best.onnx` |
| Training config | `backend/model.ipynb` CONFIG cell (`SEED=42`) |
| Noise filter params | `backend/noise_filtering.ipynb` → `filter_noise()` |
| Runtime metrics | `backend/runs/dataset_yolov8/results.csv` + `results.png` |
| API config | `api/predict_api.py` constants (`CONF_THRESHOLD`, `TILE_SIZE`, `IOU_MERGE`, ...) |
| Image hosting | Cloudinary credentials in `.env` / environment (`CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`) |
| Deployment manifest | `render.yaml` |

---
