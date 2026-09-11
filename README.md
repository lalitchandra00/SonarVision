<div align="center">

# SonarVision

**End-to-end side-scan sonar object detection**

Cleaner input · YOLOv8s detector · 4 inference modes · deployable REST API

</div>

SonarVision detects submerged objects — pipes, wrecks, cylinders, ghost nets, aircraft and humans — in side-scan sonar imagery. Because raw sonar frames are dominated by speckle, salt-and-pepper and Gaussian noise, the project couples a sonar-specific denoising stage directly to the detector: **the exact same preprocessing used during training is applied again at inference time**, so the model always sees clean, contrast-stable input.

The system ships as a complete, reproducible ML pipeline:

- **Jupyter notebooks** for noise filtering, batch dataset preparation, training, and all prediction modes;
- **Trained model artifacts** — `best.pt` (Ultralytics) and `best.onnx` (runtime-optimized);
- **A pure-ONNX FastAPI service** that runs the model with `onnxruntime` (no PyTorch at runtime), small enough for free-tier Cloud hosts;
- **Raw sonar log ingestion** — parses `.xtf` survey logs, renders them to waterfall imagery, runs detection per tile, merges seam duplicates and geotags every detection.

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
│  ──▶ JSON: base64 annotated image(s) + structured detections                  │
└────────────────────────────────────────────────────────────────────────────┘
```

A single source of truth — `filter_noise()` / `preprocess_for_model()` in `backend/noise_filtering.ipynb` — is loaded by every consumer (training prep notebooks, prediction notebooks, CLI scripts and the FastAPI service), guaranteeing there is **no preprocessing mismatch** between training and inference.

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Sonar-aware preprocessing** | Median blur → bilateral filter → light CLAHE, applied identically in training and inference |
| **Detector** | YOLOv8s trained for 6 submarine-object classes |
| **Runtime-optimized inference** | Pure `onnxruntime` (`YOLOOnnx`), ~150 MB RSS vs ~1.5 GB for ultralytics+torch |
| **4 inference modes** | Single image, video, live webcam, and raw `.xtf`/`.jsf` survey logs |
| **Geotagged log detections** | Every shipwreck/target box mapped to ping-row, interpolated lat/lon/time → CSV + GeoJSON |
| **Tile stitching** | Large logs are sliced into 1024×1024 overlapping tiles and duplicate detections merged across seams |
| **Deployable service** | FastAPI + `render.yaml` blueprint — runs on Render's free tier (512 MB) |
| **Resumable training** | Per-epoch checkpoints to Google Drive; a dead Colab session just re-runs and picks up where it left off |
| **Reproducible** | Fixed seed, auto-derived `data.yaml`, committed artifacts and run metrics |

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

The training set is built from two paired Hugging Face sources. **`sonar_filtered_dataset`** provides the pre-denoised training/validation images; **`sonar_dataset`** provides the matching YOLO `.txt` labels:

| Dataset | URL | Size | Contents |
|---------|-----|------|----------|
| **Training images (filtered)** | [lalitchandra00/sonar_filtered_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_filtered_dataset) | ~1.5 GB | `noise_filtered_training.zip` — pre-denoised train/val images (1334 + 33) |
| **Labels** | [lalitchandra00/sonar_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_dataset) | ~6 GB | `dataset_final.zip` — original images + YOLO `.txt` label files |

Only the label `.txt` files are taken from `sonar_dataset`; all training images come from the noise-filtered set, matched 1-to-1 by filename. This pairing is produced by `backend/noise_filtering.ipynb`, whose `filter_archive_to()` denoises every image of `dataset_final.zip` into `noise_filtered_training/{train_filtered,val_filtered,test_filtered}/` — the release that is now stored on Hugging Face as `sonar_filtered_dataset`.

| Split | Filtered images | Labels |
|-------|-----------------|--------|
| Train | 1334 | 1334 |
| Val | 33 | 33 |

The training notebooks verify 1-to-1 image↔label pairing and **derive the 6 class names from the label files themselves** (robust to any class ordering), then write `data.yaml` at runtime.

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

All notebooks auto-install any missing dependency (`ultralytics`, `pyxtf`) and load the noise filter + model the same way, so **`Run All` is all that is needed**.

### Image prediction — `backend/predictions/image_prediction.ipynb`

Reads `input/image/*`, runs `filter_noise → preprocess_for_model → YOLO`, draws boxes only when a detection exceeds `CONF_THRESHOLD` (0.25), and saves annotated output.

### Video prediction — `backend/predictions/video_prediction.ipynb`

Frame-by-frame: noise-filter → predict (`conf=0.50`). A cleaned copy of the whole video goes to `output/noise_filter/.../videos/`; **only frames with detections** are saved as boxed PNGs to `output/predictions/video_prediction/`.

### Realtime prediction — `backend/predictions/realtime_prediction.ipynb`

Live webcam detection with an inline side-by-side view (raw left / model view right). Frames are throttled to ~1 FPS, raw and noise-filtered copies are archived, detection frames are saved, and the loop stops after `RUN_SECONDS` or `stop_loop = True`.

### Log prediction — `backend/predictions/log_prediction.ipynb` (`.xtf` surveys)

The end-to-end sonar-log workflow:

1. **Extract** — `pyxtf` parses the `.xtf`; sonar pings are sorted by time, stacked per channel (port/starboard), normalised to 16-bit and rendered to 8-bit waterfall strips (dB + percentile stretch).
2. **Tile** — strips are sliced into 1024×1024 tiles (128 px overlap).
3. **Filter & predict** — every tile passes through the same `preprocess_for_model` then the detector.
4. **Merge** — duplicate detections spanning tile seams are merged by IoU + row-closeness (`merge_detections`, `IOU_MERGE=0.35`).
5. **Visualise** — annotated tiles + boxed strip overviews.
6. **Geotag** — each box is mapped to a ping row; lat/lon/time are interpolated from the file's navigation packets → `detections.csv` and `detections.geojson`.
7. **Review video** — an optional MP4 of the survey.

Core helpers live in **`backend/sonar_ingest.py`** and are importable by any notebook, script, or the API (`extract_survey`, `render_waterfall`, `make_tiles`, `merge_detections`, `to_u16`, `stack_channel`).

---

## REST API

`api/predict_api.py` exposes every inference mode as a single FastAPI service. The model runs through a pure `onnxruntime` inference shim (`YOLOOnnx`) — no `torch`/`ultralytics` at runtime, keeping the resident set around **~150 MB** (vs ~1.5 GB) so it fits comfortably in a free-tier container.

Responses are JSON and contain **already-drawn images** as base64 data-URIs — vivid per-class bounding boxes with a black-backed `"class conf%"` label — plus structured detection metadata.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | App info, classes, default confidence, links to `/docs` |
| `GET`  | `/health` | Always 200; reports `model_loaded` / `model_error` |
| `GET`  | `/ready` | Readiness probe — 503 until the model has warmed up |
| `POST` | `/predict/image` | Upload one image → boxed PNG + detections |
| `POST` | `/predict/video` | Upload a video → boxed frames (1 frame / 5 s) + per-frame detections |
| `POST` | `/predict/realtime` | Upload a webcam image frame → instant detection (frontend capture loop) |
| `POST` | `/predict/log` | Upload `.xtf` / `.jsf` → boxed overviews + annotated tiles + merged, geotagged detections |

All prediction endpoints accept the uploaded file as multipart field `file` and an optional `conf` query parameter (`default 0.70`, range `0.01–1.0`).

### Example request

```bash
# Image
curl -X POST http://localhost:8000/predict/image \
     -F "file=@input/image/test1.jpeg;type=image/jpeg"

# Sonar survey log
curl -X POST http://localhost:8000/predict/log \
     -F "file=@input/logs/xtf_files/survey1.xtf;type=application/octet-stream"
```

### Example response (image)

```json
{
  "success": true,
  "width": 1280,
  "height": 720,
  "conf_threshold": 0.7,
  "elapsed_ms": 212.4,
  "detections": [
    {
      "class_id": 5,
      "class": "human",
      "confidence": 0.9132,
      "bbox": { "x1": 210.5, "y1": 84.1, "x2": 380.2, "y2": 201.9 }
    }
  ],
  "annotated_image": "data:image/png;base64,iVBORw0KGgo...",
  "annotated_image_path": "output/predictions/image_prediction/prediction_a1b2c3d4.png"
}
```

### Video response

```json
{
  "success": true,
  "fps": 30.0,
  "total_frames": 900,
  "frames_with_detections": 4,
  "conf_threshold": 0.7,
  "elapsed_ms": 8840.1,
  "frames": [
    { "frame_id": 150, "image": "data:image/png;base64,...", "image_path": "output/predictions/video_prediction/frame_000150.png", "detections": [...] }
  ]
}
```

### Log response

```json
{
  "success": true,
  "survey_name": "survey1",
  "conf_threshold": 0.7,
  "elapsed_ms": 48210.2,
  "channels": [
    {
      "label": "ch0",
      "side": "port",
      "width": 1105,
      "height": 6314,
      "overview_image": "data:image/png;base64,...",
      "detections": [...]
    }
  ],
  "tiles": [
    { "row0": 0, "row1": 1024, "image": "data:image/png;base64,...", "detections": [...] }
  ]
}
```

> **Runtime settings** (defined in `predict_api.py`): `CONF_THRESHOLD=0.70`, `TILE_SIZE=1024`, `TILE_OVERLAP=128`, `IOU_MERGE=0.35`. Videos are sampled 1 frame per 5 seconds so responses stay fast. The output directory falls back to `/tmp/sonarvision` automatically when the repo is read-only (free-tier Render mounts it so).

### Run the API locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn api.predict_api:app --host 0.0.0.0 --port 8000
# or
python api/predict_api.py
```

Interactive Swagger UI: <http://localhost:8000/docs>

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

output/
  predictions/                  annotated image / video / realtime frames
  log_prediction/
    images/                     annotated tiles + strip overviews
    geotag/                     detections.csv + detections.geojson
    video/                      optional MP4 survey review
  noise_filter/
    input_noise_filter/         cleaned copies of image / video / realtime inputs
    log_images/                 cleaned copies of survey tiles
```

---


---

## Reproducibility

| What | Where |
|------|-------|
| Training dataset | `lalitchandra00/sonar_filtered_dataset` |
| Dataset preparation | `backend/noise_filtering.ipynb` → `filter_archive_to()` (denoise + match labels) |
| Trained weights | `backend/best/best.pt`, `backend/best/best.onnx` (committed) + `MyDrive/SIH/SonarVision/backend/best/` |
| Training config | `backend/model.ipynb` CONFIG cell (`SEED=42`) |
| Noise filter params | `backend/noise_filtering.ipynb` → `filter_noise()` |
| Runtime metrics | `backend/runs/dataset_yolov8/results.csv` + `results.png` |
| API config | `api/predict_api.py` constants (`CONF_THRESHOLD`, `TILE_SIZE`, `IOU_MERGE`, ...) |
| Deployment manifest | `render.yaml` |

---
