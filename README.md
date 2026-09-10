# SonarVision

**Side-scan sonar object detection** powered by YOLOv8s with a custom noise-filtering pipeline.

Side-scan sonar images are inherently noisy — speckle, salt-and-pepper, and Gaussian noise degrade model performance. SonarVision addresses this end-to-end: a sonar-specific denoising stage precedes training and runs identically at inference, ensuring the model always sees clean, contrast-enhanced input.

---

## End-to-End Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐    ┌──────────────┐
│ Raw sonar   │───▶│  Noise       │───▶│  Dataset    │───▶│  Training      │───▶│  Prediction  │
│ images      │    │  filtering   │    │  prep       │    │  (YOLOv8s)     │    │              │
│             │    │              │    │             │    │  on Colab      │    │  img / vid / │
│ input/      │    │  noise_      │    │  filtered   │    │  + Drive       │    │  realtime    │
│ *.jpeg      │    │  filtering.  │    │  images +   │    │  checkpoints   │    │              │
│ *.mp4       │    │  ipynb       │    │  matched    │    │                │    │  best.pt     │
│             │    │              │    │  labels     │    │                │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────────┘    └──────────────┘
```

---

## Repository Structure

```
SonarVision/
├── backend/
│   ├── noise_filtering.ipynb       # Noise filter + batch dataset preprocessing
│   ├── model.ipynb                 # YOLOv8s training (Google Colab → Google Drive)
│   ├── best.pt                     # Trained model weights (served from GitHub)
│   ├── runs/                       # Training artifacts (ignored by .gitignore)
│   ├── training_checkpoints/       # Auto-resume checkpoints for Colab (ignored)
│   └── predictions/
│       ├── image_prediction.ipynb  # Single image inference
│       ├── video_prediction.ipynb  # Video frame-by-frame detection
│       └── realtime_prediction.ipynb  # Live webcam detection
├── input/
│   ├── image/                      # Test images (test1–test5.jpeg/jpg)
│   ├── video/                      # Test videos (Side Scan Sonar.mp4)
│   └── realtime/                   # Captured webcam frames
├── output/
│   ├── predictions/                # Annotated output (image / video / realtime)
│   ├── noise_filter/               # Noise-filtered copies of inputs
│   └── test_predictions/           # Dataset2 test-set results
├── requirements.txt                # Python dependencies (noise filtering only)
└── .gitignore
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the noise-filtering pipeline (NumPy, OpenCV, Matplotlib, tqdm).

> **Note:** `ultralytics` (YOLOv8) is installed automatically inside the training and prediction notebooks — it is not listed in `requirements.txt` because training is done on Google Colab.

### 2. Obtain the trained model

`backend/best.pt` is committed directly to this repository (~22.5 MB). If the file is missing or you need to retrain, see **Stage 3 — Training** below.

### 3. Run prediction

Open any of the three prediction notebooks in `backend/predictions/` and run all cells. Place your inputs in the corresponding `input/` subdirectory first.

---

## Datasets

Two Hugging Face datasets are used; the filtered dataset supplies clean images, and the original supplies YOLO-format labels:

| Dataset | URL | Size | Contents |
|---------|-----|------|----------|
| **Noise-filtered images** | [lalitchandra00/sonar_filtered_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_filtered_dataset) | ~1.5 GB | `noise_filtered_training.zip` — pre-denoised train/val images (1334 + 33) |
| **Original + labels** | [lalitchandra00/sonar_dataset](https://huggingface.co/datasets/lalitchandra00/sonar_dataset) | ~6 GB | `dataset_final.zip` — original images + YOLO `.txt` label files |

> Only **label `.txt` files** are extracted from the original dataset; all training images come from the noise-filtered set. Filenames are matched 1-to-1.

---

## Stage 1 — Noise Filtering (`backend/noise_filtering.ipynb`)

### What it does

Sonar images are cleaned in three steps, then resized to the model input size:

| Step | Operation | Parameters | Purpose |
|------|-----------|------------|---------|
| 1 | Median blur | `ksize=5` | Removes speckle / salt-and-pepper noise (sonar-typical) |
| 2 | Bilateral filter | `d=9, σColor=50, σSpace=50` | Edge-preserving smoothing; eliminates residual Gaussian noise |
| 3 | Light CLAHE | `clipLimit=1.2, tileGridSize=16×16` | Gentle local contrast enhancement (noise-safe) |

The result is converted to BGR 3-channel and resized to **1024×1024** (`preprocess_for_model`).

### Key functions

- **`filter_noise(image, ...)`** — pure filtering; accepts any ndarray, returns uint8 BGR 3-channel
- **`preprocess_for_model(image, target_w=1024, target_h=1024)`** — denoise + resize; output is model-ready
- **`filter_archive_to(archive, subset, out_dir)`** — batch filter every image of a given subset from the in-memory zip

### Modes

- **Local (Windows):** reads `dataset_final.zip` from disk into memory
- **Google Colab:** downloads `dataset_final.zip` from Hugging Face into memory

### Output layout

```
noise_filtered_training/
├── train_filtered/    # ~1334 filtered images
├── val_filtered/      # ~33 filtered images
└── test_filtered/     # filtered test set
```

### Usage at prediction time

All three prediction notebooks dynamically load `filter_noise` / `preprocess_for_model` from `noise_filtering.ipynb` via `load_noise_filter_from_nb()`. The same function is used in both training prep and inference — no preprocessing mismatch.

---

## Stage 2 — Dataset Preparation (`backend/model.ipynb`, cells 4–5)

The training notebook assembles the final dataset at runtime:

1. **Download filtered images** from Hugging Face → extract `train_filtered/` → `DATASET/train/images/`, `val_filtered/` → `DATASET/val/images/`
2. **Download original dataset** from Hugging Face → extract only `dataset_final/train/labels/*.txt` and `dataset_final/val/labels/*.txt` → `DATASET/{train,val}/labels/`
3. **Verify** every image has a matching label file (filename stem check, 1-to-1)
4. **Derive class names** dynamically from label file prefixes (robust to any ordering)
5. **Write `data.yaml`** at runtime

### Expected dataset sizes

| Split | Images | Labels | Total annotations |
|-------|--------|--------|-------------------|
| Train | 1334 | 1334 | 8004 (6 classes × 1334) |
| Val   | 33     | 33     | 198  (6 classes × 33)  |

---

## Stage 3 — Training (`backend/model.ipynb`)

Training runs on **Google Colab** with automatic checkpointing to **Google Drive** so it survives session resets.

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_SIZE` | `s` | YOLOv8s (small) — good T4 balance |
| `IMGSZ` | `640` | Training image size |
| `BATCH` | `16` | Reduce to 8 on low-VRAM GPUs |
| `EPOCHS` | `120` | Budget — early stopping stops sooner |
| `PATIENCE` | `30` | Stop if val mAP50-95 plateaus |
| `DEVICE` | auto | Detects CUDA T4; falls back to CPU |
| `cos_lr` | True | Cosine annealing LR schedule |
| `amp` | True | Mixed precision (fast + safe on T4) |
| `close_mosaic` | 10 | Disable mosaic augmentation last 10 epochs |
| `SEED` | 42 | Reproducibility |

### Drive layout (`MyDrive/SIH/SonarVision/backend/`)

| Path | Contents |
|------|----------|
| `runs/dataset_yolov8/` | Full run: `results.csv`, training graphs, confusion matrix, `weights/last.pt` |
| `training_checkpoints/` | `last.pt` + `state.json` — auto-resume source when Colab session dies |
| `best/` | `best.pt`, `best.onnx` (exported), `training_summary.txt` |

### Auto-resume

Every finished epoch copies `last.pt` to `training_checkpoints/`. If Colab crashes, re-running the notebook resumes from that checkpoint automatically. A `DONE` marker is written only after the full pipeline completes.

### Outputs

- `backend/best/best.pt` → best weights
- `backend/best/best.onnx` → ONNX export (same input size)
- `backend/best/training_summary.txt` — class list, best epoch, mAP scores

### Training metrics to check

After training, `results.csv` and `results.png` show per-epoch loss and validation mAP. Strong results: val mAP50 > 0.70. If mAP is low, try `MODEL_SIZE='m'` or `IMGSZ=1024`.

---

## Stage 4 — Prediction

### Image Prediction (`backend/predictions/image_prediction.ipynb`)

**Pipeline per image:**
1. Load raw image from `input/image/`
2. Noise-filter (`preprocess`) → 1024×1024 clean BGR
3. YOLOv8 prediction (`conf=0.25`)
4. Show annotated image (boxes only if detections found)
5. Save to `output/predictions/image_prediction/` + `output/noise_filter/input_noise_filter/images/`

### Video Prediction (`backend/predictions/video_prediction.ipynb`)

**Pipeline per frame:**
1. Read frame from video (`input/video/`)
2. Noise-filter (`filter_noise`)
3. YOLOv8 prediction (`conf=0.50`)
4. Write noise-filtered frame to cleaned video copy (`output/noise_filter/input_noise_filter/videos/`)
5. If detection found: save annotated frame to `output/predictions/video_prediction/`

### Realtime Prediction (`backend/predictions/realtime_prediction.ipynb`)

**Live webcam detection:**
1. Capture webcam frame
2. Noise-filter + YOLOv8 prediction (`conf=0.50`)
3. Inline side-by-side display (raw left, model view right)
4. Save raw frame to `input/realtime/`, noise-filtered to `output/noise_filter/input_noise_filter/real_time/`
5. Save detection frame to `output/predictions/realtime_prediction/image/`
6. Throttled to ~1 FPS; stops after `RUN_SECONDS` or set `stop_loop = True`

### Output paths

| Mode | Predictions | Noise-filtered |
|------|-------------|----------------|
| Image | `output/predictions/image_prediction/` | `output/noise_filter/input_noise_filter/images/` |
| Video | `output/predictions/video_prediction/` | `output/noise_filter/input_noise_filter/videos/` |
| Realtime | `output/predictions/realtime_prediction/image/` | `output/noise_filter/input_noise_filter/real_time/` |

---

## Known Issues & Fixes

These are in the prediction notebooks only and need to be updated manually:

### 1. Prediction notebooks point to old `OceanScan` root path

All three prediction notebooks have `ROOT = Path(r"D:\1. Project Program\1.SIH\OceanScan")`. Update to your SonarVision path:

```python
# Before (prediction notebooks)
ROOT = Path(r"D:\1. Project Program\1.SIH\OceanScan")

# After
ROOT = Path(r"D:\1. Project Program\1.SIH\SonarVision")
```

And update `BEST` and `NF_NB` paths accordingly:
```python
BEST = ROOT / "backend" / "best.pt"
NF_NB = ROOT / "backend" / "noise_filtering.ipynb"
```

### 2. Hardcoded 4 classes vs 6 training classes

Prediction notebooks define:
```python
CLASS_NAMES = {0: "body", 1: "plane", 2: "ship", 3: "target"}
```

Training derives **6 classes** from label filename prefixes. After training, read the class list from `backend/best/training_summary.txt` and update all three prediction notebooks accordingly.

### 3. `requirements.txt` does not include `ultralytics`

`ultralytics` is installed via `%pip install` inside each notebook. If you want to run predictions from a standalone `.py` script, add it to `requirements.txt`:

```
ultralytics>=8.2.40
```

---

## Reproducibility

| What | Where |
|------|-------|
| Training dataset | `lalitchandra00/sonar_filtered_dataset` + labels from `lalitchandra00/sonar_dataset` |
| Trained weights | `backend/best.pt` (this repo) + `MyDrive/SIH/SonarVision/backend/best/` (Google Drive) |
| Training config | `backend/model.ipynb` config cell |
| Noise filter params | `backend/noise_filtering.ipynb` (`filter_noise` function) |
| Seed | `SEED = 42` (set in training config) |
