"""SonarVision prediction REST API.

Exposes the four inference modes from ``backend/predictions/*.ipynb`` as a
single FastAPI service:

    POST /predict/image     - one image upload  -> boxed image + detections
    POST /predict/video     - one video upload  -> boxed frame images + detections
    POST /predict/realtime  - captured clip     -> boxed frame images (same core as video)
    POST /predict/log       - .xtf/.jsf log     -> boxed tiles + strip overviews + detections

Every response is JSON with base64-encoded, already-drawn images: vivid
per-class bounding boxes with a black-backed "class confidence%" label on the
top-left of every box (matching the style used in the prediction notebooks).

Run:
    python -m uvicorn api.predict_api:app --host 0.0.0.0 --port 8000
or  python api/predict_api.py

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from backend.sonar_ingest import (  # noqa: E402
    extract_survey,
    make_tiles,
    merge_detections,
    render_waterfall,
)

CONF_THRESHOLD = 0.70
TILE_SIZE = 1024
TILE_OVERLAP = 128
IOU_MERGE = 0.35
ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_LOG_EXTS = {".xtf", ".jsf"}
IMAGE_OUTPUT_DIR = ROOT / "output" / "predictions" / "image_prediction"
VIDEO_OUTPUT_DIR = ROOT / "output" / "predictions" / "video_prediction"
REALTIME_OUTPUT_DIR = ROOT / "output" / "predictions" / "realtime_prediction" / "image"

# --------------------------------------------------------------------------
# Model / noise-filter loading (same source of truth as the notebooks)
# --------------------------------------------------------------------------

_model = None
_CLASS_NAMES = {}
_CLASS_COLORS = {}
_noise_ns = {}
_model_lock = threading.Lock()
_warm_thread = None


def _load_noise_filter_from_nb(nb_path: Path) -> dict:
    import json

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = {"__name__": "noise_filtering_mod", "cv2": cv2, "np": np, "Path": Path}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def filter_noise" in source or "def preprocess_for_model" in source:
            exec(source, ns)  # noqa: S102 - trusted notebook code
    return ns


def gen_colors(n):
    """Vivid per-class BGR colors - visible on black sonar backgrounds."""
    import colorsys

    return {
        i: tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0)[::-1])
        for i in range(n)
    }


def draw_boxes(img, boxes, label_map, class_colors, border_thick=4, text_scale=1.2, text_thick=3):
    """Draw vivid class-colored boxes with a larger, black-backed 'class conf%' label.

    ``boxes`` is a list of dicts: {x1,y1,x2,y2,class_id,confidence}.
    """
    if not boxes:
        return img
    for b in boxes:
        x1 = int(round(b["x1"]))
        y1 = int(round(b["y1"]))
        x2 = int(round(b["x2"]))
        y2 = int(round(b["y2"]))
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img.shape[1] - 1)
        y2 = min(y2, img.shape[0] - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        cls_id = int(b["class_id"])
        color = class_colors.get(cls_id, (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, border_thick)
        lbl = "{} {:.0f}%".format(
            label_map.get(cls_id, cls_id), float(b["confidence"]) * 100
        )
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thick)
        ty = max(y1 - 12, th + 8)
        cv2.rectangle(
            img,
            (x1, ty - th - 6),
            (min(x1 + tw + 12, img.shape[1] - 1), ty + 6),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            lbl,
            (x1 + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            text_thick,
        )
    return img


def init_model():
    global _model, _CLASS_NAMES, _CLASS_COLORS, _noise_ns
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:
            return
        from ultralytics import YOLO

        best = None
        for cand in (ROOT / "backend" / "best" / "best.pt", ROOT / "backend" / "best.pt"):
            if cand.exists():
                best = cand
                break
        if best is None:
            raise RuntimeError("best.pt not found under backend/")

        nf_nb = ROOT / "backend" / "noise_filtering.ipynb"
        if not nf_nb.exists():
            nf_nb = ROOT / "noise_filtering.ipynb"
        if not nf_nb.exists():
            raise RuntimeError("noise_filtering.ipynb not found")

        _model = YOLO(str(best))
        _CLASS_NAMES = _model.names
        _CLASS_COLORS = gen_colors(len(_CLASS_NAMES))
        _noise_ns = _load_noise_filter_from_nb(nf_nb)


def get_model():
    if _model is None:
        init_model()
    return _model


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _data_uri(img_bgr) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _save_annotated_image(img_bgr, output_dir: Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if not cv2.imwrite(str(output_path), img_bgr):
        raise HTTPException(status_code=500, detail="could not save annotated image")
    return str(output_path.relative_to(ROOT)).replace("\\", "/")


def _read_upload(data: bytes, filename: str, allow: set, what: str) -> np.ndarray:
    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in allow:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported {what} type '{suffix}'; allowed: {sorted(allow)}",
        )
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"could not read {what}: not a decodable image")
    return img


def _save_upload(data: bytes, suffix: str) -> Path:
    tmp = tempfile.gettempdir()
    path = Path(tmp) / f"sonarvision_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return path


def _boxes_from_result(result) -> list:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append(
            {
                "class_id": int(b.cls[0]),
                "confidence": float(b.conf[0]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return out


def _scale_boxes(boxes, sx, sy):
    for b in boxes:
        b["x1"] *= sx
        b["x2"] *= sx
        b["y1"] *= sy
        b["y2"] *= sy
    return boxes


def _clean(result_boxes):
    return [
        {
            "class_id": b["class_id"],
            "class": _CLASS_NAMES.get(b["class_id"], str(b["class_id"])),
            "confidence": round(float(b["confidence"]), 4),
            "bbox": {
                "x1": round(float(b["x1"]), 2),
                "y1": round(float(b["y1"]), 2),
                "x2": round(float(b["x2"]), 2),
                "y2": round(float(b["y2"]), 2),
            },
        }
        for b in result_boxes
    ]


def _predict_tile(tile_bgr, conf) -> list:
    model = get_model()
    res = model(tile_bgr, conf=conf, verbose=False)[0]
    return _boxes_from_result(res)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(
    title="SonarVision Prediction API",
    version="1.0.0",
    description="Side-scan sonar object detection: upload image / video / captured clip / .xtf log and get boxed images with object name + confidence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _start_warmup():
    """Kick off model loading in a background thread so the server accepts
    connections immediately and /health never blocks on model load."""
    global _warm_thread
    if _warm_thread is not None and _warm_thread.is_alive():
        return
    _warm_thread = threading.Thread(target=init_model, name="model-warmup", daemon=True)
    _warm_thread.start()


@app.on_event("startup")
def _startup():
    _start_warmup()


@app.get("/")
def root():
    return {
        "app": "SonarVision Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "image": "/predict/image",
            "video": "/predict/video",
            "realtime": "/predict/realtime",
            "log": "/predict/log",
        },
        "classes": _CLASS_NAMES,
        "default_conf": CONF_THRESHOLD,
        "model": str((ROOT / "backend").resolve()),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "classes": _CLASS_NAMES,
        "default_conf": CONF_THRESHOLD,
    }


@app.get("/ready")
def ready():
    """Readiness probe: 503 until the model is loaded, 200 afterwards."""
    if _model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model still loading")
    return {
        "status": "ready",
        "model_loaded": True,
        "classes": _CLASS_NAMES,
        "default_conf": CONF_THRESHOLD,
    }


@app.post("/predict/image")
def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=1.0),
):
    """Upload one image -> JSON with detections + original-size boxed PNG (base64)."""
    t0 = time.time()
    raw = _read_upload(file.file.read(), file.filename or "", ALLOWED_IMG_EXTS, "image")

    clean_orig = _noise_ns["filter_noise"](raw)  # original resolution
    clean_sq = _noise_ns["preprocess_for_model"](raw)  # 1024x1024 for the model

    res = get_model()(clean_sq, conf=conf, verbose=False)[0]
    boxes = _boxes_from_result(res)
    sx = clean_orig.shape[1] / clean_sq.shape[1]
    sy = clean_orig.shape[0] / clean_sq.shape[0]
    boxes = _scale_boxes(boxes, sx, sy)

    annot = clean_orig.copy()
    if boxes:
        annot = draw_boxes(annot, boxes, _CLASS_NAMES, _CLASS_COLORS)
    output_path = _save_annotated_image(annot, IMAGE_OUTPUT_DIR, "prediction.png")

    return {
        "success": True,
        "width": clean_orig.shape[1],
        "height": clean_orig.shape[0],
        "conf_threshold": conf,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "detections": _clean(boxes),
        "annotated_image": _data_uri(annot),
        "annotated_image_path": output_path,
    }


def _predict_video_core(data: bytes, conf: float, suffix: str = ".mp4", output_dir: Path = VIDEO_OUTPUT_DIR) -> dict:
    t0 = time.time()
    tmp = _save_upload(data, suffix)
    try:
        cap = cv2.VideoCapture(str(tmp))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="could not open video file")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = 0
        frames = []
        frame_id = 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            total += 1
            clean = _noise_ns["filter_noise"](raw)
            boxes = _boxes_from_result(get_model()(clean, conf=conf, verbose=False)[0])
            if boxes:
                annot = draw_boxes(clean.copy(), boxes, _CLASS_NAMES, _CLASS_COLORS)
                output_path = _save_annotated_image(
                    annot, output_dir, "frame_{:06d}.png".format(frame_id)
                )
                frames.append(
                    {
                        "frame_id": frame_id,
                        "image": _data_uri(annot),
                        "image_path": output_path,
                        "detections": _clean(boxes),
                    }
                )
            frame_id += 1
        cap.release()
        return {
            "success": True,
            "fps": round(float(fps), 2),
            "total_frames": total,
            "frames_with_detections": len(frames),
            "conf_threshold": conf,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "frames": frames,
        }
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/predict/video")
def predict_video(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=1.0),
):
    """Upload a video -> JSON with per-frame detections + boxed frame PNGs (only frames with detections)."""
    data = file.file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in ALLOWED_VID_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported video type '{suffix}'; allowed: {sorted(ALLOWED_VID_EXTS)}",
        )
    return _predict_video_core(data, conf, suffix, VIDEO_OUTPUT_DIR)


@app.post("/predict/realtime")
def predict_realtime(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=1.0),
):
    """Upload a captured clip (frontend webcam stream) -> boxed frame images + detections."""
    data = file.file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    suffix = Path(file.filename or "clip.webm").suffix.lower()
    if suffix not in ALLOWED_VID_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported video type '{suffix}'; allowed: {sorted(ALLOWED_VID_EXTS)}",
        )
    return _predict_video_core(data, conf, suffix, REALTIME_OUTPUT_DIR)


@app.post("/predict/log")
def predict_log(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=1.0),
):
    """Upload a sonar .xtf/.jsf log -> boxed strip overviews + annotated tiles + detections."""
    t0 = time.time()
    suffix = Path(file.filename or "log.xtf").suffix.lower()
    if suffix not in ALLOWED_LOG_EXTS:
        raise HTTPException(status_code=400, detail="only .xtf / .jsf files are supported")
    data = file.file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    tmp = _save_upload(data, suffix)
    try:
        survey = extract_survey(tmp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not parse sonar log: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)

    channels_out = []
    tiles_out = []
    for label, ch in survey["channels"].items():
        waterfall = render_waterfall(ch["array"])
        side = ch["side"]
        channel_dets = []  # merged, strip coordinates
        tiles = make_tiles(waterfall, TILE_SIZE, TILE_OVERLAP)
        tile_images = []

        for t in tiles:
            tile_bgr = _noise_ns["preprocess_for_model"](t["tile"])
            raw_boxes = _predict_tile(tile_bgr, conf)
            row0 = t["row0"]
            strip_boxes = []
            for b in raw_boxes:
                strip_boxes.append(
                    {
                        "x1": b["x1"],
                        "y1": b["y1"] + row0,
                        "x2": b["x2"],
                        "y2": b["y2"] + row0,
                        "score": b["confidence"],
                        "cls": b["class_id"],
                        "tile": str(t["row0"]),
                        "channel": label,
                    }
                )
            channel_dets.extend(strip_boxes)
            tile_images.append((t, raw_boxes))

        merged = merge_detections(channel_dets, IOU_MERGE, max(TILE_OVERLAP * 0.5, 1.0))

        # boxed strip overview
        strip_bgr = cv2.cvtColor(waterfall, cv2.COLOR_GRAY2BGR)
        strip_h = strip_bgr.shape[0]
        ch_boxes = [
            {"x1": m["x1"], "y1": m["y1"], "x2": m["x2"], "y2": m["y2"],
             "class_id": m["cls"], "confidence": m["score"]}
            for m in merged
        ]
        draw_boxes(strip_bgr, ch_boxes, _CLASS_NAMES, _CLASS_COLORS)
        if strip_bgr.shape[0] > 3000:
            s = 3000.0 / strip_bgr.shape[0]
            strip_bgr = cv2.resize(
                strip_bgr,
                (max(int(strip_bgr.shape[1] * s), 1), 3000),
                interpolation=cv2.INTER_AREA,
            )

        # boxed tiles
        tile_json = []
        for t, raw_boxes in tile_images:
            row0 = t["row0"]
            in_tile = []
            for m in merged:
                if m["y2"] >= row0 and m["y1"] < row0 + TILE_SIZE:
                    bb = {
                        "x1": m["x1"],
                        "y1": max(m["y1"] - row0, 0.0),
                        "x2": m["x2"],
                        "y2": min(m["y2"] - row0, float(TILE_SIZE)),
                        "class_id": m["cls"],
                        "confidence": m["score"],
                    }
                    in_tile.append(bb)
            tile_img = cv2.cvtColor(t["tile"], cv2.COLOR_GRAY2BGR)
            if in_tile:
                draw_boxes(tile_img, in_tile, _CLASS_NAMES, _CLASS_COLORS)
            tile_json.append(
                {
                    "row0": t["row0"],
                    "row1": t["row1"],
                    "image": _data_uri(tile_img),
                    "detections": _clean(in_tile),
                }
            )
        tiles_out.extend(tile_json)

        channels_out.append(
            {
                "label": label,
                "side": side,
                "width": strip_h and waterfall.shape[1],
                "height": strip_bgr.shape[0],
                "overview_image": _data_uri(strip_bgr),
                "detections": [
                    {
                        "class_id": m["cls"],
                        "class": _CLASS_NAMES.get(m["cls"], str(m["cls"])),
                        "confidence": round(float(m["score"]), 4),
                        "bbox": {
                            "x1": round(float(m["x1"]), 2),
                            "y1": round(float(m["y1"]), 2),
                            "x2": round(float(m["x2"]), 2),
                            "y2": round(float(m["y2"]), 2),
                        },
                    }
                    for m in merged
                ],
            }
        )

    return {
        "success": True,
        "survey_name": survey["name"],
        "conf_threshold": conf,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "channels": channels_out,
        "tiles": tiles_out,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)