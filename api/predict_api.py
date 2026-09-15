"""SonarVision prediction REST API.

Exposes the four inference modes from ``backend/predictions/*.ipynb`` as a
single FastAPI service:

    POST /predict/image     - one image upload  -> boxed image + detections
    POST /predict/video     - one video upload  -> boxed frame images + detections
    POST /predict/realtime  - captured clip     -> boxed frame images (same core as video)
    POST /predict/log       - .xtf/.jsf log     -> boxed tiles + strip overviews + detections

Every response is JSON with vivid per-class bounding boxes (black-backed
"class confidence%" labels) already drawn on the images. The boxed images are
stored locally and uploaded to Cloudinary; the JSON carries their secure URLs
instead of base64 payloads.

Cloudinary credentials are read from the environment (CLOUDINARY_URL or
CLOUDINARY_CLOUD_NAME / CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET, loaded from
`<repo>/.env`). If they are missing, uploads are skipped and URLs are ``null``.

Run:
    python -m uvicorn api.predict_api:app --host 0.0.0.0 --port 8000
or  python api/predict_api.py

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

# Force headless matplotlib backend BEFORE anything else imports it.
# Without this, importing matplotlib on a server without a display crashes
# the entire process, causing 502/503 on all routes including /docs.
import matplotlib
matplotlib.use("Agg")

import logging
import os
import sys
import tempfile
import threading
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("sonarvision.api")

load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

from backend.sonar_ingest import (  # noqa: E402
    extract_survey,
    make_tiles,
    merge_detections,
    render_waterfall,
)

CONF_THRESHOLD = 0.70

# Per-class confidence thresholds. Effective threshold per class is
# max(user conf, this class's configured value). Shipwrecks are kept at a high
# bar (0.98) to avoid false alarms; everything else uses the normal 0.70.
CLASS_CONF_THRESHOLDS = {
    0: 0.70,  # pipe
    1: 0.98,  # shipwrecks
    2: 0.70,  # cylinder
    3: 0.70,  # ghostnet
    4: 0.70,  # plane
    5: 0.70,  # human
}

TILE_SIZE = 1024
TILE_OVERLAP = 128
IOU_MERGE = 0.35
ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_LOG_EXTS = {".xtf", ".jsf"}

# Cloudinary hosting: boxed images are uploaded here and the secure URLs are
# returned in the JSON. Sub-folders mirror the input types (image/realtime/
# video/logs). Credentials come from the environment (see module docstring).
CLOUDINARY_FOLDER = "SonarVision"
CLOUDINARY_UPLOAD_WORKERS = max(1, int(os.getenv("SONARVISION_UPLOAD_WORKERS", "8")))

_cloudinary_lock = threading.Lock()
_cloudinary_configured = False


def _cloudinary_env_ready() -> bool:
    """True when full Cloudinary credentials are present in the environment."""
    return bool(
        os.getenv("CLOUDINARY_URL")
        or (
            os.getenv("CLOUDINARY_CLOUD_NAME")
            and os.getenv("CLOUDINARY_API_KEY")
            and os.getenv("CLOUDINARY_API_SECRET")
        )
    )


def _configure_cloudinary() -> None:
    """Single-time lazy configuration of the cloudinary SDK (thread-safe)."""
    global _cloudinary_configured
    if _cloudinary_configured:
        return
    with _cloudinary_lock:
        if _cloudinary_configured:
            return
        import cloudinary  # noqa: PLC0415 - lazy so import fails are surfaced by helper

        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        )
        _cloudinary_configured = True


def _upload_cloudinary(img_bgr: np.ndarray, public_id: str, folder: str) -> str | None:
    """Upload one annotated image to Cloudinary; return its secure URL.

    Returns ``None`` when credentials are missing or the upload fails
    (prediction still succeeds; only the URL is null).
    """
    if not _cloudinary_env_ready():
        return None
    try:
        _configure_cloudinary()
        import cloudinary.uploader  # noqa: PLC0415

        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            logger.warning("cloudinary: could not encode image %s", public_id)
            return None
        resp = cloudinary.uploader.upload(
            buf.tobytes(),
            folder=folder,
            public_id=public_id,
            overwrite=True,
            resource_type="image",
        )
        return resp.get("secure_url")
    except Exception as exc:  # noqa: BLE001 - one bad upload must not fail the request
        logger.warning("cloudinary upload failed (%s): %s", public_id, exc)
        return None


def _upload_many(items: list, folder: str) -> dict:
    """Upload several images in parallel; return ``{public_id: url_or_None}``.

    ``items`` is a list of ``(public_id, img_bgr)`` tuples. All uploads run
    concurrently so a request with many frames returns all URLs at once.
    """
    result: dict = {}
    if not items:
        return result
    if not _cloudinary_env_ready():
        return {pid: None for pid, _ in items}
    with ThreadPoolExecutor(max_workers=CLOUDINARY_UPLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_upload_cloudinary, img, pid, folder): pid
            for pid, img in items
        }
        for fut, pid in futures.items():
            result[pid] = fut.result()
    return result


def _effective_conf(conf: float) -> dict:
    """Per-class effective thresholds: max(user conf, configured default)."""
    return {
        c: max(conf, t)
        for c, t in CLASS_CONF_THRESHOLDS.items()
    }


def _writable_output_dir(preferred: Path) -> Path:
    """Return ``preferred`` if it is (or can be made) writable, otherwise
    fall back to a sub-directory of /tmp.  Render's free tier mounts the
    repo as read-only, so ``output/`` will not be writable there."""
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        test = preferred / ".write_test"
        test.touch()
        test.unlink()
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "sonarvision" / preferred.name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


IMAGE_OUTPUT_DIR = _writable_output_dir(ROOT / "output" / "predictions" / "image_prediction")
VIDEO_OUTPUT_DIR = _writable_output_dir(ROOT / "output" / "predictions" / "video_prediction")
REALTIME_OUTPUT_DIR = _writable_output_dir(ROOT / "output" / "predictions" / "realtime_prediction" / "image")

# --------------------------------------------------------------------------
# Model / noise-filter loading (same source of truth as the notebooks)
# --------------------------------------------------------------------------

_model = None
_CLASS_NAMES = {}
_CLASS_COLORS = {}
_noise_ns = {}
_model_lock = threading.Lock()
_warm_thread = None
_model_error: str | None = None  # set if warmup failed; surfaced in /health


def _load_noise_filter_from_nb(nb_path: Path) -> dict:
    ns = {"__name__": "noise_filtering_mod", "cv2": cv2, "np": np, "Path": Path}
    import json

    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def filter_noise" in source or "def preprocess_for_model" in source:
            exec(source, ns)  # noqa: S102 - trusted repository notebook code
    if "filter_noise" not in ns or "preprocess_for_model" not in ns:
        raise RuntimeError("noise notebook did not define filter_noise and preprocess_for_model")
    return ns


def gen_colors(n):
    """Vivid per-class BGR colors - visible on black sonar backgrounds."""
    import colorsys

    return {
        i: tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0)[::-1])
        for i in range(n)
    }


# --------------------------------------------------------------------------
# Pure onnxruntime YOLO inference  (no torch / ultralytics at runtime)
# --------------------------------------------------------------------------

class _Box:
    """Minimal shim so _boxes_from_result works without changes."""
    __slots__ = ("xyxy", "cls", "conf")

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 class_id: int, confidence: float) -> None:
        self.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
        self.cls  = [np.array(class_id,    dtype=np.float32)]
        self.conf = [np.array(confidence,  dtype=np.float32)]


class _Boxes:
    """Iterable container of _Box objects."""
    def __init__(self, box_list: list) -> None:
        self._list = [_Box(**b) for b in box_list]

    def __len__(self)  -> int:            return len(self._list)
    def __iter__(self):                   return iter(self._list)
    def __bool__(self) -> bool:           return bool(self._list)


class _Result:
    """Mimics the ultralytics result object returned by model()."""
    def __init__(self, box_list: list) -> None:
        self.boxes = _Boxes(box_list)


class YOLOOnnx:
    """Run a YOLOv8 ONNX model with onnxruntime — zero torch dependency.

    Memory footprint: ~150 MB vs ~1.5 GB for ultralytics+torch.
    Works on Render free tier (512 MB RAM).
    """

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        inp_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.ih = int(inp_shape[2]) if isinstance(inp_shape[2], int) else 640
        self.iw = int(inp_shape[3]) if isinstance(inp_shape[3], int) else 640

        # Extract class names stored in ONNX metadata by ultralytics exporter
        meta = self.session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            import ast
            raw = meta["names"]
            parsed = ast.literal_eval(raw)        # e.g. {0: 'mine', 1: 'wreck'}
            self.names: dict = {int(k): str(v) for k, v in parsed.items()}
        else:
            # Fallback: infer class count from output shape [1, 4+nc, anchors]
            out_shape = self.session.get_outputs()[0].shape
            nc = max(int(out_shape[1]) - 4, 1)
            self.names = {i: str(i) for i in range(nc)}

    # ------------------------------------------------------------------
    def __call__(self, img_bgr: np.ndarray,
                 conf: float = 0.5,
                 class_conf: dict | None = None,
                 verbose: bool = False) -> list:
        """Run inference; return [_Result] to match ultralytics interface.

        ``class_conf`` is an optional per-class override map {class_id: min_conf}.
        The effective threshold for a class is ``max(conf, class_conf[cls])``.
        """
        h0, w0 = img_bgr.shape[:2]

        # Pre-process: resize → RGB → [0,1] → [1,3,H,W]
        inp = cv2.resize(img_bgr, (self.iw, self.ih))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]          # [1, 3, H, W]

        raw = self.session.run(None, {self.input_name: inp})[0]  # [1, 4+nc, 8400]
        pred = raw[0]                                            # [4+nc, 8400]

        # YOLOv8 output is [4+nc, 8400]; some exporters transpose it.
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T                                         # → [8400, 4+nc]

        cx, cy, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        class_scores  = pred[:, 4:]                              # [8400, nc]
        class_ids     = np.argmax(class_scores, axis=1)          # [8400]
        confidences   = class_scores[np.arange(len(class_scores)), class_ids]

        # Per-anchor threshold: base ``conf``, raised per class where configured.
        thr = np.full(class_scores.shape[1], conf, dtype=np.float32)
        if class_conf:
            for c, t in class_conf.items():
                if 0 <= int(c) < len(thr):
                    thr[int(c)] = max(conf, t)

        mask = confidences >= thr[class_ids]
        if not mask.any():
            return [_Result([])]

        sx, sy = w0 / self.iw, h0 / self.ih
        x1 = (cx[mask] - pw[mask] / 2) * sx
        y1 = (cy[mask] - ph[mask] / 2) * sy
        x2 = (cx[mask] + pw[mask] / 2) * sx
        y2 = (cy[mask] + ph[mask] / 2) * sy
        cls   = class_ids[mask]
        confs = confidences[mask]

        # NMS per class using cv2.dnn (available in opencv-python-headless)
        boxes_out: list = []
        for c in np.unique(cls):
            idx  = cls == c
            # cv2.dnn.NMSBoxes wants [x, y, w, h]
            bx   = np.stack([x1[idx], y1[idx],
                             x2[idx] - x1[idx], y2[idx] - y1[idx]], axis=1)
            sc   = confs[idx].tolist()
            keep = cv2.dnn.NMSBoxes(bx.tolist(), sc,
                                    score_threshold=float(conf),
                                    nms_threshold=0.45)
            if len(keep) > 0:
                keep = np.asarray(keep).flatten()
                for k in keep:
                    boxes_out.append({
                        "x1": float(x1[idx][k]), "y1": float(y1[idx][k]),
                        "x2": float(x2[idx][k]), "y2": float(y2[idx][k]),
                        "class_id":   int(c),
                        "confidence": float(confs[idx][k]),
                    })

        return [_Result(boxes_out)]


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
        if isinstance(label_map, dict):
            class_name = label_map.get(cls_id, str(cls_id))
        else:
            class_name = label_map[cls_id] if cls_id < len(label_map) else str(cls_id)
        lbl = "{} {:.0f}%".format(class_name, float(b["confidence"]) * 100)
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
    global _model, _CLASS_NAMES, _CLASS_COLORS, _noise_ns, _model_error
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:
            return

        # Use ONNX model with pure onnxruntime — no torch/ultralytics needed.
        # This keeps memory ~150 MB (vs ~1.5 GB with torch) so it fits in
        # Render's free-tier 512 MB container.
        best_onnx = None
        for cand in (
            ROOT / "backend" / "best" / "best.onnx",
            ROOT / "backend" / "best.onnx",
        ):
            if cand.exists():
                best_onnx = cand
                break

        if best_onnx is None:
            raise RuntimeError(
                "best.onnx not found under backend/best/. "
                "Make sure the ONNX model is committed to the repo."
            )

        nf_nb = ROOT / "backend" / "noise_filtering.ipynb"
        if not nf_nb.exists():
            nf_nb = ROOT / "noise_filtering.ipynb"
        if not nf_nb.exists():
            raise RuntimeError("noise_filtering.ipynb not found")

        _model = YOLOOnnx(str(best_onnx))
        _CLASS_NAMES = _model.names
        _CLASS_COLORS = gen_colors(len(_CLASS_NAMES))
        _noise_ns = _load_noise_filter_from_nb(nf_nb)
        _model_error = None


def get_model():
    """Return the loaded model, or raise 503 if warmup failed."""
    if _model is None:
        if _model_error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not available: {_model_error}",
            )
        # Model is still loading in background thread — wait briefly
        init_model()
    return _model


def _require_noise_ns():
    """Raise 503 if the noise filter functions failed to load."""
    if not _noise_ns:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Noise filter not loaded yet. Model warmup may have failed. Check /health.",
        )
    return _noise_ns


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

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
    def class_name(class_id):
        if isinstance(_CLASS_NAMES, dict):
            return _CLASS_NAMES.get(class_id, str(class_id))
        return _CLASS_NAMES[class_id] if class_id < len(_CLASS_NAMES) else str(class_id)

    return [
        {
            "class_id": b["class_id"],
            "class": class_name(b["class_id"]),
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


def _predict_tile(tile_bgr, conf, class_conf=None) -> list:
    model = get_model()
    res = model(tile_bgr, conf=conf, class_conf=class_conf, verbose=False)[0]
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


def _warmup_safe():
    """Wrapper around init_model that catches all exceptions so a missing
    model file or bad notebook never kills the worker process.
    The error is stored in _model_error and surfaced via /health."""
    global _model_error
    try:
        init_model()
        logger.info("Model loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        _model_error = str(exc)
        logger.error("Model warmup failed: %s", exc)


def _start_warmup():
    """Kick off model loading in a background thread so the server accepts
    connections immediately and /health never blocks on model load."""
    global _warm_thread
    if _warm_thread is not None and _warm_thread.is_alive():
        return
    _warm_thread = threading.Thread(target=_warmup_safe, name="model-warmup", daemon=True)
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
        "class_conf_thresholds": _effective_conf(CONF_THRESHOLD),
        "model": str((ROOT / "backend").resolve()),
    }


@app.get("/health")
def health():
    """Always returns 200. Use model_loaded / model_error to check state."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_error": _model_error,
        "classes": _CLASS_NAMES,
        "default_conf": CONF_THRESHOLD,
        "class_conf_thresholds": _effective_conf(CONF_THRESHOLD),
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
        "class_conf_thresholds": _effective_conf(CONF_THRESHOLD),
    }


@app.post("/predict/image")
def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=1.0),
):
    """Upload one image -> JSON with detections + boxed image hosted on Cloudinary."""
    t0 = time.time()
    try:
        ns = _require_noise_ns()
        raw = _read_upload(file.file.read(), file.filename or "", ALLOWED_IMG_EXTS, "image")

        clean_orig = ns["filter_noise"](raw)
        clean_sq = ns["preprocess_for_model"](raw)

        res = get_model()(clean_sq, conf=conf, class_conf=_effective_conf(conf), verbose=False)[0]
        boxes = _boxes_from_result(res)
        sx = clean_orig.shape[1] / clean_sq.shape[1]
        sy = clean_orig.shape[0] / clean_sq.shape[0]
        boxes = _scale_boxes(boxes, sx, sy)

        annot = clean_orig.copy()
        if boxes:
            annot = draw_boxes(annot, boxes, _CLASS_NAMES, _CLASS_COLORS)
        # Use unique filename per request to avoid concurrent-request collisions
        img_token = uuid.uuid4().hex[:8]
        out_name = "prediction_{}.png".format(img_token)
        _save_annotated_image(annot, IMAGE_OUTPUT_DIR, out_name)
        annotated_image_url = _upload_cloudinary(
            annot, "prediction_" + img_token, CLOUDINARY_FOLDER + "/image"
        )

        return {
            "success": True,
            "width": clean_orig.shape[1],
            "height": clean_orig.shape[0],
            "conf_threshold": conf,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "detections": _clean(boxes),
            "annotated_image_url": annotated_image_url,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image prediction failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="image prediction failed: {}".format(exc)) from exc


def _predict_video_core(
    data: bytes,
    conf: float,
    suffix: str = ".mp4",
    output_dir: Path = VIDEO_OUTPUT_DIR,
    sample_interval: int = 1,          # 1 = every frame; N = every Nth frame
) -> dict:
    t0 = time.time()
    ns = _require_noise_ns()
    tmp = _save_upload(data, suffix)
    cap = None
    try:
        cap = cv2.VideoCapture(str(tmp))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="could not open video file")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = 0
        frames = []
        upload_items = []  # (public_id, img_bgr) uploaded in parallel after the loop
        frame_id = 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            total += 1
            # Skip frames that are not on the sample boundary
            if (frame_id % sample_interval) != 0:
                frame_id += 1
                continue
            clean = ns["filter_noise"](raw)
            boxes = _boxes_from_result(
                get_model()(clean, conf=conf, class_conf=_effective_conf(conf), verbose=False)[0]
            )
            if boxes:
                annot = draw_boxes(clean.copy(), boxes, _CLASS_NAMES, _CLASS_COLORS)
                img_token = uuid.uuid4().hex[:8]
                _save_annotated_image(
                    annot, output_dir, "frame_{}_{:06d}.png".format(img_token, frame_id)
                )
                upload_items.append((f"frame_{img_token}_{frame_id:06d}", annot))
                frames.append(
                    {
                        "frame_id": frame_id,
                        "image_url": None,  # filled after parallel upload
                        "detections": _clean(boxes),
                    }
                )
            frame_id += 1

        urls = _upload_many(upload_items, CLOUDINARY_FOLDER + "/video")
        for entry, (pid, _) in zip(frames, upload_items):
            entry["image_url"] = urls.get(pid)

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
        # Always release the capture and clean temp file
        if cap is not None:
            cap.release()
        tmp.unlink(missing_ok=True)


@app.post("/predict/video")
def predict_video(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=1.0),
):
    """Upload a video -> one JSON with per-frame detections + Cloudinary URLs (only frames with detections).

    Samples 1 frame every 3 seconds to keep response times fast.
    For a 30 fps video this means 1 frame every 90 frames.
    """
    data = file.file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in ALLOWED_VID_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported video type '{suffix}'; allowed: {sorted(ALLOWED_VID_EXTS)}",
        )

    # Read just enough of the file to detect FPS before passing full data
    # We write to a temp file to probe FPS, then reuse the same bytes.
    tmp_probe = _save_upload(data, suffix)
    try:
        probe = cv2.VideoCapture(str(tmp_probe))
        raw_fps = probe.get(cv2.CAP_PROP_FPS) if probe.isOpened() else 0.0
        probe.release()
    finally:
        tmp_probe.unlink(missing_ok=True)

    # Sample 1 frame every 3 seconds; fall back to every 90th frame if FPS unknown
    fps = raw_fps if raw_fps and raw_fps > 0 else 30.0
    sample_interval = max(1, int(round(fps * 3)))
    logger.info("Video prediction: fps=%.2f  sample_interval=%d (1 frame / 3 s)", fps, sample_interval)

    return _predict_video_core(data, conf, suffix, VIDEO_OUTPUT_DIR, sample_interval=sample_interval)


@app.post("/predict/realtime")
def predict_realtime(
    file: UploadFile = File(...),
    conf: float = Query(CONF_THRESHOLD, ge=0.01, le=1.0),
):
    """Receive a single image frame from the frontend webcam and return detections instantly.

    The frontend captures one frame every few seconds and sends it here as an image.
    This endpoint treats it exactly like /predict/image — one frame in, one prediction out.
    Supported formats: .jpg, .jpeg, .png, .bmp, .webp
    """
    t0 = time.time()
    try:
        ns = _require_noise_ns()
        data = file.file.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="empty upload")

        raw = _read_upload(data, file.filename or "frame.jpg", ALLOWED_IMG_EXTS, "image frame")

        clean_orig = ns["filter_noise"](raw)
        clean_sq   = ns["preprocess_for_model"](raw)

        res = get_model()(clean_sq, conf=conf, class_conf=_effective_conf(conf), verbose=False)[0]
        boxes = _boxes_from_result(res)
        sx = clean_orig.shape[1] / clean_sq.shape[1]
        sy = clean_orig.shape[0] / clean_sq.shape[0]
        boxes = _scale_boxes(boxes, sx, sy)

        annot = clean_orig.copy()
        if boxes:
            annot = draw_boxes(annot, boxes, _CLASS_NAMES, _CLASS_COLORS)

        img_token   = uuid.uuid4().hex[:8]
        out_name    = "realtime_{}.png".format(img_token)
        _save_annotated_image(annot, REALTIME_OUTPUT_DIR, out_name)
        annotated_image_url = _upload_cloudinary(
            annot, "realtime_" + img_token, CLOUDINARY_FOLDER + "/realtime"
        )

        return {
            "success":              True,
            "width":                clean_orig.shape[1],
            "height":               clean_orig.shape[0],
            "conf_threshold":       conf,
            "elapsed_ms":           round((time.time() - t0) * 1000, 1),
            "detections":           _clean(boxes),
            "annotated_image_url":  annotated_image_url,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Realtime frame prediction failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="realtime prediction failed: {}".format(exc)) from exc


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
    all_upload_items: list[tuple[str, np.ndarray]] = []
    # Track (channel_index, public_id) for overviews so we can fill URLs later
    overview_upload_map: list[tuple[int, str]] = []
    # Track (tile_index_in_tiles_out, public_id) for tiles
    tile_upload_map: list[tuple[int, str]] = []

    for label, ch in survey["channels"].items():
        ch_idx = len(channels_out)
        waterfall = render_waterfall(ch["array"])
        side = ch["side"]
        channel_dets = []  # merged, strip coordinates
        tiles = make_tiles(waterfall, TILE_SIZE, TILE_OVERLAP)
        tile_images = []

        ns = _require_noise_ns()
        for t in tiles:
            tile_bgr = ns["preprocess_for_model"](t["tile"])
            raw_boxes = _predict_tile(tile_bgr, conf, _effective_conf(conf))
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

        ov_token = uuid.uuid4().hex[:8]
        ov_pid = "overview_{}_{}".format(ov_token, label)
        all_upload_items.append((ov_pid, strip_bgr))
        overview_upload_map.append((ch_idx, ov_pid))

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
                    "image_url": None,  # filled after parallel upload
                    "detections": _clean(in_tile),
                }
            )
            tile_token = uuid.uuid4().hex[:8]
            tile_pid = "tile_{}_{}".format(tile_token, row0)
            all_upload_items.append((tile_pid, tile_img))
            tile_upload_map.append((len(tiles_out) + len(tile_json) - 1, tile_pid))

        tiles_out.extend(tile_json)

        def _cls_name(cls_id):
            """Safe class name lookup whether _CLASS_NAMES is a dict or list."""
            if isinstance(_CLASS_NAMES, dict):
                return _CLASS_NAMES.get(cls_id, str(cls_id))
            return _CLASS_NAMES[cls_id] if cls_id < len(_CLASS_NAMES) else str(cls_id)

        channels_out.append(
            {
                "label": label,
                "side": side,
                "width": int(waterfall.shape[1]),
                "height": int(strip_bgr.shape[0]),
                "overview_image_url": None,  # filled after parallel upload
                "detections": [
                    {
                        "class_id": m["cls"],
                        "class": _cls_name(m["cls"]),
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

    urls = _upload_many(all_upload_items, CLOUDINARY_FOLDER + "/logs")

    for ch_idx, ov_pid in overview_upload_map:
        channels_out[ch_idx]["overview_image_url"] = urls.get(ov_pid)

    for tile_idx, tile_pid in tile_upload_map:
        tiles_out[tile_idx]["image_url"] = urls.get(tile_pid)

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