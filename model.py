"""
model.py

Utilities to:
- load a YOLO model from saved weights (local project folder)
- run inference on an image/frame
- save a copy of weights and optionally export ONNX/TorchScript

Used by app.py (Flask dashboard).
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import numpy as np
import cv2
from typing import List, Dict

# ---------------- CONFIG ----------------
# Root of your project (folder where app.py, best.pt, last.pt are kept)
PROJECT_DIR = Path(".").resolve()

# Folder where extra copies / exports will be stored
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"

# Fallback backbone if no custom weights are found
DEFAULT_BASE_WEIGHTS = "yolov8n.pt"  # or "yolov10n.pt" if you have it

# detection thresholds (tweak in app if you like)
CONFIDENCE_THRESHOLD = 0.35

# Whether to try exporting ONNX / TorchScript when dashboard starts
ENABLE_EXPORT = False  # keep False for smoother startup
# ----------------------------------------


def find_weights(project_dir: Path = PROJECT_DIR) -> Path | None:
    """
    Search for best.pt or last.pt in the current project folder first,
    then in saved_models, then any 'weights' subfolders.
    """
    # 1 Look in project root (where you placed best.pt / last.pt)
    for name in ("best.pt", "last.pt"):
        p = project_dir / name
        if p.exists():
            print(f"Using weights from project root: {p}")
            return p

    # 2) Look in saved_models
    if SAVED_MODELS_DIR.exists():
        pts = sorted(SAVED_MODELS_DIR.glob("*.pt"))
        if pts:
            print(f"Using weights from saved_models: {pts[0]}")
            return pts[0]

    # 3) Search for 'weights' folders under project_dir
    for root in project_dir.rglob("weights"):
        for candidate in Path(root).glob("*.pt"):
            # Prefer best/last if found
            if candidate.name in ("best.pt", "last.pt"):
                print(f"Using weights from run folder: {candidate}")
                return candidate
            return candidate

    print("No custom weights found; will fall back to default backbone.")
    return None


def load_model(weights: Path | None = None, device: str | None = None) -> YOLO:
    """
    Load ultralytics YOLO model.
    - weights: Path to .pt file. If None, load DEFAULT_BASE_WEIGHTS.
    - device: 'cpu' or '0' (GPU index). YOLO picks automatically if None.
    """
    if weights is None:
        print("No weights provided. Loading default backbone:", DEFAULT_BASE_WEIGHTS)
        return YOLO(DEFAULT_BASE_WEIGHTS)

    print("Loading model from weights:", weights)
    # NOTE: If you want to force CPU, you can add: model.to("cpu")
    model = YOLO(str(weights))
    return model


def save_weights_copy(src: Path, dest_dir: Path = SAVED_MODELS_DIR) -> Path:
    """
    Save a time-stamped copy of weights into saved_models.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


def try_export(model: YOLO, out_dir: Path = SAVED_MODELS_DIR) -> Dict[str, str]:
    """
    Attempt ONNX and TorchScript exports. Returns dict of successful exports.
    This can be a bit slow - controlled by ENABLE_EXPORT.
    """
    if not ENABLE_EXPORT:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, str] = {}

    try:
        # Ultralytics chooses filename, but we keep directory organized
        model.export(format="onnx", imgsz=640, opset=12, project=str(out_dir), name="onnx_export")
        exported["onnx"] = str(out_dir / "onnx_export" / "model.onnx")
    except Exception as e:
        print("ONNX export failed:", e)

    try:
        model.export(format="torchscript", imgsz=640, project=str(out_dir), name="ts_export")
        exported["torchscript"] = str(out_dir / "ts_export" / "model.torchscript")
    except Exception as e:
        print("TorchScript export failed:", e)

    return exported


# ------------- inference helpers -----------------
def predict_image(model: YOLO, image_bgr: np.ndarray, conf: float | None = None):
    """
    Run inference on a BGR image (numpy array), returns ultralytics Results.
    Example usage:
      res = predict_image(model, img_bgr)
    """
    kwargs = {}
    if conf is not None:
        kwargs["conf"] = conf
    results = model(image_bgr, imgsz=640, verbose=False, **kwargs)
    return results


def parse_detections(results, conf_thresh: float = CONFIDENCE_THRESHOLD) -> List[dict]:
    """
    Convert ultralytics Results to a list of detections:
    [{'xyxy':[x1,y1,x2,y2], 'conf':0.9, 'cls':0}, ...]
    """
    dets: List[dict] = []
    try:
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return dets

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else [0] * len(confs)

        for i, c in enumerate(confs):
            if c < conf_thresh:
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            dets.append(
                {
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(c),
                    "cls": int(cls[i]),
                }
            )
    except Exception as e:
        print("Failed to parse detections:", e)
    return dets


def annotate_image_np(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw boxes + labels on an image (BGR numpy) and return annotated image.
    """
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        conf = d["conf"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"fire {conf:.2f}"
        ((text_w, text_h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - text_h - 6), (x1 + text_w, y1), (0, 0, 255), -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
