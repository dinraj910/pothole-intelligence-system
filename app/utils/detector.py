"""
Multi-Model Road Damage Detection — Inference Engine
=====================================================
Runs 3 YOLOv8s models (RDD2022 · Pothole-600 · Kaggle) on every image,
merges predictions with cross-model NMS, and draws color-coded bounding
boxes with consensus indicators.

Transfer chain:  COCO → Model A → Model B → Model C
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO


# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "RoadDamageAI_Phase1" / "weights"

# ── Model registry ───────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "model_a": {
        "file": "model_a_rdd2022.pt",
        "name": "RDD2022 Dashcam",
        "short": "A",
        "color_bgr": (50, 50, 255),      # Red in BGR
        "color_hex": "#ff3232",
        "conf": 0.05,
    },
    "model_b": {
        "file": "model_b_pothole600.pt",
        "name": "Pothole-600 Close-up",
        "short": "B",
        "color_bgr": (50, 220, 50),      # Green in BGR
        "color_hex": "#32dc32",
        "conf": 0.10,
    },
    "model_c": {
        "file": "model_c_kaggle.pt",
        "name": "Kaggle Street",
        "short": "C",
        "color_bgr": (255, 100, 50),     # Blue in BGR
        "color_hex": "#3264ff",
        "conf": 0.05,
    },
}


# ── Load models (singleton) ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_models() -> dict:
    """Load all available YOLO models once and cache them."""
    models = {}
    for mid, info in MODEL_REGISTRY.items():
        weight_path = WEIGHTS_DIR / info["file"]
        if weight_path.exists():
            models[mid] = YOLO(str(weight_path))
            print(f"  ✓ Loaded {mid}: {info['name']} ({info['file']})")
        else:
            print(f"  ✗ {mid} not found at {weight_path}")
    if not models:
        raise FileNotFoundError(
            f"No model weights found in {WEIGHTS_DIR}. "
            "Ensure model_a_rdd2022.pt, model_b_pothole600.pt, "
            "and model_c_kaggle.pt are present."
        )
    return models


# ── IoU helper ───────────────────────────────────────────────────────────
def _compute_iou(box1: list, box2: list) -> float:
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ── Cross-model NMS ─────────────────────────────────────────────────────
def _cross_model_nms(detections: list, iou_threshold: float = 0.45) -> list:
    """Merge overlapping detections from all models, tracking consensus."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept, suppressed = [], set()
    for i, det_i in enumerate(sorted_dets):
        if i in suppressed:
            continue
        det_i["agreeing_models"] = [det_i["model_id"]]
        for j, det_j in enumerate(sorted_dets):
            if j <= i or j in suppressed:
                continue
            if _compute_iou(det_i["bbox"], det_j["bbox"]) > iou_threshold:
                det_i["agreeing_models"].append(det_j["model_id"])
                suppressed.add(j)
        det_i["consensus"] = len(det_i["agreeing_models"])
        kept.append(det_i)
    return kept


# ── Draw annotated image ────────────────────────────────────────────────
def _draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw color-coded bounding boxes with labels on the image."""
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = det["color_bgr"]
        model_letter = MODEL_REGISTRY[det["model_id"]]["short"]
        label = f"{model_letter}: {det['class_name']} {det['confidence']:.0%}"

        # Thicker box if multiple models agree
        thickness = 3 if det.get("consensus", 1) > 1 else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background + text
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        label_y = max(y1 - 5, text_size[1] + 5)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_size[1] - 6),
            (x1 + text_size[0] + 6, label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            annotated, label, (x1 + 3, label_y - 2),
            font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


# ── Public API ───────────────────────────────────────────────────────────
def run_inference(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    conf: float | None = None,
) -> dict:
    """
    Run the full multi-model pipeline on an image.

    Returns a dict with:
        detections  — list of merged detections
        count       — total detection count
        model_counts— per-model raw detection counts
        consensus   — how many detections had multi-model agreement
    """
    models = _load_models()
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    all_raw_detections = []
    model_counts = {}

    # ── Step 1: Run every model ──────────────────────────────────────
    for mid, model in models.items():
        info = MODEL_REGISTRY[mid]
        threshold = conf if conf is not None else info["conf"]
        results = model(image, conf=threshold, iou=0.45, verbose=False)
        raw_count = 0

        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                raw_count += 1
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                all_raw_detections.append({
                    "model_id": mid,
                    "model_name": info["name"],
                    "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    "confidence": float(result.boxes.conf[i]),
                    "class_id": int(result.boxes.cls[i]),
                    "class_name": result.names[int(result.boxes.cls[i])],
                    "color_bgr": info["color_bgr"],
                    "color_hex": info["color_hex"],
                })
        model_counts[mid] = raw_count

    # ── Step 2: Cross-model NMS ──────────────────────────────────────
    final_detections = _cross_model_nms(all_raw_detections, iou_threshold=0.45)

    # ── Step 3: Draw and save ────────────────────────────────────────
    annotated = _draw_detections(image, final_detections)
    cv2.imwrite(str(output_path), annotated)

    # ── Step 4: Build metadata ───────────────────────────────────────
    consensus_count = sum(1 for d in final_detections if d.get("consensus", 1) > 1)

    # Clean detections for JSON serialization (remove numpy/tuple fields)
    clean_detections = []
    for det in final_detections:
        clean_detections.append({
            "model_id": det["model_id"],
            "model_name": det["model_name"],
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 4),
            "consensus": det.get("consensus", 1),
            "agreeing_models": det.get("agreeing_models", [det["model_id"]]),
            "bbox": [round(v, 1) for v in det["bbox"]],
            "color_hex": det["color_hex"],
        })

    return {
        "detections": clean_detections,
        "count": len(final_detections),
        "model_counts": model_counts,
        "consensus_detections": consensus_count,
        "models_loaded": list(models.keys()),
    }


def run_inference_on_array(image: np.ndarray, conf: float | None = None) -> dict:
    """
    Run the full multi-model pipeline on a numpy BGR image array.
    Used by the live-camera API (no file I/O needed).

    Returns the same structure as run_inference(), plus:
        damage_pct — % of frame area covered by bounding boxes
    """
    models = _load_models()

    h, w = image.shape[:2]
    frame_area = h * w if h * w > 0 else 1

    all_raw_detections = []
    model_counts = {}

    for mid, model in models.items():
        info = MODEL_REGISTRY[mid]
        threshold = conf if conf is not None else info["conf"]
        results = model(image, conf=threshold, iou=0.45, verbose=False)
        raw_count = 0

        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                raw_count += 1
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                all_raw_detections.append({
                    "model_id": mid,
                    "model_name": info["name"],
                    "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    "confidence": float(result.boxes.conf[i]),
                    "class_id": int(result.boxes.cls[i]),
                    "class_name": result.names[int(result.boxes.cls[i])],
                    "color_bgr": info["color_bgr"],
                    "color_hex": info["color_hex"],
                })
        model_counts[mid] = raw_count

    final_detections = _cross_model_nms(all_raw_detections, iou_threshold=0.45)
    consensus_count = sum(1 for d in final_detections if d.get("consensus", 1) > 1)

    # Damage % — union of all bounding box areas relative to frame area
    box_area = 0
    for det in final_detections:
        x1, y1, x2, y2 = det["bbox"]
        box_area += max(0, x2 - x1) * max(0, y2 - y1)
    damage_pct = min(100.0, (box_area / frame_area) * 100)

    clean = []
    for det in final_detections:
        clean.append({
            "model_id": det["model_id"],
            "model_name": det["model_name"],
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 4),
            "consensus": det.get("consensus", 1),
            "agreeing_models": det.get("agreeing_models", [det["model_id"]]),
            "bbox": [round(v, 1) for v in det["bbox"]],
            "color_hex": det["color_hex"],
        })

    return {
        "detections": clean,
        "count": len(final_detections),
        "model_counts": model_counts,
        "consensus_detections": consensus_count,
        "models_loaded": list(models.keys()),
        "damage_pct": round(damage_pct, 2),
    }


def get_model_info() -> list[dict]:
    """Return metadata about all registered models for the UI."""
    info_list = []
    for mid, info in MODEL_REGISTRY.items():
        weight_path = WEIGHTS_DIR / info["file"]
        info_list.append({
            "id": mid,
            "name": info["name"],
            "short": info["short"],
            "color_hex": info["color_hex"],
            "conf": info["conf"],
            "available": weight_path.exists(),
            "file": info["file"],
        })
    return info_list
