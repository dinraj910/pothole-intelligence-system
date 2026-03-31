"""YOLOv8 inference helper for pothole detection."""

from functools import lru_cache
from pathlib import Path
from typing import Union

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"


@lru_cache(maxsize=1)
def _load_model() -> YOLO:
	"""Load the YOLO model once and reuse it."""
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
	return YOLO(str(MODEL_PATH))


def run_inference(image_path: Union[str, Path], output_path: Union[str, Path], conf: float = 0.05) -> Path:
	"""Run detection on an image and write the plotted result to disk."""

	model = _load_model()

	input_path = Path(image_path)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	results = model(str(input_path), conf=conf)
	plotted = results[0].plot()  # BGR numpy array with boxes drawn

	cv2.imwrite(str(output_path), plotted)
	return output_path
