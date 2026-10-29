
from pathlib import Path

import cv2
import onnxruntime as ort

from src.person_detect import detect_people


class PersonDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            root = Path(__file__).resolve().parent.parent
            model_path = root / "models" / "yolox_nano.onnx"

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def detect(self, frame):
        """Return person boxes as (x1, y1, x2, y2, confidence)."""
        return detect_people(self.session, frame)
