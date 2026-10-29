
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class BodyReID:
    """Extract normalized body features using the ONNX Re-ID model."""

    def __init__(self, model_path="models/person_reid_youtu_fp32.onnx"):
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path.resolve()}")

        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        batch = self.session.get_inputs()[0].shape[0]
        self.batch_size = batch if isinstance(batch, int) and batch > 0 else 1

        self.mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        ).reshape(1, 1, 3)

        print("Body Re-ID model loaded.")

    def extract(self, person_crop):
        """Return a normalized 768-dimensional body feature vector."""
        if person_crop is None or person_crop.size == 0:
            raise ValueError("Person crop is empty.")

        image = cv2.resize(person_crop, (128, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))[None, ...]
        if self.batch_size > 1:
            image = np.repeat(image, self.batch_size, axis=0)

        features = self.session.run(
            [self.output_name],
            {self.input_name: np.ascontiguousarray(image)},
        )[0][0]

        features = features.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(features)

        if not np.isfinite(features).all() or not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("The model returned an empty feature vector.")

        return features / norm

    @staticmethod
    def similarity(first, second):
        """Cosine similarity between two normalized body feature vectors."""
        return float(np.dot(first, second))
