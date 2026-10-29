
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolox_nano.onnx"

INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45
PERSON_CLASS_ID = 0

# YOLO boxes can be tight around the visible pixels. Add a small margin so
# the displayed crop includes the whole visible person, especially shoes and
# hair, while still clipping safely to the camera frame.
BOX_PAD_X = 0.08
BOX_PAD_TOP = 0.06
BOX_PAD_BOTTOM = 0.05


def preprocess(frame):
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)

    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8
    )
    padded[:resized_height, :resized_width] = resized

    # Official YOLOX ONNX exports expect BGR values in the 0..255 range.
    image = padded.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(image), scale


def decode_predictions(output):
    """Decode the raw stride-grid output of the bundled YOLOX Nano export."""
    predictions = np.asarray(output, dtype=np.float32).copy()
    grids, strides = [], []
    for stride in (8, 16, 32):
        size = INPUT_SIZE // stride
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        grid = np.stack((x, y), axis=-1).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((len(grid), 1), stride))
    grid = np.concatenate(grids)
    stride = np.concatenate(strides)
    if predictions.shape != (1, len(grid), 85):
        raise ValueError(f"Unexpected YOLOX output shape: {predictions.shape}")
    predictions[..., :2] = (predictions[..., :2] + grid) * stride
    predictions[..., 2:4] = np.exp(np.clip(predictions[..., 2:4], -20, 20)) * stride
    return predictions[0]


def detect_people(session, frame):
    input_name = session.get_inputs()[0].name
    image, scale = preprocess(frame)
    output = decode_predictions(session.run(None, {input_name: image})[0])

    height, width = frame.shape[:2]
    boxes = []
    scores = []

    for detection in output:
        objectness = float(detection[4])
        class_scores = detection[5:]
        person_score = objectness * float(class_scores[PERSON_CLASS_ID])

        if person_score < CONFIDENCE_THRESHOLD:
            continue

        cx, cy, bw, bh = map(float, detection[:4])

        x1 = int((cx - bw / 2) / scale)
        y1 = int((cy - bh / 2) / scale)
        x2 = int((cx + bw / 2) / scale)
        y2 = int((cy + bh / 2) / scale)

        box_width = x2 - x1
        box_height = y2 - y1
        x1 -= int(round(box_width * BOX_PAD_X))
        x2 += int(round(box_width * BOX_PAD_X))
        y1 -= int(round(box_height * BOX_PAD_TOP))
        y2 += int(round(box_height * BOX_PAD_BOTTOM))

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(person_score)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) == 0:
        return []

    people = []
    for index in np.asarray(indices).reshape(-1):
        x, y, bw, bh = boxes[int(index)]
        people.append((x, y, x + bw, y + bh, scores[int(index)]))

    return people


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLOX model not found: {MODEL_PATH}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            for x1, y1, x2, y2, score in detect_people(session, frame):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(
                    frame,
                    f"PERSON {score:.2f}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("YOLOX Full-Body Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
