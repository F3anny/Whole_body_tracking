
"""Live person detection with YOLOX and face recognition with ArcFace.

Run from the project root:
    python -m src.detect

YOLOX detects whole people.
ArcFace recognizes identities when a face is visible.
The face mesh is omitted because this version does not use Haar.
"""

from pathlib import Path

import cv2
import onnxruntime as ort

from src.align import align_face_5pt
from src.person_detect import detect_people
from src.recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
)


ROOT = Path(__file__).resolve().parent.parent
FACE_DB = ROOT / "data" / "db" / "face_db.npz"
FACE_MODEL = ROOT / "models" / "embedder_arcface.onnx"
PERSON_MODEL = ROOT / "models" / "yolox_nano.onnx"


def draw_label(frame, text, xy, color, scale=0.65):
    """Draw readable text with a dark outline."""
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA,
    )


def main():
    for path in (FACE_DB, FACE_MODEL, PERSON_MODEL):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # YOLOX detects people, not faces.
    person_detector = ort.InferenceSession(
        str(PERSON_MODEL),
        providers=["CPUExecutionProvider"],
    )

    # ArcFace identifies a face once a compatible face crop is available.
    embedder = ArcFaceEmbedderONNX(
        model_path=str(FACE_MODEL),
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(
        load_db_npz(FACE_DB),
        dist_thresh=0.34,
    )

    # No Haar detector is created or used in this file.
    # YOLOX provides the large full-body boxes.
    # Face recognition needs a separate face detector or face landmarks
    # to locate and align a face inside the person crop.

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Camera not opened. Try camera index 1.")

    print("YOLOX full-body detection. Press Q to quit; +/- changes face threshold.")
    print("Note: identity recognition is not active until a non-Haar face detector is connected.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            people = detect_people(person_detector, frame)

            for x1, y1, x2, y2, confidence in people:
                # Large rectangle around the entire detected person.
                color = (0, 255, 255)
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 3
                )
                draw_label(
                    frame,
                    f"PERSON {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    color,
                )

            draw_label(
                frame,
                f"People: {len(people)} | Face IDs: {len(matcher._names)}",
                (15, 30),
                (255, 255, 255),
            )
            draw_label(
                frame,
                "YOLOX body detection only - face identity not yet connected",
                (15, 60),
                (0, 255, 255),
                0.48,
            )

            cv2.imshow("YOLOX Full-Body Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = min(
                    1.20, matcher.dist_thresh + 0.01
                )
                print(f"Face threshold: {matcher.dist_thresh:.2f}")
            elif key in (ord("-"), ord("_")):
                matcher.dist_thresh = max(
                    0.05, matcher.dist_thresh - 0.01
                )
                print(f"Face threshold: {matcher.dist_thresh:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
