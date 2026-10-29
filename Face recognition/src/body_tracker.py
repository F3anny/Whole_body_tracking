
from pathlib import Path
import os

import cv2
import numpy as np
import onnxruntime as ort

from src.person_detect import detect_people
from src.body_reid import BodyReID


ROOT = Path(__file__).resolve().parent.parent
PERSON_MODEL = ROOT / "models" / "yolox_nano.onnx"
BODY_MODEL = ROOT / "models" / "person_reid_youtu_fp32.onnx"
DATABASE = ROOT / "data" / "db" / "body_target.npz"

MAX_SAMPLES = 30


class BodyFallbackTracker:
    """Track a person by body appearance when their face is hidden."""

    def __init__(self, similarity_threshold=0.70, model_path=None, database=None):
        self.database = Path(database) if database is not None else DATABASE
        detector_path = Path(model_path) if model_path else PERSON_MODEL

        if not detector_path.exists():
            raise FileNotFoundError(
                f"Person detection model not found: {detector_path}"
            )

        if not BODY_MODEL.exists():
            raise FileNotFoundError(
                f"Body Re-ID model not found: {BODY_MODEL}"
            )

        self.detector = ort.InferenceSession(
            str(detector_path),
            providers=["CPUExecutionProvider"],
        )
        self.reid = BodyReID(str(BODY_MODEL))
        self.similarity_threshold = similarity_threshold

        self.profiles = {}
        self.active_name = "Unnamed target"
        self.samples = self._load_samples()
        self.last_box = None
        self.last_people = []
        self.last_people_count = 0
        self.last_similarity = None
        self.last_status = "SEARCHING"
        self.pending_box = None
        self.pending_count = 0
        self.pending_matches = {}
        self.last_matches = {}

        print(f"Loaded {len(self.samples)} body samples.")

    @property
    def samples(self):
        return self.profiles.setdefault(self.active_name, [])

    @samples.setter
    def samples(self, value):
        self.profiles[self.active_name] = value

    def select_person(self, name):
        name = " ".join(name.split())
        if not name or len(name) > 40 or not name.isprintable():
            raise ValueError("Enter a name between 1 and 40 characters.")
        existing = next((key for key in self.profiles if key.casefold() == name.casefold()), name)
        self.active_name = existing
        self.profiles.setdefault(existing, [])
        self.reset()

    def rename_person(self, name):
        old_name = self.active_name
        samples = self.samples
        name = " ".join(name.split())
        if not name or len(name) > 40 or not name.isprintable():
            raise ValueError("Enter a name between 1 and 40 characters.")
        if any(key.casefold() == name.casefold() and key != old_name for key in self.profiles):
            raise ValueError("That name already belongs to another profile.")
        del self.profiles[old_name]
        self.profiles[name] = samples
        self.active_name = name
        self.reset()

    def _load_samples(self):
        if not self.database.exists():
            return []

        with np.load(self.database) as data:
            if "model" not in data or str(data["model"]) != BODY_MODEL.name:
                print("Body samples use the previous model. Press A to enroll again.")
                return []
            features = data["features"].astype(np.float32)
            names = data["names"].astype(str) if "names" in data else np.full(len(features), "Unnamed target")
            active = str(data["active_name"]) if "active_name" in data else "Unnamed target"

        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError(
                f"Unexpected body feature shape: {features.shape}"
            )

        if names.shape != (len(features),):
            raise ValueError("Body names and sample counts do not match.")
        self.profiles = {}
        for name, feature in zip(names, features):
            norm = np.linalg.norm(feature)
            if np.isfinite(feature).all() and norm > 1e-12:
                self.profiles.setdefault(str(name), []).append(feature / norm)

        self.active_name = active
        return self.profiles.get(active, [])

    def save_samples(self):
        rows = [(name, feature) for name, samples in self.profiles.items() for feature in samples]
        features = np.asarray([feature for _, feature in rows], dtype=np.float32).reshape(-1, 768)

        self.database.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.database.with_suffix(".npz.tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, features=features,
                                names=np.asarray([name for name, _ in rows], dtype=str),
                                active_name=np.asarray(self.active_name), model=np.asarray(BODY_MODEL.name))
        os.replace(temporary, self.database)
        print(f"Saved {len(rows)} body samples across {sum(bool(s) for s in self.profiles.values())} people.")

    @staticmethod
    def crop(frame, box):
        x1, y1, x2, y2 = map(int, box)
        height, width = frame.shape[:2]

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def enroll(self, frame, box):
        """Enroll one manually selected person using their body crop."""
        crop = self.crop(frame, box)

        if crop is None or crop.shape[0] < 40 or crop.shape[1] < 20:
            self.last_status = "ENROLLMENT_FAILED"
            print("Body crop is too small. Show the full body.")
            return False

        if len(self.samples) >= MAX_SAMPLES:
            print(f"Maximum of {MAX_SAMPLES} samples reached.")
            return False

        try:
            feature = self.reid.extract(crop)
        except (ValueError, cv2.error) as exc:
            print(f"Body enrollment failed: {exc}")
            return False

        self.samples.append(feature.copy())
        self.last_box = tuple(map(int, box))
        self.last_similarity = 1.0
        self.last_status = "BODY_ENROLLED"

        print(f"Enrolled body sample {len(self.samples)}/{MAX_SAMPLES}.")
        return True

    def clear_samples(self):
        """Clear samples from memory; save replacements with save_samples()."""
        self.samples.clear()
        self.last_box = None
        self.last_similarity = None
        self.last_status = "BODY_SAMPLES_CLEARED"
        self.pending_box = None
        self.pending_count = 0
        self.pending_matches = {}
        self.last_matches = {}

    def reset(self):
        """Reset tracking without deleting enrolled body samples."""
        self.last_box = None
        self.last_similarity = None
        self.last_status = "SEARCHING"
        self.pending_box = None
        self.pending_count = 0
        self.pending_matches = {}
        self.last_matches = {}

    @staticmethod
    def overlap(first, second):
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area = lambda b: max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        return intersection / max(area(first) + area(second) - intersection, 1)

    def _match(self, feature):
        if not self.samples:
            return None

        return max(
            self.reid.similarity(sample, feature)
            for sample in self.samples
        )

    def update(self, frame):
        """Identify each visible body against every saved named profile."""
        people = detect_people(self.detector, frame)
        self.last_people = people
        self.last_people_count = len(people)
        self.last_matches = {}
        self.last_box = None
        self.last_similarity = None
        profiles = {name: samples for name, samples in self.profiles.items() if samples}
        if not profiles:
            self.pending_matches = {}
            self.last_status = "BODY_NOT_ENROLLED" if people else "TARGET_LOST"
            return None

        proposed = {}
        ambiguous = False
        for x1, y1, x2, y2, confidence in people:
            box = (x1, y1, x2, y2)
            crop = self.crop(frame, box)
            if crop is None:
                continue
            try:
                feature = self.reid.extract(crop)
            except (ValueError, cv2.error):
                continue
            scores = sorted(((max(self.reid.similarity(sample, feature) for sample in samples), name)
                             for name, samples in profiles.items()), reverse=True)
            score, name = scores[0]
            if score < self.similarity_threshold:
                continue
            if len(scores) > 1 and score - scores[1][0] < 0.08:
                ambiguous = True
                continue
            proposed.setdefault(name, []).append((box, score))

        pending = {}
        for name, candidates in proposed.items():
            # One saved identity cannot be assigned to two visible people.
            candidates.sort(key=lambda candidate: candidate[1], reverse=True)
            if len(candidates) > 1 and candidates[0][1] - candidates[1][1] < 0.08:
                ambiguous = True
                continue
            box, score = candidates[0]
            previous = self.pending_matches.get(name)
            count = previous[1] + 1 if previous and self.overlap(previous[0], box) > 0.3 else 1
            pending[name] = (box, count)
            if count >= 3:
                self.last_matches[box] = (name, score)
        self.pending_matches = pending
        if self.last_matches:
            self.last_status = "BODY_MATCH"
            # Optional hardware follows only the explicitly selected profile.
            for box, (name, score) in self.last_matches.items():
                if name == self.active_name:
                    self.last_box = box
                    self.last_similarity = score
                    return box, score
            return None
        self.last_status = ("AMBIGUOUS_BODY" if ambiguous else "CONFIRMING_BODY" if pending
                            else "UNKNOWN_BODY" if people else "TARGET_LOST")
        return None
