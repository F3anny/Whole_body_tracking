"""Regression checks for raw YOLOX decoding and conservative target matching."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import cv2

from src.person_detect import decode_predictions, detect_people, preprocess
from src.body_tracker import BodyFallbackTracker


class FakeDetector:
    def get_inputs(self):
        return [type("Input", (), {"name": "images"})()]

    def run(self, names, inputs):
        raw = np.zeros((1, 3549, 85), np.float32)
        # A 160x240 person centered at (160, 160), stride 8.
        index = 20 * 52 + 20
        raw[0, index, :4] = [0, 0, np.log(20), np.log(30)]
        raw[0, index, 4:6] = .95
        return [raw]


class FakeReID:
    feature = np.array([1., 0.], np.float32)

    def extract(self, crop):
        return self.feature.copy()

    def similarity(self, a, b):
        return float(np.dot(a, b))


class DetectorTests(unittest.TestCase):
    def test_raw_grid_decoding(self):
        raw = FakeDetector().run(None, None)[0]
        result = decode_predictions(raw)
        np.testing.assert_allclose(result[20*52+20, :4], [160, 160, 160, 240], rtol=1e-5)
        self.assertEqual(raw[0, 20*52+20, 0], 0)  # does not mutate inference output

    def test_realistic_raw_output_yields_visible_person_box(self):
        boxes = detect_people(FakeDetector(), np.zeros((416, 416, 3), np.uint8))
        self.assertEqual(len(boxes), 1)
        x1, y1, x2, y2, score = boxes[0]
        self.assertLess(x1, 80)
        self.assertGreater(x2, 240)
        self.assertGreater(y2-y1, 240)
        self.assertGreater(score, .9)

    def test_bgr_input_is_preserved(self):
        image = np.full((416, 416, 3), [10, 20, 30], np.uint8)
        tensor, _ = preprocess(image)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [10, 20, 30])


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "target.npz"
        with patch("src.body_tracker.ort.InferenceSession", return_value=FakeDetector()), patch("src.body_tracker.BodyReID", return_value=FakeReID()):
            self.tracker = BodyFallbackTracker(database=self.db)
        self.tracker.samples = [np.array([1., 0.], np.float32)]
        self.frame = np.zeros((416, 416, 3), np.uint8)
        self.people = [(80, 40, 240, 280, .95)]

    def update(self, people=None):
        with patch("src.body_tracker.detect_people", return_value=self.people if people is None else people):
            return self.tracker.update(self.frame)

    def test_requires_three_consecutive_matches(self):
        self.assertIsNone(self.update())
        self.assertIsNone(self.update())
        self.assertIsNotNone(self.update())

    def test_other_person_is_rejected_and_previous_box_cleared(self):
        for _ in range(3):
            self.update()
        self.tracker.reid.feature = np.array([0., 1.], np.float32)
        self.assertIsNone(self.update())
        self.assertIsNone(self.tracker.last_box)
        self.assertEqual(self.tracker.last_status, "UNKNOWN_BODY")

    def test_lookalikes_are_ambiguous(self):
        self.assertIsNone(self.update(self.people + [(250, 40, 400, 280, .9)]))
        self.assertEqual(self.tracker.last_status, "AMBIGUOUS_BODY")

    def test_lost_person_requires_confirmation_again(self):
        for _ in range(3):
            self.update()
        self.assertIsNone(self.update([]))
        self.assertIsNone(self.tracker.last_box)
        self.assertIsNone(self.update())

    def test_unenrolled_people_still_available_for_rectangles(self):
        self.tracker.clear_samples()
        self.assertIsNone(self.update())
        self.assertEqual(self.tracker.last_people_count, 1)

    def test_clear_is_persisted(self):
        self.tracker.samples = [np.ones(768, np.float32) / np.sqrt(768)]
        self.tracker.save_samples()
        self.tracker.clear_samples()
        self.tracker.save_samples()
        self.assertEqual(self.tracker._load_samples(), [])

    def test_existing_unnamed_samples_can_be_named(self):
        samples = self.tracker.samples.copy()
        self.tracker.rename_person("Alice")
        self.assertEqual(self.tracker.active_name, "Alice")
        np.testing.assert_array_equal(self.tracker.samples, samples)
        self.assertNotIn("Unnamed target", self.tracker.profiles)

    def test_named_profiles_persist_and_clearing_one_keeps_other(self):
        self.tracker.samples = [np.ones(768, np.float32) / np.sqrt(768)]
        self.tracker.rename_person("Alice")
        self.tracker.select_person("Bob")
        self.tracker.samples = [np.eye(1, 768, dtype=np.float32)[0]]
        self.tracker.save_samples()
        with patch("src.body_tracker.ort.InferenceSession", return_value=FakeDetector()), patch("src.body_tracker.BodyReID", return_value=FakeReID()):
            loaded = BodyFallbackTracker(database=self.db)
        self.assertEqual(set(loaded.profiles), {"Alice", "Bob"})
        self.assertEqual(loaded.active_name, "Bob")
        loaded.clear_samples()
        loaded.save_samples()
        with np.load(self.db) as data:
            self.assertEqual(data["names"].tolist(), ["Alice"])

    def test_legacy_samples_preserved_as_unnamed(self):
        from src.body_tracker import BODY_MODEL
        np.savez(self.db, features=np.ones((12, 768), np.float32), model=np.asarray(BODY_MODEL.name))
        with patch("src.body_tracker.ort.InferenceSession", return_value=FakeDetector()), patch("src.body_tracker.BodyReID", return_value=FakeReID()):
            loaded = BodyFallbackTracker(database=self.db)
        self.assertEqual(len(loaded.samples), 12)
        self.assertEqual(loaded.active_name, "Unnamed target")

    def test_confusing_identity_profiles_stay_unknown(self):
        self.tracker.profiles["Lookalike"] = [self.tracker.samples[0].copy()]
        for _ in range(3):
            self.assertIsNone(self.update())
        self.assertEqual(self.tracker.last_matches, {})
        self.assertEqual(self.tracker.last_status, "AMBIGUOUS_BODY")

    def test_two_named_people_recognized_together(self):
        self.tracker.rename_person("Alice")
        self.tracker.profiles["Bob"] = [np.array([0., 1.], np.float32)]
        people = self.people + [(250, 40, 400, 280, .9)]
        with patch.object(self.tracker.reid, "extract", side_effect=[np.array([1., 0.]), np.array([0., 1.])] * 3):
            for _ in range(3):
                self.update(people)
        self.assertEqual({name for name, score in self.tracker.last_matches.values()}, {"Alice", "Bob"})

    def test_name_change_requires_confirmation_again(self):
        for _ in range(3):
            self.update()
        self.tracker.rename_person("Alice")
        self.assertIsNone(self.update())

    def test_empty_or_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            self.tracker.rename_person("  ")
        self.tracker.profiles["Alice"] = []
        with self.assertRaises(ValueError):
            self.tracker.rename_person("alice")

    def test_camera_name_entry_saves_existing_samples(self):
        from src.face_tracking import main
        self.tracker.samples = [np.ones(768, np.float32) / np.sqrt(768)] * 12
        self.tracker.last_people = self.people
        camera = Mock()
        camera.isOpened.return_value = True
        keys = [ord(c) for c in "nAlice"] + [13, ord("q")]
        camera.read.side_effect = [(True, self.frame.copy()) for _ in keys]
        with patch("sys.argv", ["tracker"]), patch("src.face_tracking.BodyFallbackTracker", return_value=self.tracker), patch.object(self.tracker, "update", return_value=None), patch("src.face_tracking.cv2.VideoCapture", return_value=camera), patch("src.face_tracking.cv2.imshow"), patch("src.face_tracking.cv2.destroyAllWindows"), patch("src.face_tracking.cv2.waitKey", side_effect=keys):
            main()
        with np.load(self.db) as data:
            self.assertEqual(data["names"].tolist(), ["Alice"] * 12)
        camera.release.assert_called_once()

    def test_enrollment_keeps_name_when_enter_has_no_detection_and_capture_briefly_drops(self):
        from src.face_tracking import main
        feature = np.eye(1, 768, dtype=np.float32)[0]
        self.tracker.samples = [feature.copy()] * 12
        self.tracker.rename_person("Fanny")
        self.tracker.reid.feature = np.roll(feature, 1)
        # Enter while no person is detected. A repeated A during the waiting
        # state must not reopen name entry. One missed capture frame pauses.
        steps = [(0, ord("a"), False), (1, ord("B"), False),
                 (2, ord("o"), False), (3, ord("b"), False), (4, 13, False),
                 (5, -1, False), (6, ord("a"), True), (7, -1, True),
                 (8, -1, True), (11, -1, True), (12, -1, False),
                 (13, -1, True), (18, -1, True), (24, -1, True),
                 (30, -1, True), (33, -1, True), (34, ord("q"), True)]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, self.frame.copy()) for _ in steps]
        updates = iter(steps)
        def update(frame):
            _, _, visible = next(updates)
            self.tracker.last_people = self.people if visible else []
            return None
        with patch("sys.argv", ["tracker"]), patch("src.face_tracking.BodyFallbackTracker", return_value=self.tracker), patch.object(self.tracker, "update", side_effect=update), patch("src.face_tracking.cv2.VideoCapture", return_value=camera), patch("src.face_tracking.cv2.imshow"), patch("src.face_tracking.cv2.destroyAllWindows"), patch("src.face_tracking.cv2.waitKey", side_effect=[key for _, key, _ in steps]), patch("src.face_tracking.time.monotonic", side_effect=[when for when, _, _ in steps]), patch("src.face_tracking.draw_label") as labels:
            main()
        with np.load(self.db) as data:
            names = data["names"].tolist()
            self.assertEqual(names.count("Fanny"), 12)
            self.assertGreaterEqual(names.count("Bob"), 4)
        texts = [call.args[1] for call in labels.call_args_list]
        self.assertTrue(any("Name accepted: Bob" in text for text in texts))
        self.assertTrue(any("capture paused" in text for text in texts))
        self.assertTrue(any("Saved Bob:" in text for text in texts))


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Path(__file__).parent / "fixtures" / "bus.jpg"
        if not cls.fixture.exists():
            raise unittest.SkipTest("Public multi-person fixture not downloaded")
        cls.tracker = BodyFallbackTracker(database=cls.fixture.parent / "nonexistent_target.npz")
        cls.frame = cv2.imread(str(cls.fixture))
        cls.tracker.update(cls.frame)
        cls.people = cls.tracker.last_people.copy()

    def test_real_multi_person_detection(self):
        self.assertGreaterEqual(len(self.people), 3)
        for x1, y1, x2, y2, _ in self.people:
            self.assertGreater(y2-y1, 300)

    def test_real_model_rejects_other_people_entering_alone(self):
        tracker = self.tracker
        tracker.clear_samples()
        self.assertTrue(tracker.enroll(self.frame, self.people[0][:4]))
        tracker.reset()
        with patch("src.body_tracker.detect_people", return_value=[self.people[0]]):
            for _ in range(3):
                result = tracker.update(self.frame)
            self.assertIsNotNone(result)
        for person in self.people[1:]:
            with patch("src.body_tracker.detect_people", return_value=[person]):
                for _ in range(3):
                    self.assertIsNone(tracker.update(self.frame))
                self.assertEqual(tracker.last_status, "UNKNOWN_BODY")


if __name__ == "__main__":
    unittest.main()
