"""Full-body detection and appearance tracking.

This is kept as the project's main camera entry point for compatibility, but
it no longer loads or calls any face detector, face embedder, or face tracker.

Run from the project root:
    python -m src.face_tracking

Controls:
    q  quit
    r  reset body tracking
    a  enter a name and start guided enrollment
    n  name or rename the selected person's saved samples
    e  add a pose for the selected person
    c  clear only the selected person's body samples
    +/- change body similarity threshold
"""

import argparse
import time

import cv2

from src.body_tracker import BodyFallbackTracker


def draw_label(frame, text, xy, color, scale=0.65):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def body_position_signal(box, shape, dead_zone=0.07):
    """Return normalized body-center position for the pan indicator."""
    x1, y1, x2, y2 = box
    height, width = shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    error_x = (cx - width / 2.0) / max(width / 2.0, 1.0)
    error_y = (cy - height / 2.0) / max(height / 2.0, 1.0)

    horizontal = "CENTER"
    vertical = "CENTER"
    if error_x < -dead_zone:
        horizontal = "LEFT"
    elif error_x > dead_zone:
        horizontal = "RIGHT"
    if error_y < -dead_zone:
        vertical = "UP"
    elif error_y > dead_zone:
        vertical = "DOWN"

    return error_x, error_y, horizontal, vertical


def direction_to_angle(position):
    _, _, horizontal, vertical = position
    if vertical == "UP":
        return "UP", 0.0
    if vertical == "DOWN":
        return "DOWN", 180.0
    if horizontal == "LEFT":
        return "LEFT", 0.0
    if horizontal == "RIGHT":
        return "RIGHT", 180.0
    return "CENTER", 90.0


def main():
    parser = argparse.ArgumentParser(description="Full-body tracking camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--body-threshold", type=float, default=0.70)
    parser.add_argument("--mqtt", action="store_true", help="Enable optional pan hardware")
    parser.add_argument("--name", help="Select a saved person for extra samples or hardware tracking")
    args = parser.parse_args()

    body_tracker = BodyFallbackTracker(
        similarity_threshold=args.body_threshold,
    )
    if args.name:
        body_tracker.select_person(args.name)

    publish_angle = lambda angle: False
    disconnect_mqtt = lambda: None
    if args.mqtt:
        from mqtt.client import connect_mqtt, disconnect_mqtt, publish_angle
        if not connect_mqtt():
            disconnect_mqtt()
            raise RuntimeError("Could not connect to MQTT. Run without --mqtt for camera only.")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        disconnect_mqtt()
        raise RuntimeError(f"Camera {args.camera} is not available.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_angle = None
    capture_start = None
    capture_previous_box = None
    last_capture = 0.0
    name_mode = None
    name_buffer = ""
    waiting_name = None
    ready_frames = 0
    ready_box = None
    capture_last_frame = None
    capture_last_seen = None
    previous_samples = []
    pose_counts = [0, 0, 0, 0]
    notice = "A: enroll a named person | N: name existing samples | Tab: select person"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()
            body_result = body_tracker.update(frame)
            known_box = body_result[0] if body_result is not None else None
            now = time.monotonic()
            if waiting_name is not None:
                if len(body_tracker.last_people) == 1:
                    box = body_tracker.last_people[0][:4]
                    ready_frames = ready_frames + 1 if ready_box is not None and body_tracker.overlap(ready_box, box) > 0.3 else 1
                    ready_box = box
                    notice = f"Name accepted: {waiting_name}. Hold still ({ready_frames}/3)"
                    if ready_frames >= 3:
                        body_tracker.select_person(waiting_name)
                        previous_samples = body_tracker.samples.copy()
                        body_tracker.clear_samples()
                        capture_start = now
                        capture_last_frame = now
                        capture_last_seen = now
                        capture_previous_box = box
                        pose_counts = [0, 0, 0, 0]
                        last_capture = now
                        waiting_name = None
                else:
                    ready_frames = 0
                    ready_box = None
                    notice = f"Name accepted: {waiting_name}. Waiting for ONE person in view (Esc cancels)"
            if capture_start is not None:
                elapsed = now - capture_start
                poses = ("FACE CAMERA", "TURN RIGHT SIDE", "FACE AWAY / BACK", "TURN LEFT SIDE")
                if len(body_tracker.last_people) != 1:
                    # A missed detection must not discard the entered name or
                    # consume a pose's capture window.
                    capture_start += now - capture_last_frame
                    notice = f"{body_tracker.active_name}: capture paused - show ONE person"
                    if now - capture_last_seen > 2.0:
                        capture_start = None
                        body_tracker.samples = previous_samples
                        body_tracker.reset()
                        waiting_name = body_tracker.active_name
                        ready_frames = 0
                        ready_box = None
                        notice = f"Keeping name {waiting_name}. Waiting to restart poses."
                elif elapsed >= 24:
                    capture_start = None
                    if all(pose_counts):
                        body_tracker.save_samples()
                        notice = f"Saved {body_tracker.active_name}: {len(body_tracker.samples)} samples"
                    else:
                        body_tracker.samples = previous_samples
                        body_tracker.reset()
                        waiting_name = body_tracker.active_name
                        ready_frames = 0
                        ready_box = None
                        notice = f"Missing poses for {waiting_name}. Keeping name and restarting."
                else:
                    box = body_tracker.last_people[0][:4]
                    if capture_previous_box is not None and body_tracker.overlap(capture_previous_box, box) < 0.3:
                        capture_start = None
                        body_tracker.samples = previous_samples
                        body_tracker.reset()
                        waiting_name = body_tracker.active_name
                        ready_frames = 0
                        ready_box = None
                        notice = f"Movement interrupted capture. Keeping name {waiting_name}."
                    else:
                        capture_last_seen = now
                        capture_previous_box = box
                        phase = int(elapsed // 6)
                        notice = f"{poses[phase]} ({phase+1}/4) - {6-int(elapsed % 6)}s"
                        # Allow three seconds to turn before taking clean samples.
                        if elapsed % 6 >= 3 and now - last_capture >= 1:
                            if body_tracker.enroll(raw_frame, box):
                                pose_counts[phase] += 1
                            last_capture = now
                capture_last_frame = now
            if capture_start is not None:
                body_result = None
                known_box = None

            # Every detected person gets a whole-body rectangle. Only the
            # body matching the enrolled appearance is green; others are red.
            for x1, y1, x2, y2, confidence in body_tracker.last_people:
                box = (x1, y1, x2, y2)
                match = body_tracker.last_matches.get(box) if capture_start is None else None
                is_known = match is not None
                color = (0, 255, 0) if is_known else (0, 0, 255)
                label = (
                    f"{match[0]} {match[1]:.2f}"
                    if is_known
                    else f"UNKNOWN PERSON {confidence:.2f}"
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                draw_label(frame, label, (x1, max(28, y1 - 10)), color)

            if body_result is not None:
                active_box, similarity = body_result
                position = body_position_signal(active_box, frame.shape)
                direction, angle = direction_to_angle(position)
                if args.mqtt and angle != last_angle:
                    if publish_angle(angle):
                        last_angle = angle
                draw_label(frame, f"BODY: {direction} ({angle:.0f} deg)",
                           (20, 40), (0, 255, 0))
                draw_label(frame, f"Body similarity: {similarity:.2f}",
                           (20, 72), (0, 255, 0))
            elif body_tracker.last_people:
                draw_label(frame, body_tracker.last_status,
                           (20, 40), (0, 0, 255))
            else:
                draw_label(frame, "NO PERSON DETECTED", (20, 40), (0, 165, 255))
            if body_result is None and args.mqtt and last_angle not in (None, 90.0):
                if publish_angle(90.0):
                    last_angle = 90.0

            prompt = f"{'Enroll name' if name_mode == 'enroll' else 'Rename selected person'}: {name_buffer}_  [Enter / Esc]" if name_mode else notice
            draw_label(frame, prompt, (20, 100), (0, 255, 255), scale=0.50)

            draw_label(
                frame,
                f"Selected: {body_tracker.active_name} | samples: {len(body_tracker.samples)} | saved people: {sum(bool(s) for s in body_tracker.profiles.values())}",
                (20, frame.shape[0] - 50), (255, 255, 255), scale=0.55,
            )
            draw_label(frame, "A enroll | N rename | Tab select | E add pose | C clear | Q quit",
                       (20, frame.shape[0] - 20), (255, 255, 255), scale=0.50)
            cv2.imshow("Full Body Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if name_mode:
                if key == 27:
                    name_mode = None
                elif key in (8, 127):
                    name_buffer = name_buffer[:-1]
                elif key in (10, 13):
                    try:
                        if name_mode == "rename":
                            body_tracker.rename_person(name_buffer)
                            body_tracker.save_samples()
                            notice = f"Saved name: {body_tracker.active_name}"
                        else:
                            cleaned_name = " ".join(name_buffer.split())
                            if not cleaned_name or len(cleaned_name) > 40:
                                raise ValueError("Type a name before pressing Enter.")
                            waiting_name = cleaned_name
                            ready_frames = 0
                            ready_box = None
                            notice = f"Name accepted: {waiting_name}. Stand alone with whole body visible."
                        name_mode = None
                    except ValueError as exc:
                        notice = str(exc)
                elif 32 <= key <= 126 and len(name_buffer) < 40:
                    name_buffer += chr(key)
                continue
            if ord("A") <= key <= ord("Z"):
                key += 32
            if key == ord("q"):
                break
            if key == 27 and (waiting_name is not None or capture_start is not None):
                if capture_start is not None:
                    body_tracker.samples = previous_samples
                    body_tracker.reset()
                capture_start = None
                waiting_name = None
                ready_frames = 0
                notice = "Enrollment cancelled. Previous samples kept. A to enroll."
                continue
            if key == ord("r"):
                body_tracker.reset()
            elif key == ord("a"):
                if capture_start is None and waiting_name is None:
                    name_mode = "enroll"
                    name_buffer = ""
            elif key == ord("n") and capture_start is None and waiting_name is None:
                name_mode = "rename"
                name_buffer = ""
            elif key == 9 and capture_start is None and waiting_name is None:
                names = [name for name, samples in body_tracker.profiles.items() if samples]
                if names:
                    index = names.index(body_tracker.active_name) if body_tracker.active_name in names else -1
                    body_tracker.select_person(names[(index + 1) % len(names)])
                    notice = f"Selected: {body_tracker.active_name}"
            elif key == ord("e") and capture_start is None and waiting_name is None:
                if len(body_tracker.last_people) == 1:
                    body_tracker.enroll(raw_frame, body_tracker.last_people[0][:4])
                    body_tracker.save_samples()
                else:
                    notice = "Sample refused: exactly one person must be visible"
            elif key == ord("c"):
                waiting_name = None
                if capture_start is not None:
                    body_tracker.samples = previous_samples
                capture_start = None
                body_tracker.clear_samples()
                body_tracker.save_samples()
                notice = f"Cleared {body_tracker.active_name} only. A to enroll a person."
            elif key in (ord("+"), ord("=")):
                body_tracker.similarity_threshold = min(
                    0.99, body_tracker.similarity_threshold + 0.01
                )
            elif key in (ord("-"), ord("_")):
                body_tracker.similarity_threshold = max(
                    0.05, body_tracker.similarity_threshold - 0.01
                )
    finally:
        if capture_start is not None:
            body_tracker.samples = previous_samples
        if args.mqtt and last_angle != 90.0:
            publish_angle(90.0)
        cap.release()
        disconnect_mqtt()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
