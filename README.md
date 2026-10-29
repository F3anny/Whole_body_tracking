# Whole Body Tracking

A Windows webcam application that detects people, enrolls their body appearance under a name, and displays that name when a matching person is visible. Guided enrollment captures front, right side, back, and left side views to help match people as they turn.

This is appearance-based person re-identification. Clothing, lighting, occlusion, and similar outfits can cause incorrect or missed matches. It does not prove someone's identity, and reliable recognition from behind has not been established through a representative live evaluation.

## Features

- Whole-body rectangles for detected people, including before enrollment.
- Multiple named profiles, saved locally between sessions.
- Four-pose enrollment with a countdown and confirmation when saved.
- Green named boxes for confirmed matches; red boxes for unknown people.
- Three-frame confirmation and rejection of ambiguous identity matches.
- Name retention while waiting for a clear view; brief detection gaps pause capture.
- Optional MQTT output for external pan hardware.

## Requirements

- Windows with a working webcam and camera permission enabled.
- Python 3.10 or newer; development checks were run with Python 3.13.
- Internet access for installing dependencies and downloading the body model.
- CPU inference is supported; no GPU is required.

The supported instructions below use PowerShell. The current camera backend is Windows DirectShow; Linux and macOS camera support has not been implemented or tested.

## Install

```powershell
git clone https://github.com/F3anny/Whole_body_tracking.git
cd Whole_body_tracking
cd "Face recognition"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe models/download_reid.py
```

The last command downloads the official OpenCV Zoo YoutuReID model (about 107 MB) and verifies its published SHA-256. Keep the connection open until it prints **Official body model downloaded and verified**. The small YOLOX Nano detector is included in the repository. Larger model files and local virtual environments are excluded from Git.

Always create a new virtual environment after cloning. Copying a virtual environment from another computer can leave it pointing to a Python installation that does not exist on yours.

## Run

From the `Face recognition` directory:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking
```

After installation, you can also double-click `Start Body Tracking.bat`. It uses the local virtual environment when available, otherwise the system `python`.

Select another camera if needed:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --camera 1
```

`python -m src.camera` opens the same body tracker. The older `src.recognize` entry point also redirects to tracking, but imports legacy face modules that require additional dependencies. Use `src.face_tracking` for the supported body-only workflow.

## Enroll a person with their name

1. Have only that person stand in view, far enough back for their head and feet to be visible.
2. Click the camera window, press **A** once, type their name, and press **Enter**.
3. Look for **Name accepted**. The application keeps the name and waits for exactly one person in three consecutive frames before starting capture. Do not type the name again.
4. Follow the prompts: face camera, right side, back, and left side. Each pose lasts six seconds, with three seconds allowed for turning before samples are taken.
5. Wait for **Saved [name]: [count] samples**. Closing the application before this message does not complete enrollment.
6. Repeat with a different name for the next person. Other saved people remain enrolled.

Brief detection gaps pause the timer. Longer gaps, abrupt movement, or missing pose samples restart the poses with the same name. Enrollment may therefore take longer than 24 seconds. Previous samples remain saved until a complete replacement capture succeeds.

Names entered in the camera window support up to 40 printable ASCII characters. Entering an existing name replaces that person's samples after a successful capture. Names are matched without regard to case when selecting existing profiles.

## Controls

Click the camera window before using the keyboard. Shortcut keys work in either case; while typing a name, letters are entered as text.

| Key | Action |
| --- | --- |
| A | Enter a name and begin guided enrollment |
| N | Rename the selected profile; does not capture a new person |
| Enter | Submit the typed name |
| Backspace | Edit the typed name |
| Esc | Cancel name entry, waiting, or capture |
| Tab | Select the next saved person |
| E | Add one pose to the selected person's samples; exactly one person must be visible |
| C | Clear only the selected person's samples and save that change |
| R | Reset matching confirmation without deleting enrollment |
| + / - | Raise / lower the similarity threshold |
| Q | Quit when outside name entry |

The selected name and sample count appear at the bottom. To name previously saved unnamed samples, select **Unnamed target** with Tab, press N, enter the name, and press Enter.

You can also select a profile at startup:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --name "Fanny"
```

## Read the results

- **Green name and score:** a saved appearance match confirmed over three consecutive frames.
- **Red UNKNOWN PERSON:** the detection has no confirmed named match.
- **BODY_NOT_ENROLLED:** there are no enrolled body samples.
- **CONFIRMING_BODY:** a possible match is waiting for confirmation.
- **AMBIGUOUS_BODY:** competing matches are too similar to select confidently.
- **NO PERSON DETECTED:** the body detector did not find anyone in the frame.

The default match threshold is 0.70. Lowering it makes matches easier and increases the chance of false matches. Different enrolled people can each show their own name in the same frame; one name is never assigned to two boxes in one frame.

## Saved data and repository contents

Names and body features are stored locally in `Face recognition/data/db/body_target.npz`, which is created when samples are saved. Clones start with no enrolled people. Legacy face profiles do not enroll someone in the body tracker.

The Git ignore rules exclude enrollment images, body and face databases, enrollment backups, runtime logs, virtual environments, and the large downloadable models. These files remain on the local machine and are not part of the intended upload. Do not manually force-add personal enrollment data.

```text
Face recognition/
  src/                   Detection, body features, tracking, and legacy face tools
  models/                Included detector, model downloader, and third-party license
  tests/                 Regression and optional real-image integration tests
  mqtt/                  Optional MQTT client
  hardware/              Hardware integration directory
  requirements.txt       Body tracking dependencies
  requirements-mqtt.txt  Optional MQTT dependencies
  Start Body Tracking.bat
```

Legacy face scripts remain in `src/` for reference. They are not required for named body enrollment and need separate face models and MediaPipe dependencies.

## Optional MQTT hardware

The webcam application works without a broker. To enable hardware output, install the optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-mqtt.txt
```

Configure the broker host, port, username, password, and topic in `mqtt/client.py` for your own setup, start your broker, then run:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking --mqtt --name "Fanny"
```

The selected person's position produces numeric angles on the configured topic. Missing selected targets return the output to center. The repository contains placeholder credentials; do not commit real credentials. A complete device firmware or wiring guide is not included.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Everyone is unknown | Complete named enrollment and wait for the saved-name message; check the selected profile's sample count. |
| Name accepted but poses do not start | Keep exactly one person in view and show their whole body. The name stays accepted while the camera waits. |
| Name prompt seems to repeat | Close the old running camera and restart the updated code. Press A only once, enter the name, and watch for Name accepted. |
| Capture repeatedly restarts | Improve lighting, move farther back, keep other people out of view, and turn slowly while staying in frame. |
| N keeps asking for a name | N is the rename shortcut. Use A to capture a new person. |
| Camera will not open | Close other camera apps, check Windows camera permissions, and try `--camera 1`. |
| Body model not found | Run `python models/download_reid.py` using the installed environment and wait for verification. |
| Missing Python module | Install `requirements.txt` using the same Python executable used to launch the tracker. |
| Model download fails | Check internet access and retry the downloader. It requires HTTP range support and verifies the final file before saving. |
| Person stops matching after changing clothes | Re-enroll their profile in the new outfit. |

## Tests and validation

From `Face recognition`, after installing dependencies and the body model:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite contains 20 tests covering detector decoding, input preprocessing, confirmation, rejection, ambiguity, named profile persistence, legacy sample migration, renaming, multiple simultaneous identities, and the name-entry/capture flow with missing detections.

Two real-image integration tests use an optional public fixture. To enable them locally:

```powershell
New-Item -ItemType Directory -Force tests/fixtures
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg" -OutFile tests/fixtures/bus.jpg
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Without the fixture those integration tests are skipped. The image is excluded from Git; its upstream license applies. In the local photo test, three people were detected and cross-person similarities with the replacement model were about 0.31–0.40, below the default 0.70 threshold. All 20 tests passed locally with the fixture. Camera 0 also returned a 640×480 frame.

These checks do not establish real-world identification accuracy. Test each enrolled person facing forward, sideways, and backward, then test unenrolled people, similar outfits, multiple people, crossings, and re-entry under your actual lighting. No deployment accuracy percentage has been measured.

## Models and attribution

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX): bundled Nano person detector; Apache 2.0. [Reference ONNX inference](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py).
- [OpenCV Zoo YoutuReID](https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtureid): downloaded full-precision 768-dimensional body features; Apache 2.0. License text is in `Face recognition/models/YoutuReID-LICENSE.txt`.
- [Ultralytics test image](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg): optional local test fixture, subject to its upstream license.

The previous quantized body model and legacy face models are not used by the supported body tracking workflow and are not uploaded. Third-party licenses apply to their respective components. No license for this project's original code has been selected yet.
