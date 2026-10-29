# Whole Body Tracking

The complete setup, enrollment, controls, troubleshooting, model attribution, and test instructions are in the [repository README](../README.md).

After installing dependencies and downloading the body model, run from this folder:

```powershell
.\.venv\Scripts\python.exe -m src.face_tracking
```

Click the camera window, press **A**, type the person's name, and press **Enter**. Wait for **Name accepted**, follow the four pose prompts, then wait for **Saved [name]**. Use **N** only to rename existing samples.
