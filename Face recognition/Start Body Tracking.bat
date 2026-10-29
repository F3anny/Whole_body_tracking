@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto system_python
".venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 goto system_python
".venv\Scripts\python.exe" -m src.face_tracking %*
if errorlevel 1 pause
exit /b
:system_python
python -m src.face_tracking %*
if errorlevel 1 pause
