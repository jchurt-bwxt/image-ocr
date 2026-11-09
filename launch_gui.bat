@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Create the .venv environment and install requirements as described in README.md first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "launch_gui.pyw"
