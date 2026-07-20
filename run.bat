@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title FlyGym v2 Tracker

REM ============================================================================
REM  THIS FILE IS A LAUNCHER. IT USED TO BE THE APPLICATION.
REM
REM  It was a numbered menu over `python -m flygym_tracker.cli`: [1] run,
REM  [2] draw vial positions, [3] replay, [4] noise floor, [5] free the camera.
REM  Every one of those is a BUTTON IN THE WINDOW now, so a menu here would be a
REM  second way to do the same five jobs -- and a second place for the paths, the
REM  defaults and the wording to drift out of step with the app.
REM
REM  WHAT IS LEFT IS THE PART A WINDOW CANNOT DO FOR ITSELF: find a Python, put
REM  the package on the path, and check the imports the app needs BEFORE the app
REM  tries to open. A missing PySide6 has to be reported by something that is not
REM  PySide6, or the operator gets a traceback with no window behind it.
REM
REM  THE PATHS ARE NOT SET HERE ANY MORE. The app owns them -- config file, vial
REM  positions and output folder are chosen at the top of its window and saved.
REM  Keeping a copy here meant choosing an output folder in the app and still
REM  getting results somewhere else.
REM ============================================================================

REM ---- locate Python (prefer the py launcher) ----
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo.
  echo   ERROR: Python 3 was not found on this computer.
  echo   Install it from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)

REM ---- make the package importable without installing it ----
set "PYTHONPATH=%CD%\src"

REM ---- the imports the window needs before it can report anything itself ----
%PY% -c "import numpy, cv2, pandas, yaml, openpyxl, PySide6" >nul 2>&1
if errorlevel 1 (
  echo.
  echo   Some required Python packages are missing.
  set "INSTALL="
  set /p INSTALL="  Install them now? [Y/n]: "
  if /I "!INSTALL!"=="n" exit /b 1
  echo.
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo   Package install failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)

REM ---- OpenCV must be the GUI build for the vial editor and the live monitor ----
REM      The app itself does not need it: it draws with Qt, not with cv2. But the
REM      tools it LAUNCHES as child processes (draw vial positions, replay with
REM      the monitor) do, and finding that out from a child process that silently
REM      fails to open a window is worse than being told here.
%PY% -c "from flygym_tracker.gui_support import has_gui_support; raise SystemExit(0 if has_gui_support() else 3)" >nul 2>&1
if errorlevel 3 (
  echo.
  echo   NOTE: this OpenCV build has no GUI support ^(opencv-python-headless^).
  echo   Drawing vial positions and the live monitor cannot open a window.
  echo   The rest of the app is unaffected - it does not use OpenCV to draw.
  set "FIXCV="
  set /p FIXCV="  Fix it now? [Y/n]: "
  if /I not "!FIXCV!"=="n" (
    %PY% -m pip uninstall -y opencv-python-headless
    %PY% -m pip install opencv-python
  )
)

REM ---- the one line this file exists for ----
%PY% -m flygym_tracker.cli gui
if errorlevel 1 (
  echo.
  echo   The app closed with an error. The message above is the reason.
  echo.
  pause
)
exit /b 0
