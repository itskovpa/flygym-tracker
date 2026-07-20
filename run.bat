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

REM ---- OpenCV's GUI build is no longer needed BY THE APP ----
REM      It used to be checked here and offered as a fix, because the app launched
REM      the vial editor and the live monitor as child processes with cv2 windows.
REM      It does not any more: drawing vial positions, replaying a recording,
REM      measuring the noise floor and learning the drum faces all happen inside
REM      the app's own window, drawn with Qt (see gui/video_stage.py). OpenCV is
REM      still used for the MATHS - masks, contours, frame differences - and that
REM      works identically in the headless build.
REM
REM      The `cli` subcommands (select-vials, edit-rois, replay --monitor) DO still
REM      open cv2 windows when run from a terminal, and they check for themselves.

REM ---- the one line this file exists for ----
%PY% -m flygym_tracker.cli gui
if errorlevel 1 (
  echo.
  echo   The app closed with an error. The message above is the reason.
  echo.
  pause
)
exit /b 0
