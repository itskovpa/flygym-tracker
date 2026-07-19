@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title FlyGym v2 Tracker

REM ============================================================================
REM  THE PATHS ARE NO LONGER EDITED HERE. The app owns them: choose the config
REM  file, the vial-position folder and the output folder at the top of its
REM  window, and this menu reads back whatever the app last saved. Editing them
REM  in Notepad was the thing the app was built to replace, and keeping a second
REM  copy in this file would mean choosing an output folder in the app and still
REM  getting results somewhere else from option [1].
REM
REM  BIN_SECONDS is NOT here either. It decides what one row of the results
REM  MEANS, which makes it a measurement parameter: it lives in the config YAML
REM  next to the thresholds it will be compared against, and it is a settings row
REM  in the app.
REM ============================================================================
set "CONFIG=config\flygym_rig.yaml"
set "CALIB=calib_faces"
set "OUTDIR=output"

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

REM ---- check dependencies once ----
%PY% -c "import numpy, cv2, pandas, yaml, openpyxl" >nul 2>&1
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

REM ---- the app needs PySide6; everything else works without it ----
set "HAVE_GUI=1"
%PY% -c "import PySide6" >nul 2>&1
if errorlevel 1 (
  set "HAVE_GUI="
  echo.
  echo   The settings app needs PySide6, which is not installed.
  echo   Without it, everything still works from this menu and from the terminal.
  set "INSTALLQT="
  set /p INSTALLQT="  Install PySide6 now? [Y/n]: "
  if /I not "!INSTALLQT!"=="n" (
    %PY% -m pip install PySide6
    if not errorlevel 1 set "HAVE_GUI=1"
  )
)

REM ---- OpenCV must be the GUI build for the vial editor and the live monitor ----
REM      The settings APP does not need it: it draws with Qt, not with cv2.
%PY% -c "from flygym_tracker.gui_support import has_gui_support; raise SystemExit(0 if has_gui_support() else 3)" >nul 2>&1
if errorlevel 3 (
  echo.
  echo   NOTE: this OpenCV build has no GUI support ^(opencv-python-headless^).
  echo   Drawing vial positions and the live monitor cannot open a window.
  echo   The settings app is unaffected - it does not use OpenCV to draw.
  set "FIXCV="
  set /p FIXCV="  Fix it now? [Y/n]: "
  if /I not "!FIXCV!"=="n" (
    %PY% -m pip uninstall -y opencv-python-headless
    %PY% -m pip install opencv-python
  )
)

:menu
cls
REM ---- read the paths back from the app, so this menu and the app agree ----
for /f "usebackq tokens=1,* delims==" %%a in (`%PY% -m flygym_tracker.cli gui --print-paths 2^>nul`) do set "%%a=%%b"

echo ============================================
echo    FlyGym v2  -  Drosophila Activity Tracker
echo ============================================
echo.
echo    config : %CONFIG%
echo    calib  : %CALIB%
echo    output : %OUTDIR%
echo    ^(change these in the app - option [A]^)
echo.
REM  THE NUMBERS DO NOT MOVE. [1]..[5] mean what they have always meant, because they are what is
REM  written on the note stuck to the rig and what is in the operator's fingers. The app is a NEW
REM  letter at the top and the DEFAULT on Enter, which is how it becomes the obvious way in
REM  without breaking a habit. (tests/test_cli.py asserts both halves of this.)
echo    [A]  Settings and camera        THE APP: all settings, live preview, folders
echo    [S]  Settings, old style        tracking + camera sliders; superseded by [A]
echo.
echo    [1]  Start experiment           (asks about vial positions, then tracks)
echo    [2]  Draw vial positions        (16 polygons on the live feed; both faces)
echo    [3]  Replay a recorded video
echo    [4]  Measure noise floor
echo    [5]  Free the camera            (find what is holding it and stop it)
echo    [Q]  Quit
echo.
set "CH="
set /p CH="   Choose [A]: "
if not defined CH set "CH=A"

if /I "%CH%"=="A" goto app
if /I "%CH%"=="S" goto settings
if /I "%CH%"=="1" goto run
if /I "%CH%"=="2" goto selectvials
if /I "%CH%"=="3" goto replay
if /I "%CH%"=="4" goto noise
if /I "%CH%"=="5" goto freecam
if /I "%CH%"=="Q" exit /b 0
goto menu

:app
if not defined HAVE_GUI (
  echo.
  echo   PySide6 is not installed, so the app cannot open.
  echo     %PY% -m pip install PySide6
  echo   Option [S] opens the old OpenCV settings panel in the meantime.
  echo.
  pause
  goto menu
)
echo.
echo   Everything is in one window: tracking and camera settings, a live preview
echo   to set exposure and gain against, and the config/vial/output folders.
echo.
echo   The camera is NOT opened until you press "Open camera" - only one program
echo   at a time may use it, so the app does not take it just for being open.
echo   Camera rows are either an explicit value this software sends, or
echo   "camera default", which means NOTHING is sent and the camera keeps
echo   whatever MVS was set to.
echo.
%PY% -m flygym_tracker.cli gui --config "%CONFIG%"
goto done

:settings
echo.
echo   The OLD panel. Prefer [A]: this one needs the OpenCV GUI build, has no
echo   preview to judge exposure against, and will be removed once the app grows
echo   its run view. Drag a slider or use the arrow keys; s saves, q closes.
echo   Camera rows are either an explicit value this software sends, or "camera
echo   default", which means NOTHING is sent and the camera keeps what MVS set.
echo   Image width/height only take effect when a run starts.
echo.
%PY% -m flygym_tracker.cli settings --config "%CONFIG%"
goto done

:freecam
echo.
echo   Only one program at a time may use the camera. The usual culprit is a
echo   leftover Bonsai started with --no-editor: it has NO WINDOW, so the rig
echo   looks idle while the camera is still locked.
echo.
%PY% -m flygym_tracker.cli free-camera
goto done

:run
echo.
echo   The round starts by offering the vial positions saved in "%CALIB%".
echo   Press ENTER to reuse them, or answer n to draw them again on the live feed.
echo   In the monitor window, press t for the tracking + camera sliders (drag them
echo   while the run continues; s in that window saves them to "%CONFIG%").
echo   To set them BEFORE starting - including the image size, which cannot change
echo   mid-run at all - quit here and choose [A] (or the older [S]) instead.
echo   Then: close the monitor window, or press Ctrl+C here, to stop the experiment.
echo   IMPORTANT: make sure MVS and the settings app are CLOSED - the camera
echo   allows only one program at a time.
echo.
%PY% -m flygym_tracker.cli run --config "%CONFIG%" --calib "%CALIB%" --out "%OUTDIR%" --monitor
goto done

:selectvials
echo.
echo   Draw one polygon around each vial on ONE face; the other face gets the same
echo   positions automatically, so 16 drawings cover all 32 vials.
echo     left click = add a corner        ENTER     = this vial is done, next one
echo     BACKSPACE  = undo a corner       u         = redo the previous vial
echo     c          = clear this vial     SPACE     = freeze the picture to click
echo     q / ESC    = stop early, keeping the vials drawn so far
echo   IMPORTANT: make sure MVS and the settings app are CLOSED - the camera
echo   allows only one program at a time.
echo.
set "VID="
set /p VID="   Video to draw on (blank = draw on the LIVE camera): "
if not defined VID (
  %PY% -m flygym_tracker.cli select-vials --out "%CALIB%" --config "%CONFIG%"
) else (
  %PY% -m flygym_tracker.cli select-vials --out "%CALIB%" --video "%VID%"
)
goto done

:replay
echo.
echo   Replaying the SAME clip after changing a setting is how the tracking is tuned:
echo   adjust in the app, watch, then replay again.
echo.
set "VID="
set /p VID="   Path to the video file: "
if not defined VID goto menu
%PY% -m flygym_tracker.cli replay --video "%VID%" --config "%CONFIG%" --calib "%CALIB%" --out "%OUTDIR%" --monitor
goto done

:noise
echo.
%PY% -m flygym_tracker.cli noise --config "%CONFIG%" --calib "%CALIB%"
goto done

:done
echo.
if errorlevel 1 (
  echo   ---------------------------------------------------
  echo   Finished with an ERROR - the message above says why.
  echo   ---------------------------------------------------
) else (
  echo   Done. Results are in: %OUTDIR%
)
echo.
pause
goto menu
