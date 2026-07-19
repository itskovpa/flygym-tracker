@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title FlyGym v2 Tracker

REM ============ settings you may want to edit ============
set "CONFIG=config\flygym_rig.yaml"
set "CALIB=calib_faces"
set "OUTDIR=output"
set "BIN_SECONDS=60"
REM =======================================================

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

:menu
cls
echo ============================================
echo    FlyGym v2  -  Drosophila Activity Tracker
echo ============================================
echo.
echo    config : %CONFIG%
echo    calib  : %CALIB%
echo    output : %OUTDIR%     bin: %BIN_SECONDS%s
echo.
echo    [1]  Start experiment      (live camera + live monitor)
echo    [2]  Calibrate vial ROIs   (do this before a new experiment)
echo    [3]  Replay a recorded video
echo    [4]  Measure noise floor
echo    [Q]  Quit
echo.
set "CH="
set /p CH="   Choose [1]: "
if not defined CH set "CH=1"

if /I "%CH%"=="1" goto run
if /I "%CH%"=="2" goto calib
if /I "%CH%"=="3" goto replay
if /I "%CH%"=="4" goto noise
if /I "%CH%"=="Q" exit /b 0
goto menu

:run
echo.
echo   Starting acquisition. Close the monitor window, or press Ctrl+C here, to stop.
echo   IMPORTANT: make sure MVS is CLOSED - the camera allows only one program at a time.
echo.
%PY% -m flygym_tracker.cli run --config "%CONFIG%" --calib "%CALIB%" --out "%OUTDIR%" --bin-seconds %BIN_SECONDS% --monitor
goto done

:calib
echo.
set "VID="
set /p VID="   Video file to calibrate from (blank = grab live from camera): "
if not defined VID (
  %PY% -m flygym_tracker.cli calibrate --from-camera --out "%CALIB%" --config "%CONFIG%" --wizard
) else (
  %PY% -m flygym_tracker.cli calibrate-faces --video "%VID%" --out "%CALIB%" --config "%CONFIG%"
)
goto done

:replay
echo.
set "VID="
set /p VID="   Path to the video file: "
if not defined VID goto menu
%PY% -m flygym_tracker.cli replay --video "%VID%" --config "%CONFIG%" --calib "%CALIB%" --out "%OUTDIR%" --bin-seconds %BIN_SECONDS% --monitor
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
