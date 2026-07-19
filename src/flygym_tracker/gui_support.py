"""Detect whether this OpenCV build can actually open a window, and say what to do if not.

`opencv-python-headless` installs the SAME `cv2` module name as `opencv-python` but is built with
``GUI: NONE``, so `cv2.imshow` raises a confusing "The function is not implemented. Rebuild the
library with Windows, GTK+ 2.x or Cocoa support" error. It is commonly pulled in as a transitive
dependency of some other package, silently shadowing a working install.

Every interactive feature here (calibration wizard, ROI editor, live monitor) needs a window, so we
check up front and fail with an actionable message instead of a traceback from deep inside cv2.
"""
from __future__ import annotations

import re


def headless_package_installed() -> bool:
    """True if the GUI-less `opencv-python-headless` distribution is installed."""
    try:
        from importlib.metadata import distributions
    except Exception:  # pragma: no cover - importlib.metadata is stdlib on supported versions
        return False
    for dist in distributions():
        try:
            name = (dist.metadata["Name"] or "").strip().lower()
        except Exception:
            continue
        if name == "opencv-python-headless":
            return True
    return False


def gui_backend() -> str:
    """Return the GUI backend reported by the cv2 build (e.g. 'WIN32UI', 'GTK3', 'NONE', 'UNKNOWN')."""
    try:
        import cv2
        for line in cv2.getBuildInformation().splitlines():
            m = re.match(r"\s*GUI:\s*(\S+)", line)
            if m:
                return m.group(1).strip().upper()
    except Exception:
        pass
    return "UNKNOWN"


def has_gui_support() -> bool:
    """True if this cv2 build reports a usable GUI backend."""
    backend = gui_backend()
    return backend not in ("NONE", "UNKNOWN", "")


def gui_diagnosis(feature: str = "this feature") -> str:
    """A human-readable, copy-pasteable explanation of how to get a working GUI."""
    lines = [
        f"{feature} needs to open a window, but this OpenCV build has no GUI support",
        f"(cv2 reports  GUI: {gui_backend()}).",
        "",
    ]
    if headless_package_installed():
        lines += [
            "Cause: 'opencv-python-headless' is installed. It provides the same 'cv2' module but is",
            "built without any window support, so it shadows the normal build.",
            "",
            "Fix (run these two commands):",
            "    python -m pip uninstall -y opencv-python-headless",
            "    python -m pip install opencv-python",
        ]
    else:
        lines += [
            "Fix: install the GUI-enabled OpenCV build:",
            "    python -m pip uninstall -y opencv-python-headless",
            "    python -m pip install opencv-python",
        ]
    lines += [
        "",
        "Then re-run. (Headless machines can still RUN experiments - only the interactive",
        "windows are unavailable; the tracker itself logs to CSV without a display.)",
    ]
    return "\n".join(lines)


def require_gui(feature: str = "this feature") -> None:
    """Raise SystemExit(2) with an actionable message if no window can be opened."""
    if not has_gui_support():
        raise SystemExit("\nERROR: " + gui_diagnosis(feature) + "\n")
