"""Make `flygym_tracker` importable in tests without an install (src layout)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# NO TEST RUN MAY OPEN A REAL CAMERA. Discovering a webcam means opening it -- UVC has no
# handle-free enumeration -- and the window now scans for every camera at startup. Without this,
# several hundred window constructions across the suite would each probe the developer's own
# devices: slow, dependent on whatever happens to be plugged in, and it flicks the machine's camera
# light on over and over while the tests run.
#
# Set here rather than in a fixture because the scan is reached deep inside `MainWindow`, which
# tests construct wholesale. `setdefault`, so a deliberate override still wins.
os.environ.setdefault("FLYGYM_NO_CAMERA_SCAN", "1")
