"""Make `flygym_tracker` importable in tests without an install (src layout)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
