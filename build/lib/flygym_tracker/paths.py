"""Where the program lives, and where the operator's own files go -- which are NOT the same place.

RUNNING FROM A CLONE they are the same folder and none of this matters: the config templates, the
operator's `flygym_rig.local.yaml`, `gui_state.json` and `output/` all sit in the repo, which is
writable because it is somebody's working copy.

INSTALLED, THEY MUST NOT BE. A customer's copy lands in `C:\\Program Files\\FlyGym Tracker`, and
Windows does not let a normal user write there. Everything this program saves -- the machine's own
config, the window's remembered paths, the vial positions, and the RESULTS OF AN EXPERIMENT --
would fail to write. On modern Windows it fails in the worst possible way: not with an error, but
silently, into a per-user shadow copy under `AppData\\Local\\VirtualStore`, so the app looks like
it saved and the files are not where anyone will look for them.

That failure would land on a three-day experiment. Hence two roots:

    bundle_root()      READ-ONLY. The shipped config templates. Inside the install directory.
    user_data_root()   WRITABLE. Settings, calibration and results. Under the user's own profile.

`sys.frozen` is what PyInstaller sets on a built app, and it is the only thing that distinguishes
the two situations -- so it is asked once, here, rather than in each of the four places that used
to walk up from `__file__`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: The folder created under the user's profile for an INSTALLED copy. Documents rather than
#: AppData: the results of an experiment are the scientist's own data, and data a person is
#: expected to open, copy to a colleague and back up does not belong in a hidden folder.
INSTALLED_FOLDER_NAME = "FlyGym Tracker"


def is_frozen() -> bool:
    """True when running from a PyInstaller build rather than from a source clone."""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Where the program's OWN files are -- config templates, and nothing writable.

    Frozen, this is the install directory (`sys._MEIPASS` covers the one-file case, where the
    bundle is unpacked to a temp folder that exists only while the app runs). From source it is the
    repo root, three levels up from this file: `<repo>/src/flygym_tracker/paths.py`.
    """
    if is_frozen():
        base = getattr(sys, "_MEIPASS", None)
        return Path(base) if base else Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def user_data_root() -> Path:
    """Where everything this program SAVES goes. Always writable, never inside the install.

    `FLYGYM_DATA_DIR` overrides it, which is what a shared rig or a locked-down lab machine needs
    -- a fixed folder on a data drive that several accounts can reach.
    """
    override = os.environ.get("FLYGYM_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if not is_frozen():
        # A clone is somebody's working copy: keep the old behaviour exactly, so a developer's
        # config, state file and output folder stay where they have always been.
        return bundle_root()
    return _documents_dir() / INSTALLED_FOLDER_NAME


def _documents_dir() -> Path:
    """The user's Documents folder, asking Windows rather than assuming `~/Documents`.

    IT IS OFTEN NOT `~/Documents`. With OneDrive Known Folder Move -- which is on by default on
    consumer Windows and is on THIS rig -- Documents is redirected into the OneDrive folder, and
    the literal `~/Documents` either does not exist or is a stale empty one. Writing a three-day
    experiment into the wrong one of those is how results go missing.
    """
    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes

            # SHGetFolderPathW with CSIDL_PERSONAL (5) resolves the redirect.
            buffer = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buffer) == 0 and buffer.value:
                return Path(buffer.value)
        except Exception:
            pass                      # fall through to the home-directory guess
    guess = Path.home() / "Documents"
    return guess if guess.is_dir() else Path.home()


def ensure_user_data_root() -> Path:
    """`user_data_root()`, created if it is not there. Never raises.

    A folder that cannot be created is not a reason to refuse to start -- the callers all have a
    read-only fallback -- but it IS a reason for the path to still be returned, so whatever fails
    next fails naming the folder the operator would have to fix.
    """
    root = user_data_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return root


def default_output_dir() -> str:
    """Where results go before anyone chooses otherwise.

    ABSOLUTE WHEN INSTALLED, and that is the point. The default was the relative `"output"`, which
    resolves against the working directory -- and a desktop shortcut starts the app in whatever
    folder Windows feels like, so the same button would write a run's results to a different place
    depending on how the app was launched.
    """
    if not is_frozen():
        return "output"
    return str(user_data_root() / "output")


def default_config_path() -> str:
    """This machine's own config file, before anyone chooses otherwise.

    THE SAME RELATIVE-PATH TRAP as `default_output_dir`, and it bit harder here because it is
    hidden: `"config/flygym_rig.local.yaml"` resolves against the working directory, so an
    installed copy launched from a desktop shortcut would create -- and then read -- its settings
    from whatever folder Windows started it in. Two launches from two places would each have their
    own settings, and neither would be findable.
    """
    if not is_frozen():
        return "config/flygym_rig.local.yaml"
    return str(user_data_root() / "config" / "flygym_rig.local.yaml")


def default_calib_dir() -> str:
    """Where the vial positions live before anyone chooses otherwise. See `default_output_dir`."""
    if not is_frozen():
        return "calib_faces"
    return str(user_data_root() / "calib_faces")
