"""Where the app remembers which config, which vial folder and which output folder.

WHAT THIS REPLACES. `run.bat` carried these in a header block the operator was told to edit:

    set "CONFIG=config\\flygym_rig.yaml"
    set "CALIB=calib_faces"
    set "OUTDIR=output"

Editing a batch file in Notepad to choose an output folder is precisely what "I want all the
settings in one usable GUI, not as command line prompts" was about, so they become fields with file
pickers, and this module is where the answers live between sessions.

A VISIBLE JSON FILE, NOT QSettings AND NOT THE REGISTRY. `QSettings` on Windows writes to
HKEY_CURRENT_USER, where a scientist cannot see it, cannot copy it to the second rig, cannot put it
in a backup and cannot delete it when it goes wrong. This is one readable file next to the program,
and "delete it and start again" is a support instruction that works.

IT IS LOADED DEFENSIVELY, FOR THE SAME REASON `config.py` MERGES ONTO PACKAGED DEFAULTS. This file
is written by a future version of the app and read by whichever version the rig happens to have. So
loading NEVER raises: a missing file, an empty file, a truncated file, a hand-edit that broke the
JSON, a key that is now a different type, a key that no longer exists -- every one of those loads
the defaults for the parts it cannot use and keeps the parts it can. A startup KeyError on a lab
machine at the beginning of an experiment is a worse outcome than a forgotten folder, every time.

NOT HERE: `bin_seconds`. It looks like a fourth item in that batch-file block, and it is not the
same kind of thing. It decides what one row of the results MEANS, which makes it a measurement
parameter belonging in the experiment's YAML next to the thresholds it will be compared against --
and, through `run_meta.json`, in the record of the run. `settings_model.build_app_settings` puts it
on screen as a settings row instead, which is how it stays in exactly one place.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

#: Where the file lives: next to the repo, not in the user profile, so it travels with the install
#: and is visible in the same folder the operator already opens.
STATE_FILENAME = "gui_state.json"

#: Bumped only when a change cannot be handled by the merge below -- which, so far, none has been.
#: Present from version 1 so a future migration has something to branch on rather than guessing.
SCHEMA_VERSION = 1

#: The shape, and the only place a default is written down. `_coerce` is driven off these types.
DEFAULTS: Dict[str, Any] = {
    "version": SCHEMA_VERSION,
    #: THIS MACHINE'S OWN config, not the shipped template. It layers on top of
    #: `config/flygym_rig.yaml` (see `config.load_config`), so the rig gets every value the
    #: template carries plus whatever was tuned here -- and the template stays what a fresh clone
    #: gets rather than a record of the last operator's experiment.
    "config_path": "config/flygym_rig.local.yaml",
    "calib_dir": "calib_faces",
    "output_dir": "output",
    "recent_configs": [],
    #: The last recording a video job was pointed at, so the file picker opens where the operator
    #: was rather than at the top of the disk. Listed HERE because `save_state` writes only the
    #: keys in this table -- a key the window sets but this dict does not know about is dropped on
    #: the way to disk, silently, and looks like a setting that will not stick.
    "last_video": "",
}

#: How many entries the config dropdown keeps. Small: this is a shortcut, not a history feature,
#: and a long list is harder to pick from than a short one.
MAX_RECENT = 8


def default_state() -> Dict[str, Any]:
    return json.loads(json.dumps(DEFAULTS))       # deep copy, no aliasing of the list


def state_path(root: str) -> str:
    return os.path.join(root, STATE_FILENAME)


def _coerce(key: str, value: Any) -> Any:
    """`value` if it matches the shape `key` is supposed to have, else the default for `key`.

    A hand-edited file that put a number where a path goes, or a string where the recent-list goes,
    must not reach the widgets: a `QComboBox` handed an int raises somewhere far from the cause.
    """
    default = DEFAULTS[key]
    if isinstance(default, list):
        if not isinstance(value, list):
            return list(default)
        return [str(v) for v in value if isinstance(v, str) and v.strip()][:MAX_RECENT]
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if value is None:
        return default
    return str(value)


def load_state(root: str) -> Dict[str, Any]:
    """The saved state merged onto the defaults. NEVER raises -- see the module docstring."""
    state = default_state()
    path = state_path(root)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return state                       # missing, unreadable, or not JSON: defaults, silently
    if not isinstance(raw, dict):
        return state
    for key in DEFAULTS:
        if key in raw:
            state[key] = _coerce(key, raw[key])
    # Unknown keys are dropped rather than carried: they belong to a version that knows what they
    # mean, and this one would only write them back out wrong.
    state["version"] = SCHEMA_VERSION
    return state


def save_state(root: str, state: Dict[str, Any]) -> bool:
    """Write the state. True on success; a failure is reported, never raised.

    A read-only install directory is a real deployment (a shared rig, a locked-down lab machine),
    and being unable to remember a folder is not a reason to refuse to run.
    """
    clean = default_state()
    for key in DEFAULTS:
        if key in state:
            clean[key] = _coerce(key, state[key])
    try:
        with open(state_path(root), "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
            f.write("\n")
        return True
    except OSError:
        return False


def remember_config(state: Dict[str, Any], path: str) -> List[str]:
    """Move `path` to the front of the recent list. Returns the new list.

    Most-recent-first with duplicates removed, because the two configs anyone alternates between
    (the rig's and the one they are trying) should both stay one click away.
    """
    if not path:
        return list(state.get("recent_configs") or [])
    recent = [p for p in (state.get("recent_configs") or []) if p and p != path]
    recent.insert(0, path)
    state["recent_configs"] = recent[:MAX_RECENT]
    return state["recent_configs"]
