"""Can this rig start an experiment right now, and if not, what is the one thing to fix?

WHY A STRIP OF SENTENCES AND NOT A LOG. The failures this catches are the ones that cost days
rather than minutes, and they all look identical while they are happening: a run starts, frames are
processed, a CSV fills up, and nothing is wrong until someone reads the results. A calibration
bundle with no face templates attributes every vial on face B to face A for three days
(`cli.face_id_readiness` was written for exactly that). A camera value armed with no camera open
puts 0.1 fps into the config the next run reads. An output folder that cannot be written to is
found at the END of the first bin.

None of these produce an error at the moment they matter, so none of them can be reported by
failing. They have to be VISIBLE BEFORE the start button, in a sentence with no jargon in it, next
to a button that fixes the specific thing.

THE RULES THIS FILE KEEPS.

  * NEVER PROBE THE CAMERA. USB3 Vision is exclusive; a readiness check that opened the camera to
    see whether it opens would be the thing holding it. Camera state is reported from what the app
    already knows (`CameraState`), never from a fresh open.
  * A CHECK THAT CANNOT BE MADE SAYS SO. `Check.state` has three values, not two: an unknown is
    `UNKNOWN` and reads "not checked", never a tick. Claiming readiness nobody measured is the
    same class of error as claiming a measurement nobody made.
  * NO FILESYSTEM WRITES. `_writable` uses `os.access` rather than touching a probe file into the
    operator's output folder.

Toolkit-free on purpose (see `settings_model`): the CLI can print this, the Qt strip renders it,
and `tests/test_readiness.py` drives it with plain temp folders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

OK = "ok"
BAD = "bad"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Check:
    """One readiness line: a state, a sentence anyone can act on, and how to fix it.

    `fix_action` is a stable identifier (not a label) so the UI can wire a button to it without
    matching on prose -- prose gets reworded, and a fix button that silently stops working because
    a sentence changed is worse than no button.
    """

    key: str
    state: str
    sentence: str
    fix_label: str = ""
    fix_action: str = ""

    @property
    def ok(self) -> bool:
        return self.state == OK

    def mark(self) -> str:
        """A tick, a cross, or a dash -- the same three glyphs the CLI and the strip both use."""
        return {OK: "ok", BAD: "X", UNKNOWN: "-"}[self.state]


def _writable(path: str) -> bool:
    """True if `path` (or the nearest existing parent) can be written to. Writes nothing."""
    probe = os.path.abspath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return os.access(probe, os.W_OK)


def check_config(path: Optional[str]) -> Check:
    if not path:
        return Check("config", BAD, "No config file is chosen, so there is nothing to run from.",
                     "Choose a config file", "pick_config")
    if not os.path.isfile(path):
        return Check("config", BAD, "The config file %s does not exist." % path,
                     "Choose a config file", "pick_config")
    return Check("config", OK, "Config file: %s" % path)


def check_calibration(path: Optional[str]) -> Check:
    """Is there a vial-position bundle to run against?

    Only the presence of `calibration.json` is checked, not its contents. Whether the bundle can
    IDENTIFY the two drum faces is a different and much subtler question, already answered by
    `cli.face_id_readiness` at run start; duplicating half of it here would produce two verdicts
    that could disagree.
    """
    if not path:
        return Check("calibration", BAD, "No vial-position folder is chosen.",
                     "Choose a folder", "pick_calib")
    bundle = os.path.join(path, "calibration.json")
    if not os.path.isfile(bundle):
        return Check("calibration", BAD,
                     "No vial positions in %s - they have to be drawn once before a run." % path,
                     "Draw vial positions", "draw_vials")
    return Check("calibration", OK, "Vial positions: %s" % path)


def check_output(path: Optional[str]) -> Check:
    if not path:
        return Check("output", BAD, "No output folder is chosen, so results have nowhere to go.",
                     "Choose a folder", "pick_output")
    if os.path.isdir(path) and not _writable(path):
        return Check("output", BAD, "The output folder %s cannot be written to." % path,
                     "Choose a folder", "pick_output")
    if not os.path.isdir(path):
        if not _writable(path):
            return Check("output", BAD,
                         "The output folder %s does not exist and cannot be created." % path,
                         "Choose a folder", "pick_output")
        return Check("output", OK, "Results will go to %s (created at the start of the run)."
                     % path)
    return Check("output", OK, "Results go to %s" % path)


def check_camera(state: str, detail: str = "") -> Check:
    """Camera ownership, from what the app already knows. NEVER opens anything.

    "Not open" is not a failure: the app deliberately does not take the camera until asked, because
    an app that grabs an exclusive device at launch is an app that blocks the rig. It is reported
    as UNKNOWN rather than OK because the limits on screen are then the rig camera's documented
    ones rather than this sensor's, and a tick would claim otherwise.
    """
    if state == "streaming":
        return Check("camera", OK, "Camera is open and this software has it%s."
                     % (" - %s" % detail if detail else ""))
    if state in ("error_busy", "error_other"):
        return Check("camera", BAD,
                     "The camera cannot be opened%s." % (" - %s" % detail if detail else ""),
                     "See what is holding it", "free_camera")
    return Check("camera", UNKNOWN,
                 "Camera is not open. Settings can still be edited; the limits shown are the rig "
                 "camera's documented ones, not this sensor's.",
                 "Open camera", "open_camera")


def check_unverified(never_checked: Sequence[str], labels: Optional[dict] = None) -> Check:
    """The `settings_controller` hazard, surfaced where an operator will meet it before saving."""
    if not never_checked:
        return Check("unverified", OK, "Every camera value on screen was read from a camera or "
                                       "left at the camera default.")
    labels = labels or {}
    named = ", ".join(labels.get(k, k) for k in never_checked)
    return Check("unverified", BAD,
                 "%s was set with no camera open, so it is the lowest legal value rather than "
                 "anything a sensor confirmed." % named.capitalize(),
                 "Open camera", "open_camera")


def check_unsaved(n_changed: int, config_path: Optional[str]) -> Check:
    if not n_changed:
        return Check("unsaved", OK, "No unsaved setting changes.")
    return Check("unsaved", UNKNOWN,
                 "%d setting change(s) are not in %s yet, so a run started now would use the "
                 "file's old values."
                 % (n_changed, config_path or "the config file"),
                 "Save to config", "save")


@dataclass
class Readiness:
    """The whole strip, in the order an operator would work through it."""

    checks: List[Check] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when nothing is a cross. An UNKNOWN never blocks -- "the camera is not open" is a
        legitimate state to edit settings in, and a strip that demanded a perfect score before it
        stopped nagging would be a strip nobody reads."""
        return not any(c.state == BAD for c in self.checks)

    def problems(self) -> List[Check]:
        return [c for c in self.checks if c.state == BAD]

    def text(self) -> str:
        """The strip as plain text -- what the CLI would print, and what a test asserts on."""
        return "\n".join("[%s] %s" % (c.mark(), c.sentence) for c in self.checks)


def evaluate(*, config_path=None, calib_dir=None, output_dir=None, camera_state="closed",
             camera_detail="", never_checked=(), labels=None, n_changed=0) -> Readiness:
    """Every check, in operator order: what to run, what to run it on, where it goes, then health."""
    return Readiness([
        check_config(config_path),
        check_calibration(calib_dir),
        check_output(output_dir),
        check_camera(camera_state, camera_detail),
        check_unverified(list(never_checked), labels),
        check_unsaved(n_changed, config_path),
    ])
