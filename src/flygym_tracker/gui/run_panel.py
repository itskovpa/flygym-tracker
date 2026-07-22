"""The run band: start/stop, what the run is doing right now, and the jobs that used to be in run.bat.

WHY THIS IS A BAND AND NOT A SECOND WINDOW. Exposure and gain are tuned by looking at the picture,
and the threshold is tuned by looking at what the vials report. A design that makes you switch
windows to see the effect of the knob you are turning is the cv2 panel's problem again, one level
up. The settings stay on screen, editable, while this band says what the run is doing with them.

NO TERMINAL PROMPT ANYWHERE. `run.bat` was a numbered menu over `python -m flygym_tracker.cli`, so
every job it offered was one the operator reached by reading a list and typing a digit. They are
buttons now.

AND THEY NO LONGER LAUNCH ANYTHING. These buttons used to start CHILD PROCESSES, one OpenCV window
each. That was not laziness: process DPI awareness is 0 (UNAWARE) before `QApplication(...)` and 2
(PER_MONITOR_AWARE) after, and `live_vial_selector.screen_view_limit` depends on the process
staying UNAWARE, because an AUTOSIZE OpenCV window is laid out at the frame's own pixel size and
`SM_CXFULLSCREEN` is the only thing that reports the desktop in that same coordinate space. On this
machine's 2880x1800 panel at 200% scaling, a DPI-aware process put the bottom rows of every frame
below the screen edge -- "not visible, and impossible to click... exactly the part of the tube the
operator most needs to enclose".

THE CONSTRAINT WENT AWAY WITH THE OpenCV WINDOW. A letterboxed Qt widget inside a layout is handed
a rectangle and fits the frame into it; it cannot run off the screen edge at any DPI, so nothing
needs to measure the desktop and nothing needs a second process to stay unaware of it. Each button
now emits `tool_requested`, and the window switches `video_stage` into the mode that does the job
IN the picture. See `video_stage`'s module docstring.

THE TOOL STRIP WRAPS. A QHBoxLayout's minimum width is the sum of its children; measured, this row
alone reported 1532 px against a 1440 px desktop. See `flow_layout` -- fourth occurrence of that
bug class on this project.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from flygym_tracker.gui.flow_layout import flow_strip
from flygym_tracker.gui.run_controller import DONE, FAILED, IDLE, RUNNING, STARTING, STOPPING

#: The drum: two faces of sixteen vials. Laid out as the rig is, so a dead column on screen is a
#: dead column on the drum without anyone having to map indices.
VIALS_PER_FACE = 16
N_FACES = 2


# `VialStrip` USED TO LIVE HERE and has been deleted rather than left beside its replacement.
# It coloured one cell per vial by activity and its own docstring called it "NOT A MEASUREMENT
# DISPLAY... a presence check" -- which was honest, and left the operator unable to tell 3 from 300
# without opening the file afterwards. `results_panel.VialActivityGrid` shows the NUMBER, next to
# the rows actually written to activity.csv. Keeping both would be two displays of one measurement,
# updated from the same signal by different code, free to disagree.
class RunPanel(QWidget):
    """Start/stop plus the live readout, and the four jobs that were in `run.bat`'s menu."""

    start_requested = Signal()
    stop_requested = Signal()
    #: A cv2 tool the operator asked for, by stable action name rather than by button label -- a
    #: button that silently stops working because a label was reworded is worse than no button.
    tool_requested = Signal(str)
    #: A behavioural parameter to plot, by `behaviour_series` field name.
    plot_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._running = False
        self._stage_busy = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.start_button = QPushButton("Start experiment")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start_requested)
        controls.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("role", "danger")
        self.stop_button.setToolTip(
            "Finish the current bin, write the last rows and close the files. The run is never "
            "killed outright - that would abandon a partial bin and truncate the CSV.")
        self.stop_button.clicked.connect(self.stop_requested)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.stop_button)

        self.state_label = QLabel("No run in progress")
        self.state_label.setProperty("role", "note")
        self.state_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        controls.addWidget(self.state_label, 1)

        # The most-looked-at numbers in the app, in tabular monospace so they do not jitter as
        # they update. `elapsed` and `frames` are COUNTED BY THE PIPELINE and passed through; this
        # label samples nothing and computes nothing.
        self.readout = QLabel("")
        self.readout.setProperty("role", "readout")
        controls.addWidget(self.readout)
        outer.addLayout(controls)

        # THE PLOTS. A picker rather than a button per parameter: there are thirteen of them and a
        # row of thirteen buttons would be the tool strip's width problem all over again.
        from flygym_tracker.gui.behaviour_series import PLOTTABLE

        plot_row = QHBoxLayout()
        plot_row.setSpacing(8)
        plot_label = QLabel("PLOT")
        plot_label.setProperty("role", "grouptitle")
        plot_row.addWidget(plot_label)
        self.plot_box = QComboBox()
        for field, label in PLOTTABLE:
            self.plot_box.addItem(label, field)
        self.plot_box.setToolTip(
            "A behavioural parameter to plot as a timeseries, 8x2 vials per drum face. Each one "
            "opens its own dock, which can be floated, tabbed or closed.")
        plot_row.addWidget(self.plot_box, 1)
        self.plot_button = QPushButton("Show graph")
        self.plot_button.setProperty("role", "ghost")
        self.plot_button.clicked.connect(
            lambda: self.plot_requested.emit(self.plot_box.currentData()))
        plot_row.addWidget(self.plot_button)
        outer.addLayout(plot_row)

        # THE JOBS THAT WERE IN run.bat. Buttons, not a numbered menu, and each one names what it
        # opens rather than what it is called internally.
        #
        # ON A WRAPPING STRIP, which is not cosmetic. A QHBoxLayout's minimum width is the SUM of
        # its children: measured, this row alone reported 1532 px, and the rig laptop's whole
        # desktop is 1440 px wide -- so the window could not be fully seen and the buttons past the
        # edge could not be reached. See `flow_layout` for the full measurement; it is the FOURTH
        # time something on this project has been laid out wider or taller than that screen.
        tools_widget, tools_row = flow_strip(spacing=8)
        for action, label, tip in (
            ("draw_vials", "Draw vial positions",
             "Draw the vials on the picture in this window, on the live camera or on a "
             "recording. Click corners; the keys and the buttons under the picture do the same."),
            ("replay", "Replay a recording",
             "Run the identical pipeline against a recorded video instead of the camera, and "
             "watch it in this window."),
            ("noise", "Measure noise floor",
             "Measure the static-rig noise floor, watching the rig while it is measured. The "
             "suggested thresholds land on the settings rows, unsaved."),
            ("mark_band", "Mark marker band",
             "Drag across the two bright LED slots to tell the software where the marker band "
             "is, instead of it guessing from brightness every frame. Saved with the vial "
             "positions."),
            ("learn_faces", "Learn drum faces",
             "Learn one marker template per drum face while the drum turns. Without this, "
             "everything is recorded as one face."),
            ("free_camera", "Free the camera...",
             "Name what is holding the camera, and offer to stop it. Nothing is stopped "
             "without a yes."),
        ):
            button = QPushButton(label)
            button.setProperty("role", "ghost")
            button.setToolTip(tip)
            button.clicked.connect(
                lambda _checked=False, a=action: self.tool_requested.emit(a))
            tools_row.addWidget(button)
            setattr(self, "tool_%s_button" % action, button)
        outer.addWidget(tools_widget)

    #: The video jobs. All of them want frames, and there is one picture to show them in.
    VIDEO_TOOLS = ("draw_vials", "mark_band", "replay", "noise", "learn_faces")

    # -- state ----------------------------------------------------------------------------------
    def set_stage_busy(self, busy: bool) -> None:
        """A video job already has the picture, so nothing may start a second one."""
        self._stage_busy = bool(busy)
        self._refresh_tools()

    def _refresh_tools(self) -> None:
        blocked = self._running or self._stage_busy
        for action in self.VIDEO_TOOLS:
            getattr(self, "tool_%s_button" % action).setEnabled(not blocked)

    def set_run_state(self, state: str, detail: str = "") -> None:
        """Enable exactly the actions that are legal now, and say what is happening in a sentence."""
        running = state in (STARTING, RUNNING, STOPPING)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(state in (STARTING, RUNNING))
        # A video job wants frames, and the run has the camera. Offering the button anyway would be
        # offering a job that can only fail with the SDK's culprit-free error.
        self._running = running
        self._refresh_tools()
        self.state_label.setText(_state_sentence(state, detail))

    def set_progress(self, payload: dict) -> None:
        """Render one throttled snapshot. Every figure here was counted by the pipeline."""
        elapsed = float(payload.get("elapsed_s") or 0.0)
        self.readout.setText(
            "%s   %d frames   %.1f fps   %d rot   face %s%s" % (
                _hms(elapsed), int(payload.get("frames") or 0),
                float(payload.get("fps_est") or 0.0), int(payload.get("n_rotations") or 0),
                payload.get("face") or "?", _rotation_readout(payload)))


def _rotation_readout(payload: dict) -> str:
    """"   rot-signal 0.14 / 0.20 STATIONARY" -- the LIVE rotation signal beside its trip line.

    THE POINT IS TO MAKE A FAKE ROTATION VISIBLE. The count says a rotation HAPPENED; this says how
    close the detector is to calling one right now: `disp` is the per-frame displacement the whole
    decision rests on, and the second number is the threshold it must cross. Watching `disp` sit
    just under the line and then jump while the drum is still is how the operator SEES fly motion
    being misread as rotation -- which is exactly what this readout was asked for.

    Rendered only when the detector actually produces the signal (the adaptive one does; the
    fixed-threshold one does not), so it is never a fake 0.
    """
    rot = payload.get("rotation") or {}
    disp = rot.get("disp")
    enter = rot.get("enter")
    if disp is None or enter is None:
        return ""
    flag = "ROTATING" if float(disp) >= float(enter) else "stationary"
    return "   rot-signal %.3f / %.3f %s%s" % (
        float(disp), float(enter), flag, _roi_tag(payload.get("rotation_roi")))


def _roi_tag(roi) -> str:
    """" · band 485-594" / " · band?" / " · full" -- where the rotation signal is measured, so the
    operator can see the marker-band ROI is engaged (or still locating) rather than guess. Empty
    when the pipeline sent no label (older run) so the readout is unchanged for it."""
    if not roi:
        return ""
    roi = str(roi)
    if roi.startswith("marker_band rows="):
        return " · band %s" % roi.split("=", 1)[1]
    if roi.startswith("marker_band"):
        return " · band?"          # asked for the band but still locating it
    return " · full"


def _state_sentence(state: str, detail: str) -> str:
    """One sentence per state. The detail is appended, never substituted -- a bare reason with no
    state reads as an error even when it is an ordinary refusal."""
    base = {
        IDLE: "No run in progress",
        STARTING: "Starting the run",
        RUNNING: "Run in progress - camera and algorithm settings are live",
        STOPPING: "Stopping - finishing the current bin",
        DONE: "Run finished",
        FAILED: "Run could not start",
    }.get(state, state)
    return "%s - %s" % (base, detail) if detail else base


def _hms(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return "%d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


# `launch_cli_tool` USED TO LIVE HERE and has been deleted rather than left unused. It started
# `python -m flygym_tracker.cli <args>` as a child process so the cv2 tools could run in a
# DPI-UNAWARE process -- `live_vial_selector.screen_view_limit` depends on that, because an
# AUTOSIZE OpenCV window is laid out at the frame's own pixel size and there is no other way to
# find out whether that fits the desktop.
#
# NOTHING IN THIS WINDOW OPENS AN OpenCV WINDOW ANY MORE (see `video_stage`), so the constraint is
# gone with the thing that had it. Keeping the helper would leave a working, tested-looking way to
# launch a second copy of a tool that is already in this window -- against a camera this process
# may be holding exclusively. The CLI subcommands themselves are untouched and still work from a
# terminal; this was only the GUI's way of shelling out to them.
