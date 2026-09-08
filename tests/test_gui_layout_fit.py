"""Nothing in the window is cropped, drawn over something else, or squeezed to nothing.

REPORTED FROM THE RIG: "there are a lot of overlaps, and many things are cropped or hard to use."
Measured on the rig laptop's 1440 x 852 work area before this file existed: the run's status read
"Run in progress - camera and algorith", the camera scanner's advice was cut mid-word, the same
status label was 6 px wide at 1280 x 720, and at that size the picture's caption was laid out 16 px
INTO the picture with the run controls off the bottom of the window.

Two rules, checked here by walking every visible widget rather than by looking:

  * A label or button must be as wide as its text. The exception is `ElidedLabel`, which is the
    one widget ALLOWED to be narrower than its sentence -- it says so with "..." and a tooltip.
  * No two visible siblings overlap.

Plus the mechanism that keeps the picture from being drawn over: the stage's minimum height
follows what is actually under the picture at the current width.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import (QAbstractButton, QAbstractScrollArea, QComboBox, QLabel,
                               QSplitter, QStackedWidget, QTabWidget, QWidget)

from flygym_tracker.config import load_config
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.elided_label import ElidedLabel
from flygym_tracker.gui.main_window import MainWindow
from flygym_tracker.gui.run_controller import RUNNING


# =============================================================================================
# ElidedLabel: the one widget allowed to be narrower than its sentence
# =============================================================================================
def test_an_elided_label_keeps_the_whole_sentence_and_shows_the_cut(qapp):
    label = ElidedLabel("no cameras detected - check the USB cable, and close the MVS Viewer")
    label.resize(120, 20)
    qapp.processEvents()
    assert label.text().endswith("MVS Viewer"), "text() must be the full sentence"
    assert label.is_elided
    assert "MVS Viewer" in label.toolTip(), "the cut sentence must be readable in the tooltip"


def test_an_elided_label_never_asks_the_window_to_be_as_wide_as_its_sentence(qapp):
    label = ElidedLabel("x" * 400)
    assert label.minimumSizeHint().width() < 200
    assert label.sizeHint().width() > 1000, "but it does ask for the room when there is some"


def test_an_elided_label_that_fits_has_no_tooltip_and_no_cut(qapp):
    label = ElidedLabel("short")
    label.resize(400, 20)
    qapp.processEvents()
    assert not label.is_elided
    assert label.toolTip() == ""


def test_an_owners_tooltip_wins_over_the_automatic_one(qapp):
    label = ElidedLabel("a very long sentence that will certainly be cut at this width")
    label.setToolTip("the owner's explanation")
    label.resize(60, 20)
    qapp.processEvents()
    assert label.toolTip() == "the owner's explanation"


# =============================================================================================
# The window, walked
# =============================================================================================
@pytest.fixture
def window(qapp, tmp_path):
    state = gui_state.default_state()
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"), state=state,
                     root=str(tmp_path), camera_factory=lambda: None, confirm=lambda text: True)
    win.show()
    yield win
    win.session.shutdown()


def _visible(win):
    """Every visible widget of the CENTRAL column -- the picture, the measurement, the run band.

    The settings dock is left out on purpose: its rows are a grid that trades label width for
    value width by design, and the offscreen test font is not the rig's."""
    root = win._central_scroll
    return [w for w in root.findChildren(QWidget) if w.isVisible() and w.width() > 0]


def _describe(w):
    text = ""
    for attr in ("text", "currentText"):
        f = getattr(w, attr, None)
        if callable(f):
            try:
                text = f() or ""
            except Exception:
                text = ""
            if text:
                break
    return "%s %r" % (type(w).__name__, text[:50])


def cropped(win):
    """Visible labels, buttons and pickers narrower than their own text (elided labels excepted)."""
    out = []
    for w in _visible(win):
        if isinstance(w, ElidedLabel) or not isinstance(w, (QLabel, QAbstractButton, QComboBox)):
            continue
        if isinstance(w, QLabel) and w.wordWrap():
            continue
        if w.minimumWidth() == w.maximumWidth():          # deliberately fixed (the +/- steppers)
            continue
        if isinstance(w, QComboBox):                      # a picker elides its own current entry
            continue
        if w.sizeHint().width() > w.width() + 2:
            out.append("%s needs %d px, has %d" % (_describe(w), w.sizeHint().width(), w.width()))
    return out


def overlapping(win):
    """Pairs of visible siblings whose rectangles intersect (stacks and splitters excepted)."""
    by_parent = {}
    for w in _visible(win):
        p = w.parentWidget()
        if p is None or isinstance(p, (QStackedWidget, QSplitter, QTabWidget, QAbstractScrollArea)):
            continue
        by_parent.setdefault(p, []).append(w)
    out = []
    for kids in by_parent.values():
        for i, a in enumerate(kids):
            for b in kids[i + 1:]:
                inter = a.geometry().intersected(b.geometry())
                if inter.width() > 2 and inter.height() > 2:
                    out.append("%s over %s (%dx%d)" % (_describe(a), _describe(b),
                                                       inter.width(), inter.height()))
    return out


def _run_in_progress(window):
    """The state that grew the notes: a run posting progress, results showing, tracking on."""
    window._on_run_state(RUNNING, "replaying")
    window.stage.show_run()
    window.results.setVisible(True)
    window.results.set_progress({
        "vial_results": {i: (120, 900, 0.3) for i in range(1, 33)},
        "face": "A", "pixel_threshold": 15.0,
        "video": {"frames_written": 4820, "frames_dropped": 0, "bytes": 337 << 20, "fps": 10.0,
                  "error": None}})
    window.run_panel.set_progress({"elapsed_s": 63.0, "frames": 1260, "fps_est": 20.0,
                                   "n_rotations": 3, "face": "A",
                                   "rotation": {"disp": 0.039, "enter": 0.15},
                                   "rotation_roi": "full"})


@pytest.mark.parametrize("size", [(1440, 852), (1280, 720)])
def test_nothing_is_cropped_or_overlapping_while_idle(qapp, window, size):
    window.resize(*size)
    qapp.processEvents()
    assert cropped(window) == []
    assert overlapping(window) == []


@pytest.mark.parametrize("size", [(1440, 852), (1280, 720)])
def test_nothing_is_cropped_or_overlapping_during_a_run(qapp, window, size):
    """THE REPORTED CASE. The notes grow when a run posts to them; the picture must not be drawn over."""
    window.resize(*size)
    _run_in_progress(window)
    qapp.processEvents()
    assert cropped(window) == []
    assert overlapping(window) == []


def test_the_run_status_is_readable_not_six_pixels_wide(qapp, window):
    """The status sentence was `Ignored` width and got squeezed to 6 px at 1280 x 720."""
    window.resize(1280, 720)
    _run_in_progress(window)
    qapp.processEvents()
    label = window.run_panel.state_label
    assert label.width() >= label.minimumSizeHint().width() >= 60
    assert "Run in progress" in label.text()


def test_the_numbers_have_a_line_of_their_own_and_only_once_there_are_any(qapp, window):
    """Beside the status sentence the readout cut it to "camera and algorith"; on its own line it
    does not. Hidden until a run counts something, so an idle window carries no blank row."""
    qapp.processEvents()
    assert not window.run_panel.readout.isVisible()
    _run_in_progress(window)
    qapp.processEvents()
    assert window.run_panel.readout.isVisible()
    assert window.run_panel.readout.y() > window.run_panel.state_label.y(), "not on its own line"


# =============================================================================================
# The stage's minimum follows what is under the picture
# =============================================================================================
def test_the_stage_never_promises_less_height_than_its_caption_and_strip_need(qapp, window):
    """At a narrow width the drawing strip wraps to two rows and the caption to two lines; the
    minimum the stage reports to the splitter must include both, or they land on the picture."""
    stage = window.stage
    stage.resize(600, 400)
    stage.caption.setText("FILE  a_long_recording_name.avi  (recorded - not the camera)   -   "
                          "16 vial(s) loaded - drag a corner to adjust, or click to start vial 17")
    stage._show_mode("draw")
    qapp.processEvents()
    need = (stage.view.minimumHeight() + stage.caption.heightForWidth(stage.width())
            + stage.bars.currentWidget().layout().heightForWidth(stage.width()))
    assert stage.minimumHeight() >= need, "%d promised, %d needed" % (stage.minimumHeight(), need)
    assert stage.minimumHeight() >= stage.MIN_HEIGHT


# =============================================================================================
# A graph cannot take the whole window
# =============================================================================================
def test_opening_graphs_leaves_the_picture_and_controls_on_screen(qapp, window):
    """REPORTED FROM THE RIG as "it looks like the old version": two graphs opened, the left dock
    area grew to the graph's own hint, and the central column was squeezed to nothing."""
    window.resize(1440, 852)
    qapp.processEvents()
    for index in (0, 1):
        window.run_panel.plot_box.setCurrentIndex(index)
        window.run_panel.plot_button.click()
        qapp.processEvents()
    central = window._central_scroll
    assert central.width() >= window.MIN_CENTRAL_WIDTH, "the central column is %d px" % central.width()
    assert central.minimumWidth() >= window.MIN_CENTRAL_WIDTH, "nothing stops a drag from crushing it"
    assert window.stage.isVisible() and window.stage.width() > 300
