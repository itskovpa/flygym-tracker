"""Screen space: the settings dock, the collapsible experiment header, and the vials on the run.

All three are the rig owner's calls, and all three are about the picture getting the room:

  * "the settings menu on the left is occupying too much screen space with a lot of dead space --
    make it an expandable/collapsable dockable popup window in a separate window";
  * "for the Config in the top menu - also make that section collapsable";
  * "vial markers stay on during the recording".
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import numpy as np                                                     # noqa: E402


def _square(x, y, w=20, h=30):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


@pytest.fixture
def window(qapp, tmp_path):
    """A real `MainWindow`, shown. Local rather than shared: the one in `test_gui_main_window`
    belongs to that file's fixtures, and a fixture moved to `conftest` for one caller becomes a
    thing every GUI test file silently depends on."""
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    state = gui_state.default_state()
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"), state=state,
                     root=str(tmp_path), camera_factory=lambda: None,
                     confirm=lambda text: True)
    win.show()
    qapp.processEvents()
    yield win
    win.run.shutdown()
    win.session.shutdown()


# =============================================================================================
# The settings dock
# =============================================================================================
def test_the_settings_can_be_closed_and_reopened(qapp, window):
    """A dock that can be closed needs a visible switch to reopen it, or closing it once removes
    the settings from the app as far as the operator can tell."""
    assert window.settings_dock.isVisible()
    window.status_bar.settings_button.setChecked(False)
    qapp.processEvents()
    assert not window.settings_dock.isVisible()

    window.status_bar.settings_button.setChecked(True)
    qapp.processEvents()
    assert window.settings_dock.isVisible()


def test_closing_the_dock_by_itself_updates_the_button(qapp, window):
    """One of them driving the other only would leave the button showing "on" over a dock that is
    not there -- and the next click would then appear to do nothing."""
    window.settings_dock.close()
    qapp.processEvents()
    assert not window.status_bar.settings_button.isChecked()


def test_the_settings_can_float_as_their_own_window(qapp, window):
    """The point of a dock rather than a collapsible pane: the settings can sit on a second
    monitor while the rig picture fills this one."""
    from PySide6.QtWidgets import QDockWidget

    features = window.settings_dock.features()
    assert features & QDockWidget.DockWidgetFeature.DockWidgetFloatable
    assert features & QDockWidget.DockWidgetFeature.DockWidgetClosable
    window.settings_dock.setFloating(True)
    qapp.processEvents()
    assert window.settings_dock.isFloating()
    assert window.settings_view.isVisible(), "floating the dock lost the settings"


def test_the_window_still_fits_the_rig_laptop_with_the_dock_open(qapp, window):
    """MEASURED REGRESSION. Moving the settings into a dock stopped them sharing width with the
    centre and started them ADDING to it: the window's minimum went to 1752 px on a 1440 px
    desktop. The status bar's non-wrapping sentence had been demanding 1186 px of that all along,
    and only stayed under the limit while it was the widest thing in the window."""
    assert window.minimumSizeHint().width() <= 1400, window.minimumSizeHint().width()


# =============================================================================================
# The collapsible experiment header
# =============================================================================================
def test_the_experiment_paths_collapse(qapp, window):
    bar = window.session_bar
    assert bar.is_expanded()
    bar.set_expanded(False)
    qapp.processEvents()
    assert not bar.is_expanded()
    assert not bar.body.isVisible()
    assert bar.isVisible(), "collapsing hid the whole band instead of its rows"


def test_a_collapsed_header_still_says_which_experiment_is_set_up(qapp, window):
    """A collapsed section showing only its title would hide the answer to "am I about to
    overwrite yesterday's output folder"."""
    bar = window.session_bar
    bar.set_expanded(False)
    assert bar.summary.text().strip(), "the collapsed header says nothing about what is chosen"
    for part in (bar.config_path(), bar.calib_field.value(), bar.output_field.value()):
        if part:
            assert part in bar.summary.toolTip(), "the full path is not reachable"


def test_the_summary_follows_a_changed_path(qapp, window):
    bar = window.session_bar
    bar.output_field.set_value("/tmp/somewhere_else")
    bar._on_path_picked("/tmp/somewhere_else")
    assert "somewhere_else" in bar.summary.text()


# =============================================================================================
# The vials stay on the picture while they are measured
# =============================================================================================
class _Box:
    stats = (0, 0)

    def take(self):
        return None


class _Session:
    def __init__(self):
        self.latest = _Box()
        self.is_open = True
        self.measured_fps = 0.0
        self.tap = None

    def attach_tap(self, job):
        return False

    def detach_tap(self):
        pass


class _Run:
    def __init__(self):
        self.latest = _Box()


def test_the_vials_are_drawn_on_the_run(qapp):
    """They are not decoration: which pixels went into each row of activity.csv is decided
    entirely by these shapes, and a run watched without them is a drum and a table of numbers with
    no way to see that one outline has slipped onto the tube next door."""
    from flygym_tracker.gui.vial_overlay import RunVialOverlay
    from flygym_tracker.gui.video_stage import RUN, VideoStage

    stage = VideoStage(_Session(), _Run())
    overlay = RunVialOverlay([_square(10, 10), _square(50, 10)])
    stage.set_run_overlay(overlay)
    stage.show_run()
    assert stage.mode == RUN
    assert stage.view.overlay is overlay, "the vials are not on the run's picture"


def test_the_overlay_tints_by_what_each_vial_reports(qapp):
    from flygym_tracker.gui.vial_overlay import RunVialOverlay

    overlay = RunVialOverlay([_square(0, 0), _square(40, 0)])
    overlay.set_activity({0: (120, 900, 0.3), 17: (300, 900, 0.5)})
    assert overlay.activity[0] == 120
    # Global id 17 is face B's vial 2, which shares face A's coordinates on this rig.
    assert overlay.activity[1] == 300


def test_an_unreported_vial_is_left_untinted_rather_than_drawn_as_zero(qapp):
    """"No reading" and "a reading of zero" are different facts, and only one drum face is visible
    at a time -- so the other face's sixteen SHOULD be untinted."""
    from flygym_tracker.gui.vial_overlay import RunVialOverlay

    overlay = RunVialOverlay([_square(0, 0), _square(40, 0)])
    overlay.set_activity({0: (120, 900, 0.3)})
    assert 1 not in overlay.activity


def test_the_overlay_is_built_from_the_saved_bundle(qapp, tmp_path):
    """One face's worth of shapes: both faces share identical coordinates on this rig, so drawing
    both would draw every shape twice in the same place."""
    from flygym_tracker.calibration import (build_two_face_calibration_from_polygons,
                                            load_calibration, save_calibration)
    from flygym_tracker.gui.vial_overlay import overlay_from_calibration

    frame = np.full((120, 240), 80, dtype=np.uint8)
    polygons = [_square(10 + 40 * i, 20) for i in range(3)]
    calib, masks, _ = build_two_face_calibration_from_polygons(
        polygons, frame, (240, 120), faces=("A", "B"))
    out = str(tmp_path / "b")
    save_calibration(calib, masks, out)

    overlay = overlay_from_calibration(load_calibration(out))
    assert overlay is not None
    assert len(overlay.polygons) == 3, "expected one face's vials, got %d" % len(overlay.polygons)


def test_a_bundle_that_cannot_be_read_costs_the_overlay_and_not_the_run(qapp):
    from flygym_tracker.gui.vial_overlay import overlay_from_calibration

    assert overlay_from_calibration(None) is None
    assert overlay_from_calibration(object()) is None
