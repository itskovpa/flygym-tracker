"""The palette's derivation, and the font rule that produced a warning on every launch.

TWO THINGS ARE GUARDED HERE.

1. THE DERIVED COLOURS ARE STILL DERIVED. `settings_panel`'s colours are BGR triples because that
   is what cv2 takes; read as RGB they are a different colour entirely (amber becomes cyan). A
   hex literal typed into the stylesheet would make the app and the cv2 panel disagree while a
   comment in each claimed they matched -- which is what happened for years. The assertion is that
   the app's colour is COMPUTED from the panel's constant, so changing one changes both.

2. NO FONT SIZE IS IN PIXELS. See `theme`'s module docstring. Under the old global
   `QWidget { font-size: 13px; }` every widget in the app reported `font().pointSize() == -1`
   (measured, PySide6 6.11 / Qt 6.11), which is the precondition for
   `QFont::setPointSize: Point size <= 0 (-1), must be greater than 0` -- the line the operator saw
   on every startup. A warning on every launch trains an operator to ignore output, which is
   expensive on a rig whose real failures also arrive on stderr.

   The precondition is a whole class of call sites, not one, so the test guards the CLASS: no
   px font-size anywhere in the stylesheet, and a base font with a positive point size. A later
   edit that "just" wants 13px back reopens it, and this is what says so.
"""
from __future__ import annotations

import re

from flygym_tracker.gui import theme
from flygym_tracker.settings_model import COLOR_DEFAULT, COLOR_VALUE


# =============================================================================================
# The derivation
# =============================================================================================
def test_the_imposed_colour_is_computed_from_the_cv2_panels_own_constant():
    """Not a hex literal. `COLOR_VALUE = (0, 235, 255)` is BGR and draws AMBER; anyone hand-copying
    those three numbers into a stylesheet would have written CYAN."""
    assert theme.IMPOSED == theme.bgr_to_hex(COLOR_VALUE)
    assert theme.IMPOSED == "#FFEB00", "the panel draws amber; the app must not drift to cyan"


def test_the_default_colour_is_computed_too():
    assert theme.DEFAULT_GREEN == theme.bgr_to_hex(COLOR_DEFAULT)


def test_bgr_to_hex_swaps_the_channels_and_clamps():
    assert theme.bgr_to_hex((0, 235, 255)) == "#FFEB00"
    assert theme.bgr_to_hex((-5, 300, 128)) == "#80FF00"


def test_unsaved_is_not_a_second_yellow(qapp=None):
    """#FFC800 sat 8 degrees from IMPOSED amber. "This value is imposed" and "this value is
    unsaved" are different facts an operator acts on differently, and at 2 am on a dim monitor
    those two yellows were one colour."""
    assert theme.UNSAVED == "#A78BFA"
    assert theme.UNSAVED != theme.IMPOSED


def test_focus_is_its_own_hue_and_never_the_imposed_one():
    """AMBER IS RESERVED: any amber on screen means the software is touching the sensor. A focus
    ring in the imposed colour would dilute the one channel that matters most."""
    assert theme.FOCUS not in (theme.IMPOSED, theme.DEFAULT_GREEN, theme.UNSAVED)


def test_amber_appears_in_the_stylesheet_only_where_it_carries_meaning():
    """The value numeral, the save button's outline and its hover wash. Nothing else may be amber,
    or the reservation is decorative rather than readable."""
    occurrences = theme.STYLESHEET.count(theme.IMPOSED)
    assert occurrences <= 4, \
        "amber is used %d times; it is reserved for 'the software is imposing this'" % occurrences


# =============================================================================================
# The font warning
# =============================================================================================
def test_no_font_size_in_the_stylesheet_is_measured_in_pixels():
    """THE REGRESSION. A px font-size makes `pointSize()` return -1 for every widget it reaches,
    which is the precondition for the startup warning."""
    px_rules = re.findall(r"font-size:\s*[\d.]+px", theme.STYLESHEET)
    assert px_rules == [], \
        "px font sizes reopen the QFont::setPointSize(-1) warning: %r" % px_rules


def test_every_font_size_in_the_stylesheet_is_a_positive_point_size():
    sizes = [float(m) for m in re.findall(r"font-size:\s*([\d.]+)pt", theme.STYLESHEET)]
    assert sizes, "the stylesheet declares no font sizes at all"
    assert all(size > 0 for size in sizes), sizes


def test_the_base_application_font_has_a_positive_point_size(qapp):
    """-1 is Qt's "unset" sentinel and the value the warning complains about."""
    font = theme.base_font()
    assert font.pointSize() > 0


def test_applying_the_theme_leaves_every_widget_with_a_valid_point_size(qapp):
    """The end-to-end version: build a widget under the real stylesheet and ask it what the old
    code answered -1 to."""
    from PySide6.QtWidgets import QLabel, QPushButton, QWidget

    theme.apply(qapp)
    try:
        host = QWidget()
        for widget in (QLabel("x", host), QPushButton("y", host), host):
            widget.ensurePolished()
            assert widget.font().pointSize() > 0, \
                "%s has pointSize %d" % (type(widget).__name__, widget.font().pointSize())
    finally:
        qapp.setStyleSheet("")


# =============================================================================================
# Geometry constants the layout tests measure against
# =============================================================================================
def test_the_value_and_action_columns_fit_inside_the_panes_minimum_width():
    """The 168px value column plus the 28px action column must land inside the 560px pane with
    horizontal scrolling permanently off. Measured, not estimated -- this is the exact axis that
    once produced a 1222px minimum inside a 584px viewport."""
    assert theme.VALUE_WIDTH == 168
    assert theme.ACTION_WIDTH == 28
    # spine + label breathing room + value + action, well inside 560
    assert theme.VALUE_WIDTH + theme.ACTION_WIDTH + 120 <= 560


def test_the_stepper_buttons_fit_inside_the_value_column():
    """168 = 24 + 4 + field + 4 + 24. If the buttons grew past the column the field would be
    squeezed to nothing, which `tests/test_gui_layout.py` would then report as an unhittable
    control."""
    assert 2 * theme.STEP_BUTTON + 8 < theme.VALUE_WIDTH


def test_reduce_motion_collapses_every_duration(monkeypatch):
    """One flag, set by the test suite, so no test ever waits on an animation."""
    monkeypatch.setattr(theme, "REDUCE_MOTION", True)
    assert theme.motion_ms(theme.FLASH_MS) == 0
    monkeypatch.setattr(theme, "REDUCE_MOTION", False)
    assert theme.motion_ms(theme.FLASH_MS) == theme.FLASH_MS


# =============================================================================================
# The state spine's redundant shape channel
# =============================================================================================
def test_the_spine_distinguishes_imposed_from_default_by_colour():
    assert theme.spine_color(blocked=False, imposed=True) == theme.IMPOSED
    assert theme.spine_color(blocked=False, imposed=False) == theme.DEFAULT_GREEN
    assert theme.spine_color(blocked=True, imposed=True) == theme.BLOCKED


def test_a_non_nullable_setting_draws_no_spine_at_all():
    """It has no sensor to impose on, so a state mark would report a state that does not exist."""
    assert theme.spine_color(blocked=False, imposed=False, nullable=False) == "transparent"


def test_the_spine_also_differs_in_SHAPE_not_only_colour(qapp):
    """MANDATORY, not optional. Green-vs-amber is the worst confusion pair under deuteranopia, so
    the fastest-to-scan channel in the design cannot be colour alone: imposed draws a full-height
    bar, camera-default draws a short centred tick."""
    from flygym_tracker.config import load_config
    from flygym_tracker.gui.setting_row import NullableSettingRow
    from flygym_tracker.settings_controller import SettingsController
    from flygym_tracker.settings_model import build_app_settings

    controller = SettingsController(build_app_settings(load_config(path="config/flygym_rig.yaml")))
    row = NullableSettingRow(controller.model.get("source.camera.gain_db"), controller)
    row.show()
    assert row.spine._shape == "tick", "a default row must not draw a full bar"
    row.arm_button.click()
    qapp.processEvents()
    assert row.spine._shape == "full", "an imposed row must draw a full bar"
