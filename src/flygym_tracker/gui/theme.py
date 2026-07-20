"""Colours, type scale and one-line style rules, DERIVED from the cv2 panel's constants.

THE BUG THIS FILE IS BUILT AROUND. `settings_panel`'s colours are BGR triples, because that is the
channel order `cv2.rectangle` and `cv2.putText` take. `COLOR_VALUE = (0, 235, 255)` therefore
renders AMBER (#FFEB00). Read as RGB -- which is what anyone hand-copying it into a stylesheet
would do -- the same three numbers are CYAN (#00EBFF). A comment in the panel claimed cyan for
years while the panel drew amber, so a hand-written hex literal here would have made the app cyan
where the panel is amber, with a comment in both places claiming they matched.

So the tuple is CONVERTED, in code, by `bgr_to_hex`. The panel and the app cannot drift apart
because there is only one set of numbers and one direction of derivation. If the rig owner decides
the imposed colour should be cyan after all, changing `settings_model.COLOR_VALUE` changes both
surfaces at once, which is the property that was missing.

WHICH COLOUR WAS INTENDED IS A JUDGEMENT CALL, AND IT WAS MADE THIS WAY: the app matches what the
panel ACTUALLY DRAWS (amber), not what its comment claimed (cyan). The operator has been reading
this panel for the life of the project; recolouring the tool they know, to chase a comment, is the
change with a real cost.

WHY THERE IS NO DARK/LIGHT SWITCH. The rig room is dark and the monitor is dim; the cv2 panel is
dark and the app matches it. One appearance, matching the tool beside it, beats two that have to be
kept in step by hand.

=================================================================================================
AMBER IS RESERVED. This is the one rule the whole palette hangs on: no hover, selection, scrollbar,
focus ring or ordinary button may be amber. Any amber on screen means THIS SOFTWARE IS TOUCHING THE
SENSOR. Diluting it across chrome turns the single most consequential channel in the app into
decoration. Focus is steel blue, unsaved is violet, and neither is ever the imposed colour.

VIOLET FOR UNSAVED, NOT #FFC800. The retired CHANGED yellow sat 8 degrees from IMPOSED amber --
"this value is imposed" and "this value is unsaved" are different facts an operator acts on
differently, and at 2 am on a dim monitor those two yellows were the same colour.

=================================================================================================
FONT SIZES ARE IN POINTS, DELIBERATELY, AND THIS IS A BUG FIX.

The previous stylesheet opened with `QWidget { font-size: 13px; }`. MEASURED on this machine
(PySide6 6.11 / Qt 6.11, windows11 style): under that rule EVERY widget in the app reports
`font().pointSize() == -1` and `font().pixelSize() == 13`. A pixel-size font simply has no point
size, so -1 is Qt's "unset" sentinel.

That is the precondition for `QFont::setPointSize: Point size <= 0 (-1), must be greater than 0`,
which the operator saw on every launch. Any code anywhere -- Qt's own styles, a platform theme, a
future widget here -- that reads a font's point size and sets it back emits that warning the moment
the point size is -1. It is not one bad call site; it is a whole class of them, held open by one
global rule.

Declaring the scale in POINTS closes the class: `pointSize()` is valid for every widget, so there
is nothing for such a call to go wrong on, whoever makes it. `tests/test_gui_theme.py` asserts that
no font-size in this stylesheet is in px and that the base application font has a positive point
size, so the precondition cannot be reintroduced by a later edit that "just" wants 13px back.

HONESTLY STATED: the warning itself could NOT be reproduced on this machine -- not offscreen, not
under the real windows11 style, not through popups, tooltips, message boxes or a full repaint. So
this removes the documented precondition rather than a call site anyone has seen fire, and it
needs one launch on the rig to confirm the line is actually gone.
"""
from __future__ import annotations

from flygym_tracker.settings_model import (
    COLOR_BLOCKED as _BGR_BLOCKED,
    COLOR_DEFAULT as _BGR_DEFAULT,
    COLOR_VALUE as _BGR_VALUE,
)


def bgr_to_hex(bgr) -> str:
    """``(B, G, R)`` -> ``"#RRGGBB"``. The only place the channel order is dealt with."""
    b, g, r = (int(max(0, min(255, c))) for c in bgr)
    return "#%02X%02X%02X" % (r, g, b)


# -- the three derived semantics. Never hand-write these as hex; see the module docstring. ------
#: "this software is leaving this alone" -- the camera keeps whatever MVS set.
DEFAULT_GREEN = bgr_to_hex(_BGR_DEFAULT)      # #78C878
#: "this software is imposing this value on the sensor". RESERVED -- nothing else may be amber.
IMPOSED = bgr_to_hex(_BGR_VALUE)              # #FFEB00
#: "this cannot be edited right now" -- and the row says why.
BLOCKED = bgr_to_hex(_BGR_BLOCKED)            # #5F5F5F

# -- the neutral ground ------------------------------------------------------------------------
# Replaces the old warm sepia ramp (#1A1614 / #241F1C / #2C2622 / #463F3C), which spanned very
# little luminance -- panels did not separate, so the surface read flat-brown rather than
# instrument. A cool near-black with crisp hairlines reads as an oscilloscope bezel, and it stops
# amber sitting on a warm field where it loses separation.
INK_0 = "#0B0D0F"        # window backdrop
INK_1 = "#12161A"        # pane
INK_2 = "#181D22"        # row at rest
INK_3 = "#1F262C"        # editor well / row hover
INK_4 = "#2A333B"        # pressed
RULE = "#232B31"         # 1px hairline between rows
RULE_STRONG = "#35414A"  # group divider, splitter handle, pane frame

#: Deliberately NOT #FFFFFF. Pure white on near-black, in a dark room, beside an 850 nm backlight,
#: is a glare source over an eight-hour watch; a dark-adapted operator squints and loses the row.
TEXT = "#E8EDF1"
TEXT_DIM = "#8A97A2"     # help lines and notes
TEXT_FAINT = "#5B676F"   # units, tertiary

#: Steel blue. Focus rings ONLY -- a third hue on purpose, because focus must be neither amber
#: (imposed) nor green (left alone).
FOCUS = "#7FB2D9"
#: Violet. "Edited but not written to the config file" -- see the docstring on why not #FFC800.
UNSAVED = "#A78BFA"

WARN = "#FFA000"
ERROR = "#FF5B4D"
GOOD = DEFAULT_GREEN

# Backwards-compatible aliases. `readiness_strip` and `camera_status` import these by name, and a
# rename would be churn in files this change has no other reason to touch.
BG = INK_0
PANEL = INK_1
ROW = INK_2
DIM = TEXT_DIM
CHANGED = UNSAVED

#: Camera-ownership dot colours, one per `CameraState`. Colour is never the only channel -- every
#: state also carries a full sentence -- but it is the one that works from across the room.
STATE_COLORS = {
    "closed": TEXT_DIM,
    "opening": WARN,
    "streaming": GOOD,
    "closing": WARN,
    "error_busy": ERROR,
    "error_other": ERROR,
}

# -- type scale, IN POINTS (see the module docstring: px here is a bug, not a preference) --------
FONT_UI = '"Segoe UI Variable Text", "Segoe UI", sans-serif'
#: Tabular by construction. A value stepping 9.9 -> 10.0 must not shift its digits sideways, or a
#: readout that updates twice a second jitters while the operator is watching the number.
FONT_MONO = '"Cascadia Mono", "Consolas", monospace'

PT_BASE = 10        # row labels, buttons          (~13 px at 96 dpi)
PT_SMALL = 8        # help lines, notes, hints     (~11 px)
PT_TINY = 7         # badges                       (~9 px)
PT_VALUE = 15       # the numeral -- the hero      (~20 px)
PT_HERO = 21        # the delivered-fps readout    (~28 px)

# -- geometry constants the layout tests measure against ----------------------------------------
#: The value cell. Fixed, so every numeral in the pane lands in ONE vertical stripe -- that column
#: alignment is what makes lab gear read as precise, and an HBox-with-stretch cannot produce it.
#: 168 = 24px decrement + 4 + 112 field + 4 + 24 increment.
VALUE_WIDTH = 168
#: The revert column. The old "back to default" button was capped at 150px and ate the row; the
#: reclaimed width is what pays for the 15pt numeral inside the same 560px pane minimum.
ACTION_WIDTH = 28
#: Flanking stepper buttons. See `stepper.StepperField` -- these are REAL widgets, and this is
#: their real, hit-testable size.
STEP_BUTTON = 24
#: Single-line row. Direction 1 proposed 44px two-line rows and admitted it drops the pane from
#: ~12 settings to ~8; the user said explicitly that everything must be on one window, so the help
#: line is revealed on hover/focus instead and the row stays at one line.
ROW_HEIGHT = 34

#: 1px dotted leader between label and value. On fractional HiDPI a 1px dotted line aliases into an
#: uneven grey, so it is a constant that can be turned off in one line rather than a style rule
#: spread through the file. Unverified on the rig's own monitor.
DOTTED_LEADERS = False

#: Collapse every animation duration to zero. `tests/conftest.py` sets it, so the suite never waits
#: on an animation and stays deterministic without pytest-qt. Motion in this app is two 100-150ms
#: cues that carry information; everything else is instant on purpose, because anything that moves
#: in the periphery of a 48-hour experiment is a distraction that accrues.
REDUCE_MOTION = False

#: The applied-flash and the state-spine crossfade. The only two animations in the app.
FLASH_MS = 140
SPINE_FADE_MS = 120


def motion_ms(duration: int) -> int:
    """`duration`, or 0 when motion is reduced. The single gate every animation asks."""
    return 0 if REDUCE_MOTION else int(duration)


STYLESHEET = """
QWidget { background: %(ink0)s; color: %(text)s; font-family: %(ui)s; font-size: %(pt_base)dpt; }
QScrollArea, QScrollArea > QWidget > QWidget { background: %(ink1)s; }
QToolTip {
    background: %(ink3)s; color: %(text)s; border: 1px solid %(rule_strong)s; padding: 4px 6px;
}

/* Groups have NO frame. Deleting six nested rounded QGroupBox borders is, on its own, most of the
   difference between "dull" and "instrument": boxes inside boxes inside a splitter is 1998 chrome,
   and it costs the vertical space the rows want. Grouping is carried by a tracked uppercase title
   and one full-width hairline instead. */
QLabel[role="grouptitle"] { color: %(dim)s; font-size: %(pt_small)dpt; font-weight: 700; }
QLabel[role="groupcount"] {
    color: %(dim)s; font-family: %(mono)s; font-size: %(pt_small)dpt; font-weight: 600;
}
QFrame[role="rule"] { background: %(rule_strong)s; border: none; }
QFrame[role="hairline"] { background: %(rule)s; border: none; }

QLabel[role="note"] { color: %(dim)s; font-size: %(pt_small)dpt; }
QLabel[role="help"] { color: %(dim)s; font-size: %(pt_small)dpt; }
QLabel[role="blocked"] { color: %(blocked)s; font-size: %(pt_small)dpt; }
QLabel[role="default"] {
    color: %(default)s; font-size: %(pt_small)dpt; font-weight: 700;
}
QLabel[role="hint"] {
    color: %(dim)s; font-family: %(mono)s; font-size: %(pt_small)dpt;
}
QLabel[role="label"] { color: %(text)s; font-weight: 500; }
QLabel[role="label"][live="off"] { color: %(dim)s; }
QLabel[role="badge"] {
    color: %(faint)s; border: 1px solid %(rule)s; border-radius: 3px;
    padding: 0 4px; font-size: %(pt_tiny)dpt; font-weight: 600;
}
QLabel[role="warnbadge"] {
    color: %(warn)s; border: 1px solid %(warn)s; border-radius: 3px;
    padding: 0 4px; font-size: %(pt_tiny)dpt; font-weight: 600;
}
QLabel[role="notice"] { color: %(warn)s; font-size: %(pt_small)dpt; }
QLabel[role="hero"] {
    color: %(text)s; font-family: %(mono)s; font-size: %(pt_hero)dpt; font-weight: 600;
}
QLabel[role="readout"] {
    color: %(text)s; font-family: %(mono)s; font-size: %(pt_small)dpt;
}

/* THE VALUE FIELD. A "well": recessed, boxed and lit, against a default row's flat open hole.
   That shape difference is the channel that survives colour-blindness and a dim monitor.

   NOTE the absence of ::up-button / ::down-button rules. That is deliberate and it is the point of
   `stepper.StepperField`: the native sub-controls are switched OFF in code
   (`setButtonSymbols(NoButtons)`), so there is no sub-control geometry for a stylesheet to get
   wrong and none for Qt to hit-test. See that module for the measured bug. */
QAbstractSpinBox {
    background: %(ink3)s; color: %(imposed)s; font-family: %(mono)s;
    font-size: %(pt_value)dpt; font-weight: 600;
    border: 1px solid %(rule)s; border-radius: 3px; padding: 1px 6px;
    selection-background-color: %(focus)s; selection-color: %(ink0)s;
}
QAbstractSpinBox:focus { border: 1px solid %(focus)s; background: %(ink4)s; }
QAbstractSpinBox:disabled { color: %(blocked)s; border-color: %(blocked)s; background: %(ink2)s; }

QLineEdit {
    background: %(ink3)s; border: 1px solid %(rule)s; border-radius: 3px; padding: 4px 6px;
}
QLineEdit:focus { border-color: %(focus)s; }
QLineEdit:read-only { color: %(dim)s; }

QPushButton, QToolButton {
    background: %(ink2)s; border: 1px solid %(rule)s; border-radius: 3px; padding: 4px 10px;
    color: %(text)s;
}
QPushButton:hover, QToolButton:hover { background: %(ink3)s; border-color: %(rule_strong)s; }
QPushButton:pressed, QToolButton:pressed { background: %(ink4)s; }
QPushButton:focus, QToolButton:focus { border-color: %(focus)s; }
QPushButton:disabled, QToolButton:disabled {
    color: %(blocked)s; border-color: %(blocked)s; background: %(ink1)s;
}

/* The step buttons. Real widgets with their own geometry -- what is drawn IS what is clicked. */
QPushButton[role="step"] {
    background: %(ink2)s; border: 1px solid %(rule)s; border-radius: 3px;
    color: %(text)s; font-family: %(mono)s; font-weight: 700; padding: 0px;
}
QPushButton[role="step"]:hover { background: %(ink4)s; border-color: %(focus)s; }
QPushButton[role="step"]:pressed { background: %(focus)s; color: %(ink0)s; }
QPushButton[role="step"]:disabled {
    color: %(blocked)s; border-color: %(rule)s; background: %(ink1)s;
}

/* Ghost buttons: the arm affordance and the revert glyph. No fill, so they do not compete with
   the numeral, but they are still explicitly-clicked buttons with their own rect. A large
   forgiving hit target is precisely the WRONG shape for the one action that writes to the sensor,
   which is why arming is not a click anywhere on the cell. */
QPushButton[role="ghost"] {
    background: transparent; border: 1px solid %(rule)s; border-radius: 3px;
    color: %(dim)s; padding: 3px 8px; font-size: %(pt_small)dpt;
}
QPushButton[role="ghost"]:hover { color: %(text)s; border-color: %(focus)s; }
QPushButton[role="ghost"]:disabled { color: %(blocked)s; border-color: %(rule)s; }
QToolButton[role="revert"] {
    background: transparent; border: none; color: %(faint)s; padding: 0px; font-weight: 700;
}
QToolButton[role="revert"]:hover { color: %(imposed)s; }

/* The one filled button in the app, and a knowing exception to the amber rule: saving is the
   action that needs weight, and an outline says so without a wide amber fill next to an 850 nm
   rig. Flagged here rather than left to be discovered. */
QPushButton[role="primary"] {
    background: transparent; border: 1px solid %(imposed)s; color: %(imposed)s;
    border-radius: 3px; padding: 5px 14px; font-weight: 600;
}
QPushButton[role="primary"]:hover { background: rgba(255, 235, 0, 0.10); }
QPushButton[role="primary"]:disabled { border-color: %(rule)s; color: %(blocked)s; }
QPushButton[role="danger"] {
    background: transparent; border: 1px solid %(error)s; color: %(error)s;
    border-radius: 3px; padding: 5px 14px; font-weight: 600;
}
QPushButton[role="danger"]:hover { background: rgba(255, 91, 77, 0.12); }
QPushButton[role="danger"]:disabled { border-color: %(rule)s; color: %(blocked)s; }

QComboBox {
    background: %(ink2)s; border: 1px solid %(rule)s; border-radius: 3px; padding: 4px 6px;
}
QComboBox:hover { border-color: %(rule_strong)s; }
QComboBox QAbstractItemView {
    background: %(ink2)s; border: 1px solid %(rule_strong)s; selection-background-color: %(ink4)s;
}
QListWidget, QTreeWidget, QTableWidget {
    background: %(ink1)s; border: 1px solid %(rule)s; border-radius: 3px;
}
QScrollArea { border: none; }
QSplitter::handle { background: %(rule_strong)s; }
QProgressBar {
    background: %(ink2)s; border: 1px solid %(rule)s; border-radius: 3px; text-align: center;
    color: %(text)s; font-family: %(mono)s; font-size: %(pt_small)dpt;
}
QProgressBar::chunk { background: %(focus)s; border-radius: 2px; }
QScrollBar:vertical { background: %(ink1)s; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: %(rule_strong)s; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: %(faint)s; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
""" % {
    "ink0": INK_0, "ink1": INK_1, "ink2": INK_2, "ink3": INK_3, "ink4": INK_4,
    "rule": RULE, "rule_strong": RULE_STRONG,
    "text": TEXT, "dim": TEXT_DIM, "faint": TEXT_FAINT,
    "default": DEFAULT_GREEN, "imposed": IMPOSED, "blocked": BLOCKED,
    "warn": WARN, "error": ERROR, "focus": FOCUS,
    "ui": FONT_UI, "mono": FONT_MONO,
    "pt_base": PT_BASE, "pt_small": PT_SMALL, "pt_tiny": PT_TINY,
    "pt_value": PT_VALUE, "pt_hero": PT_HERO,
}


def base_font():
    """The application font, with a REAL point size. See the module docstring's font section.

    Built here rather than in `app.main` so that the tests can assert on the same object the app
    installs, and so a second entry point cannot forget it.
    """
    from PySide6.QtGui import QFont

    font = QFont("Segoe UI Variable Text")
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPointSize(PT_BASE)
    return font


def apply(app) -> None:
    """Install the font and the stylesheet, in that order.

    ORDER MATTERS: the stylesheet's `font-family` cascades from the application font for anything
    it does not name, so setting the font afterwards would leave already-polished widgets on the
    default one.
    """
    app.setFont(base_font())
    app.setStyleSheet(STYLESHEET)


def spine_color(*, blocked: bool, imposed: bool, nullable: bool = True) -> str:
    """The 2px state spine's colour. Colour is the FAST channel, never the only one -- see
    `setting_row.StateSpine`, which also changes SHAPE (filled vs outlined) for the same states."""
    if blocked:
        return BLOCKED
    if not nullable:
        return "transparent"
    return IMPOSED if imposed else DEFAULT_GREEN
