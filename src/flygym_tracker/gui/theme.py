"""Colours and one-line style rules, DERIVED from the cv2 panel's constants rather than restated.

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
change with a real cost. The comment in `settings_panel` has been corrected instead. The half that
carries the meaning survives either choice -- green is "we are leaving this alone", the bright
colour is "we are imposing this" -- because `COLOR_DEFAULT`'s green is channel-symmetric.

WHY THERE IS NO DARK/LIGHT SWITCH. The rig room is dark and the monitor is dim; the cv2 panel is
dark and the app matches it. One appearance, matching the tool beside it, beats two that have to be
kept in step by hand.
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


#: "this software is leaving this alone" -- the camera keeps whatever MVS set.
DEFAULT_GREEN = bgr_to_hex(_BGR_DEFAULT)      # #78C878
#: "this software is imposing this value on the sensor".
IMPOSED = bgr_to_hex(_BGR_VALUE)              # #FFEB00
#: "this cannot be edited right now" -- and the row says why.
BLOCKED = bgr_to_hex(_BGR_BLOCKED)            # #5F5F5F

BG = "#1A1614"
PANEL = "#241F1C"
ROW = "#2C2622"
TEXT = "#FFFFFF"
DIM = "#969696"
RULE = "#463F3C"
CHANGED = "#FFC800"
WARN = "#FFA000"
ERROR = "#FF5B4D"
GOOD = "#78C878"

#: Camera-ownership dot colours, one per `CameraState`. Colour is never the only channel -- every
#: state also carries a full sentence -- but it is the one that works from across the room.
STATE_COLORS = {
    "closed": DIM,
    "opening": WARN,
    "streaming": GOOD,
    "closing": WARN,
    "error_busy": ERROR,
    "error_other": ERROR,
}

STYLESHEET = """
QWidget { background: %(bg)s; color: %(text)s; font-size: 13px; }
QGroupBox {
    border: 1px solid %(rule)s; border-radius: 6px; margin-top: 22px;
    padding: 10px 8px 8px 8px; background: %(panel)s;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 6px;
    color: %(text)s; font-weight: 600;
}
QLabel[role="note"] { color: %(dim)s; font-size: 11px; }
QLabel[role="help"] { color: %(dim)s; font-size: 11px; }
QLabel[role="blocked"] { color: %(blocked)s; font-size: 11px; }
QLabel[role="default"] { color: %(default)s; font-weight: 600; }
QLabel[role="badge"] {
    color: %(dim)s; border: 1px solid %(rule)s; border-radius: 3px;
    padding: 0 4px; font-size: 10px;
}
QLabel[role="warnbadge"] {
    color: %(warn)s; border: 1px solid %(warn)s; border-radius: 3px;
    padding: 0 4px; font-size: 10px;
}
QLabel[role="notice"] { color: %(warn)s; font-size: 11px; }
QAbstractSpinBox {
    background: %(row)s; color: %(imposed)s; font-weight: 600;
    border: 1px solid %(rule)s; border-radius: 4px; padding: 3px 4px; min-width: 110px;
}
QAbstractSpinBox:disabled { color: %(blocked)s; border-color: %(blocked)s; }
QLineEdit { background: %(row)s; border: 1px solid %(rule)s; border-radius: 4px; padding: 3px 5px; }
QLineEdit:read-only { color: %(dim)s; }
QPushButton, QToolButton {
    background: %(row)s; border: 1px solid %(rule)s; border-radius: 4px; padding: 4px 10px;
}
QPushButton:hover, QToolButton:hover { border-color: %(text)s; }
QPushButton:disabled, QToolButton:disabled { color: %(blocked)s; border-color: %(blocked)s; }
QComboBox { background: %(row)s; border: 1px solid %(rule)s; border-radius: 4px; padding: 3px 5px; }
QListWidget { background: %(panel)s; border: 1px solid %(rule)s; border-radius: 4px; }
QScrollArea { border: none; }
QSplitter::handle { background: %(rule)s; }
""" % {
    "bg": BG, "panel": PANEL, "row": ROW, "text": TEXT, "dim": DIM, "rule": RULE,
    "default": DEFAULT_GREEN, "imposed": IMPOSED, "blocked": BLOCKED, "warn": WARN,
}
