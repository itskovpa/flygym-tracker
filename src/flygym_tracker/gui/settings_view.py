"""Every setting, grouped, with the save/reset pair and the counter that reads the tri-state.

ONE GROUP BOX PER `model.groups()`, IN MODEL ORDER. The order is the model's, not this file's, so a
row added to `build_app_settings` appears here with nothing to change -- and the cv2 panel and the
app present the same settings in the same order, which is what lets an operator move between them
without relearning the layout.

THE GROUP TITLE CARRIES A LIVE COUNT: "Camera - 3 of 5 left at camera default". That is the fourth
channel of the tri-state (after widget shape, colour and the words on the row), and it is the one
that works without reading anything: an operator can see how much the software is imposing from
across the room, and notice a row that got armed by accident.

THE GROUP NOTE IS VERBATIM FROM `build_camera_settings`. "camera not open - limits are the rig
camera's, not live" is the sentence that stops a documented range being read as a measured one, and
it is not reworded here for the same reason the block reasons are not: two wordings of one fact is
one fact and a future disagreement.

WHAT REBUILDS, AND WHEN. Opening the camera changes the LIMITS, and a start-only write changes them
again -- a new Width legitimately changes the legal ExposureTime and AcquisitionFrameRate ranges.
`frame_source.ranges()` caches at open and `refresh_ranges()` existed with nothing calling it, so a
spinbox built once at startup would go on offering a value the SDK now REJECTS. On a start-only
node a rejected value is a failed run, not a slightly wrong picture, so the camera rows are rebuilt
from fresh limits at both moments.
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from flygym_tracker.gui import theme

from flygym_tracker.gui.setting_row import NullableSettingRow, SettingRow
from flygym_tracker.settings_model import build_camera_settings


class SettingsView(QWidget):
    """The scrolling column of groups, plus the change banner underneath it."""

    changed = Signal()
    save_requested = Signal()

    def __init__(self, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.rows: Dict[str, SettingRow] = {}
        self._group_boxes: Dict[str, QGroupBox] = {}
        self._group_notes: Dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        #: STICKY, AT THE TOP OF THE PANE. The user asked for speed to iterate: typing "exp" to
        #: reach exposure beats scrolling a thirty-row column, and it costs nothing structurally.
        #: It is also the inert widget that takes the window's initial focus -- Qt auto-focuses the
        #: first focusable widget of a shown window, which would otherwise be a settings spinbox,
        #: and a stray keypress at 2 am would then edit a camera setting. The worst a stray
        #: keystroke can do HERE is hide some rows.
        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText("Type to filter the settings...")
        self.filter_box.setClearButtonEnabled(True)
        self.filter_box.textChanged.connect(self._on_filter)
        outer.addWidget(self.filter_box)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        # NO HORIZONTAL SCROLLBAR, EVER. Caught by rendering the window offscreen: the long
        # arm-button labels ("Set a value (camera not open - starts at 0.1 fps)") pushed the
        # content wider than the pane, Qt grew a horizontal scrollbar, and the VALUE CONTROLS went
        # off the right edge -- a settings window whose settings are off-screen until you scroll
        # sideways, which nobody does. With this off, the layout must fit; the help lines wrap and
        # the buttons stay reachable.
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #: Wide enough for the longest row (label + badges + a spinbox + the way back to default)
        #: without the splitter being able to squeeze a control out of reach.
        self.setMinimumWidth(560)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 8, 8, 8)
        self._body_layout.setSpacing(10)
        self._build_groups()
        self._body_layout.addStretch(1)
        self.scroll.setWidget(self._body)
        outer.addWidget(self.scroll, 1)

        outer.addWidget(self._change_banner())
        self.refresh()

    # -- construction --------------------------------------------------------------------------
    def _build_groups(self) -> None:
        """A group is a tracked uppercase title, a live count, a hairline, and its rows.

        THE QGroupBox FRAME IS GONE, DELIBERATELY. A bordered box containing rows containing
        bordered fields containing bordered badges is four levels of frame, and boxes-inside-boxes
        is most of what made this surface read as dull rather than as an instrument. Precision
        gear groups things with hairlines and alignment, not nested containers. The frame is also
        the most expensive chrome on screen in vertical pixels, which the rows want.

        `QGroupBox` is still the widget, because `refresh_titles` sets the title through it and
        `_group_boxes` is part of this class's shape -- it just carries no border. Restyling
        rather than replacing keeps the title/count path exactly as it was.
        """
        model = self.controller.model
        for group in model.groups():
            box = QGroupBox(group)
            # FLAT, AND WITH NO BORDER. The title stays on the QGroupBox rather than moving to a
            # QLabel of its own, because `refresh_titles` writes the live count into it and
            # `tests/test_gui_settings_view.py` reads it back from there -- one string, one owner.
            # Uppercasing it here would have meant forking that string, so the tracking and the
            # caps from the visual direction are dropped rather than duplicating the count.
            box.setFlat(True)
            layout = QVBoxLayout(box)
            layout.setContentsMargins(0, 6, 0, 0)
            layout.setSpacing(1)

            # The hairline that replaces the frame. One rule under the title does the grouping
            # that four nested borders were doing, and costs one pixel instead of ten.
            rule = QFrame()
            rule.setProperty("role", "rule")
            rule.setFixedHeight(1)
            layout.addWidget(rule)

            note = QLabel(model.group_notes.get(group, ""))
            note.setProperty("role", "note")
            note.setWordWrap(True)
            # Allowed to shrink, so a long note wraps instead of widening the pane past its half of
            # the splitter -- see `setting_row._wrapping_label`.
            note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
            note.setVisible(bool(note.text()))
            layout.addWidget(note)
            self._group_notes[group] = note

            # ROWS ARE NOT CARDS. Thirty rounded tiles is thirty boxes of noise; a 1px hairline
            # between rows separates them for a tenth of the ink and none of the vertical space.
            # None after the last row -- a trailing rule reads as an empty row below the group.
            first = True
            for setting in model.settings:
                if setting.group != group:
                    continue
                if not first:
                    hairline = QFrame()
                    hairline.setProperty("role", "hairline")
                    hairline.setFixedHeight(1)
                    layout.addWidget(hairline)
                first = False
                row_class = NullableSettingRow if setting.nullable else SettingRow
                row = row_class(setting, self.controller)
                row.edited.connect(self._on_row_edited)
                layout.addWidget(row)
                self.rows[setting.key] = row

            self._group_boxes[group] = box
            self._body_layout.addWidget(box)

    def _change_banner(self) -> QWidget:
        banner = QWidget()
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(8, 4, 8, 4)

        self.change_label = QLabel("no changes")
        # VIOLET, NOT A SECOND YELLOW. "Unsaved" and "imposed on the sensor" are different facts
        # an operator acts on differently; the retired #FFC800 sat 8 degrees from the imposed
        # amber and the two were one colour on a dim monitor at 2 am.
        self.change_label.setStyleSheet(
            "color: %s; font-family: %s;" % (theme.UNSAVED, theme.FONT_MONO))
        layout.addWidget(self.change_label, 1)

        # The one outlined-amber control in the app, and a knowing exception to the reservation
        # rule: saving is the action that needs weight. Flagged in `theme` rather than left to be
        # discovered by whoever next wonders why amber appears on a button.
        self.save_button = QPushButton("Save to config")
        self.save_button.setProperty("role", "primary")
        self.save_button.clicked.connect(self.save_requested)
        layout.addWidget(self.save_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setProperty("role", "ghost")
        self.reset_button.setToolTip("Put every row back to what the config file says.")
        self.reset_button.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_button)
        return banner

    # -- events --------------------------------------------------------------------------------
    def _on_row_edited(self, key: str) -> None:
        self.refresh_titles()
        self.changed.emit()

    def _on_filter(self, text: str) -> None:
        """Hide rows whose label or help does not mention `text`. Hiding only -- never disabling.

        A hidden row still holds its value and is still saved: filtering is a way to find a setting
        in a list, not a way to exclude one from the file.
        """
        needle = (text or "").strip().lower()
        for key, row in self.rows.items():
            haystack = "%s %s %s" % (row.setting.label, row.setting.help, key)
            row.setVisible(not needle or needle in haystack.lower())

    def _on_reset(self) -> None:
        """Reset is NOT confirmed: it restores the file's own values, which the file still holds.

        Blocked rows are left alone by the controller -- letting reset move a row that refuses a
        drag would be the one way round the guard.
        """
        self.controller.reset()
        self.refresh()
        self.changed.emit()

    # -- refresh -------------------------------------------------------------------------------
    def refresh(self) -> None:
        for row in self.rows.values():
            row.refresh()
        self.refresh_titles()

    def refresh_titles(self) -> None:
        """Group titles, the change count, and the save/reset enablement."""
        for group, box in self._group_boxes.items():
            box.setTitle(self.controller.group_title(group))
        n = len(self.controller.changed())
        self.change_label.setText("no changes" if not n else
                                  "%d unsaved change%s" % (n, "" if n == 1 else "s"))
        self.save_button.setEnabled(bool(n))
        self.reset_button.setEnabled(bool(n))

    def set_status(self, text: str) -> None:
        """Append the last save result to the change line, so a save leaves a visible trace.

        `save_settings_to_yaml` changes a file the operator is not looking at; a save with no
        output on screen leaves them unable to tell a write from a silently skipped one, and the
        next run becomes a mystery. The CLI's saver prints for the same reason.
        """
        n = len(self.controller.changed())
        base = "no changes" if not n else "%d unsaved change%s" % (n, "" if n == 1 else "s")
        self.change_label.setText("%s   -   %s" % (base, text) if text else base)

    def rebuild_camera_rows(self, config, camera) -> None:
        """Re-derive the camera rows' LIMITS from `camera` (or the fallbacks when it is None).

        Only the limits and hints are adopted -- the VALUES stay exactly as the model holds them.
        Rebuilding values here would silently overwrite an operator's unsaved edit with whatever
        the sensor happens to be doing, which is the "impose whatever it is doing" behaviour the
        whole tri-state exists to prevent.
        """
        try:
            settings, notes = build_camera_settings(config, camera)
        except Exception:
            return
        for fresh in settings:
            row = self.rows.get(fresh.key)
            if row is None:
                continue
            stored = self.controller.model.get(fresh.key)
            # Widen the limits to keep whatever the model already holds reachable: a row the config
            # set to 88 fps must not become uneditable because the fresh range stops at 60.
            if stored.value is not None:
                fresh.lo = min(float(fresh.lo), float(stored.value))
                fresh.hi = max(float(fresh.hi), float(stored.value))
            stored.lo, stored.hi, stored.step = fresh.lo, fresh.hi, fresh.step
            stored.default_hint = fresh.default_hint
            row.rebuild_limits(stored)
        for group, text in (notes or {}).items():
            label = self._group_notes.get(group)
            if label is not None:
                label.setText(text)
                label.setVisible(bool(text))
        self.refresh_titles()
