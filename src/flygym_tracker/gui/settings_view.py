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
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

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
        model = self.controller.model
        for group in model.groups():
            box = QGroupBox(group)
            layout = QVBoxLayout(box)
            layout.setSpacing(2)

            note = QLabel(model.group_notes.get(group, ""))
            note.setProperty("role", "note")
            note.setWordWrap(True)
            # Allowed to shrink, so a long note wraps instead of widening the pane past its half of
            # the splitter -- see `setting_row._wrapping_label`.
            note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
            note.setVisible(bool(note.text()))
            layout.addWidget(note)
            self._group_notes[group] = note

            for setting in model.settings:
                if setting.group != group:
                    continue
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
        layout.addWidget(self.change_label, 1)

        self.save_button = QPushButton("Save to config")
        self.save_button.clicked.connect(self.save_requested)
        layout.addWidget(self.save_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Put every row back to what the config file says.")
        self.reset_button.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_button)
        return banner

    # -- events --------------------------------------------------------------------------------
    def _on_row_edited(self, key: str) -> None:
        self.refresh_titles()
        self.changed.emit()

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
