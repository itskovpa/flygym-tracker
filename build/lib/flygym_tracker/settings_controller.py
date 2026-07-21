"""The RULES a settings surface obeys, with no toolkit underneath them.

WHY THIS IS NOT IN THE WIDGET. `settings_panel.SettingsWindow` proved the semantics -- a value is
reported only when it actually MOVES, a refusal is shown rather than reverted, a blocked row
refuses every route including reset -- but it proved them inside a class that owns a cv2 window,
so every test of a RULE had to go through a window to reach it. Porting that shape to Qt would
have bought the same problem in a new toolkit: the invariants that protect a days-long experiment
would only be checkable through offscreen widgets, on a machine that has neither a display nor a
camera.

So the rules live here, in plain Python, and the Qt layer is a renderer that calls them. Every
invariant in `tests/test_settings_controller.py` runs with no QApplication at all. What the widget
tests then have to prove is much smaller and much more honest: that the buttons are wired to these
functions, and that a stray wheel or click cannot reach one.

THE FOUR RULES, and the failure each one is made of:

  1. A COMMIT IS A THREE-STEP LOOP, and the widget is never the source of truth. The model clamps,
     snaps and casts; whatever it stores is what gets displayed back. A spinbox that kept its own
     idea of the value would show 29.7 fps on a row the model rounded to 29.5, and the operator
     would believe the picture rather than the file.
  2. BLOCKED MEANS BLOCKED AT THE FUNNEL, not at the control. `setEnabled(False)` is an affordance
     only -- measured: a programmatic `setValue` on a disabled QSpinBox succeeds and emits. So the
     refusal is here, where every route (typing, stepping, arming, returning to default, reset)
     has to pass.
  3. ARMING A CAMERA ROW WITH NO CAMERA IS A REAL HAZARD, and it is guarded on the irreversible
     step. See `arming_plan`.
  4. RETURNING TO "camera default" CANNOT UN-SEND. See `to_default_notice`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from flygym_tracker.settings_model import (
    CAMERA_START_ONLY_KEYS,
    DEFAULT_TEXT,
    Setting,
    SettingsModel,
    format_value,
    save_settings_to_yaml,
    start_only_block_reason,
)

#: What a row armed with no camera attached is badged with, everywhere it appears -- on the row,
#: in the readiness strip, and in the sentence `save` refuses on. One string, so the operator reads
#: the same words in all three places and recognises the second and third from the first.
NEVER_CHECKED_BADGE = "never checked against a camera"


# ==========================================================================================
# Results (plain data, so a test asserts on a value rather than on a rendered string)
# ==========================================================================================
@dataclass(frozen=True)
class CommitResult:
    """What one attempted edit actually did.

    `moved` and `applied` are deliberately separate. A value can be stored (and therefore saved to
    the YAML later) while the RUN refuses it -- that is the "not applied to this run" case, and
    conflating the two is how a panel ends up either silently reverting an operator's typing or
    silently claiming a change reached the sensor.
    """

    key: str
    moved: bool                       # the stored value actually changed
    stored: Any                       # what the model holds now (None = "camera default")
    applied: bool = True              # the change router accepted it
    refused: Optional[str] = None     # block reason; set only when NOTHING was stored
    message: str = ""                 # one operator-readable line

    @property
    def blocked(self) -> bool:
        return self.refused is not None


@dataclass(frozen=True)
class SaveResult:
    """The outcome of a save, including the one case where a save is refused."""

    saved: bool
    path: Optional[str]
    notes: List[str] = field(default_factory=list)
    message: str = ""
    needed_confirm: bool = False


@dataclass(frozen=True)
class ArmingPlan:
    """What "Set a value..." will do, decided BEFORE the operator presses it.

    Separate from `arm()` so the button can put the landing value in its own label -- the number
    has to be readable without pressing anything, because the press is the thing being decided.
    """

    key: str
    landing: float
    blind: bool                       # no camera hint was available: this is a guess, not a read
    button_text: str                  # short: it must not force the settings pane wider
    warning: str                      # shown BESIDE the button; "" when there is nothing to warn
    tooltip: str


# ==========================================================================================
# Rule 3: arming a camera row
# ==========================================================================================
def arming_plan(setting: Setting) -> ArmingPlan:
    """Where "Set a value..." lands on a row that is currently at the camera default.

    THE HAZARD, MEASURED. `SettingsModel.nudge` lands on `default_hint` -- what the sensor says it
    is doing right now -- so arming the exposure row on an open camera writes back the 5000 us it
    was already using, and nothing physical changes; only provenance does. That is the good case,
    and it is the case the design was written against.

    With NO CAMERA OPEN there is no hint, and `nudge` documents its fallback as `lo`. On this rig,
    with `config/flygym_rig.yaml` (every camera key null), that fallback is 0.1 fps, 9 us, 0.0 dB
    and 32x32 px. Arming a row there and saving would write 0.1 fps into the config the next
    experiment starts from -- a ruined run, produced by one button press that looked like it was
    reading the camera. The camera is closed for most tuning, so this is the COMMON case, not the
    edge one.

    THE FIX IS NOT TO BLOCK THE ROW. Editing the config away from the rig is a workflow the CLI
    supports (`settings` needs no camera at all, by design), and taking it away would be a
    regression to buy a guard. Nor is it to invent a plausible number: a "typical values" table is
    wrong the day a different camera is fitted, and it would be indistinguishable on screen from a
    value actually read from a sensor.

    Instead the guess is made VISIBLE and the guard is moved to the irreversible step:
      * the button names the landing value before it is pressed, and says why it is a guess;
      * the armed row carries the `NEVER_CHECKED_BADGE`;
      * the readiness strip shows a cross while any such row exists;
      * `SettingsController.save` REFUSES to write it without a confirmation naming value and file.
    Reversible actions stay unconfirmed; the one that writes a file the next run reads does not.

    THE BUTTON LABEL IS SHORT AND THE WARNING SITS BESIDE IT. The first version put the whole
    sentence on the button ("Set a value (camera not open - starts at the lowest legal value,
    0.1 fps)"), and rendering the window offscreen showed what that costs: the button forced the
    settings pane wider than its half of the splitter, Qt grew a horizontal scrollbar, and the
    value controls went off the right edge. So the LANDING VALUE stays on the button, where it is
    read before the press, and `warning` goes on the hint line immediately to its left -- same row,
    same glance, no layout damage.
    """
    hint = setting.default_hint
    blind = hint is None
    landing = float(setting.lo) if blind else float(hint)
    probe = _as_probe(setting, landing)
    if not blind:
        return ArmingPlan(
            key=setting.key, landing=landing, blind=False,
            button_text="Set %s" % format_value(probe),
            warning="",
            tooltip=("Starts at %s, which is what the camera reports it is doing right now, so "
                     "nothing about the picture changes. What changes is that this software "
                     "begins imposing the value instead of leaving it alone."
                     % format_value(probe)),
        )
    return ArmingPlan(
        key=setting.key, landing=landing, blind=True,
        button_text="Set %s" % format_value(probe),
        warning="no camera - that is the lowest legal value, not a reading",
        tooltip=("No camera is open, so there is nothing to read the current value from and this "
                 "starts at the LOWEST LEGAL value, %s. That is a guess, not a measurement. The "
                 "row will be marked '%s' until a camera confirms it, and saving it to the config "
                 "file will ask first." % (format_value(probe), NEVER_CHECKED_BADGE)),
    )


def _as_probe(setting: Setting, value: float) -> Setting:
    """A throwaway `Setting` holding `value`, purely so `format_value` renders it in real units.

    The bounds are widened around `value` for the same reason `format_hint` does it: this is a
    number being REPORTED, and clamping it into the row's range to print it would make the display
    disagree with what the press will actually do.
    """
    return Setting(
        key=setting.key, label=setting.label, value=value,
        lo=min(float(setting.lo), float(value)), hi=max(float(setting.hi), float(value)),
        step=setting.step, kind=setting.kind, group=setting.group, help="", unit=setting.unit,
    )


# ==========================================================================================
# Rule 4: leaving "camera default" is cheap; coming back to it is not
# ==========================================================================================
def to_default_notice(setting: Setting, *, camera_open: bool, recording: bool = False) -> str:
    """What the operator must be told when a row goes back to "camera default". "" = nothing.

    `HikCameraSource._set_live_float(None)` cannot un-send. The camera is running at the last value
    written and will keep running at it; clearing the row only stops this software from imposing
    it, now and at the next start. An operator who reads "camera default" and expects the sensor to
    revert has been misled by the only word available.

    NOT A MODAL. This is the action an operator repeats while tuning -- set a value, look, put it
    back, look again -- and a dialog on a repeated action is a dialog that gets dismissed without
    reading, which then trains the same reflex on the two dialogs that DO matter (stopping another
    program, closing with unsaved changes). The transition is already recorded either way:
    `apply_setting` logs ``None -> value`` and ``value -> None`` as `setting_change` events, and
    that log is what survives into the analysis. So the words go on the row, in front of the
    operator, and nothing has to be clicked away.

    WITH THE CAMERA CLOSED THIS IS EMPTY, and that is the point of the rule rather than an
    exemption from it: nothing was ever sent, so there is nothing to un-send, and this is plain
    config editing. Most tuning happens with the camera closed, which is what keeps the notice rare
    enough to still be read when it does appear.
    """
    if not camera_open or setting.value is None:
        return ""
    running = format_value(setting)
    text = ("The camera is running at %s and will KEEP running at %s. Returning to '%s' only "
            "stops this software from imposing it - from now on, and at the next start."
            % (running, running, DEFAULT_TEXT))
    if recording:
        text += " This run continues to record at %s." % running
    return text


# ==========================================================================================
# The controller
# ==========================================================================================
class SettingsController:
    """One `SettingsModel`, plus the rules for changing it. No toolkit, no I/O except `save`.

    Args:
        model: the rows to edit.
        block_reason: ``(key) -> reason | None``, asked before every mutation. ONE callable with
            `TrackerPipeline.setting_block_reason`'s exact signature, so Stage 1 can pass a shim
            over `start_only_block_reason` and Stage 2 can pass the pipeline's bound method with
            nothing else changing. There is deliberately no two-branch default here: a helper that
            decides for itself whether the stream is live is a second opinion about the one fact
            `HikCameraSource.is_acquiring` exists to be the single source of.
        on_change: ``(key, value) -> bool | None``, the router. `False` means "this run cannot
            apply it", which is SHOWN and never reverted -- the value is still what the config file
            will get. None means nothing is running, which is Stage 1's normal state.
        config_path: the YAML `save` writes to, for the messages that have to name it.
        camera_open: ``() -> bool``. A CALLABLE, not a boolean, because the answer changes while
            the surface is up -- it decides whether returning a row to "camera default" needs the
            "the camera will keep running at this value" sentence or nothing at all.
    """

    def __init__(
        self,
        model: SettingsModel,
        *,
        block_reason: Optional[Callable[[str], Optional[str]]] = None,
        on_change: Optional[Callable[[str, Any], Any]] = None,
        config_path: Optional[str] = None,
        camera_open: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.model = model
        self._block_reason = block_reason
        self.on_change = on_change
        self.config_path = config_path
        self.camera_open = camera_open if camera_open is not None else (lambda: False)
        #: keys armed while no camera was open, i.e. holding a value nothing has ever confirmed.
        #: Cleared when the row goes back to the default, or when a camera confirms the value.
        self._never_checked: set = set()

    # -- queries ---------------------------------------------------------------------------
    def block_reason(self, key: str) -> Optional[str]:
        """Why `key` cannot be edited right now, or None. Never raises: a diagnostic that throws
        would take down the surface it is meant to annotate."""
        if self._block_reason is None:
            return None
        try:
            return self._block_reason(key)
        except Exception:
            return None

    def is_blocked(self, key: str) -> bool:
        return self.block_reason(key) is not None

    def changed(self) -> List[Setting]:
        return self.model.changed()

    def never_checked(self) -> List[str]:
        """Keys currently holding a value that no camera has ever confirmed, in model order."""
        return [s.key for s in self.model.settings if s.key in self._never_checked]

    def default_counts(self, group: str) -> tuple:
        """``(n_at_default, n_nullable)`` for `group` -- the group title's live count.

        This is the fourth, group-level channel for the tri-state: an operator who reads
        "3 of 5 left at camera default" knows how much the software is imposing without reading a
        single row, and notices a row that got armed by accident.
        """
        rows = [s for s in self.model.settings if s.group == group and s.nullable]
        return (sum(1 for s in rows if s.value is None), len(rows))

    def group_title(self, group: str) -> str:
        at_default, total = self.default_counts(group)
        if not total:
            return group
        return "%s - %d of %d left at %s" % (group, at_default, total, DEFAULT_TEXT)

    # -- the one funnel --------------------------------------------------------------------
    def commit(self, key: str, value: Any) -> CommitResult:
        """Store `value` under `key`, and tell the router only if the stored value actually MOVED.

        THE ORDER IS THE POINT. The block check comes first, so a refusal stores nothing at all.
        Then the model coerces -- clamp, snap, cast -- and the coerced value is what everything
        downstream sees, including the widget that gets written back to. A no-op edit returns
        early: a spinbox emits on every keystroke and on programmatic writes, and a router that
        logged a `setting_change` per emission would bury the real transitions in its own noise.
        """
        reason = self.block_reason(key)
        if reason is not None:
            return CommitResult(key=key, moved=False, stored=self.model.value(key),
                                applied=False, refused=reason,
                                message="%s: %s" % (self.model.get(key).label, reason))
        before = self.model.value(key)
        after = self.model.set(key, value)
        return self._report(key, before, after)

    def arm(self, key: str) -> CommitResult:
        """Take a nullable row OUT of "camera default", onto `arming_plan`'s landing value.

        A row armed with no camera to read from is remembered (see `arming_plan` and `save`).
        """
        setting = self.model.get(key)
        if not setting.nullable:
            return CommitResult(key=key, moved=False, stored=setting.value, applied=False,
                                message="%s has no camera default" % setting.label)
        plan = arming_plan(setting)
        result = self.commit(key, plan.landing)
        if result.moved and plan.blind:
            self._never_checked.add(key)
        elif result.moved:
            self._never_checked.discard(key)
        return result

    def to_default(self, key: str) -> CommitResult:
        """Put a nullable row back to "impose nothing", through the same funnel as any other edit.

        Goes out to the router too: clearing a value mid-run is a measurement-regime change like
        any other, and `apply_setting` logs it as one.
        """
        setting = self.model.get(key)
        if not setting.nullable:
            return CommitResult(key=key, moved=False, stored=setting.value, applied=False,
                                message="%s has no camera default to return to" % setting.label)
        reason = self.block_reason(key)
        if reason is not None:
            return CommitResult(key=key, moved=False, stored=setting.value, applied=False,
                                refused=reason, message="%s: %s" % (setting.label, reason))
        before = self.model.value(key)
        after = self.model.to_default(key)
        self._never_checked.discard(key)
        return self._report(key, before, after)

    def confirm_against_camera(self, values: Dict[str, Any]) -> None:
        """Mark rows as checked, given what a now-open camera reports (`current_values()` shape).

        Only the KEY is cleared, not the value: an operator who armed the exposure row blind at
        9 us and then opened the camera still has 9 us in the row, and it is still their number to
        change -- but it is no longer unverifiable, because a sensor is now attached to confirm it
        against and the row's help line shows what that sensor says.
        """
        for attr in list(values or {}):
            self._never_checked.discard("source.camera.%s" % attr)

    def reset(self) -> List[str]:
        """Every row back to the config file's value -- except the blocked ones. Returns the keys
        that moved.

        A blocked row refuses a drag; letting reset move it anyway would be the one way round the
        guard, and would leave the surface showing a value the run is not using.
        """
        held = {s.key: s.value for s in self.model.settings if self.is_blocked(s.key)}
        moved = self.model.reset()
        for key, value in held.items():
            self.model.set(key, value)
        moved = [k for k in moved if k not in held]
        for key in moved:
            self._never_checked.discard(key)
            if self.on_change is not None:
                try:
                    self.on_change(key, self.model.value(key))
                except Exception:
                    pass
        return moved

    def _report(self, key: str, before: Any, after: Any) -> CommitResult:
        """The ONE place a stored change becomes an `on_change` call."""
        if after == before:
            return CommitResult(key=key, moved=False, stored=after)
        setting = self.model.get(key)
        if self.on_change is None:
            return CommitResult(key=key, moved=True, stored=after,
                                message="%s = %s" % (setting.label, format_value(setting)))
        try:
            accepted = self.on_change(key, after)
        except Exception as exc:
            return CommitResult(key=key, moved=True, stored=after, applied=False,
                                message="%s: change failed (%s)" % (setting.label, exc))
        if accepted is False:
            return CommitResult(
                key=key, moved=True, stored=after, applied=False,
                message="not applied to this run - takes effect at next start")
        return CommitResult(key=key, moved=True, stored=after,
                            message="%s = %s" % (setting.label, format_value(setting)))

    # -- saving ----------------------------------------------------------------------------
    def unverified_save_warning(self) -> str:
        """The sentence `save` refuses on, or "" when there is nothing unverified to write.

        Names every value and the file, because "are you sure?" is not a question anybody can
        answer -- the numbers are the whole content of the decision.
        """
        rows = [s for s in self.model.changed() if s.key in self._never_checked]
        if not rows:
            return ""
        lines = ["These camera values were never checked against a camera - they were chosen with "
                 "no camera open, so they are the lowest legal value rather than anything the "
                 "sensor confirmed:", ""]
        for s in rows:
            lines.append("    %s = %s" % (s.label, format_value(s)))
        lines.append("")
        lines.append("Writing them to %s means the next experiment starts with them."
                     % (self.config_path or "the config file"))
        return "\n".join(lines)

    def save(self, *, confirm: Callable[[str], bool]) -> SaveResult:
        """Write the changed rows to `config_path`, asking `confirm` first if any are unverified.

        `confirm` is REQUIRED and has no default, deliberately the same shape as
        `camera_lock.release_camera(holders, confirm=...)`: there must be no call that can reach an
        irreversible write without a decision, and a headless test must be able to make that
        decision without a dialog existing.

        An ordinary save is NOT confirmed. It rewrites only the values that changed and preserves
        every comment, so it is neither destructive nor surprising; over-confirming is what trains
        an operator to click past the one dialog that mattered.
        """
        if not self.config_path:
            return SaveResult(saved=False, path=None,
                              message="no config file to save to - open one first")
        changed = self.model.changed()
        if not changed:
            return SaveResult(saved=False, path=self.config_path,
                              message="nothing changed - nothing to save")
        warning = self.unverified_save_warning()
        if warning and not confirm(warning):
            return SaveResult(saved=False, path=self.config_path, needed_confirm=True,
                              message="not saved - the unchecked camera values were not written")
        n = len(changed)
        try:
            notes = save_settings_to_yaml(self.config_path, self.model)
        except Exception as exc:
            return SaveResult(saved=False, path=self.config_path,
                              message="save FAILED: %s" % exc)
        self._never_checked.clear()
        return SaveResult(saved=True, path=self.config_path, notes=list(notes or []),
                          needed_confirm=bool(warning),
                          message="wrote %d change(s) to %s" % (n, self.config_path))


# ==========================================================================================
# The Stage 1 block-reason shim
# ==========================================================================================
def camera_block_reason(source_getter: Callable[[], Any]) -> Callable[[str], Optional[str]]:
    """`(key) -> reason | None` for an app that owns the camera and has no pipeline.

    This is the ONE provider Stage 1 passes, and it is a call to the same
    `settings_model.start_only_block_reason` the pipeline uses -- not a second implementation of
    the rule. Stage 2 replaces this shim with `pipeline.setting_block_reason`, whose signature is
    identical, and nothing in the settings surface changes.

    `closable=True` because this caller CAN close the camera: telling an operator who is looking at
    a "Close camera" button to go and use a command-line tool instead would be absurd.
    `recording=False` because Stage 1 has no run -- when it does, the pipeline's own method takes
    over and answers with the recording wording.
    """
    def reason(key: str) -> Optional[str]:
        return start_only_block_reason(key, source_getter(), recording=False, closable=True)

    return reason


def is_camera_key(key: str) -> bool:
    """True for the five ``source.camera.*`` rows, whose limits change when a camera opens."""
    return str(key).startswith("source.camera.")


def is_start_only(key: str) -> bool:
    return key in CAMERA_START_ONLY_KEYS
