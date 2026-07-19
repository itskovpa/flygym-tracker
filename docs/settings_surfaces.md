# Two settings front ends, and the date one of them goes away

**Decision, 2026-07-19: `flygym_tracker.gui` is the canonical settings surface.** The cv2
`settings_panel.SettingsWindow` is now legacy and is scheduled for removal — not maintained in
parallel.

## Why both exist right now

They edit the same `SettingsModel` through the same comment-preserving `save_settings_to_yaml`, so
they cannot disagree about what a value *is*. What they can disagree about is everything else:
which rows are shown, when a row is blocked, what a refusal says, and what a stray interaction
does. **Nothing asserts that the two behave identically**, and on a rig that is sold to customers
two interaction models over one model is a support liability rather than a feature.

The cv2 panel stays reachable only because it covers one thing the app does not yet:

* the monitor's `t` key, which opens it **during a run**, over the live tracking window.

That is the whole remaining justification. `run --settings` and `run.bat`'s `[S]` are conveniences
that duplicate what `[A]` now does better.

## What the app already does better

* No OpenCV **GUI** build required — it draws with Qt (`tests/test_settings_model_isolation.py`
  proves the value layer imports with cv2 and PySide6 both blocked).
* A live preview beside the settings, so exposure and gain are set by looking at the picture.
* The config / vial-positions / output folders, which used to be edited in `run.bat` by hand.
* A readiness strip, and a save that refuses to write a camera value no camera ever confirmed.
* Invariant 2 is structural rather than defended: a row at "camera default" has **no editor in the
  widget tree at all**, where the cv2 panel relies on drawing an empty track.

## Removal plan

Delete `settings_panel.SettingsWindow` and everything below it in that module (`SliderRect`,
`layout`, `value_at`, `hit`, `render`, `_draw_*`, `key_hint_lines`, `fit_text`, `decode_key`,
`startup_banner`, `panel_size`) **when the app grows its run view** — i.e. when the monitor's `t`
key has an equivalent. At that point also remove:

* `cli._cmd_settings` and the `settings` subcommand,
* `run.bat`'s `[S]` entry and its `:settings` block,
* the `--settings` flag on `run` and `replay`,
* the `t` binding in `monitor.py`.

`settings_model.py` and `settings_controller.py` **stay** — they are the shared layer, and the app
is built on them. `gui/theme.py` derives its colours from `settings_model`'s BGR constants, so
while both surfaces exist they cannot drift apart on the green/imposed meaning; that derivation is
a stopgap for exactly this period and can be inlined once the panel is gone.

## The colour question, settled

`settings_model.COLOR_VALUE = (0, 235, 255)` is a **BGR** triple, because that is what cv2 takes,
so the cv2 panel has always drawn the imposed-value colour as **amber `#FFEB00`**. A comment in
`settings_panel` called it "cyan" for a long time — cyan is what those three numbers are when read
as RGB, which is what anyone hand-copying the tuple into a stylesheet would do.

The app matches **what the panel actually draws** (amber), and the wrong comment was corrected.
`gui/theme.bgr_to_hex` converts the tuple in code rather than restating it as a hex literal, so
changing `COLOR_VALUE` changes both surfaces at once. If the rig owner would rather have cyan, that
is now a one-line change in `settings_model` — and it is a decision worth making deliberately, not
by transcription.
