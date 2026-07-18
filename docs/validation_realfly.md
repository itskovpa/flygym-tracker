# Real-fly validation (2026-07-18)

First validation with live flies (several flies loaded in **vial 7**, top row, 2nd from right).
90 s uncompressed capture (1802 frames @ 20 fps) captured live via `HikCameraSource` while the rig ran
its rotation program.

## Results

- **Live capture** from the HikRobot camera via the tool: working, steady 20 fps.
- **Rotation detection**: the rig's motion is a regular oscillation — rotation frames ~9 mean|diff|,
  static dwells drop to ~1.6 (the true sensor-noise floor). The detector tracked all 29 cycles.
- **Fly shadows**: clearly resolvable dark blobs (IR silhouettes) — SNR is not a concern.
- **Per-vial activity** (the payoff): differencing consecutive frames *within* the static dwells and
  counting thresholded motion pixels per vial → **vial 7 = 1.000, all others ≤ 0.024 (42:1)** in the
  idealized measurement; **~5:1** through the full `TrackerPipeline`. Vial 7 is unambiguously identified.
- **Dynamics**: a per-vial, per-time-bin activity series is produced (vial 7 varied 10k–28k
  motion-px per 10 s bin over the run).

Tuned parameters → `config/flygym_rig.yaml`.

## Two findings that shape the method

1. **Measure WITHIN dwells, not across them.** The activity signal is fly motion during the brief
   static holds (drum fully stopped, identical pose → any pixel change is a fly). Differencing
   *across* dwells fails because the rig rocks to **different tilt angles** each dwell, and the
   out-of-plane tilt change swamps fly motion (2D registration can't undo it). The pipeline already
   differences consecutive *stationary* frames, i.e. within-dwell — correct by construction.
2. **Compression matters.** The signal is invisible in an MJPG recording (block-noise swamps it) but
   clean on uncompressed live frames. Run the tool live (uncompressed); don't tune on MJPG video.

## Open refinement: two tilt poses

The rig rocks between (at least) two tilt poses. A single-pose calibration measures the matching-pose
dwells well but attributes off-pose dwells with slightly misaligned ROIs (structure leaks into
neighbouring vials → the 5:1 vs 42:1 gap). Options, best first:
1. **Rig pauses at one consistent tilt** — eliminates the problem; ~42:1 out of the box.
2. **Skip off-pose dwells** — gate measurement on registration residual (measure only dwells matching
   the calibration pose). Simple pipeline addition.
3. **Calibrate both poses** — reuse the two-"face" mechanism, classify each dwell by pose. Most complete.
