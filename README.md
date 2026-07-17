# flygym-tracker

Drosophila locomotor-activity tracker for the **FlyGym v2** rig — a rotating 32-vial drum, back-lit
at 850 nm, imaged as IR silhouettes by a single HikRobot camera.

It measures **population activity per vial** over long experiments (hours→days) by frame-differencing
each vial's back-lit ROI during stationary periods, gating out the rotation intervals, and keeping
vial identity across rotations. Output is a tidy per-vial-per-time-bin CSV/Excel table plus an events
log of the rotation stimuli.

> This is an **activity-index** system (how much are the flies moving), not individual-fly tracking —
> the proven approach for this readout (cf. pySolo-Video, ethoscope), and the right one for multi-fly
> back-lit silhouettes.

See **[DESIGN.md](DESIGN.md)** for the full spec and **`docs/frame_annotated.png`** for the rig.

## Status

Under active construction. The rig-independent core (capture, calibration, rotation detection,
activity metric, logging, tests) is being built and validated on the **empty** rig. Final activity-
threshold tuning and shadow-SNR validation are deferred until flies are loaded (the "last bit").

## Layout

```
DESIGN.md                     authoritative spec
config/default_config.yaml    tunable parameters
src/flygym_tracker/           the package (see DESIGN.md §4)
tests/                        fly-independent unit tests
calib/                        calibration bundle (json + illum masks) — produced by `calibrate`
docs/                         reference frames
```

## Quick start (once built)

```
pip install -r requirements.txt
python -m flygym_tracker.cli noise      --config config/default_config.yaml   # measure noise floor
python -m flygym_tracker.cli calibrate  --frame docs/frame_full.png           # build ROI calibration
python -m flygym_tracker.cli run        --config config/default_config.yaml   # live tracking
python -m flygym_tracker.cli replay     --video path/to/clip.avi              # offline on a recording
```

Requires the HikRobot **MVS** runtime installed (for live capture); the `MvImport` Python SDK is
loaded from the MVS install directory at runtime.
