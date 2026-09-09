# Spatial heatmaps and occupancy

Open the Spatial activity heatmap visualizer while running or replaying a recording. Select face A or B. The views update every 10 seconds of recording time and at run end.

- **Accumulated activity**: the fraction of measured frame pairs with a motion detection at each pixel. Raw detection counts remain uint64; the displayed rate accounts for unequal observation of the two faces. It still depends on the activity pixel threshold.
- **Live detection**: the detector's current pixel mask, displayed on the current video frame.
- **Mean image (frame sum)**: the exact uint64 sum of stationary grayscale frames divided by each pixel's sample count. Display contrast is rescaled without modifying the stored sum. This has no activity-threshold dependence, but fixed dark objects remain visible.
- **Relative occupancy (lighting corrected)**: cumulative mean relative darkness after per-vial brightness normalization and rolling background subtraction. It uses no activity-detection threshold and needs no separate empty-vial recording.

## Moving background

The background window defaults to 120 seconds and can be set to 10–600 seconds in the occupancy view. It measures recording time, with separate samples for each face; rotating, settling and unknown-face frames are excluded. Up to 32 samples are spaced through that window. A background estimate becomes available after at least four samples and uses their temporal 90th percentile at each pixel. Estimation runs off the acquisition thread. The most recently completed background is used for subsequent frames.

Each vial's current image is divided by its spatial 90th-percentile intensity to compensate for multiplicative lighting changes. At a pixel with normalized intensity J and estimated background B, the accumulated signal is `clip((B - J) / B, 0, 1)`. Its mean includes frames with a usable background. Warm-up frames are counted in the raw mean image but excluded from corrected occupancy. The view reports the corrected-frame count. Changing the window keeps previously corrected history and relearns the background; it does not retroactively recompute earlier frames. The window choice is remembered locally, and changes during a run are logged.

Longer footage helps only when it reveals unobstructed background. A moving window prevents an indefinitely old reference from dominating, but adaptation lags behind changes. Short windows can absorb slow or resting flies; long windows react more slowly to drift. A permanently covered pixel cannot reveal its hidden background. Noise, saturation, high fly density, changes in lighting within a vial and residual registration errors can all affect the result. Relative darkness is an occupancy proxy, **not a calibrated fraction of time occupied**: a half-dark fly present for a quarter of frames would ideally produce 12.5% mean relative darkness, not 25%.

## Alignment and calibration

Cumulative statistics are translated into the first stationary frame's coordinates separately for each face using the accepted registration offsets. Translation uses integer slicing without interpolation or edge wrapping. Pixels excluded by masks or outside the aligned image do not enter the averages. This corrects accepted translational shifts; it does not compensate for unknown shifts, rotations within a frame or deformation.

Vial edits now preserve learned face-identification templates when image dimensions match. Missing face templates can cause the existing pipeline to attribute measurements to one face; resolve the face-ID warning before interpreting A/B maps. Learned templates must be relearned if the camera geometry or marker appearance changes.

The accumulation approach follows the standard per-pixel image-sum operation described in [OpenCV's accumulation documentation](https://docs.opencv.org/4.13.0/d7/df3/group__imgproc__motion.html). FlyGym retains exact uint64 sums for its uint8 grayscale input and uses float64 for normalization and corrected-darkness accumulation.
