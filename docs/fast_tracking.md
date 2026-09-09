# Experimental fast centroid tracking

In the desktop application's **Tracking** menu select **Fast centroids (thread)** or
**Fast centroids (separate process)**. Enable fly tracking (not activity-only mode), then
start a new run/replay. Menu changes apply to the next run. Threshold and blob-size
controls are in the same menu; the background window uses the spatial panel's saved
window at run start. Existing spatial maps retain their existing measurement method.

Open **Spatial activity heatmap → Fast centroid tracking** or **Fast background subtraction**.
These views display matched worker frames, not an overlay paired with a newer camera frame.
Green points are centroid measurements; orange squares are unresolved groups. The readout
separates submitted, completed, dropped, pending and warm-up frames and reports mean processing
time and queue delay. Preview refresh is limited to 5 Hz; analysis is attempted for every
stationary frame. No camera-rate guarantee is implied by a responsive preview.

Configuration equivalent:

```yaml
tracking:
  mode: fast
  backend: process  # or thread
  background_window_s: 120
  fast:
    threshold: 0.15
    min_area: 8
    max_single_area: 300
    max_speed: 150.0
    max_gap_s: 0.25
    max_group_s: 1.0
```

The worker computes a shared P90 brightness reference from the current union of vial masks,
normalizes the entire uint8 frame, and maintains a separate full-frame temporal-P90 background
for each drum face. The reference calculation uses a 256-bin histogram with the same linear
interpolation as NumPy's percentile. Background sampling is at most 32 frames/window and
recomputation is no faster than window/16 seconds (minimum 1 second). Calculations use float32
frame buffers; percentile estimates and output statistics retain floating-point precision.

Thresholded connected components are measured within each vial mask. Vectorized centroid
distances and previous positions associate detections with temporary track labels. Multiple
previous tracks entering one connected component retain their count as an unresolved group;
no watershed splitting or invented individual positions are used. Groups expire after the
configured duration. A large blob without prior separate tracks has unknown multiplicity.
Tracks reset at drum rotation; backgrounds persist separately for each face.

The prototype does not provide verified identities or accurate counts through all crossings.
Threshold fluctuations, missed detections, and incorrect associations remain possible. Full-frame
backgrounds currently remain in camera coordinates; registration offsets adjust ROI selection,
but are not applied to background alignment. Camera/drum drift may therefore leave edge residuals.

## Results and performance

Each run writes `fast_tracking_<stamp>.csv` in the selected results folder. Each row contains
the face, local vial ID, dwell, timestamp, number of centroid detections, retained group count,
unknown groups, measured centroid positions, and group records. Coordinates are local to the
vial's current bounding rectangle. Distances are cumulative **within the dwell** and must not
be summed across rows. Observed consecutive-frame displacement and straight-line displacement
across gaps are separate. No inferred path through a group is counted as observed movement.
The legacy behaviour CSV is not populated by this experimental tracker because its columns
have different semantics.

Both backends use a bounded eight-frame queue. Full queues reject tracking frames explicitly;
activity processing continues. Accepted frames are drained on clean shutdown, including those
queued before a rotation (each carries its original face/dwell). Final tracking metrics in the
pipeline summary include completed frames, failures, queue/processing times, and dropped frames.
The process backend isolates OpenCV from the application's shared OpenCV lock. It also incurs
frame-copy and interprocess transfer costs. Benchmark the entire pipeline, not only the matcher,
before choosing a backend or camera rate. The source version supports both backends; frozen
executable packaging still needs a separate deployment test.
