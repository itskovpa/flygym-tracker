# FlyGym v2 — Drosophila locomotor-activity tracker

Authoritative design spec. Executor agents implement modules against the interfaces defined here.
Do not change the shared contracts in `src/flygym_tracker/types.py` or the config schema without
updating this document.

## 1. Goal

Measure **population locomotor activity per vial** in the FlyGym v2 rig, over long experiments
(hours→days), as a reliable time series per vial. This is an activity-index problem (how much are
the flies moving), **not** individual-fly trajectory tracking. The proven method — used by
pySolo-Video / ethoscope — is **per-vial ROI frame-differencing during stationary periods**.

## 2. The rig (established from a real full-res frame, `docs/frame_annotated.png`)

- Camera: HikRobot MV-CA013-A0UM, mono, 1280×1024, ~30 fps, serial `DA4282883`. 850 nm back-light,
  visible-cut filter → the camera sees back-lit **silhouettes** (flies = dark blobs on bright glow).
- Holder is a rotating **drum**. The camera sees **one face at a time**: **16 vial slots = 2 rows × 8**
  (4 left + 4 right, split by a central gap). Total across both faces = 32.
- Vials are vertical glowing translucent columns; mouths at the outer top/bottom edges, tube bodies
  converge on a central mounting frame.
- The two blinding-bright horizontal slots in the middle are **LED shining through frame hardware,
  not vials** — always masked out.
- **Back-lighting is non-uniform / left-biased**: columns 1–7 well lit; the **rightmost column
  falls into darkness** and is often untrackable. Track only the lit portion of each tube.
- **Tubes can be missing** (currently 2 slots empty on the visible side). Calibration must detect
  tube **presence** per slot and skip empties.
- The drum **rotates back and forth, flipping ~180°** to alternately present **Face A** and
  **Face B** to the single camera (CONFIRMED) — each flip also agitates the flies (forces climbing).
  Face B appears geometrically transformed (mirror/flip) vs Face A, so **each face is calibrated
  independently**; Face B's calibration is captured the first time a flip is observed (deferred until
  the rig can run). **~15–25 flies per vial** → strong frame-diff signal, population metric is correct.

### Vial numbering (canonical)
Per face: row-major, left→right, top row then bottom row. Upper row = 1..8, lower row = 9..16.
Global vial id across faces = `face_index * 16 + local_id`. Face "A" = the face visible in the
reference frame; "B" = the other face (calibrated when first observed).

## 3. Architecture

Standalone Python package `flygym_tracker`. OpenCV for CV; direct live capture from the camera via
the vendored HikRobot `MvImport` SDK; offline dev/replay from recorded video. **No Bonsai dependency**
(Bonsai remains available only as an optional recorder for capturing dev clips).

```
frame source ──▶ rotation state machine ──▶ (stationary?) ──▶ per-vial activity ──▶ binner ──▶ logger
                        │                                          ▲
                        └── on stationary-onset: face id (marker) + ROI re-registration ──┘
```

Process online; persist only the per-vial activity table + an events log (tiny). Do **not** store
full video in production (days of 1.3 MP @ 30 fps is untenable) — optional periodic snapshot only.

## 4. Modules & responsibilities

Each module is one file under `src/flygym_tracker/`. Interfaces reference dataclasses in `types.py`.

| Module | Responsibility |
|---|---|
| `types.py` | Shared dataclasses + enums (PROVIDED — see §6). Do not fork these types. |
| `config.py` | Load/validate YAML config into a `Config` object; defaults from `config/default_config.yaml`. |
| `frame_source.py` | `FrameSource` ABC; `VideoFileSource` (cv2); `HikCameraSource` (MvImport live). |
| `calibration.py` | Auto-detect vial lattice + illuminated mask + tube presence on a still → `Calibration` bundle (JSON + mask PNGs). Also `load()`. |
| `rotation.py` | `RotationDetector`: global frame-diff → STATIONARY/ROTATING with hysteresis + debounce. |
| `markers.py` | `MarkerDetector`: generic contour/Hu-moment fiducial → face id. Written before any physical marker existed; kept as the fallback, NOT the rig's path. |
| `marker_band.py` | `MarkerBandDetector`: decodes the rig's real IR-sticker band → face id + vial column spans. **This is the face-ID scheme the run path uses**; validated on `Good Markers.avi`. |
| `face_learning.py` | `FaceLearner` / `learn_faces`: watch the drum flip and register one `MarkerBandDetector` template per face, then merge them into the bundle. |
| `activity.py` | `ActivityMeter`: per-vial frame-diff activity within (bbox ∩ illum mask ∩ present); per-frame + bin aggregation. |
| `registration.py` | Align current stationary frame to the calibration frame (translation/rotation) so ROIs stay locked under drift. |
| `logger.py` | `ActivityLogger`: append `ActivityRecord`/`EventRecord` to CSV/XLSX; rolling files; resumable. |
| `pipeline.py` | `TrackerPipeline`: wire source→rotation→(face/register)→activity→binner→logger; the run loop. |
| `cli.py` | Entry points: `calibrate`, `run`, `replay`, `noise` (measure noise floor). |

## 5. Algorithms

### 5.1 Rotation vs stationary (`rotation.py`)
- Per frame, global motion `m = mean(|cur − prev|)` over the illuminated region (or full frame if no
  calibration yet). During a rotation the whole scene moves → `m` spikes; stationary → `m` ≈ noise.
- Hysteresis + debounce: state → ROTATING when `m > enter_threshold`; → STATIONARY when
  `m < exit_threshold` for `debounce_frames` consecutive frames. First `min_stationary_frames` after
  a stationary onset are "settling" (skip for activity; used for re-registration).
- Emit `EventRecord` on every transition (rotation_start/rotation_end) — these are the forced-activity
  stimulus timestamps.
- Thresholds seeded from the measured **noise floor** (`cli noise`) and refined; must sit far above
  static-rig noise and far below rotation motion (large separation expected — verify on a real
  rotation clip).

### 5.2 Face id + registration (stationary onset)
- On each stationary onset (after settling): run the marker detector → face name. The detector in
  the run path is `marker_band.MarkerBandDetector`, rebuilt from the templates in the bundle by
  `calibration.marker_detector_from_calibration` (validated 43/43 on real footage).
  `markers.MarkerDetector` remains only for a bundle carrying old-style contour `signature`s.
- **Identification failure NEVER invents a face** (`pipeline._handle_onset`). It logs
  `marker_absent` and then:
  - keeps the **last confidently identified** face, if there has been one;
  - before the first identification, attributes activity to **no face at all** until one arrives.
    A short gap at the start of a multi-day run is recoverable; mislabelled vial identities are not.
  - a **single-face** bundle (or a run with no detector able to discriminate) keeps the old
    default-face behaviour — with one face there is nothing to get wrong.
  A run whose bundle covers 2 faces but carries no marker templates cannot identify anything, and
  says so loudly at startup (`cli.face_id_readiness`) rather than silently producing half the data.
- Templates are learned once per rig by `face_learning.learn_faces`, straight after the vials are
  drawn: it watches the drum through at least one flip and registers a template from the first
  settled dwell of each distinct face. `calibration.attach_face_templates` merges them into the
  saved bundle **additively** — the hand-drawn polygons are never rewritten.
- `registration.py`: estimate small (dx,dy) aligning the current frame's rigid structure to the
  calibration/reference frame (phase correlation on the masked frame). Apply the offset to ROIs.
  Two mis-registration guards, both log `mis_registration` and keep ROIs at their calibration anchors:
  (a) reject if the correlation **residual** is too large; (b) reject if the **shift magnitude**
  exceeds `max_shift` (default 0.4x the tightest vial-center pitch). Guard (b) is essential because
  the vial lattice is periodic — phase correlation can lock onto a whole vial-pitch offset with HIGH
  confidence (low residual, so guard (a) is blind to it), which would alias every ROI onto its
  neighbour. Real drift after the drum returns to pose is far smaller than a pitch.

### 5.3 Per-vial activity (`activity.py`)
For each **stationary, non-settling** frame, for each **present** vial on the current face:
- Effective mask `M = (illum_mask == 255) ∩ vial.bbox`.
- `diff = |cur − prev_stationary|` (reference = previous frame classified stationary on the same face;
  reset the reference after any rotation / face change).
- `motion = diff > pixel_threshold` within `M`. `pixel_threshold = noise_mean + k*noise_std`
  (k configurable, default 5).
- Per-frame: `motion_px = count(motion)`, `active_fraction = motion_px / lit_area_px`.
- **Bin** (default 60 s wall-clock): accumulate per vial → `ActivityRecord`:
  `motion_px_sum`, `active_fraction_mean` (mean over stationary frames in bin),
  `n_stationary_frames`, `n_rotating_frames`, `lit_area_px`, `present`.
  Log per-frame count so a bin straddling a rotation is interpretable.
- `active_fraction` (area-normalized) is the primary cross-vial-comparable readout, since lit area
  differs per vial. `motion_px_sum` is the raw total.

### 5.4 Calibration (`calibration.py`)
Two ways to produce a calibration bundle, both emitting the IDENTICAL `Calibration` bundle:
- **(A) Manual ROI wizard — PRIMARY / reliable path (user-endorsed).** The user iterates over the 16
  slots on a captured face frame, drawing/adjusting each vial ROI and marking present/absent. Robust
  against non-uniform light and missing tubes. See §5.5.
- **(B) Auto-detect — optional accelerator.** Best-effort lattice detection that **pre-seeds** the
  wizard's initial boxes; the user then confirms/nudges. Never the sole path.

**Auto-detect** — input: one representative still of a face (empty rig OK). Steps:
1. Illuminated mask: threshold the bright back-lit region; morphological cleanup → `illum_mask_<face>.png`.
2. Central hardware band: detect the ultra-bright horizontal slots + frame → exclude from illum mask.
3. Vial lattice: from column/row intensity profiles, find the two tube bands (rows) and 8 column
   centers per band (expect 4+4 split by the central gap). Produce 16 bboxes.
4. Tube presence: per slot, test for a glowing column (mean brightness / structure in the lit band).
   Empty slot (dark, or no tube walls) → `present=false`.
5. Emit `Calibration` JSON + mask PNG(s). Also write `calib/overlay_<face>.png` for human check.

### 5.5 Manual calibration wizard (PRIMARY)
- Interactive per-face: show the captured frame; for slot 1..16 the user draws a rectangle (e.g.
  `cv2.selectROI`), or accepts the auto-seeded box; a keypress marks the slot **absent** (skip).
  Optionally pre-seed all 16 boxes from auto-detect (B) so the user only nudges.
- Within each accepted box the **illuminated sub-mask** is auto-derived (threshold the bright pixels
  inside the box) so the user draws boxes, not pixel masks; the central-hardware band stays excluded.
- Emit the SAME `Calibration` bundle (JSON + `illum_mask_<face>.png` + `overlay_<face>.png`).
- Keep the interactive driver THIN; put bundle-building (boxes + present flags + frame → Calibration
  + masks) in a pure, unit-testable function `build_calibration_from_boxes(...)`.
- The JSON also stays hand-editable as a final escape hatch.

## 6. Shared data contracts — see `src/flygym_tracker/types.py` (provided, authoritative)

- `TrackState` enum: `STATIONARY`, `ROTATING`, `SETTLING`, `UNKNOWN`.
- `VialROI(id,row,col,x,y,w,h,present,quad=None)`. `quad` = 4 corners `[[x,y]]*4` clockwise from
  top-left, following the vial's real outline. The drum is CYLINDRICAL, so edge tubes curve away and
  an axis-aligned rectangle cannot follow them (measured: edge vials only 0.28–0.50 lit fraction).
  When `quad` is set the per-vial measurement mask is `illum_mask ∩ polygon(quad)`; `x,y,w,h` remain
  the crop bounding box. `quad=None` behaves exactly as before quads existed (old bundles stay valid,
  verified byte-identical). Edited by hand once per experiment via `roi_editor.run_roi_editor`
  (CLI: `edit-rois`), then transferred to the other face — the faces present in the SAME orientation
  (identity, NOT mirrored; verified by correlation), so shapes copy across and snap to the target
  face's own marker-derived column spans.
- `FaceCalibration(name, vials: list[VialROI], illum_mask_path, marker)`.
- `Calibration(image_width,image_height,faces: dict[str,FaceCalibration],created,notes)` + `to_json/from_json/load`.
- `ActivityRecord(...)` — one row per vial per bin (schema = §5.3 fields).
- `EventRecord(run_id, iso_time, elapsed_s, event, detail)`.
- `Frame(image: np.ndarray uint8 HxW, index:int, t_monotonic:float, t_wall_iso:str)`.

## 7. Output format

- `activity.csv` / `.xlsx` — long/tidy, one row per (vial, bin). Columns:
  `run_id, bin_start_iso, bin_end_iso, elapsed_s, face, vial_id, row, col, present,
   n_stationary_frames, n_rotating_frames, motion_px_sum, active_fraction_mean, lit_area_px`.
- `events.csv` — `run_id, iso_time, elapsed_s, event, detail`.
- `run_meta.json` — config snapshot, calibration hash, camera settings, start/stop, versions.
- Rolling: new file per day (`activity_YYYYMMDD.csv`); resumable (append; recover last bin on restart).

## 8. Testing (fly-independent — do now)

- `test_activity.py`: synthetic frames with a **known** number of changed pixels in known vial ROIs
  → assert `motion_px`/`active_fraction` match exactly.
- `test_rotation.py`: synthetic sequence (quiet frames + injected global-motion frames) → assert
  state transitions + event emission at the right indices.
- `test_logger.py`: schema correctness, append/resume, bin boundaries, xlsx+csv parity.
- `test_calibration.py`: synthetic lattice image (known grid, one empty slot) → assert 16 ROIs, right
  empty flagged, mask excludes central band.
- `test_registration.py`: shift a frame by known (dx,dy) → detector recovers it within tolerance.

## 9. Build order & status

1. **Now (fly-independent):** types, config, frame_source, logger, rotation, activity, registration,
   calibration, markers-framework, pipeline, tests, CLI. Validate on the **empty rig**: live capture,
   noise floor, calibration on the real face, rotation detection on a real (empty) rotation clip.
2. **Deferred ("last bit", needs flies loaded):** tune `pixel_threshold`/`k` and the activity metric
   against real fly motion; confirm shadow SNR; validate per-vial activity vs. eyeball/known stimulus.
   Marker decode is DONE (`marker_band.py`), as is learning a template per face (`face_learning.py`).

## 10. Confirmed parameters (from the rig owner)

- Experiment length **hours to 1–3 days**. Activity bin is **user-defined** (default 60 s), settable
  via config AND a CLI flag — must be easy to change per experiment.
- **15–25 flies per vial** → population frame-diff metric (not single-fly tracking).
- Rotation **flips ~180°** presenting Face A/B alternately (see §2). Built for 2 faces; markers give
  face id; each face calibrated independently.
- Fly shadows are clearly visible when flies are loaded (owner confirmed) — so SNR is not a concern;
  only the numeric activity threshold needs tuning on real flies (the deferred "last bit", §9).
