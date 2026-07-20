"""Shared data contracts for flygym_tracker.

AUTHORITATIVE. Every module imports these types; do not fork or redefine them. Changing a field
here is a spec change — update DESIGN.md too.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class TrackState(str, Enum):
    UNKNOWN = "unknown"
    ROTATING = "rotating"
    SETTLING = "settling"      # just became stationary; used for re-registration, excluded from activity
    STATIONARY = "stationary"  # measurable


@dataclass
class Frame:
    """One acquired frame. `image` is HxW uint8 grayscale."""
    image: "object"            # np.ndarray uint8, HxW  (typed as object to avoid importing numpy here)
    index: int
    t_monotonic: float         # seconds from a monotonic clock (for intervals)
    t_wall_iso: str            # wall-clock ISO 8601 (for logging)


@dataclass
class VialROI:
    """A single vial slot on a face. bbox in pixel coords of the full frame.

    `quad` (OPTIONAL) is a 4-corner polygon — ``[[x, y], [x, y], [x, y], [x, y]]``, clockwise
    from the TOP-LEFT corner — that follows the vial's real outline. The drum is cylindrical,
    so tubes near the left/right edges curve away and are foreshortened; an axis-aligned
    rectangle cannot follow them (measured: edge vials only 0.28–0.50 "lit fraction"). The
    quad is edited by hand once per experiment via `roi_editor.run_roi_editor`.

    `x, y, w, h` REMAIN the bounding box and stay the cheap crop rectangle; when a quad is
    present the per-vial measurement mask is ``illum_mask ∩ polygon(quad)`` inside that crop
    (see `pipeline.TrackerPipeline._bbox_submask`). Callers that edit a quad must keep the
    bbox in sync — `calibration.sync_bbox_to_quad` does it.

    ``quad=None`` means "no polygon" and behaves EXACTLY as before quads existed, which is what
    keeps calibration bundles written by older versions valid.

    `polygon` (OPTIONAL) is the SAME idea with an arbitrary vertex count -- ``[[x, y], ...]``,
    3 or more points, in the order the operator clicked them. It is what
    `live_vial_selector.select_vials_live` produces: the rig owner draws every vial by hand on
    the live feed at the start of a session, so the shape is whatever it takes to follow that
    tube (4 corners, 6, 12 -- the drum is cylindrical and no fixed vertex count fits every slot).

    PRECEDENCE, enforced in `pipeline.TrackerPipeline._bbox_submask`:
    ``polygon`` (if set) > ``quad`` (if set) > plain bbox. A vial carrying neither behaves
    exactly as it always has, which is what keeps older bundles valid.
    """
    id: int                    # local id within the face, 1..16 (row-major, top row 1..8, bottom 9..16)
    row: int                   # 0 = upper, 1 = lower
    col: int                   # 0..7, left to right
    x: int
    y: int
    w: int
    h: int
    present: bool = True       # False = empty/missing tube slot, skip in activity
    quad: Optional[list] = None  # [[x,y]] * 4, clockwise from top-left; None = plain bbox
    polygon: Optional[list] = None  # [[x,y]] * N (N >= 3), click order; None = fall back to quad/bbox

    def __post_init__(self) -> None:
        """Normalise `quad`/`polygon` to ``list[list[int]]`` so JSON round-trips are byte-identical.

        Accepts anything point-shaped (tuples, numpy rows, floats) and stores plain ints, so a
        shape built by the editor/selector, loaded from JSON or produced by `transfer_quads` all
        compare equal.
        """
        if self.quad is not None:
            pts = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in self.quad]
            if len(pts) != 4:
                raise ValueError(
                    "VialROI.quad must have exactly 4 corners (clockwise from top-left), got %d"
                    % len(pts)
                )
            self.quad = pts
        if self.polygon is not None:
            pts = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in self.polygon]
            if len(pts) < 3:
                raise ValueError(
                    "VialROI.polygon needs at least 3 vertices to enclose an area, got %d"
                    % len(pts)
                )
            self.polygon = pts


@dataclass
class FaceCalibration:
    name: str                          # "A" or "B"
    vials: list                        # list[VialROI]
    illum_mask_path: str               # PNG, full-frame, 255 = trackable lit pixel, 0 = excluded
    marker: Optional[dict] = None      # marker template/id/bbox; None until markers exist


@dataclass
class Calibration:
    image_width: int
    image_height: int
    faces: dict                        # dict[str, FaceCalibration]
    created: str = ""
    notes: str = ""

    def to_json(self, path: str) -> None:
        def enc(o):
            if isinstance(o, (Calibration, FaceCalibration, VialROI)):
                return asdict(o)
            raise TypeError(type(o))
        payload = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "created": self.created,
            "notes": self.notes,
            "faces": {k: asdict(v) for k, v in self.faces.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "Calibration":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        faces = {}
        for name, fc in d["faces"].items():
            vials = [VialROI(**v) for v in fc["vials"]]
            faces[name] = FaceCalibration(
                name=fc["name"], vials=vials,
                illum_mask_path=fc["illum_mask_path"], marker=fc.get("marker"),
            )
        return Calibration(
            image_width=d["image_width"], image_height=d["image_height"],
            faces=faces, created=d.get("created", ""), notes=d.get("notes", ""),
        )

    # convenience
    def resolve_mask_paths(self, base_dir: str) -> None:
        """Make illum_mask_path absolute relative to the calibration bundle dir."""
        for fc in self.faces.values():
            if not os.path.isabs(fc.illum_mask_path):
                fc.illum_mask_path = os.path.join(base_dir, fc.illum_mask_path)


# ---- Output records ---------------------------------------------------------

#: CSV/XLSX column order for the activity table (one row per vial per bin).
ACTIVITY_COLUMNS = [
    "run_id", "bin_start_iso", "bin_end_iso", "elapsed_s", "face", "vial_id", "row", "col",
    "present", "n_stationary_frames", "n_rotating_frames", "motion_px_sum",
    "active_fraction_mean", "lit_area_px",
]


@dataclass
class ActivityRecord:
    run_id: str
    bin_start_iso: str
    bin_end_iso: str
    elapsed_s: float
    face: str
    vial_id: int               # global id = face_index*16 + local_id
    row: int
    col: int
    present: bool
    n_stationary_frames: int
    n_rotating_frames: int
    motion_px_sum: int
    active_fraction_mean: float
    lit_area_px: int

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in ACTIVITY_COLUMNS}


#: CSV column order for the BEHAVIOUR table -- one row per vial per dwell, from `fly_tracking.
#: summarize`. A SEPARATE FILE from activity.csv, keyed by `run_id`/`elapsed_s`/`vial_id` so the
#: two join, because they are different measurements at different scales: activity is a
#: frame-difference over a whole bin and needs nothing but the ROI, while these are fly-level
#: statistics that only exist when the drum is still and the tracker could see individual animals.
#: Widening activity.csv would also have meant every existing analysis script seeing new columns.
BEHAVIOUR_COLUMNS = [
    "run_id", "iso_time", "elapsed_s", "face", "vial_id",
    # height / climbing
    "mean_height", "median_height", "frac_above_mid", "max_height",
    # counts
    "n_blobs_mean", "est_n_flies_mean",
    # speed / path
    "mean_speed", "median_speed", "p90_speed", "mean_speed_norm",
    "total_path_length", "median_path_length",
    # diagnostics -- READ THESE BEFORE QUOTING ANYTHING ABOVE. A low `mean_fragment_frames` means
    # the tracks were shredded by crowding, and every per-fly figure above is then a statement
    # about fragments rather than about flies.
    "mean_fragment_frames", "n_tracks", "n_frames",
]

EVENT_COLUMNS = ["run_id", "iso_time", "elapsed_s", "event", "detail"]


@dataclass
class EventRecord:
    run_id: str
    iso_time: str
    elapsed_s: float
    event: str                 # rotation_start | rotation_end | face_change | calibration | marker_absent | mis_registration
    detail: str = ""

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in EVENT_COLUMNS}
