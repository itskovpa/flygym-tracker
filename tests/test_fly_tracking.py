"""Tests for flygym_tracker.fly_tracking.

Every fixture here is SYNTHETIC and DETERMINISTIC: a bright "tube" ROI with dark discs drawn at
known pixel positions, so each assertion checks a number we constructed rather than a number we
eyeballed. No randomness, no real frames, no fly data required.

The disc geometry mirrors the real rig (DESIGN.md §2): dark silhouettes on a bright back-lit
tube, discs of ~5 px radius (~80 px area, the measured real-fly size), and where a test needs an
illumination gradient it uses one at least as steep as the rig's real left-right falloff.
"""
import math

import numpy as np
import pytest

from flygym_tracker.fly_tracking import (
    Blob,
    DetectParams,
    FrameStats,
    Track,
    VialTracker,
    detect_flies,
    estimate_single_fly_area,
    link_blobs,
    project_heights,
    summarize,
)


# ---- synthetic scene helpers -------------------------------------------------
H, W = 200, 120           # a vial-shaped frame (tall and narrow, like a tube)
BG_LEVEL = 170            # tube glow, matching the real rig's ~160-180 counts
FLY_LEVEL = 100           # silhouette core; real flies sit 60-70 counts below background
FLY_RADIUS = 5            # -> ~80 px area, the measured real single-fly area


def make_scene(centers, radius=FLY_RADIUS, gradient=0.0, bg=BG_LEVEL, fly=FLY_LEVEL):
    """A bright frame with dark discs at `centers`, plus an optional left-right gradient.

    `gradient` is the TOTAL grey-level swing from the left edge to the right edge. The rig's
    real falloff is ~68 counts across the frame (measured per-vial medians 116..184), so a test
    passing at gradient=120 is testing well past the real condition.

    Returns ``(frame_uint8, roi_mask_bool)``; the ROI is the whole frame minus a 2 px rim.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    ramp = (xx / max(1, W - 1)) * gradient
    img = np.clip(bg - ramp, 0, 255).astype(np.float64)

    for (cx, cy) in centers:
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        # Silhouette depth is preserved relative to the LOCAL background, which is what a
        # back-lit shadow does physically (it attenuates the glow it sits on).
        img[disc] = np.clip(img[disc] - (bg - fly), 0, 255)

    roi = np.zeros((H, W), dtype=bool)
    roi[2:H - 2, 2:W - 2] = True
    return img.astype(np.uint8), roi


def sorted_centroids(blobs):
    return sorted((round(b.centroid[0], 3), round(b.centroid[1], 3)) for b in blobs)


# ---- detect_flies: basic detection ------------------------------------------
def test_detect_flies_finds_exactly_n_discs_at_known_positions():
    centers = [(30, 40), (80, 60), (55, 120), (25, 160), (95, 25)]
    frame, roi = make_scene(centers)

    blobs = detect_flies(frame, roi)

    assert len(blobs) == len(centers)

    # Every known disc centre is matched by exactly one blob centroid within 1 px.
    found = sorted_centroids(blobs)
    for (cx, cy) in sorted(centers):
        matches = [
            b for b in blobs
            if math.hypot(b.centroid[0] - cx, b.centroid[1] - cy) <= 1.0
        ]
        assert len(matches) == 1, "no unique blob within 1 px of disc %r (found %r)" % (
            (cx, cy), found)

    # A radius-5 disc is 81 px by construction; areas must land on it, not merely "be positive".
    expected_area = int(((np.mgrid[0:11, 0:11][0] - 5) ** 2
                         + (np.mgrid[0:11, 0:11][1] - 5) ** 2 <= 25).sum())
    for b in blobs:
        assert abs(b.area - expected_area) <= 6
        assert b.is_merged is False
        assert b.n_flies == 1
        # bbox must actually contain the centroid.
        x, y, w, h = b.bbox
        assert x <= b.centroid[0] <= x + w
        assert y <= b.centroid[1] <= y + h
        # A silhouette is DARK: its mean grey must sit below the tube glow.
        assert b.mean_intensity < BG_LEVEL


def test_detect_flies_survives_a_strong_left_right_illumination_gradient():
    """THE point of the relative threshold (DESIGN.md §2: the back-light is left-biased).

    With a 120-count swing across the frame, the left background (170) is brighter than the
    right background (50) by more than the entire fly-to-background contrast, so ANY fixed grey
    threshold either floods the right side or misses the left. The local-background residual
    must still find all five discs at the same positions.
    """
    centers = [(10, 40), (35, 80), (60, 120), (85, 150), (110, 30)]
    flat_frame, roi = make_scene(centers, gradient=0.0)
    grad_frame, _ = make_scene(centers, gradient=120.0)

    # Sanity-check the fixture: the gradient really does span more than the fly contrast, i.e.
    # no single global grey level can separate flies from background in this image.
    left_bg = int(grad_frame[5, 3])
    right_bg = int(grad_frame[5, W - 4])
    assert left_bg - right_bg > (BG_LEVEL - FLY_LEVEL)

    flat_blobs = detect_flies(flat_frame, roi)
    grad_blobs = detect_flies(grad_frame, roi)

    assert len(flat_blobs) == len(centers)
    assert len(grad_blobs) == len(centers), (
        "the gradient cost us blobs -> the threshold is not background-relative")

    for (cx, cy) in centers:
        near = [b for b in grad_blobs
                if math.hypot(b.centroid[0] - cx, b.centroid[1] - cy) <= 1.0]
        assert len(near) == 1, "disc %r lost or split under the gradient" % ((cx, cy),)


def test_detect_flies_empty_roi_and_blank_frame_return_no_blobs():
    frame, roi = make_scene([(50, 50)])

    # All-False mask -> nothing to look at, and no crash.
    assert detect_flies(frame, np.zeros((H, W), dtype=bool)) == []

    # A completely uniform frame has no flies (and no divide-by-zero on sigma == 0).
    blank = np.full((H, W), BG_LEVEL, dtype=np.uint8)
    assert detect_flies(blank, roi) == []

    # An all-zero (dark) frame likewise.
    assert detect_flies(np.zeros((H, W), dtype=np.uint8), roi) == []


def test_detect_flies_rejects_noise_specks_and_thin_wall_edges():
    """min_area kills specks; min_thickness/max_aspect kill the tube-wall seam."""
    centers = [(30, 60), (85, 140)]
    frame, roi = make_scene(centers)

    # A 1x1 dark speck: real, but far too small to be a fly.
    frame[10, 10] = 20
    # A tube-wall edge: 3 px wide, spanning nearly the whole ROI vertically.
    frame[5:H - 5, 100:103] = 60

    blobs = detect_flies(frame, roi)

    assert len(blobs) == 2
    for (cx, cy) in centers:
        assert any(math.hypot(b.centroid[0] - cx, b.centroid[1] - cy) <= 1.0 for b in blobs)
    # Nothing survived from the wall column.
    assert all(not (100 <= b.centroid[0] <= 103) for b in blobs)


def test_detect_flies_rejects_mismatched_mask_shape():
    frame, _ = make_scene([(50, 50)])
    with pytest.raises(ValueError):
        detect_flies(frame, np.ones((H + 5, W), dtype=bool))


# ---- merging -----------------------------------------------------------------
def test_two_touching_discs_form_one_merged_blob_estimated_at_two_flies():
    """15-25 flies per vial (DESIGN.md §10) WILL touch; report it instead of hiding it."""
    # Six well-separated singles calibrate the single-fly area, plus one touching pair.
    singles = [(25, 20), (25, 60), (25, 100), (25, 140), (25, 180), (60, 20)]
    pair = [(90, 100), (98, 100)]              # 8 px apart, radius 5 each -> overlapping
    frame, roi = make_scene(singles + pair)

    blobs = detect_flies(frame, roi)

    # 6 singles + 1 merged component.
    assert len(blobs) == len(singles) + 1

    merged = [b for b in blobs if b.is_merged]
    assert len(merged) == 1, "the touching pair should be flagged exactly once"
    m = merged[0]
    assert m.n_flies == 2, "merged blob estimated at %d flies, expected 2" % m.n_flies

    # It sits between the two discs it is made of.
    assert 90 <= m.centroid[0] <= 98
    assert abs(m.centroid[1] - 100) <= 1.5

    # Every isolated disc stays a single fly.
    for b in blobs:
        if not b.is_merged:
            assert b.n_flies == 1

    # And its area really is ~2x a single's.
    single_area = estimate_single_fly_area([b for b in blobs if not b.is_merged])
    assert 1.5 * single_area <= m.area <= 2.2 * single_area


def test_estimate_single_fly_area_is_robust_to_a_tail_of_merged_clumps():
    def blob(area):
        return Blob(centroid=(0.0, 0.0), area=area, bbox=(0, 0, 1, 1), mean_intensity=0.0)

    # 8 singles near 80 px + 4 clumps (2x, 3x, 5x). The plain mean would be dragged to ~145.
    areas = [78, 80, 82, 79, 81, 80, 83, 77] + [160, 162, 240, 400]
    est = estimate_single_fly_area([blob(a) for a in areas])
    assert 76 <= est <= 84, "trimmed estimate %.1f drifted off the single-fly mode" % est
    assert est < float(np.mean(areas))

    # Even when clumps are the MAJORITY it must still find the small mode.
    areas2 = [80, 80, 81] + [160] * 8
    assert 78 <= estimate_single_fly_area([blob(a) for a in areas2]) <= 84

    assert estimate_single_fly_area([]) == 0.0


def test_is_merged_and_n_flies_never_contradict_each_other():
    """`n_flies > 1` implies `is_merged`. Ungated, area/single rounds to 2 from 1.5x while
    is_merged only trips at 1.8x, leaving blobs claiming to be 2 flies AND not merged."""
    singles = [(25, 20), (25, 60), (25, 100), (25, 140), (25, 180), (60, 20)]
    pair = [(90, 100), (98, 100)]
    frame, roi = make_scene(singles + pair)

    for mult in (1.4, 1.8, 2.5, 10.0):
        blobs = detect_flies(frame, roi, DetectParams(merge_area_mult=mult))
        assert blobs
        for b in blobs:
            if b.n_flies > 1:
                assert b.is_merged, (
                    "blob area=%d claims %d flies but is_merged=False (mult=%.1f)"
                    % (b.area, b.n_flies, mult))
            if not b.is_merged:
                assert b.n_flies == 1

    # A mult so high that nothing is ever called merged -> everything is exactly one fly.
    blobs = detect_flies(frame, roi, DetectParams(merge_area_mult=100.0))
    assert all(b.n_flies == 1 and not b.is_merged for b in blobs)


def test_detect_flies_accepts_a_pinned_single_fly_area():
    frame, roi = make_scene([(30, 50), (80, 150)])
    # Pin the single-fly area at half the true disc area -> every disc reads as ~2 flies.
    p = DetectParams(single_fly_area=40.0)
    blobs = detect_flies(frame, roi, p)
    assert len(blobs) == 2
    for b in blobs:
        assert b.is_merged is True
        assert b.n_flies == 2


# ---- link_blobs --------------------------------------------------------------
def _blobs_at(points):
    return [Blob(centroid=(float(x), float(y)), area=80, bbox=(int(x) - 5, int(y) - 5, 11, 11),
                 mean_intensity=100.0)
            for (x, y) in points]


def test_link_blobs_matches_a_known_uniform_displacement():
    prev = _blobs_at([(10, 10), (50, 40), (90, 120)])
    shift = (3.0, -4.0)                                   # |shift| = 5 px
    cur = _blobs_at([(x + shift[0], y + shift[1]) for (x, y) in
                     [(10, 10), (50, 40), (90, 120)]])

    matches = link_blobs(prev, cur, max_dist=10.0)
    assert matches == [(0, 0), (1, 1), (2, 2)]


def test_link_blobs_matches_correctly_when_the_input_order_is_shuffled():
    prev = _blobs_at([(10, 10), (50, 40), (90, 120)])
    # Same three flies moved +2 px in x, but presented in a different order.
    cur = _blobs_at([(92, 120), (12, 10), (52, 40)])

    matches = link_blobs(prev, cur, max_dist=10.0)
    assert matches == [(0, 1), (1, 2), (2, 0)]


def test_link_blobs_distance_gate_drops_far_pairs_and_starts_new_tracks():
    prev = _blobs_at([(10, 10), (50, 40)])
    cur = _blobs_at([(12, 10), (300, 300)])               # 2nd is way outside the gate

    matches = link_blobs(prev, cur, max_dist=10.0)
    assert matches == [(0, 0)]                            # prev[1] ends, cur[1] starts fresh


def test_link_blobs_is_one_to_one_and_handles_empty_inputs():
    # Two prev blobs, one cur blob -> only ONE match (no double-booking).
    prev = _blobs_at([(10, 10), (14, 10)])
    cur = _blobs_at([(12, 10)])
    matches = link_blobs(prev, cur, max_dist=20.0)
    assert len(matches) == 1
    assert matches[0][1] == 0

    assert link_blobs([], cur, 10.0) == []
    assert link_blobs(prev, [], 10.0) == []
    assert link_blobs(prev, cur, 0.0) == []


# ---- axis projection / heights ----------------------------------------------
def test_project_heights_matches_known_geometry_along_an_explicit_axis():
    """A disc at the top of the tube must read height ~= 1, at the bottom ~= 0."""
    top = (60, 10)
    middle = (60, 100)
    bottom = (60, 190)
    frame, roi = make_scene([top, middle, bottom])

    # Vial axis in IMAGE coords: y grows downward, so "bottom of the vial" is large y.
    axis = ((60.0, 190.0), (60.0, 10.0))       # (bottom point, top point)

    blobs = detect_flies(frame, roi)
    assert len(blobs) == 3
    heights = project_heights(blobs, axis, roi)

    by_y = sorted(zip((b.centroid[1] for b in blobs), heights))
    (y_top, h_top), (y_mid, h_mid), (y_bot, h_bot) = by_y

    assert h_top == pytest.approx(1.0, abs=0.02), "top disc height %.3f, expected ~1" % h_top
    assert h_mid == pytest.approx(0.5, abs=0.02)
    assert h_bot == pytest.approx(0.0, abs=0.02), "bottom disc height %.3f, expected ~0" % h_bot


def test_project_heights_from_a_direction_vector_uses_the_roi_extent():
    frame, roi = make_scene([(60, 10), (60, 190)])
    blobs = detect_flies(frame, roi)
    assert len(blobs) == 2

    # Direction vector only: "up the image" is (0, -1). Extent comes from the ROI (rows 2..197).
    heights = project_heights(blobs, (0.0, -1.0), roi)
    hi = max(heights)
    lo = min(heights)
    assert hi == pytest.approx(1.0, abs=0.05)
    assert lo == pytest.approx(0.0, abs=0.05)

    # A horizontal axis makes both discs (same x) sit at the same height.
    flat = project_heights(blobs, (1.0, 0.0), roi)
    assert flat[0] == pytest.approx(flat[1], abs=0.02)


def test_project_heights_clips_into_the_unit_interval():
    frame, roi = make_scene([(60, 10), (60, 190)])
    blobs = detect_flies(frame, roi)
    # A deliberately SHORT axis (rows 150..50) leaves both discs outside its span.
    heights = project_heights(blobs, ((60.0, 150.0), (60.0, 50.0)), roi)
    assert all(0.0 <= h <= 1.0 for h in heights)
    assert max(heights) == 1.0 and min(heights) == 0.0


def test_resolve_axis_rejects_degenerate_input():
    frame, roi = make_scene([(60, 100)])
    blobs = detect_flies(frame, roi)
    with pytest.raises(ValueError):
        project_heights(blobs, ((10.0, 10.0), (10.0, 10.0)), roi)   # identical endpoints
    with pytest.raises(ValueError):
        project_heights(blobs, (0.0, 0.0), roi)                      # zero-length direction
    with pytest.raises(ValueError):
        project_heights(blobs, (1.0, 2.0, 3.0), roi)                 # wrong shape


# ---- VialTracker -------------------------------------------------------------
AXIS = ((60.0, 190.0), (60.0, 10.0))          # bottom -> top, 180 px long


def test_vial_tracker_recovers_known_speed_and_path_length():
    """One disc climbing 4 px per frame for 10 frames, at 20 fps.

    Expected, by construction:
      path length   = 9 steps x 4 px            = 36 px
      duration      = 9 intervals / 20 fps      = 0.45 s
      speed         = 36 / 0.45                 = 80 px/s
      normalized    = 36 px / 180 px axis       = 0.2 height units -> 0.4444 units/s
    """
    fps = 20.0
    tr = VialTracker(fps=fps, max_link_dist=15.0)

    n_frames, step = 10, 4
    for i in range(n_frames):
        y = 150 - i * step                      # climbing: y decreases
        frame, roi = make_scene([(60, y)])
        stats = tr.update(frame, roi, AXIS)
        assert stats.n_blobs == 1
        assert stats.est_n_flies == 1
        assert stats.index == i
        assert stats.t == pytest.approx(i / fps)

    # One unbroken fragment: the disc never left the gate.
    assert len(tr.tracks) == 1
    track = tr.tracks[0]
    assert track.n_frames == n_frames
    assert tr.mean_fragment_frames == pytest.approx(float(n_frames))

    expected_path = (n_frames - 1) * step       # 36 px
    assert track.path_length == pytest.approx(expected_path, abs=1.0)
    assert track.duration_s == pytest.approx((n_frames - 1) / fps)
    assert track.speed == pytest.approx(expected_path * fps / (n_frames - 1), rel=0.05)   # 80 px/s

    expected_norm = expected_path / 180.0       # axis length
    assert track.path_length_norm == pytest.approx(expected_norm, rel=0.05)
    assert track.speed_norm == pytest.approx(expected_norm * fps / (n_frames - 1), rel=0.05)

    # Height rose monotonically from the start position toward the top of the tube.
    assert track.heights[-1] > track.heights[0]
    start_h = (190 - 150) / 180.0
    end_h = (190 - (150 - (n_frames - 1) * step)) / 180.0
    assert track.heights[0] == pytest.approx(start_h, abs=0.02)
    assert track.heights[-1] == pytest.approx(end_h, abs=0.02)


def test_vial_tracker_summarize_matches_hand_computed_geometry():
    """Two discs at KNOWN heights, one climbing and one still, over 5 frames at 10 fps."""
    fps = 10.0
    tr = VialTracker(fps=fps, max_link_dist=15.0)

    n_frames, step = 5, 6
    # The climber starts at y=97 (height 0.5167) so every one of its samples is STRICTLY above
    # mid-height; starting at y=90 would put the first sample exactly on 0.5, where
    # `frac_above`'s strict '>' makes the expected answer a coin-flip on float rounding.
    start_y = 97
    for i in range(n_frames):
        climber_y = start_y - i * step          # 97, 91, 85, 79, 73
        frame, roi = make_scene([(35, climber_y), (90, 190)])   # 2nd disc parked at the bottom
        stats = tr.update(frame, roi, AXIS)
        assert stats.n_blobs == 2

    s = tr.summarize()

    assert s["n_frames"] == n_frames
    assert s["n_blobs_mean"] == pytest.approx(2.0)
    assert s["est_n_flies_mean"] == pytest.approx(2.0)

    # Heights: the parked disc is at y=190 -> h=0; the climber runs 97..73 -> h 0.517..0.650.
    climber_heights = [(190 - (start_y - i * step)) / 180.0 for i in range(n_frames)]
    expected_pool = climber_heights + [0.0] * n_frames
    assert s["mean_height"] == pytest.approx(float(np.mean(expected_pool)), abs=0.02)
    assert s["median_height"] == pytest.approx(float(np.median(expected_pool)), abs=0.02)
    assert s["max_height"] == pytest.approx(max(climber_heights), abs=0.02)
    # Exactly half the observations (the 5 climber frames) sit above mid-height.
    assert s["frac_above_mid"] == pytest.approx(0.5, abs=0.01)

    # Two unbroken fragments, so the diagnostic reports full-length tracks.
    assert s["n_tracks"] == 2
    assert s["mean_fragment_frames"] == pytest.approx(float(n_frames))

    # Path: climber moved 4 x 6 = 24 px, the parked disc 0 px.
    assert s["total_path_length"] == pytest.approx(24.0, abs=1.5)
    # Speeds: 24 px over 0.4 s = 60 px/s, and 0 px/s.
    assert s["p90_speed"] == pytest.approx(60.0, rel=0.1)
    assert s["mean_speed"] == pytest.approx(30.0, rel=0.15)
    assert s["median_speed"] == pytest.approx(30.0, rel=0.15)

    # Every documented key is present and is a plain scalar (joinable to ActivityRecord).
    for key in ("mean_height", "median_height", "frac_above_mid", "max_height", "n_blobs_mean",
                "est_n_flies_mean", "mean_speed", "median_speed", "p90_speed",
                "total_path_length", "mean_fragment_frames", "n_tracks"):
        assert key in s
        assert isinstance(s[key], (int, float))


def test_vial_tracker_fragments_when_a_fly_jumps_past_the_link_gate():
    """A too-big jump breaks the track in two — and `mean_fragment_frames` says so."""
    tr = VialTracker(fps=20.0, max_link_dist=10.0)
    positions = [150, 146, 142, 60, 56, 52]     # the 142 -> 60 step is 82 px, way over the gate
    for y in positions:
        frame, roi = make_scene([(60, y)])
        tr.update(frame, roi, AXIS)

    assert len(tr.tracks) == 2
    assert sorted(t.n_frames for t in tr.tracks) == [3, 3]
    assert tr.mean_fragment_frames == pytest.approx(3.0)
    assert tr.summarize()["n_tracks"] == 2


def test_vial_tracker_handles_frames_with_no_flies():
    tr = VialTracker(fps=20.0)
    blank = np.full((H, W), BG_LEVEL, dtype=np.uint8)
    roi = np.zeros((H, W), dtype=bool)
    roi[2:H - 2, 2:W - 2] = True

    for _ in range(3):
        stats = tr.update(blank, roi, AXIS)
        assert stats.n_blobs == 0
        assert stats.est_n_flies == 0
        assert stats.heights == []
        assert math.isnan(stats.mean_height)
        assert math.isnan(stats.frac_above(0.5))

    s = tr.summarize()
    assert s["n_frames"] == 3
    assert s["n_tracks"] == 0
    assert s["n_blobs_mean"] == 0.0
    assert s["total_path_length"] == 0.0
    # Undefined, NOT zero: "no flies" must not read as "all flies at the bottom".
    for key in ("mean_height", "median_height", "frac_above_mid", "max_height",
                "mean_speed", "median_speed", "p90_speed", "mean_fragment_frames"):
        assert math.isnan(s[key]), "%s should be nan for an empty vial, got %r" % (key, s[key])


def test_vial_tracker_accepts_explicit_timestamps_and_uint8_masks():
    tr = VialTracker(fps=20.0, max_link_dist=20.0)
    times = [0.0, 0.5, 1.0]                     # 2 s^-1 sampling, NOT the nominal 20 fps
    for i, t in enumerate(times):
        frame, roi = make_scene([(60, 150 - i * 10)])
        stats = tr.update(frame, (roi * 255).astype(np.uint8), AXIS, t=t)
        assert stats.t == t

    track = tr.tracks[0]
    assert track.times == times
    assert track.duration_s == pytest.approx(1.0)
    # 20 px of travel over 1.0 s of WALL time -> 20 px/s (not 20 px * 20 fps / 2).
    assert track.speed == pytest.approx(20.0, rel=0.1)


def test_vial_tracker_rejects_bad_fps():
    with pytest.raises(ValueError):
        VialTracker(fps=0)
    with pytest.raises(ValueError):
        VialTracker(fps=-5)


def test_default_link_dist_scales_inversely_with_fps():
    assert VialTracker.default_link_dist(20.0) == pytest.approx(6.0)
    assert VialTracker.default_link_dist(10.0) == pytest.approx(12.0)
    assert VialTracker(fps=20.0).max_link_dist == pytest.approx(6.0)


# ---- summarize as a pure function --------------------------------------------
def test_summarize_pure_function_on_hand_built_inputs():
    frames = [
        FrameStats(t=0.0, index=0, n_blobs=2, est_n_flies=3, heights=[0.2, 0.8],
                   total_blob_area=200),
        FrameStats(t=0.1, index=1, n_blobs=4, est_n_flies=5, heights=[0.3, 0.9, 0.95, 0.1],
                   total_blob_area=400),
    ]
    tracks = [
        Track(id=0, positions=[(0.0, 0.0), (3.0, 4.0)], heights=[0.2, 0.3], times=[0.0, 0.1]),
        Track(id=1, positions=[(0.0, 0.0)], heights=[0.8], times=[0.0]),   # 1-frame fragment
    ]
    s = summarize(frames, tracks)

    pooled = [0.2, 0.8, 0.3, 0.9, 0.95, 0.1]
    assert s["mean_height"] == pytest.approx(float(np.mean(pooled)))
    assert s["median_height"] == pytest.approx(float(np.median(pooled)))
    assert s["max_height"] == pytest.approx(0.95)
    assert s["frac_above_mid"] == pytest.approx(3 / 6)
    assert s["n_blobs_mean"] == pytest.approx(3.0)
    assert s["est_n_flies_mean"] == pytest.approx(4.0)

    # Track 0 moved 5 px (3-4-5 triangle) in 0.1 s -> 50 px/s. Track 1 has no speed at all and
    # must NOT be folded in as a zero.
    assert s["total_path_length"] == pytest.approx(5.0)
    assert s["mean_speed"] == pytest.approx(50.0)
    assert s["median_speed"] == pytest.approx(50.0)
    assert s["p90_speed"] == pytest.approx(50.0)
    assert s["mean_speed_norm"] == pytest.approx(1.0)      # 0.1 height units / 0.1 s
    assert s["n_tracks"] == 2
    assert s["mean_fragment_frames"] == pytest.approx(1.5)
    assert s["n_frames"] == 2


def test_summarize_on_nothing_at_all_is_all_nan_not_zero():
    s = summarize([], [])
    assert s["n_frames"] == 0
    assert s["n_tracks"] == 0
    assert s["total_path_length"] == 0.0
    for key in ("mean_height", "median_height", "frac_above_mid", "max_height", "n_blobs_mean",
                "est_n_flies_mean", "mean_speed", "median_speed", "p90_speed",
                "mean_fragment_frames"):
        assert math.isnan(s[key])


def test_frame_stats_frac_above_counts_strictly_above():
    fs = FrameStats(t=0.0, index=0, n_blobs=4, est_n_flies=4, heights=[0.1, 0.5, 0.6, 0.9])
    assert fs.frac_above(0.5) == pytest.approx(0.5)        # 0.5 itself does not count
    assert fs.frac_above(0.0) == pytest.approx(1.0)
    assert fs.frac_above(0.95) == pytest.approx(0.0)
    assert fs.mean_height == pytest.approx(np.mean([0.1, 0.5, 0.6, 0.9]))
    assert fs.median_height == pytest.approx(0.55)
