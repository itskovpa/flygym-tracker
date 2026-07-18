"""Synthetic, deterministic contract tests for `marker_band.MarkerBandDetector`.

Everything here is generated from `make_band_frame` -- no video, no rig, no randomness that matters.
The synthetic frame reproduces the parts of the real rig geometry the detector actually depends on:

  * two bright horizontal strips inside a dark central band,
  * one opaque sticker per vial column, alternating upper/lower across the 8 columns,
  * a large bright region along the BOTTOM of the frame -- on the real rig the illuminated stage is
    brighter and taller than either LED strip (measured rows ~937-1023 of 1024), so "the brightest
    rows in the frame" is NOT a valid way to find the band. Keeping it here pins the vertical
    search-window behaviour that handles it.

Face "A" is the pattern as built; face "B" is the SAME pattern with the two strips swapped, which is
what a 180 degree drum flip produces (see `marker_band`'s module docstring for the real-data
measurement that establishes this).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from flygym_tracker.marker_band import MarkerBandDetector, MarkerBandParams

# --- synthetic rig geometry (scaled down, but proportioned like the real 1280x1024 frame) ---
H, W = 700, 800
UPPER_ROWS = (300, 340)      # inclusive
LOWER_ROWS = (365, 405)      # inclusive
STAGE_ROW = H - 60           # illuminated stage occupies [STAGE_ROW, H)
N_VIALS = 8
VIAL_X0 = 80                 # left edge of vial column 0
VIAL_PITCH = 80              # column 0 spans [80, 159], column 1 [160, 239], ...
BG, DARK, BRIGHT = 30, 12, 250

#: ground-truth inclusive (x0, x1) of every vial column in the synthetic frame.
TRUE_SPANS = [
    (VIAL_X0 + i * VIAL_PITCH, VIAL_X0 + (i + 1) * VIAL_PITCH - 1) for i in range(N_VIALS)
]


def make_band_frame(
    swap: bool = False,
    dx: int = 0,
    dy: int = 0,
    notches: bool = False,
    symmetric: bool = False,
    left_truncate: int = 0,
    overlap: int = 0,
    sliver: tuple | None = None,
    lower_dx: int = 0,
    hot_pixels: bool = False,
    edge_glints: bool = False,
) -> np.ndarray:
    """Build a synthetic marker-band frame.

    Args:
        swap: emit the 180-degree-flipped face (upper and lower strip contents exchanged).
        dx, dy: rigid translation of the whole band, to exercise shift tolerance.
        notches: punch narrow dark slots into the lit runs, mimicking the mounting hardware that
            sits inside the real LED slot (these must be bridged, not read as vial boundaries).
        symmetric: light BOTH strips over the same columns. The two faces are then genuinely
            indistinguishable and `identify_face` must abstain.
        left_truncate: shorten the leftmost lit run by this many px, mimicking an LED slot that
            stops part-way across the end column.
        overlap: widen every lit run by this many px on each side so neighbouring runs from the two
            strips overlap, as the real ones do.
        sliver: ``(x0, x1)`` over which a lit run is reduced to a few lit rows, mimicking the real
            slot being occluded down to a thin sliver by frame hardware.
        lower_dx: shift ONLY the lower strip's pattern, so the two strips no longer share an x
            offset. A shared lit x-window plus the NCC lag search is what absorbs this.
        hot_pixels: sprinkle isolated saturated pixels through a sticker (dark) region, as a hot
            sensor pixel or an impulse noise spike would. The max-over-rows profile is by
            construction the most outlier-sensitive statistic, so these must be filtered out.
        edge_glints: add saturated but only-a-few-rows-tall patches out beyond the vial row, as the
            drum's side hardware does on the real rig. They must not stretch the analysis window.
    """
    img = np.full((H, W), BG, dtype=np.uint8)
    img[STAGE_ROW:, :] = 255                   # illuminated stage: brightest thing in the frame
    band_r0 = UPPER_ROWS[0] - 30 + dy
    band_r1 = LOWER_ROWS[1] + 30 + dy
    img[band_r0:band_r1 + 1, :] = DARK         # dark hardware surrounding the strips

    for i, (x0, x1) in enumerate(TRUE_SPANS):
        # sticker (opaque) on the upper strip for even columns, lower for odd -> the lit run for
        # column i is on the OPPOSITE strip. `swap` exchanges the two; `symmetric` lights both.
        lit_upper = symmetric or ((i % 2 == 1) != swap)
        lit_lower = symmetric or ((i % 2 == 0) != swap)
        for rows, lit in ((UPPER_ROWS, lit_upper), (LOWER_ROWS, lit_lower)):
            if not lit:
                continue
            shift = dx + (lower_dx if rows is LOWER_ROWS else 0)
            a, b = x0 + shift - overlap, x1 + shift + overlap
            if left_truncate and i == 0:
                a += left_truncate + overlap
            img[rows[0] + dy:rows[1] + 1 + dy, max(0, a):min(W, b + 1)] = BRIGHT
    if notches:
        for x in range(VIAL_X0 + 30, VIAL_X0 + N_VIALS * VIAL_PITCH, VIAL_PITCH):
            for rows in (UPPER_ROWS, LOWER_ROWS):
                img[rows[0] + dy:rows[1] + 1 + dy, x + dx:x + dx + 8] = DARK
    if sliver is not None:
        sx0, sx1 = sliver
        for rows in (UPPER_ROWS, LOWER_ROWS):
            r0, r1 = rows[0] + dy, rows[1] + dy
            mid = (r0 + r1) // 2
            block = img[r0:r1 + 1, sx0:sx1 + 1]
            keep = block[mid - r0 - 2:mid - r0 + 2].copy()      # a 4-row lit sliver survives
            block[:] = DARK
            block[mid - r0 - 2:mid - r0 + 2] = keep
    if edge_glints:
        for gx0, gx1 in ((10, 45), (750, 785)):
            img[UPPER_ROWS[0] + dy:UPPER_ROWS[0] + dy + 4, gx0:gx1] = BRIGHT
    if hot_pixels:
        rng = np.random.default_rng(0)
        for x in range(178, 224, 4):                   # inside column 1's sticker on the lower strip
            img[LOWER_ROWS[0] + dy + int(rng.integers(0, 40)), x + dx] = 255
    return img


@pytest.fixture
def registered() -> MarkerBandDetector:
    """A detector with face "A" and its 180-degree flip "B" registered."""
    det = MarkerBandDetector()
    det.register_face(make_band_frame(swap=False), "A")
    det.register_face(make_band_frame(swap=True), "B")
    return det


# ---------------------------------------------------------------------------------------
# find_strips
# ---------------------------------------------------------------------------------------
def test_find_strips_locates_both_strips_not_the_bright_stage():
    strips = MarkerBandDetector().find_strips(make_band_frame())
    assert len(strips) == 2
    (u0, u1), (l0, l1) = strips
    assert u0 < u1 < l0 < l1                       # upper strip returned first, disjoint
    assert abs(u0 - UPPER_ROWS[0]) <= 3 and abs(u1 - UPPER_ROWS[1]) <= 3
    assert abs(l0 - LOWER_ROWS[0]) <= 3 and abs(l1 - LOWER_ROWS[1]) <= 3
    # the bottom illuminated stage is brighter and taller, and must NOT be picked
    assert l1 < STAGE_ROW


def test_find_strips_rejects_a_block_too_tall_to_be_an_led_slot():
    """A tall bright region inside the search window outranks a real strip on lit mass alone."""
    img = make_band_frame()
    img[150:270, 100:600] = BRIGHT                 # 120 rows > params.max_strip_h
    strips = MarkerBandDetector().find_strips(img)
    assert len(strips) == 2
    assert abs(strips[0][0] - UPPER_ROWS[0]) <= 3
    assert abs(strips[1][1] - LOWER_ROWS[1]) <= 3


def test_find_strips_follows_a_shifted_band():
    strips = MarkerBandDetector().find_strips(make_band_frame(dy=14))
    assert len(strips) == 2
    assert abs(strips[0][0] - (UPPER_ROWS[0] + 14)) <= 3


def test_find_strips_returns_empty_without_a_band():
    det = MarkerBandDetector()
    assert det.find_strips(np.zeros((H, W), np.uint8)) == []
    assert det.find_strips(np.full((H, W), 200, np.uint8)) == []       # no contrast
    only_one = np.full((H, W), BG, np.uint8)
    only_one[UPPER_ROWS[0]:UPPER_ROWS[1], 100:700] = BRIGHT           # a single strip is not a band
    assert det.find_strips(only_one) == []


def test_find_strips_prefers_the_pair_with_the_most_lit_area_not_the_tallest():
    """A tall, narrow bright region must not outrank a short, wide LED slot."""
    img = make_band_frame()
    img[200:290, 60:180] = BRIGHT                  # 90 rows tall (taller than a strip) but narrow
    strips = MarkerBandDetector().find_strips(img)
    assert len(strips) == 2
    assert abs(strips[0][0] - UPPER_ROWS[0]) <= 3
    assert abs(strips[1][1] - LOWER_ROWS[1]) <= 3


def test_find_strips_rejects_a_faint_low_contrast_band():
    """Two horizontal bands only 12 grey levels above ground are not blindingly-lit LED slots."""
    faint = np.full((H, W), 100, np.uint8)
    faint[UPPER_ROWS[0]:UPPER_ROWS[1] + 1, 100:700] = 112
    faint[LOWER_ROWS[0]:LOWER_ROWS[1] + 1, 100:700] = 112
    assert MarkerBandDetector().find_strips(faint) == []


def test_find_strips_rejects_bright_runs_too_narrow_to_be_a_slot():
    """Tall but only a dozen columns wide: bright enough to set the percentile, not a strip."""
    img = np.full((H, W), 20, np.uint8)
    img[200:300, 400:412] = BRIGHT
    img[340:440, 400:412] = BRIGHT
    assert MarkerBandDetector().find_strips(img) == []


def test_find_strips_rejects_a_pair_too_far_apart():
    """The two LED slots straddle the rotation axis; runs 260 rows apart are unrelated hardware."""
    img = np.full((H, W), 20, np.uint8)
    img[200:240, 100:700] = BRIGHT
    img[500:540, 100:700] = BRIGHT
    assert MarkerBandDetector().find_strips(img) == []


def test_explicit_band_rows_are_honoured():
    det = MarkerBandDetector(band_rows=(250, 450))
    assert len(det.find_strips(make_band_frame())) == 2
    # a window that excludes the band finds nothing
    assert MarkerBandDetector(band_rows=(0, 200)).find_strips(make_band_frame()) == []


# ---------------------------------------------------------------------------------------
# profiles / signature
# ---------------------------------------------------------------------------------------
def test_strip_profile_is_normalized_and_fixed_length():
    det = MarkerBandDetector()
    frame = make_band_frame()
    strips = det.find_strips(frame)
    prof = det.strip_profile(frame, strips[0])
    assert prof.shape == (det.params.profile_length,)
    assert prof.dtype == np.float64
    assert 0.0 <= prof.min() and prof.max() <= 1.0
    assert prof.max() > 0.9 and prof.min() < 0.1                      # genuinely bimodal

    # a specular blob brighter than the bulk of the strip sits above the bright percentile:
    # the profile must still be bounded, so downstream thresholds stay meaningful
    hot = frame.copy()
    hot[UPPER_ROWS[0]:UPPER_ROWS[0] + 6, 300:312] = 255
    hot_prof = det.strip_profile(hot, det.find_strips(hot)[0])
    assert hot_prof.max() <= 1.0 and hot_prof.min() >= 0.0


def test_profile_reads_a_thin_lit_sliver_as_lit():
    """Only a few rows of the slot are lit -> a row-MEAN profile would call it dark; MAX must not.

    This is the real-rig case that decided the profile definition: parts of the LED slot are
    occluded down to a sliver, and a sticker is the only thing that makes a column truly dark.
    """
    det = MarkerBandDetector()
    frame = make_band_frame(sliver=(190, 229))                        # inside column 1's lit run
    prof = det.strip_profile(frame, det.find_strips(frame)[0])
    x = int(det.params.profile_length * (210 - VIAL_X0) / (N_VIALS * VIAL_PITCH))
    assert prof[x] > det.params.block_level


def test_profile_length_is_configurable():
    det = MarkerBandDetector(params=MarkerBandParams(profile_length=64))
    up, lo = det.signature(make_band_frame())
    assert up.shape == lo.shape == (64,)


def test_signature_upper_and_lower_are_anticorrelated_within_a_face():
    """The two strips carry the interleaved halves of one alternating pattern."""
    up, lo = MarkerBandDetector().signature(make_band_frame())
    assert float(np.corrcoef(up, lo)[0, 1]) < -0.5


def test_flip_swaps_the_two_profiles():
    """The central verified fact: a 180 degree flip shows the SAME pattern, strips exchanged."""
    det = MarkerBandDetector()
    a_up, a_lo = det.signature(make_band_frame(swap=False))
    b_up, b_lo = det.signature(make_band_frame(swap=True))
    assert float(np.corrcoef(a_up, b_lo)[0, 1]) > 0.95
    assert float(np.corrcoef(a_lo, b_up)[0, 1]) > 0.95


def test_signature_is_none_without_a_band():
    assert MarkerBandDetector().signature(np.zeros((H, W), np.uint8)) is None


# ---------------------------------------------------------------------------------------
# identify_face
# ---------------------------------------------------------------------------------------
def test_identify_face_distinguishes_the_two_faces(registered):
    assert registered.identify_face(make_band_frame(swap=False)) == "A"
    assert registered.identify_face(make_band_frame(swap=True)) == "B"


@pytest.mark.parametrize("dx", [-12, -6, 0, 6, 12])
@pytest.mark.parametrize("dy", [-8, 0, 8])
def test_identify_face_tolerates_small_shifts(registered, dx, dy):
    assert registered.identify_face(make_band_frame(swap=False, dx=dx, dy=dy)) == "A"
    assert registered.identify_face(make_band_frame(swap=True, dx=dx, dy=dy)) == "B"


def test_identify_face_tolerates_hardware_notches_in_the_lit_runs(registered):
    assert registered.identify_face(make_band_frame(swap=False, notches=True)) == "A"
    assert registered.identify_face(make_band_frame(swap=True, notches=True)) == "B"


@pytest.mark.parametrize("gain", [0.45, 0.7, 1.0])
def test_identify_face_tolerates_exposure_changes(registered, gain):
    """Thresholds are per-frame relative, so a global gain change must not break detection."""
    for swap, face in ((False, "A"), (True, "B")):
        dim = (make_band_frame(swap=swap).astype(np.float32) * gain).astype(np.uint8)
        assert registered.identify_face(dim) == face


def test_identify_face_tolerates_the_strips_being_offset_from_each_other(registered):
    """Both profiles must live on ONE shared x axis, or the swap comparison is meaningless."""
    assert registered.identify_face(make_band_frame(swap=False, lower_dx=18)) == "A"
    assert registered.identify_face(make_band_frame(swap=True, lower_dx=18)) == "B"


def test_lag_search_absorbs_residual_misalignment(registered):
    """Cropping + resampling removes a GLOBAL shift; the NCC lag search covers what is left."""
    offset = make_band_frame(swap=False, lower_dx=18)
    no_lag = MarkerBandDetector(
        templates=registered.to_dict()["templates"], params=MarkerBandParams(max_lag=0),
    )
    assert registered.score_faces(offset)["A"] > no_lag.score_faces(offset)["A"]


def test_identify_face_returns_none_without_a_discernible_band(registered):
    assert registered.identify_face(np.zeros((H, W), np.uint8)) is None
    assert registered.identify_face(np.full((H, W), 128, np.uint8)) is None
    stage_only = np.full((H, W), BG, np.uint8)
    stage_only[STAGE_ROW:, :] = 255
    assert registered.identify_face(stage_only) is None


def test_identify_face_returns_none_with_fewer_than_two_faces():
    det = MarkerBandDetector()
    det.register_face(make_band_frame(), "A")
    assert det.identify_face(make_band_frame()) is None


def test_identify_face_abstains_when_the_pattern_is_ambiguous(registered):
    """Both strips lit identically -> the flip is a no-op -> the face is undecidable."""
    assert registered.identify_face(make_band_frame(symmetric=True)) is None


def test_min_margin_alone_rejects_an_ambiguous_pattern(registered):
    """Isolate the margin gate from the score gate: with `min_score` disabled it must still abstain."""
    det = MarkerBandDetector(
        templates=registered.to_dict()["templates"], params=MarkerBandParams(min_score=-1.0),
    )
    ambiguous = make_band_frame(symmetric=True)
    scores = det.score_faces(ambiguous)
    assert abs(scores["A"] - scores["B"]) < det.params.min_margin
    assert det.identify_face(ambiguous) is None
    assert det.identify_face(make_band_frame(swap=True)) == "B"       # still decides clear cases


def test_identify_face_abstains_on_an_unrelated_band(registered):
    """A band whose pattern matches neither template scores below `min_score`."""
    img = np.full((H, W), BG, np.uint8)
    img[UPPER_ROWS[0] - 30:LOWER_ROWS[1] + 30, :] = DARK
    rng = np.random.default_rng(7)
    for rows in (UPPER_ROWS, LOWER_ROWS):
        for x in np.flatnonzero(rng.random(W // 20) > 0.5) * 20:
            img[rows[0]:rows[1] + 1, x:x + 20] = BRIGHT
    assert registered.identify_face(img) is None


def test_score_faces_reports_both_faces_and_a_clear_margin(registered):
    scores = registered.score_faces(make_band_frame(swap=False))
    assert set(scores) == {"A", "B"}
    assert scores["A"] > scores["B"] + registered.params.min_margin
    assert registered.score_faces(np.zeros((H, W), np.uint8)) == {}


def test_swap_check_rescues_a_template_registered_from_a_damaged_frame():
    """Face F's upper is also visible as the OTHER face's lower, so a bad capture is recoverable.

    Registering "A" from a frame whose upper strip was partly occluded gives it a poor upper
    template. `use_swap_check` substitutes face B's (good) lower template for half the comparison,
    which must raise the score for a clean face-A query -- enough here to keep the frame decidable
    when the direct-only score has already fallen through `min_score`.
    """
    damaged = make_band_frame(swap=False)
    damaged[UPPER_ROWS[0]:UPPER_ROWS[1] + 1, W // 2:] = DARK          # kill half the upper strip
    clean_a = make_band_frame(swap=False)

    scores, decisions = {}, {}
    for use_swap in (True, False):
        det = MarkerBandDetector(params=MarkerBandParams(use_swap_check=use_swap))
        det.register_face(damaged, "A")
        det.register_face(make_band_frame(swap=True), "B")
        scores[use_swap] = det.score_faces(clean_a)["A"]
        decisions[use_swap] = det.identify_face(clean_a)

    assert scores[True] > scores[False]
    assert decisions[True] == "A"
    # and the direct-only score has degraded past the confidence floor -- it abstains rather than
    # guessing, which is the correct failure mode, but it has lost the frame.
    assert decisions[False] != "A"


def test_register_face_rejects_a_frame_without_a_band():
    with pytest.raises(ValueError):
        MarkerBandDetector().register_face(np.zeros((H, W), np.uint8), "A")


# ---------------------------------------------------------------------------------------
# vial_boundaries
# ---------------------------------------------------------------------------------------
def _assert_spans_match(spans, tol=6):
    assert len(spans) == N_VIALS
    assert spans == sorted(spans)
    for (a, b), (ta, tb) in zip(spans, TRUE_SPANS):
        assert abs(a - ta) <= tol, f"{spans} vs {TRUE_SPANS}"
        assert abs(b - tb) <= tol, f"{spans} vs {TRUE_SPANS}"
    for (_, b), (a2, _) in zip(spans, spans[1:]):
        assert a2 > b, f"spans overlap: {spans}"


@pytest.mark.parametrize("swap", [False, True])
def test_vial_boundaries_recovers_the_known_block_ranges(swap):
    _assert_spans_match(MarkerBandDetector().vial_boundaries(make_band_frame(swap=swap)))


def test_vial_boundaries_bridges_hardware_notches():
    """Narrow dark slots inside a lit run must not be read as extra vial boundaries."""
    _assert_spans_match(MarkerBandDetector().vial_boundaries(make_band_frame(notches=True)))


def test_vial_boundaries_bridges_a_thin_lit_sliver():
    """A stretch of run that is only a few rows lit is still the same vial, not two."""
    _assert_spans_match(MarkerBandDetector().vial_boundaries(make_band_frame(sliver=(190, 229))))


def test_vial_boundaries_survive_hot_pixels_inside_a_sticker():
    """The profile is a per-column MAX, so isolated saturated pixels must be filtered first."""
    _assert_spans_match(MarkerBandDetector().vial_boundaries(make_band_frame(hot_pixels=True)))


def test_vial_boundaries_ignore_bright_hardware_glints_beyond_the_vial_row():
    """The analysis x-window comes from the lit-FRACTION profile, so a few saturated rows out at
    the drum's edge cannot stretch it and invent a ninth vial."""
    _assert_spans_match(MarkerBandDetector().vial_boundaries(make_band_frame(edge_glints=True)))


def test_vial_boundaries_are_disjoint_when_the_lit_runs_overlap():
    """Neighbouring runs come from different strips and can overlap; output must not."""
    spans = MarkerBandDetector().vial_boundaries(make_band_frame(overlap=10))
    assert len(spans) == N_VIALS
    for (_, b), (a2, _) in zip(spans, spans[1:]):
        assert a2 == b + 1, f"spans must tile without overlap or gaps here: {spans}"
    # the split lands on the true column boundary, within the overlap half-width
    for (_, b), (tb, _) in zip(spans, TRUE_SPANS[1:]):
        assert abs(b - (tb - 1)) <= 10, f"{spans} vs {TRUE_SPANS}"


def test_vial_boundaries_follows_a_shifted_band():
    spans = MarkerBandDetector().vial_boundaries(make_band_frame(dx=15))
    assert len(spans) == N_VIALS
    for (a, _), (ta, _) in zip(spans, TRUE_SPANS):
        assert abs(a - (ta + 15)) <= 6


@pytest.mark.parametrize("gain", [0.45, 0.7])
def test_vial_boundaries_tolerate_exposure_changes(gain):
    dim = (make_band_frame().astype(np.float32) * gain).astype(np.uint8)
    _assert_spans_match(MarkerBandDetector().vial_boundaries(dim))


def test_vial_boundaries_extrapolates_a_truncated_end_column():
    """Leftmost lit run cut short by the slot ending: the outer edge is extrapolated to full width."""
    spans = MarkerBandDetector().vial_boundaries(make_band_frame(left_truncate=30))
    assert len(spans) == N_VIALS
    first_w = spans[0][1] - spans[0][0] + 1
    median_w = float(np.median([b - a + 1 for a, b in spans[1:-1]]))
    assert first_w > 30 + 30, f"truncated column was not extended: {spans}"
    assert abs(first_w - median_w) <= 8, f"{first_w} vs median {median_w}"


def test_vial_boundaries_empty_without_a_band():
    assert MarkerBandDetector().vial_boundaries(np.zeros((H, W), np.uint8)) == []


def test_min_run_px_rejects_narrow_glints():
    img = make_band_frame()
    img[UPPER_ROWS[0]:UPPER_ROWS[1] + 1, 20:30] = BRIGHT       # 10 px glint outside the vial row
    assert len(MarkerBandDetector(min_run_px=25).vial_boundaries(img)) == N_VIALS
    assert len(MarkerBandDetector(min_run_px=5).vial_boundaries(img)) == N_VIALS + 1


# ---------------------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------------------
def test_to_dict_from_dict_round_trip(registered):
    payload = registered.to_dict()
    json.loads(json.dumps(payload))                       # must be JSON-safe
    restored = MarkerBandDetector.from_dict(payload)

    assert set(restored.templates) == set(registered.templates)
    for face, (up, lo) in registered.templates.items():
        r_up, r_lo = restored.templates[face]
        assert np.array_equal(up, r_up)
        assert np.array_equal(lo, r_lo)
    assert restored.bright_percentile == registered.bright_percentile
    assert restored.min_run_px == registered.min_run_px
    assert restored.params == registered.params
    assert restored.band_rows == registered.band_rows

    assert restored.identify_face(make_band_frame(swap=False)) == "A"
    assert restored.identify_face(make_band_frame(swap=True)) == "B"


def test_to_dict_from_dict_preserves_non_default_settings():
    det = MarkerBandDetector(
        band_rows=(250, 450), bright_percentile=98.0, min_run_px=11,
        params=MarkerBandParams(profile_length=64, max_lag=3, search_frac=(0.1, 0.9)),
    )
    det.register_face(make_band_frame(), "A")
    restored = MarkerBandDetector.from_dict(json.loads(json.dumps(det.to_dict())))
    assert restored.band_rows == (250, 450)
    assert restored.bright_percentile == 98.0
    assert restored.min_run_px == 11
    assert restored.params.profile_length == 64
    assert restored.params.max_lag == 3
    assert restored.params.search_frac == (0.1, 0.9)


def test_templates_can_be_passed_to_the_constructor(registered):
    det = MarkerBandDetector(templates=registered.to_dict()["templates"])
    assert det.identify_face(make_band_frame(swap=True)) == "B"


# ---------------------------------------------------------------------------------------
# pipeline duck-typing
# ---------------------------------------------------------------------------------------
def test_duck_types_as_a_pipeline_marker_detector(registered):
    """`TrackerPipeline` only ever calls `identify_face(gray) -> str | None` (DESIGN.md 5.2)."""
    from flygym_tracker.markers import MarkerDetector

    assert callable(registered.identify_face)
    assert hasattr(MarkerDetector, "identify_face")
    result = registered.identify_face(make_band_frame(swap=True))
    assert result is None or isinstance(result, str)
