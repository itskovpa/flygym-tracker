"""Tests for quad (4-vertex) vial ROIs: geometry, the editor's pure logic, cross-face transfer,
the pipeline measurement path, and backward compatibility with quad-less bundles.

HEADLESS BY CONSTRUCTION. Nothing here opens a cv2 window: `roi_editor` is deliberately split
into pure state/geometry (`EditorState`, `ViewTransform`, `DragController`, `decode_key`,
`handle_key`, `render_view`) and a thin `run_roi_editor` driver that does nothing but pump
events into them, so everything that can be wrong can be tested without a display.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from flygym_tracker import calibration as C
from flygym_tracker.roi_editor import (
    DragController,
    EditorState,
    Hit,
    ViewTransform,
    decode_key,
    handle_key,
    render_view,
    tinted_frame,
)
from flygym_tracker.types import Calibration, FaceCalibration, VialROI

W, H = 200, 200


def _rect_vial(vid=1, row=0, col=0, x=10, y=10, w=40, h=60, present=True, quad=None):
    return VialROI(id=vid, row=row, col=col, x=x, y=y, w=w, h=h, present=present, quad=quad)


def _face(vials, name="A", mask_path="illum_mask_A.png", marker=None):
    return FaceCalibration(name=name, vials=list(vials), illum_mask_path=mask_path, marker=marker)


def _two_col_face(name="A", spans=((10, 49), (60, 99)), bands=((10, 70), (80, 140))):
    """A tiny but structurally real face: 2 columns x 2 rows, ids 1,2 (upper) and 3,4 (lower)."""
    vials = []
    vid = 1
    for row, (y0, y1) in enumerate(bands):
        for col, (x0, x1) in enumerate(spans):
            vials.append(_rect_vial(vid, row, col, x0, y0, x1 - x0 + 1, y1 - y0))
            vid += 1
    marker = {"vial_spans": [list(s) for s in spans],
              "row_bands": {"upper": list(bands[0]), "lower": list(bands[1])}}
    return _face(vials, name=name, marker=marker)


# =========================================================================================
# 1. Quad geometry primitives
# =========================================================================================
def test_quad_from_bbox_is_clockwise_from_top_left():
    assert C.quad_from_bbox((10, 20, 30, 40)) == [[10, 20], [40, 20], [40, 60], [10, 60]]


def test_bbox_from_quad_round_trips_a_rectangle():
    box = (10, 20, 30, 40)
    assert C.bbox_from_quad(C.quad_from_bbox(box)) == box


def test_bbox_from_quad_bounds_a_skewed_quad():
    quad = [[12, 20], [38, 24], [40, 58], [10, 61]]
    assert C.bbox_from_quad(quad) == (10, 20, 30, 41)


def test_bbox_from_quad_is_never_degenerate():
    x, y, w, h = C.bbox_from_quad([[5, 5], [5, 5], [5, 5], [5, 5]])
    assert (w, h) == (1, 1) and (x, y) == (5, 5)


def test_polygon_area_shoelace():
    assert C.polygon_area(C.quad_from_bbox((0, 0, 10, 20))) == pytest.approx(200.0)


def test_quad_polygon_mask_counts_the_rectangle_exactly():
    box = (10, 20, 30, 40)
    mask = C.quad_polygon_mask(C.quad_from_bbox(box), box)
    assert mask.shape == (40, 30)
    # fillPoly is inclusive of its boundary, so a WxH rectangle rasterises to (W+1)x(H+1) minus
    # whatever falls outside the crop -- the exact count is what the pipeline will measure.
    assert int(mask.sum()) == 30 * 40


def test_sync_bbox_to_quad_recomputes_the_bbox():
    v = _rect_vial(quad=[[12, 20], [38, 24], [40, 58], [10, 61]])
    assert (v.x, v.y, v.w, v.h) == (10, 10, 40, 60)      # stale, as loaded
    C.sync_bbox_to_quad(v)
    assert (v.x, v.y, v.w, v.h) == (10, 20, 30, 41)


def test_shift_quad_matches_apply_shift_rounding():
    from flygym_tracker.registration import apply_shift
    quad = C.quad_from_bbox((10, 20, 30, 40))
    shifted = C.shift_quad(quad, 2.6, -1.4)
    box = apply_shift((10, 20, 30, 40), 2.6, -1.4)
    assert C.bbox_from_quad(shifted) == box == (13, 19, 30, 40)


# =========================================================================================
# 2. Serialization / BACKWARD COMPATIBILITY with quad-less bundles
# =========================================================================================
def test_vial_quad_normalises_to_int_lists():
    v = _rect_vial(quad=[(1.4, 2.6), np.array([3, 4]), [5, 6], (7, 8)])
    assert v.quad == [[1, 3], [3, 4], [5, 6], [7, 8]]


def test_vial_rejects_a_quad_that_is_not_four_corners():
    with pytest.raises(ValueError):
        _rect_vial(quad=[[0, 0], [1, 1], [2, 2]])


def test_calibration_round_trips_quads(tmp_path):
    quad = [[12, 20], [38, 24], [40, 58], [10, 61]]
    calib = Calibration(image_width=W, image_height=H,
                        faces={"A": _face([_rect_vial(quad=quad), _rect_vial(vid=2)])})
    path = str(tmp_path / "calibration.json")
    calib.to_json(path)
    back = Calibration.from_json(path)
    assert back.faces["A"].vials[0].quad == quad
    assert back.faces["A"].vials[1].quad is None      # a vial may keep no quad at all


def test_old_bundle_without_quad_loads_and_defaults_to_its_bbox(tmp_path):
    """A calibration.json written before quads existed has no `quad` key at all."""
    payload = {
        "image_width": W, "image_height": H, "created": "", "notes": "legacy",
        "faces": {"A": {"name": "A", "illum_mask_path": "illum_mask_A.png", "marker": None,
                        "vials": [{"id": 1, "row": 0, "col": 0, "x": 10, "y": 20,
                                   "w": 30, "h": 40, "present": True}]}},
    }
    path = str(tmp_path / "calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    calib = Calibration.from_json(path)
    v = calib.faces["A"].vials[0]
    assert v.quad is None
    assert C.vial_quad(v) == C.quad_from_bbox((10, 20, 30, 40))
    assert C.face_quads(calib.faces["A"]) == [C.quad_from_bbox((10, 20, 30, 40))]


def test_real_two_face_bundle_loads_with_quads_defaulting_from_its_boxes():
    """The REAL bundle in `calib_faces/` (32 vials, written before quads) still loads."""
    calib_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calib_faces")
    if not os.path.isdir(calib_dir):
        pytest.skip("calib_faces bundle not present")
    calib = C.load_calibration(calib_dir)
    assert sorted(calib.faces) == ["A", "B"]
    assert sum(len(fc.vials) for fc in calib.faces.values()) == 32
    for fc in calib.faces.values():
        for v in fc.vials:
            assert v.quad is None
            assert C.vial_quad(v) == C.quad_from_bbox((v.x, v.y, v.w, v.h))


# =========================================================================================
# 3. lit_fraction maths on a synthetic mask
# =========================================================================================
def _half_lit_mask():
    """Mask lit only on x in [10, 30) -- so the LEFT HALF of a 10..50 box is lit."""
    m = np.zeros((H, W), np.uint8)
    m[:, 10:30] = 255
    return m


def test_lit_fraction_is_exact_on_a_synthetic_mask():
    mask = _half_lit_mask()
    box = (10, 10, 40, 20)                       # 40 px wide, 20 lit -> exactly half
    assert C.quad_lit_fraction(C.quad_from_bbox(box), mask) == pytest.approx(0.5)


def test_lit_fraction_rises_when_the_quad_is_tightened_onto_the_lit_part():
    mask = _half_lit_mask()
    wide = C.quad_from_bbox((10, 10, 40, 20))
    tight = C.quad_from_bbox((10, 10, 20, 20))
    assert C.quad_lit_fraction(wide, mask) == pytest.approx(0.5)
    assert C.quad_lit_fraction(tight, mask) == pytest.approx(1.0)


def test_lit_fraction_of_an_off_frame_quad_is_zero():
    assert C.quad_lit_fraction(C.quad_from_bbox((-50, -50, 10, 10)), _half_lit_mask()) == 0.0


def test_editor_lit_fraction_matches_the_calibration_helper():
    mask = _half_lit_mask()
    st = EditorState(_face([_rect_vial(x=10, y=10, w=40, h=20)]), mask)
    assert st.lit_fraction(0) == pytest.approx(0.5)
    assert st.lit_fraction() == st.lit_fraction(0)


def test_editor_lit_fraction_is_nan_without_a_mask():
    st = EditorState(_face([_rect_vial()]))
    assert np.isnan(st.lit_fraction(0))


def test_editing_a_vertex_updates_the_cached_lit_fraction():
    mask = _half_lit_mask()
    st = EditorState(_face([_rect_vial(x=10, y=10, w=40, h=20)]), mask)
    assert st.lit_fraction(0) == pytest.approx(0.5)
    st.set_vertex(0, 1, 30, 10)      # pull TR left, onto the lit boundary
    st.set_vertex(0, 2, 30, 30)      # ... and BR with it
    assert st.lit_fraction(0) == pytest.approx(1.0)


# =========================================================================================
# 4. EditorState: selection, edits, undo/redo
# =========================================================================================
def _four_vial_state(mask=None):
    return EditorState(_two_col_face(), mask, image_size=(W, H))


def test_select_wraps_both_ways():
    st = _four_vial_state()
    assert st.n_vials == 4
    assert st.select(5) == 1
    assert st.select(-1) == 3
    st.next_vial(); assert st.selected == 0
    st.prev_vial(); assert st.selected == 3


def test_cycle_vertex_walks_tl_tr_br_bl_then_whole_roi():
    st = _four_vial_state()
    assert [st.cycle_vertex() for _ in range(5)] == [0, 1, 2, 3, None]


def test_move_vertex_moves_only_that_vertex():
    st = _four_vial_state()
    before = [list(p) for p in st.quad(0)]
    st.move_vertex(0, 1, 5, -3)
    after = st.quad(0)
    assert after[1] == [before[1][0] + 5, before[1][1] - 3]
    assert [after[i] for i in (0, 2, 3)] == [before[i] for i in (0, 2, 3)]
    assert st.dirty is True


def test_move_roi_translates_every_vertex():
    st = _four_vial_state()
    before = [list(p) for p in st.quad(2)]
    st.move_roi(2, -4, 7)
    assert st.quad(2) == [[p[0] - 4, p[1] + 7] for p in before]


def test_edits_are_clipped_to_the_frame():
    st = _four_vial_state()
    st.move_roi(0, -10_000, -10_000)
    assert st.quad(0) == [[0, 0]] * 4
    st.move_roi(1, 10_000, 10_000)
    assert st.quad(1) == [[W, H]] * 4


def test_undo_restores_the_previous_quad_and_redo_reapplies_it():
    st = _four_vial_state()
    original = [list(p) for p in st.quad(0)]
    st.move_vertex(0, 0, 9, 9)
    edited = [list(p) for p in st.quad(0)]
    assert edited != original

    assert st.undo() is True
    assert st.quad(0) == original
    assert st.redo() is True
    assert st.quad(0) == edited
    assert st.redo() is False          # nothing further to redo
    assert st.undo() is True
    assert st.undo() is False          # history exhausted
    assert st.quad(0) == original


def test_undo_stack_is_deep_and_ordered():
    st = _four_vial_state()
    snaps = [[list(p) for p in st.quad(0)]]
    for _ in range(5):
        st.move_vertex(0, 0, 1, 1)
        snaps.append([list(p) for p in st.quad(0)])
    for i in range(5, 0, -1):
        assert st.quad(0) == snaps[i]
        st.undo()
    assert st.quad(0) == snaps[0]


def test_a_new_edit_clears_the_redo_stack():
    st = _four_vial_state()
    st.move_vertex(0, 0, 5, 5)
    st.undo()
    st.move_vertex(0, 1, 2, 2)
    assert st.redo() is False


def test_one_drag_is_one_undo_step():
    """The driver pushes undo at mouse-DOWN; the move samples must not each push their own."""
    st = _four_vial_state()
    original = [list(p) for p in st.quad(0)]
    drag = DragController(st, ViewTransform())
    drag.on_press(*st.quad(0)[0])
    for step in range(1, 6):
        drag.on_move(original[0][0] + step, original[0][1] + step)
    drag.on_release()
    assert st.quad(0)[0] == [original[0][0] + 5, original[0][1] + 5]
    assert st.undo() is True
    assert st.quad(0) == original


def test_reset_vial_and_reset_all_restore_the_auto_quads():
    st = _four_vial_state()
    base0 = [list(p) for p in st.quad(0)]
    st.move_roi(0, 5, 5)
    st.move_roi(1, 5, 5)
    st.reset_vial(0)
    assert st.quad(0) == base0
    assert st.quad(1) != st.base_quads[1]
    st.reset_all()
    assert st.quads == st.base_quads


# =========================================================================================
# 5. copy_shape_to_all
# =========================================================================================
def test_copy_shape_to_all_preserves_each_vials_own_box_but_copies_the_taper():
    st = _four_vial_state()
    # Taper vial 0: pull both right-hand corners in by 25% of its width.
    x, y, w, h = C.bbox_from_quad(st.quad(0))
    st.set_vertex(0, 1, x + 0.75 * w, y)
    st.set_vertex(0, 2, x + 0.75 * w, y + h)
    changed = st.copy_shape_to_all(0)
    assert changed == 3

    for i in range(1, 4):
        bx, by, bw, bh = C.bbox_from_quad(st.base_quads[i])
        assert st.quad(i) == [[bx, by], [int(round(bx + 0.75 * bw)), by],
                              [int(round(bx + 0.75 * bw)), by + bh], [bx, by + bh]]


def test_copy_shape_to_all_is_idempotent():
    st = _four_vial_state()
    st.move_vertex(0, 1, -6, 0)
    st.copy_shape_to_all(0)
    once = [[list(p) for p in q] for q in st.quads]
    st.copy_shape_to_all(0)
    assert st.quads == once


def test_copy_shape_to_all_raw_makes_every_roi_the_same_size():
    st = _four_vial_state()
    st.copy_shape_to_all(0, normalize=False)
    src_area = C.polygon_area(st.quad(0))
    for i in range(1, 4):
        assert C.polygon_area(st.quad(i)) == pytest.approx(src_area, abs=1.0)


def test_copy_shape_to_all_is_undoable_in_one_step():
    st = _four_vial_state()
    before = [[list(p) for p in q] for q in st.quads]
    st.move_vertex(0, 1, -6, 0)
    st.copy_shape_to_all(0)
    st.undo()
    st.undo()
    assert st.quads == before


# =========================================================================================
# 6. hit testing
# =========================================================================================
def test_hit_test_prefers_a_vertex_then_the_interior_then_nothing():
    st = _four_vial_state()
    vx, vy = st.quad(0)[2]                              # BR corner of vial 0
    assert st.hit_test(vx + 2, vy + 2) == Hit("vertex", 0, 2)

    cx, cy = np.mean(np.asarray(st.quad(0)), axis=0)
    assert st.hit_test(cx, cy) == Hit("roi", 0, None)
    assert st.hit_test(55, 190) is None                  # empty space below the lower band


def test_hit_test_prefers_the_selected_vials_vertex_when_two_coincide():
    """Neighbouring vials can share an edge; the one being worked on must win the grab."""
    face = _two_col_face()
    st = EditorState(face, image_size=(W, H))
    st.quads[1] = [list(p) for p in st.quads[0]]        # make vial 1 sit exactly on vial 0
    st.select(1)
    assert st.hit_test(*st.quads[1][0]) == Hit("vertex", 1, 0)
    st.select(0)
    assert st.hit_test(*st.quads[0][0]) == Hit("vertex", 0, 0)


def test_hit_test_radius_is_respected():
    st = EditorState(_two_col_face(), grab_radius=3, image_size=(W, H))
    vx, vy = st.quad(0)[0]
    assert st.hit_test(vx + 2, vy) == Hit("vertex", 0, 0)
    assert (st.hit_test(vx + 9, vy) or Hit("roi", 0)).kind == "roi"


# =========================================================================================
# 7. quad -> bbox recomputation on export
# =========================================================================================
def test_to_face_calibration_writes_quads_and_resyncs_bboxes():
    st = _four_vial_state()
    st.move_vertex(0, 0, 3, 4)
    out = st.to_face_calibration()
    assert out is not st.face_cal
    for i, v in enumerate(out.vials):
        assert v.quad == st.quads[i]
        assert (v.x, v.y, v.w, v.h) == C.bbox_from_quad(st.quads[i])
    assert st.face_cal.vials[0].quad is None            # source untouched


def test_apply_quads_to_face_rejects_a_wrong_length_list():
    with pytest.raises(ValueError):
        C.apply_quads_to_face(_two_col_face(), [C.quad_from_bbox((0, 0, 5, 5))])


# =========================================================================================
# 8. transfer_quads
# =========================================================================================
def test_transfer_quads_maps_a_known_quad_onto_a_known_destination_span():
    """Source column 0 spans x=0..99 and rows y=0..99; destination spans x=200..399, y=0..199.
    A quad at 25%/75% of the source column must land at 25%/75% of the destination one."""
    src = _face([_rect_vial(1, 0, 0, 0, 0, 100, 100)], name="A",
                marker={"vial_spans": [[0, 99]], "row_bands": {"upper": [0, 100]}})
    dst = _face([_rect_vial(1, 0, 0, 200, 0, 200, 200)], name="B",
                marker={"vial_spans": [[200, 399]], "row_bands": {"upper": [0, 200]}})
    src.vials[0].quad = [[25, 25], [75, 25], [75, 75], [25, 75]]

    out = C.transfer_quads(src, dst)
    assert out.name == "B" and out is not dst
    # x: 200 + (25-0) * (399-200)/(99-0) = 200 + 25*2.0101 = 250.25 -> 250
    assert out.vials[0].quad == [[250, 50], [351, 50], [351, 150], [250, 150]]
    assert (out.vials[0].x, out.vials[0].y, out.vials[0].w, out.vials[0].h) == (250, 50, 101, 100)
    assert dst.vials[0].quad is None                     # destination untouched


def test_transfer_quads_of_an_unedited_rectangle_reproduces_the_destinations_own_box():
    """Sanity: transferring the DEFAULT quad must be (near) a no-op on the destination geometry.

    This is what makes the transfer trustworthy -- if it distorted an untouched rectangle it
    would be distorting the edited shapes too."""
    src, dst = _two_col_face("A"), _two_col_face("B", spans=((14, 55), (66, 107)),
                                                 bands=((12, 74), (84, 116)))
    edited = C.apply_quads_to_face(src, C.face_quads(src))   # materialise the auto quads verbatim
    out = C.transfer_quads(edited, dst)
    for before, after in zip(dst.vials, out.vials):
        assert abs(after.x - before.x) <= 1
        assert abs(after.y - before.y) <= 1
        assert abs(after.w - before.w) <= 2
        assert abs(after.h - before.h) <= 2


def test_transfer_quads_preserves_the_destinations_vertical_extent():
    src = _two_col_face("A", bands=((0, 100), (200, 300)))
    dst = _two_col_face("B", bands=((10, 60), (150, 200)))
    edited = C.apply_quads_to_face(src, C.face_quads(src))
    out = C.transfer_quads(edited, dst)
    for v in out.vials:
        band = (10, 60) if v.row == 0 else (150, 200)
        ys = [p[1] for p in v.quad]
        assert min(ys) >= band[0] - 1 and max(ys) <= band[1] + 1


def test_transfer_quads_passes_through_vials_the_source_has_no_quad_for():
    src, dst = _two_col_face("A"), _two_col_face("B")
    out = C.transfer_quads(src, dst)                     # src has no quads at all
    for before, after in zip(dst.vials, out.vials):
        assert after.quad is None
        assert (after.x, after.y, after.w, after.h) == (before.x, before.y, before.w, before.h)


def test_transfer_quads_accepts_explicit_spans():
    src = _face([_rect_vial(1, 0, 0, 0, 0, 100, 100, quad=[[0, 0], [100, 0], [100, 100], [0, 100]])])
    dst = _face([_rect_vial(1, 0, 0, 0, 0, 100, 100)], name="B")
    out = C.transfer_quads(src, dst, [(0, 100)], [(500, 600)],
                           src_bands={0: (0, 100)}, dst_bands={0: (0, 100)})
    assert out.vials[0].quad == [[500, 0], [600, 0], [600, 100], [500, 100]]


def test_transfer_quads_clips_to_the_image_when_asked():
    src = _face([_rect_vial(1, 0, 0, 0, 0, 100, 100, quad=[[0, 0], [100, 0], [100, 100], [0, 100]])])
    dst = _face([_rect_vial(1, 0, 0, 0, 0, 100, 100)], name="B")
    out = C.transfer_quads(src, dst, [(0, 100)], [(150, 350)],
                           src_bands={0: (0, 100)}, dst_bands={0: (0, 100)}, image_size=(200, 200))
    assert max(p[0] for p in out.vials[0].quad) == 200


def test_face_column_spans_and_row_bands_fall_back_to_the_vial_boxes():
    face = _two_col_face()
    face.marker = None                                   # no marker geometry at all
    assert C.face_column_spans(face) == [(10, 49), (60, 99)]
    assert C.face_row_bands(face) == {0: (10, 70), 1: (80, 140)}


# =========================================================================================
# 9. THE MEASUREMENT PATH: the pipeline honours the polygon
# =========================================================================================
def _pipeline_submask(tmp_path, vial, mask):
    """Build a one-vial pipeline over `mask` and return its effective per-vial bool submask."""
    from flygym_tracker.config import load_config
    from flygym_tracker.pipeline import TrackerPipeline

    mask_path = str(tmp_path / ("illum_%s.png" % vial.id))
    cv2.imwrite(mask_path, mask)
    calib = Calibration(image_width=mask.shape[1], image_height=mask.shape[0],
                        faces={"A": _face([vial], mask_path=mask_path)})
    config = load_config(overrides={
        "activity": {"pixel_threshold": 10.0},
        "rotation": {"enter_threshold": 40.0, "exit_threshold": 15.0},
    })

    class _NullSource:
        fps = 10.0

        def open(self): pass

        def read(self): return None

        def close(self): pass

    class _NullLogger:
        run_id = "t"

        def log_activity(self, records): pass

        def log_event(self, record): pass

        def close(self): pass

    pipe = TrackerPipeline(config, calib, _NullSource(), _NullLogger())
    (_bbox, submask) = pipe._face_active["A"][int(vial.id)]   # single face -> gvid == local id
    return submask


def test_pipeline_mask_is_illum_intersect_polygon(tmp_path):
    """A quad that EXCLUDES part of the illuminated mask must measure fewer pixels than the bbox.

    The quad here is a TRAPEZOID with the same bounding box as the rectangle, i.e. the pixels it
    drops are dropped by the polygon and by nothing else -- which is the whole claim.
    """
    mask = np.zeros((H, W), np.uint8)
    mask[10:70, 10:90] = 255                     # 80 x 60 lit block

    box_only = _pipeline_submask(tmp_path, _rect_vial(1, x=10, y=10, w=80, h=60), mask)
    trapezoid = [[10, 10], [90, 10], [50, 70], [10, 70]]      # right edge tapers in, as an edge vial does
    with_quad = _pipeline_submask(tmp_path, _rect_vial(2, x=10, y=10, w=80, h=60, quad=trapezoid), mask)

    lit_box, lit_quad = int(box_only.sum()), int(with_quad.sum())
    assert box_only.shape == with_quad.shape == (60, 80)      # same bbox: only the polygon differs
    assert lit_box == 80 * 60 == 4800
    # Analytic trapezoid area = (80 + 40)/2 * 60 = 3600; the inclusive raster is 3649 px (76% of
    # the box). Pinned exactly, because this number IS the measurement.
    assert C.polygon_area(trapezoid) == pytest.approx(3600.0)
    assert lit_quad == 3649
    assert lit_quad < lit_box
    # The dropped pixels are exactly the ones outside the polygon: the bottom-right corner is out,
    # the bottom-left is in.
    assert not with_quad[55, 70]
    assert with_quad[55, 20]


def test_a_quad_derived_from_a_bbox_measures_exactly_the_bbox(tmp_path):
    """THE backward-compatibility invariant: `quad_from_bbox` must be a measurement no-op.

    Every vial that has never been hand-edited gets its bbox as a rectangular quad (in the editor,
    and in `transfer_quads`' pass-through), so if that quad measured even one pixel differently
    from the plain bbox, opening and saving the editor without touching anything would change the
    experiment's numbers.
    """
    mask = np.zeros((H, W), np.uint8)
    mask[10:70, 10:90] = 255
    mask[30:40, 40:50] = 0                                   # a hole, so this is not a trivial all-ones case
    box = (10, 10, 80, 60)
    plain = _pipeline_submask(tmp_path, _rect_vial(1, x=10, y=10, w=80, h=60), mask)
    quadded = _pipeline_submask(
        tmp_path, _rect_vial(2, x=10, y=10, w=80, h=60, quad=C.quad_from_bbox(box)), mask)
    assert np.array_equal(plain, quadded)


def test_pipeline_mask_without_a_quad_is_unchanged(tmp_path):
    """quad=None must give byte-identical behaviour to the pre-quad pipeline."""
    mask = np.zeros((H, W), np.uint8)
    mask[10:70, 10:90] = 255
    mask[30:40, 40:50] = 0                        # a hole, to prove the illum mask still governs
    sub = _pipeline_submask(tmp_path, _rect_vial(1, x=10, y=10, w=80, h=60), mask)
    assert np.array_equal(sub, mask[10:70, 10:90] == 255)


def test_pipeline_quad_and_illum_are_ANDed_not_substituted(tmp_path):
    """The polygon must not resurrect pixels the illumination mask excluded."""
    mask = np.zeros((H, W), np.uint8)
    mask[10:70, 10:50] = 255                      # lit only on the LEFT half of the bbox
    quad = [[10, 10], [90, 10], [90, 70], [10, 70]]   # quad covers the whole bbox
    sub = _pipeline_submask(tmp_path, _rect_vial(1, x=10, y=10, w=80, h=60, quad=quad), mask)
    assert int(sub.sum()) == 40 * 60
    assert not sub[:, 40:].any()


def test_pipeline_registration_shifts_the_quad_with_the_bbox(tmp_path):
    from flygym_tracker.config import load_config
    from flygym_tracker.pipeline import TrackerPipeline

    mask = np.full((H, W), 255, np.uint8)
    mask_path = str(tmp_path / "illum_shift.png")
    cv2.imwrite(mask_path, mask)
    quad = [[10, 10], [50, 10], [50, 70], [10, 70]]
    vial = _rect_vial(1, x=10, y=10, w=40, h=60, quad=quad)
    calib = Calibration(image_width=W, image_height=H, faces={"A": _face([vial], mask_path=mask_path)})
    config = load_config(overrides={"activity": {"pixel_threshold": 10.0},
                                    "rotation": {"enter_threshold": 40.0, "exit_threshold": 15.0}})

    class _S:
        fps = 10.0

        def open(self): pass

        def read(self): return None

        def close(self): pass

    class _L:
        run_id = "t"

        def log_activity(self, r): pass

        def log_event(self, r): pass

        def close(self): pass

    pipe = TrackerPipeline(config, calib, _S(), _L())
    before = int(pipe._face_active["A"][1][1].sum())
    pipe._apply_registration("A", 7.0, 3.0)
    bbox, sub = pipe._face_active["A"][1]
    assert bbox == (17, 13, 40, 60)                       # bbox_from_quad(quad) shifted by (7, 3)
    assert int(sub.sum()) == before                       # same polygon, just moved


# =========================================================================================
# 10. Keyboard + view transform (the rest of the editor's pure surface)
# =========================================================================================
@pytest.mark.parametrize("code,name", [
    (2424832, "left"), (2490368, "up"), (2555904, "right"), (2621440, "down"),   # Windows
    (65361, "left"), (65364, "down"),                                            # Linux
    (ord("s"), "s"), (ord("q"), "q"), (9, "tab"), (27, "esc"), (13, "enter"),
    (26, "ctrl+z"), (25, "ctrl+y"), (-1, None),
])
def test_decode_key(code, name):
    assert decode_key(code) == name


def test_arrow_keys_nudge_the_roi_then_the_selected_vertex():
    st = _four_vial_state()
    view = ViewTransform()
    before = [list(p) for p in st.quad(0)]

    handle_key(st, view, "right")
    assert st.quad(0) == [[p[0] + 1, p[1]] for p in before]

    st.select(0, vertex=2)
    handle_key(st, view, "down")
    q = st.quad(0)
    assert q[2] == [before[2][0] + 1, before[2][1] + 1]
    assert q[0] == [before[0][0] + 1, before[0][1]]


def test_handle_key_commands_and_navigation():
    st = _four_vial_state()
    view = ViewTransform()
    assert handle_key(st, view, "s") == "save"
    assert handle_key(st, view, "q") == "quit"
    assert handle_key(st, view, "esc") == "quit"
    assert handle_key(st, view, "h") == "toggle_help"
    assert handle_key(st, view, "m") == "toggle_magnifier"
    assert handle_key(st, view, "l") == "toggle_mask"
    assert handle_key(st, view, None) is None

    handle_key(st, view, "tab"); assert st.selected == 1
    handle_key(st, view, "n"); assert st.selected == 2
    handle_key(st, view, "p"); assert st.selected == 1


def test_handle_key_undo_redo_and_copy():
    st = _four_vial_state()
    view = ViewTransform()
    before = [[list(p) for p in q] for q in st.quads]
    handle_key(st, view, "right")
    handle_key(st, view, "c")
    handle_key(st, view, "z")
    handle_key(st, view, "z")
    assert st.quads == before
    handle_key(st, view, "y")
    assert st.quads != before


def test_handle_key_zoom_keys_change_the_view():
    st = _four_vial_state()
    view = ViewTransform(zoom=1.0)
    handle_key(st, view, "+")
    assert view.zoom == pytest.approx(1.25)
    handle_key(st, view, "-")
    assert view.zoom == pytest.approx(1.0)
    handle_key(st, view, "0")
    assert view.zoom == pytest.approx(ViewTransform.fit((W, H), (1500, 860)).zoom)


def test_view_transform_round_trips_and_zooms_about_the_cursor():
    view = ViewTransform(zoom=2.0, ox=30.0, oy=10.0)
    assert view.to_screen((30, 10)) == (0.0, 0.0)
    assert view.to_image(view.to_screen((77, 55))) == pytest.approx((77.0, 55.0))

    anchor = (120.0, 80.0)
    fixed = view.to_image(anchor)
    view.zoom_by(1.5, anchor)
    assert view.zoom == pytest.approx(3.0)
    assert view.to_image(anchor) == pytest.approx(fixed)


def test_view_transform_fit_shows_the_whole_image():
    view = ViewTransform.fit((1280, 1024), (640, 512))
    assert view.zoom == pytest.approx(0.5)
    assert view.to_screen((0, 0)) == pytest.approx((0.0, 0.0))
    assert view.to_screen((1280, 1024)) == pytest.approx((640.0, 512.0))


def test_view_pan_moves_the_image_with_the_cursor():
    view = ViewTransform(zoom=2.0)
    view.pan_by(20, -10)
    assert (view.ox, view.oy) == pytest.approx((-10.0, 5.0))


def test_drag_controller_pan_and_wheel_only_touch_the_view():
    st = _four_vial_state()
    view = ViewTransform(zoom=1.0)
    drag = DragController(st, view)
    quads = [[list(p) for p in q] for q in st.quads]

    drag.on_pan_start(100, 100)
    drag.on_pan_move(110, 90)
    drag.on_wheel(1, 50, 50)
    assert view.zoom == pytest.approx(1.25)
    assert st.quads == quads
    assert st.dirty is False


def test_drag_controller_selects_and_moves_a_whole_roi():
    st = _four_vial_state()
    drag = DragController(st, ViewTransform())
    st.select(3)
    cx, cy = np.mean(np.asarray(st.quad(1)), axis=0)
    hit = drag.on_press(cx, cy)
    assert hit == Hit("roi", 1, None) and st.selected == 1
    drag.on_move(cx + 6, cy + 2)
    drag.on_release()
    assert st.quad(1) == [[p[0] + 6, p[1] + 2] for p in st.base_quads[1]]


def test_drag_controller_grab_offset_prevents_a_jump():
    """Grabbing a vertex 3 px off-centre must not teleport it under the cursor."""
    st = _four_vial_state()
    drag = DragController(st, ViewTransform())
    vx, vy = st.quad(0)[0]
    drag.on_press(vx + 3, vy + 3)
    assert st.quad(0)[0] == [vx, vy]          # untouched by the press itself
    drag.on_move(vx + 13, vy + 3)
    assert st.quad(0)[0] == [vx + 10, vy]


def test_drag_on_empty_space_starts_no_edit():
    st = _four_vial_state()
    drag = DragController(st, ViewTransform())
    quads = [[list(p) for p in q] for q in st.quads]
    assert drag.on_press(55, 190) is None
    drag.on_move(70, 195)
    drag.on_release()
    assert st.quads == quads and st.dirty is False


# =========================================================================================
# 11. Rendering is headless-safe (draws into an array, opens nothing)
# =========================================================================================
def test_render_view_returns_an_image_of_the_requested_size():
    mask = _half_lit_mask()
    st = EditorState(_two_col_face(), mask, image_size=(W, H))
    base = tinted_frame(np.full((H, W), 120, np.uint8), mask)
    out = render_view(base, st, ViewTransform.fit((W, H), (400, 300)), (400, 300),
                      hover=Hit("vertex", 0, 1), cursor_image=(40.0, 40.0),
                      show_help=True, show_magnifier=True)
    assert out.shape == (300, 400, 3) and out.dtype == np.uint8
    assert out.any()


def test_tinted_frame_marks_the_lit_pixels_only():
    mask = _half_lit_mask()
    vis = tinted_frame(np.zeros((H, W), np.uint8), mask)
    assert tuple(vis[0, 20]) == (0, 55, 0)
    assert tuple(vis[0, 0]) == (0, 0, 0)


# =========================================================================================
# 12. The DRIVER loop itself, with the cv2 window stubbed out (still no display needed)
# =========================================================================================
class _FakeWindow:
    """Stands in for the whole cv2 highgui surface `run_roi_editor` touches.

    Lets the real driver loop run headlessly: it plays a scripted key sequence, records the
    frames it was asked to show, and hands back the mouse callback so synthetic drags can be
    fired at it. If the loop ever grew logic of its own, this is what would catch it.
    """

    def __init__(self, monkeypatch, keys):
        # An idle frame (waitKeyEx -> -1) before every keystroke, which is what a real loop sees
        # ~60x a second; anything that mistakes "no key" for a key shows up here.
        self.keys = [k for key in keys for k in (-1, key)]
        self.shown = []
        self.callback = None
        self.destroyed = []
        for name, fn in [
            ("namedWindow", lambda *a, **k: None),
            ("setMouseCallback", lambda _w, cb, *a: setattr(self, "callback", cb)),
            ("imshow", lambda _w, img: self.shown.append(img)),
            ("waitKeyEx", lambda *_a: self.keys.pop(0) if self.keys else ord("q")),
            ("waitKey", lambda *_a: -1),
            ("getWindowProperty", lambda *_a: 1.0),
            ("destroyWindow", lambda w: self.destroyed.append(w)),
        ]:
            monkeypatch.setattr(cv2, name, fn)


def test_run_roi_editor_driver_loop_applies_keys_and_saves(monkeypatch):
    from flygym_tracker.roi_editor import run_roi_editor

    face = _two_col_face()
    mask = _half_lit_mask()
    frame = np.full((H, W), 100, np.uint8)
    keys = [9,                      # tab -> vial 1
            2555904, 2555904,       # right, right -> nudge whole ROI +2 px
            ord("v"),               # select vertex TL
            2621440,                # down -> nudge that vertex only
            ord("h"), ord("m"), ord("l"),   # toggle help / magnifier / mask tint
            ord("s")]               # save
    win = _FakeWindow(monkeypatch, keys)

    out = run_roi_editor(frame, face, mask)

    assert out is not None and out.name == "A"
    assert win.destroyed == ["FlyGym ROI editor"]
    assert len(win.shown) >= len(keys)
    assert all(img.shape == (860, 1500, 3) for img in win.shown)

    base = C.quad_from_bbox((face.vials[1].x, face.vials[1].y, face.vials[1].w, face.vials[1].h))
    expected = [[p[0] + 2, p[1]] for p in base]
    expected[0] = [expected[0][0], expected[0][1] + 1]
    assert out.vials[1].quad == expected
    assert out.vials[0].quad == C.quad_from_bbox(
        (face.vials[0].x, face.vials[0].y, face.vials[0].w, face.vials[0].h))
    assert (out.vials[1].x, out.vials[1].y, out.vials[1].w, out.vials[1].h) == \
        C.bbox_from_quad(expected)


def test_run_roi_editor_quit_returns_none(monkeypatch):
    from flygym_tracker.roi_editor import run_roi_editor
    _FakeWindow(monkeypatch, [ord("q")])                 # nothing edited -> quits immediately
    assert run_roi_editor(np.zeros((H, W), np.uint8), _two_col_face()) is None


def test_run_roi_editor_quit_after_an_edit_needs_confirming(monkeypatch):
    """One stray `q` must not throw away hand-placed vertices."""
    from flygym_tracker.roi_editor import run_roi_editor

    face = _two_col_face()
    base = C.quad_from_bbox((face.vials[0].x, face.vials[0].y, face.vials[0].w, face.vials[0].h))

    # nudge right, q (only ARMS the quit), s -> the edit is saved, not discarded
    _FakeWindow(monkeypatch, [2555904, ord("q"), ord("s")])
    out = run_roi_editor(np.zeros((H, W), np.uint8), _two_col_face())
    assert out is not None
    assert out.vials[0].quad == [[p[0] + 1, p[1]] for p in base]

    # nudge right, q, q -> confirmed, edits discarded
    _FakeWindow(monkeypatch, [2555904, ord("q"), ord("q")])
    assert run_roi_editor(np.zeros((H, W), np.uint8), _two_col_face()) is None


def test_a_key_between_two_quits_disarms_the_confirmation(monkeypatch):
    from flygym_tracker.roi_editor import run_roi_editor
    # edit, q (arms), tab (disarms), q (re-arms), s -> saved
    _FakeWindow(monkeypatch, [2555904, ord("q"), 9, ord("q"), ord("s")])
    assert run_roi_editor(np.zeros((H, W), np.uint8), _two_col_face()) is not None


def test_run_roi_editor_mouse_callback_drags_a_vertex(monkeypatch):
    """Fire synthetic cv2 mouse events at the registered callback and check the geometry."""
    from flygym_tracker.roi_editor import run_roi_editor

    face = _two_col_face()
    win = _FakeWindow(monkeypatch, [])
    # The callback is only live once the driver has registered it, so fire the drag from inside
    # the waitKeyEx stub -- i.e. from exactly where a real event would arrive.
    seq = []

    def waitkey(*_a):
        cb = win.callback
        if not seq:
            vx, vy = C.quad_from_bbox((face.vials[0].x, face.vials[0].y,
                                       face.vials[0].w, face.vials[0].h))[1]
            # view is fitted, so screen != image; go through the same transform the driver uses
            v = ViewTransform.fit((W, H), (1500, 860))
            sx, sy = v.to_screen((vx, vy))
            cb(cv2.EVENT_LBUTTONDOWN, sx, sy, 0, None)
            cb(cv2.EVENT_MOUSEMOVE, *v.to_screen((vx - 12, vy + 4)), 0, None)
            cb(cv2.EVENT_LBUTTONUP, *v.to_screen((vx - 12, vy + 4)), 0, None)
            seq.append(1)
            return ord("s")
        return ord("q")

    monkeypatch.setattr(cv2, "waitKeyEx", waitkey)
    out = run_roi_editor(np.zeros((H, W), np.uint8), face)
    assert out is not None
    tr = out.vials[0].quad[1]
    base_tr = C.quad_from_bbox((face.vials[0].x, face.vials[0].y,
                                face.vials[0].w, face.vials[0].h))[1]
    assert tr[0] == pytest.approx(base_tr[0] - 12, abs=1)
    assert tr[1] == pytest.approx(base_tr[1] + 4, abs=1)


# =========================================================================================
# 13. CLI `edit-rois` end to end (the interactive editor itself is stubbed out)
# =========================================================================================
def _two_face_bundle(tmp_path):
    """A minimal but structurally real 2-face bundle on disk, plus its two masks."""
    # Column 0 of the upper row is only partly lit on BOTH faces -- the left ~60% of the slot --
    # which is the synthetic stand-in for the rig's foreshortened edge vials.
    mask_a = np.zeros((H, W), np.uint8)
    mask_a[10:70, 10:35] = 255
    mask_a[10:70, 60:100] = 255
    mask_a[80:140, 10:50] = 255
    mask_a[80:140, 60:100] = 255
    mask_b = np.zeros((H, W), np.uint8)
    mask_b[12:72, 14:39] = 255
    mask_b[12:72, 64:104] = 255
    mask_b[82:142, 14:54] = 255
    mask_b[82:142, 64:104] = 255

    calib = Calibration(image_width=W, image_height=H, faces={
        "A": _two_col_face("A"),
        "B": _two_col_face("B", spans=((14, 53), (64, 103)), bands=((12, 72), (82, 142))),
    })
    calib.faces["A"].illum_mask_path = "illum_mask_A.png"
    calib.faces["B"].illum_mask_path = "illum_mask_B.png"
    d = tmp_path / "bundle"
    d.mkdir()
    cv2.imwrite(str(d / "illum_mask_A.png"), mask_a)
    cv2.imwrite(str(d / "illum_mask_B.png"), mask_b)
    cv2.imwrite(str(d / "overlay_A.png"), np.full((H, W), 90, np.uint8))
    calib.to_json(str(d / "calibration.json"))
    return str(d)


def _stub_editor(monkeypatch, edit):
    """Replace the interactive editor with `edit(face_cal) -> FaceCalibration | None`."""
    import flygym_tracker.roi_editor as R
    calls = {}

    def fake(frame_gray, face_cal, illum_mask=None, **kwargs):
        calls["frame"] = frame_gray
        calls["mask"] = illum_mask
        calls["face"] = face_cal.name
        return edit(face_cal)

    monkeypatch.setattr(R, "run_roi_editor", fake)
    return calls


def _tighten(face_cal):
    """Stand-in for an operator: shrink every quad to the left 60% of its own box."""
    quads = []
    for q in C.face_quads(face_cal):
        x, y, w, h = C.bbox_from_quad(q)
        quads.append([[x, y], [int(x + 0.6 * w), y], [int(x + 0.6 * w), y + h], [x, y + h]])
    return C.apply_quads_to_face(face_cal, quads)


def test_cli_edit_rois_saves_and_transfers_to_the_other_face(tmp_path, monkeypatch, capsys):
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    calls = _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle]) == 0

    assert calls["face"] == "A"
    assert calls["mask"] is not None                      # the editor got the illum mask
    out = C.load_calibration(bundle)
    assert all(v.quad is not None for v in out.faces["A"].vials)
    assert all(v.quad is not None for v in out.faces["B"].vials)   # transferred
    for fc in out.faces.values():
        for v in fc.vials:
            assert (v.x, v.y, v.w, v.h) == C.bbox_from_quad(v.quad)
    assert out.faces["B"].marker["quad_source"]["face"] == "A"
    assert out.faces["B"].marker["quad_source"]["n_transferred"] == 4

    text = capsys.readouterr().out
    assert "lit before -> after" in text
    assert "transferred to face(s) B" in text


def test_cli_edit_rois_keeps_mask_paths_relative(tmp_path, monkeypatch):
    """A re-saved bundle must stay movable -- no absolute paths baked into the JSON."""
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle]) == 0
    with open(os.path.join(bundle, "calibration.json"), encoding="utf-8") as f:
        raw = json.load(f)
    assert sorted(fc["illum_mask_path"] for fc in raw["faces"].values()) == [
        "illum_mask_A.png", "illum_mask_B.png"]


def test_cli_edit_rois_no_transfer_leaves_the_other_face_alone(tmp_path, monkeypatch):
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle, "--no-transfer"]) == 0
    out = C.load_calibration(bundle)
    assert all(v.quad is not None for v in out.faces["A"].vials)
    assert all(v.quad is None for v in out.faces["B"].vials)


def test_cli_edit_rois_cancel_writes_nothing(tmp_path, monkeypatch):
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    path = os.path.join(bundle, "calibration.json")
    with open(path, encoding="utf-8") as f:
        before = f.read()
    _stub_editor(monkeypatch, lambda face_cal: None)       # operator pressed 'q'
    assert main(["edit-rois", "--calib", bundle]) == 0
    with open(path, encoding="utf-8") as f:
        assert f.read() == before


def test_cli_edit_rois_uses_the_frame_argument(tmp_path, monkeypatch, capsys):
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    frame_path = str(tmp_path / "still.png")
    cv2.imwrite(frame_path, np.full((H, W), 33, np.uint8))
    calls = _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle, "--frame", frame_path]) == 0
    assert calls["frame"].shape == (H, W)
    assert int(calls["frame"][0, 0]) == 33            # the given still, not the overlay fallback
    assert frame_path in capsys.readouterr().out


def test_cli_edit_rois_reports_an_unreadable_frame(tmp_path, monkeypatch, capsys):
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle, "--frame", str(tmp_path / "nope.png")]) == 1
    assert "could not read" in capsys.readouterr().err


def test_cli_edit_rois_reports_a_bad_face(tmp_path, monkeypatch, capsys):
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle, "--face", "Z"]) == 1
    assert "not in calibration" in capsys.readouterr().err


def test_cli_edit_rois_improves_the_lit_fraction_end_to_end(tmp_path, monkeypatch):
    """The measurable claim: editing raises coverage on the partly-lit column of BOTH faces."""
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    mask_a = cv2.imread(os.path.join(bundle, "illum_mask_A.png"), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(os.path.join(bundle, "illum_mask_B.png"), cv2.IMREAD_GRAYSCALE)
    before = C.load_calibration(bundle)
    lit_a0 = C.quad_lit_fraction(C.vial_quad(before.faces["A"].vials[0]), mask_a)
    lit_b0 = C.quad_lit_fraction(C.vial_quad(before.faces["B"].vials[0]), mask_b)

    _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle]) == 0

    after = C.load_calibration(bundle)
    lit_a1 = C.quad_lit_fraction(after.faces["A"].vials[0].quad, mask_a)
    lit_b1 = C.quad_lit_fraction(after.faces["B"].vials[0].quad, mask_b)
    assert lit_a0 == pytest.approx(0.625, abs=0.02)
    assert lit_a1 > lit_a0 + 0.3
    assert lit_b1 > lit_b0 + 0.2        # the improvement carried across to the untouched face


def test_cli_edit_rois_regenerates_the_face_overlay(tmp_path, monkeypatch):
    from flygym_tracker.cli import main

    bundle = _two_face_bundle(tmp_path)
    path = os.path.join(bundle, "overlay_A.png")
    before = cv2.imread(path)
    _stub_editor(monkeypatch, _tighten)
    assert main(["edit-rois", "--calib", bundle]) == 0
    after = cv2.imread(path)
    assert after is not None and after.shape == before.shape
    assert not np.array_equal(after, before)
