"""The live per-vial readout. The recorded ROWS are graphs and a workbook now, not a table.

THE TABLE IS GONE, at the rig owner's call: it listed every row as it was written -- 32 per bin,
scrolling past faster than anyone reads -- and "it should just go as a graph". The tests that
pinned its contents went with it; the claim they protected (what is on screen is what is in the
file) now belongs to `test_matrix_export`, which compares the workbook against the CSVs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from flygym_tracker.gui.results_panel import ResultsPanel   # noqa: E402


def test_a_vial_the_pipeline_did_not_report_stays_blank_rather_than_reading_zero(qapp):
    """"No reading" and "a reading of zero" are different facts. On this rig the distinction is
    load-bearing: only one drum face is visible at a time, so the other sixteen SHOULD be blank,
    and drawing them as 0 would be the display inventing a measurement.

    THE CELL SHOWS active_fraction (the tuple's 3rd field) AS A PERCENT, not the raw motion pixels
    (1st field), so vials with different-sized ROIs compare -- 0.3 -> "30.0", a real 0.0 -> "0.0"."""
    panel = ResultsPanel()
    panel.set_progress({"vial_results": {1: (250, 900, 0.3), 2: (0, 900, 0.0)}})
    assert panel.grid.cells[0].text() == "30.0"
    assert panel.grid.cells[1].text() == "0.0", "a real zero must still be shown"
    assert panel.grid.cells[5].text() == "", "an unreported vial was drawn as a number"


def test_the_live_grid_reads_one_based_global_vial_ids(qapp):
    """THE BUG THIS PINS, seen on the rig and reported as "it can't measure 15 vials on face A and
    1 on face B at the same time".

    `pipeline` builds global ids as `face_index * 16 + v.id` with v.id running 1..16, so face A is
    1..16 and face B is 17..32 -- 1-BASED. The grid indexes its cells from 0. Reading
    `vial_results[index]` therefore shifted every reading one cell right: A1 was always blank, A2
    showed A1's number, and face A's vial 16 appeared in the cell labelled B1 -- which on screen
    read as both faces being measured at once, which is impossible.
    """
    panel = ResultsPanel()
    # A whole face A, exactly as the pipeline emits it while face A is in front of the camera. A
    # DISTINCT active_fraction per vial (gvid/100) so the placement check can tell one cell from the
    # next: vial 1 -> 0.01 -> "1.0", vial 16 -> 0.16 -> "16.0".
    panel.set_progress({"vial_results": {gvid: (10 * gvid, 900, gvid / 100.0)
                                         for gvid in range(1, 17)}})

    assert panel.grid.cells[0].text() == "1.0", "face A vial 1 is not in the first cell"
    assert panel.grid.cells[15].text() == "16.0", "face A vial 16 is not in the last cell of face A"
    for index in range(16, 32):
        assert panel.grid.cells[index].text() == "", \
            "cell %d (face B) shows a reading while face A is the visible face" % index


def test_face_b_lands_in_the_second_row_of_cells(qapp):
    panel = ResultsPanel()
    panel.set_progress({"vial_results": {gvid: (5, 900, 0.1) for gvid in range(17, 33)}})
    assert all(cell.text() == "" for cell in panel.grid.cells[:16]), "face A showed face B's data"
    assert all(cell.text() == "10.0" for cell in panel.grid.cells[16:]), "face B is not in its row"


def test_the_live_readout_is_labelled_as_not_being_in_the_file(qapp):
    """It changes faster than it can be read and it is not binned. Saying so is what stops it
    being mistaken for the measurement."""
    panel = ResultsPanel()
    panel.set_progress({"vial_results": {1: (5, 900, 0.1)}, "face": "B", "pixel_threshold": 12.0})
    note = panel.live_note.text()
    assert "not in the file" in note
    assert "pixel threshold 12.0" in note, \
        "the threshold that decides what counts as motion is not shown beside the numbers"


def test_completed_bins_are_counted_rather_than_listed(qapp):
    panel = ResultsPanel()
    panel.add_bin({"records": [{"vial_id": 1}, {"vial_id": 2}]})
    assert "2 row(s)" in panel.recorded_note.text()


def test_an_empty_bin_does_not_count_as_one(qapp):
    panel = ResultsPanel()
    panel.add_bin({"records": []})
    assert "no bins finished yet" in panel.recorded_note.text()


def test_clearing_starts_a_new_run_from_nothing(qapp):
    panel = ResultsPanel()
    panel.set_progress({"vial_results": {1: (99, 900, 0.5)}})
    panel.add_bin({"records": [{"vial_id": 1}]})
    panel.clear()
    assert panel.grid.cells[0].text() == ""
    assert "no bins finished yet" in panel.recorded_note.text()
