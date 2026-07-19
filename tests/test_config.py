"""Tests for flygym_tracker.config — defaults, merge, validation.

Run: python -m pytest tests/test_config.py -q
"""
import pytest

from flygym_tracker.config import Config, load_config


# ---- defaults --------------------------------------------------------------

def test_defaults_load():
    cfg = load_config()
    assert isinstance(cfg, Config)
    # attribute access, nested
    assert cfg.binning.bin_seconds == 60
    assert cfg.activity.k == 5.0
    assert cfg.rotation.debounce_frames == 8
    assert cfg.source.type == "camera"
    assert cfg.source.camera.serial == "DA4282883"
    assert cfg.output.format == "both"
    # dict-style access, nested
    assert cfg["binning"]["bin_seconds"] == 60
    assert cfg["source"]["camera"]["pixel_format"] == "Mono8"


def test_every_adjustable_camera_setting_defaults_to_unset():
    """The packaged defaults are the BASE layer every config merges onto, so a number here is a
    value imposed on every rig that did not explicitly override it back to null. These five used to
    hold 1280x1024 @ 20 fps, which meant a change made in MVS was silently reverted on the next
    run and "start from the MVS settings" was not expressible at all."""
    camera = load_config().source.camera
    for key in ("width", "height", "exposure_us", "gain_db", "frame_rate"):
        assert camera.get(key) is None, "%s is forced by the packaged default config" % key
    # serial and pixel_format are NOT optional: one pins which physical camera this is, the other
    # is the format the pipeline decodes.
    assert camera.serial == "DA4282883"
    assert camera.pixel_format == "Mono8"


def test_the_rig_config_leaves_the_camera_alone_too():
    camera = load_config("config/flygym_rig.yaml").source.camera
    for key in ("width", "height", "exposure_us", "gain_db", "frame_rate"):
        assert camera.get(key) is None, "%s is forced by the rig config" % key


def test_defaults_to_dict_roundtrip():
    cfg = load_config()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["binning"]["bin_seconds"] == 60
    # to_dict() is a deep copy, not a live view
    d["binning"]["bin_seconds"] = 999
    assert cfg.binning.bin_seconds == 60


def test_missing_key_raises_attribute_error():
    cfg = load_config()
    with pytest.raises(AttributeError):
        _ = cfg.this_key_does_not_exist


def test_dict_style_get_with_default():
    cfg = load_config()
    assert cfg.get("this_key_does_not_exist", "fallback") == "fallback"
    assert cfg.get("binning").bin_seconds == 60


# ---- merging -----------------------------------------------------------

def test_overrides_dict_merges_without_clobbering_siblings():
    cfg = load_config(overrides={"binning": {"bin_seconds": 30}})
    assert cfg.binning.bin_seconds == 30
    # sibling keys under binning (none here) and other top-level sections
    # must survive an untouched deep-merge
    assert cfg.activity.k == 5.0
    assert cfg.rotation.debounce_frames == 8
    assert cfg.source.camera.serial == "DA4282883"


def test_user_yaml_path_merges_over_defaults(tmp_path):
    user_cfg = tmp_path / "user.yaml"
    user_cfg.write_text(
        "binning:\n"
        "  bin_seconds: 45\n"
        "source:\n"
        "  type: video\n"
        "  video_path: clip.avi\n",
        encoding="utf-8",
    )
    cfg = load_config(path=str(user_cfg))
    assert cfg.binning.bin_seconds == 45
    assert cfg.source.type == "video"
    assert cfg.source.video_path == "clip.avi"
    # untouched nested default under the same top-level "source" section
    assert cfg.source.camera.serial == "DA4282883"
    # untouched sibling top-level section
    assert cfg.activity.k == 5.0


def test_overrides_win_over_user_yaml(tmp_path):
    user_cfg = tmp_path / "user.yaml"
    user_cfg.write_text("binning:\n  bin_seconds: 45\n", encoding="utf-8")
    cfg = load_config(path=str(user_cfg), overrides={"binning": {"bin_seconds": 15}})
    assert cfg.binning.bin_seconds == 15


def test_missing_user_yaml_path_raises():
    with pytest.raises(FileNotFoundError):
        load_config(path="this/path/does/not/exist.yaml")


# ---- validation ----------------------------------------------------------

def test_invalid_bin_seconds_raises():
    with pytest.raises(ValueError):
        load_config(overrides={"binning": {"bin_seconds": 0}})
    with pytest.raises(ValueError):
        load_config(overrides={"binning": {"bin_seconds": -5}})


def test_invalid_activity_k_raises():
    with pytest.raises(ValueError):
        load_config(overrides={"activity": {"k": -1}})


def test_invalid_debounce_frames_raises():
    with pytest.raises(ValueError):
        load_config(overrides={"rotation": {"debounce_frames": 0}})


def test_invalid_source_type_raises():
    with pytest.raises(ValueError):
        load_config(overrides={"source": {"type": "webcam"}})


def test_invalid_output_format_raises():
    with pytest.raises(ValueError):
        load_config(overrides={"output": {"format": "json"}})


def test_valid_boundary_values_do_not_raise():
    # bin_seconds > 0 (smallest positive), k >= 0 (zero allowed), debounce_frames >= 1
    cfg = load_config(
        overrides={
            "binning": {"bin_seconds": 0.001},
            "activity": {"k": 0},
            "rotation": {"debounce_frames": 1},
        }
    )
    assert cfg.binning.bin_seconds == 0.001
    assert cfg.activity.k == 0
    assert cfg.rotation.debounce_frames == 1
