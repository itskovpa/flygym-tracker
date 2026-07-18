"""Load and validate YAML configuration for flygym_tracker.

See DESIGN.md section 4 (`config.py`) and `config/default_config.yaml` for the
schema this loads. Do not fork the shared dataclasses in `types.py` here; this
module only deals with the (separate) tunable-parameters config tree.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

#: Repo root, resolved from this file's own location:
#: <repo>/src/flygym_tracker/config.py -> parents[2] == <repo>.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"

_VALID_SOURCE_TYPES = ("camera", "video")
_VALID_OUTPUT_FORMATS = ("csv", "xlsx", "both")


class Config:
    """Read-only nested-dict wrapper with both attribute and dict-style access.

    ``config.binning.bin_seconds`` and ``config["binning"]["bin_seconds"]`` are
    equivalent. Nested dicts are wrapped in `Config` lazily on access; any other
    value (numbers, strings, bools, lists, None) is returned as-is.
    """

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        # Only invoked when normal attribute lookup fails (e.g. `_data` itself
        # is a real instance attribute and never goes through here).
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"{name!r} not found in config (have: {sorted(self._data.keys())})"
            ) from None
        return _wrap(value)

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return _wrap(self._data[key]) if key in self._data else default

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((k, _wrap(v)) for k, v in self._data.items())

    def to_dict(self) -> dict:
        """Recursively unwrap back to a plain dict (e.g. for a run_meta.json snapshot)."""
        return copy.deepcopy(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        return NotImplemented

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _wrap(value: Any) -> Any:
    return Config(value) if isinstance(value, dict) else value


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base`.

    Nested dicts are merged key-by-key; any other value (including lists)
    replaces the base value outright. Returns a new dict; neither input is
    mutated.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate(data: dict) -> None:
    binning = data.get("binning") or {}
    bin_seconds = binning.get("bin_seconds")
    if not isinstance(bin_seconds, (int, float)) or isinstance(bin_seconds, bool) or bin_seconds <= 0:
        raise ValueError(f"binning.bin_seconds must be > 0, got {bin_seconds!r}")

    activity = data.get("activity") or {}
    k = activity.get("k")
    if not isinstance(k, (int, float)) or isinstance(k, bool) or k < 0:
        raise ValueError(f"activity.k must be >= 0, got {k!r}")

    rotation = data.get("rotation") or {}
    debounce_frames = rotation.get("debounce_frames")
    if not isinstance(debounce_frames, int) or isinstance(debounce_frames, bool) or debounce_frames < 1:
        raise ValueError(f"rotation.debounce_frames must be >= 1, got {debounce_frames!r}")

    source = data.get("source") or {}
    source_type = source.get("type")
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(f"source.type must be one of {_VALID_SOURCE_TYPES}, got {source_type!r}")

    output = data.get("output") or {}
    output_format = output.get("format")
    if output_format not in _VALID_OUTPUT_FORMATS:
        raise ValueError(f"output.format must be one of {_VALID_OUTPUT_FORMATS}, got {output_format!r}")


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """Load the tracker config.

    Base layer is always `config/default_config.yaml` (resolved relative to the
    repo root, independent of CWD). If `path` is given, that YAML file is
    deep-merged on top. If `overrides` is given, it is deep-merged on top of
    that. The merged result is validated and returned as a `Config`.

    Raises:
        FileNotFoundError: `path` was given but doesn't exist.
        ValueError: the merged config fails validation.
    """
    merged = _load_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        merged = _deep_merge(merged, _load_yaml(path))
    if overrides:
        merged = _deep_merge(merged, overrides)
    _validate(merged)
    return Config(merged)
