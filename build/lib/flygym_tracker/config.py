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

from flygym_tracker import paths

#: Where the SHIPPED templates are read from. The repo root when running from a clone; the install
#: directory when running from a build -- see `paths.bundle_root`, which is the one place that
#: knows the difference. Read-only either way: nothing this program saves goes here.
REPO_ROOT = paths.bundle_root()
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"

#: The rig's shipped config. A TEMPLATE: it is in version control, it is what a fresh clone gets,
#: and it deliberately leaves every camera field null so an untouched install imposes nothing on
#: the sensor.
RIG_CONFIG_PATH = REPO_ROOT / "config" / "flygym_rig.yaml"

#: Suffix marking a machine's own copy of a config. `flygym_rig.local.yaml` sits on top of
#: `flygym_rig.yaml`; it is gitignored and it is what the app writes to.
LOCAL_SUFFIX = ".local.yaml"

#: Files that are SHIPPED and must not be written to by the app. See `local_config_path`.
TRACKED_TEMPLATES = (DEFAULT_CONFIG_PATH, RIG_CONFIG_PATH)

_LOCAL_HEADER = """\
# THIS RIG'S OWN SETTINGS. Written by the app; safe to hand-edit; NOT in version control.
#
# It is a layer ON TOP OF config/flygym_rig.yaml, not a copy of it: only the values that differ
# from the shipped template are here, and everything else -- including any improvement that
# arrives with a new version of the template -- still comes through underneath. Delete this file
# to go back to the shipped values.
#
# WHY THE APP DOES NOT WRITE TO flygym_rig.yaml. That file is the template a fresh clone gets, and
# it is asserted to leave every camera field null so an untouched install imposes nothing on the
# sensor. Saving tuned values into it made the shipped default "whatever the last operator was
# trying", and the difference was invisible in a diff nobody reads before an experiment.
"""


def is_tracked_template(path) -> bool:
    """True if `path` is one of the shipped, version-controlled config files."""
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return False
    return any(resolved == template.resolve() for template in TRACKED_TEMPLATES)


def local_config_path(path=None) -> Path:
    """The machine-local sibling of `path`: ``config/x.yaml`` -> ``config/x.local.yaml``.

    A path that is ALREADY a local file is returned unchanged, so this is safe to apply twice --
    which matters, because it is called both when choosing where to save and when deciding whether
    a chosen file needs redirecting.
    """
    base = Path(path) if path else RIG_CONFIG_PATH
    if str(base).endswith(LOCAL_SUFFIX):
        return base
    local = base.with_name(base.name[:-len(base.suffix)] + LOCAL_SUFFIX)
    if paths.is_frozen() and _inside_bundle(local):
        # AN INSTALLED COPY CANNOT WRITE BESIDE ITS OWN TEMPLATE. The template lives in
        # `C:\Program Files\...`, and the machine's own settings must not. They move to the user's
        # data folder and `template_for_local` finds its way back -- see there.
        return paths.user_data_root() / "config" / local.name
    return local


def _inside_bundle(path) -> bool:
    try:
        Path(path).resolve().relative_to(paths.bundle_root().resolve())
        return True
    except (ValueError, OSError):
        return False


def template_for_local(path) -> Optional[Path]:
    """The shipped file a `*.local.yaml` layers on top of, if it exists. Else None.

    LOOKS BESIDE IT FIRST, THEN IN THE INSTALL. From a clone the two sit in the same folder and the
    first answer is right. Installed, the operator's `flygym_rig.local.yaml` is under their profile
    while the `flygym_rig.yaml` it layers on top of is in Program Files -- and without the second
    lookup the local file would be read as the WHOLE config rather than as an override, silently
    dropping every value the template supplies.
    """
    text = str(path)
    if not text.endswith(LOCAL_SUFFIX):
        return None
    template = Path(text[:-len(LOCAL_SUFFIX)] + ".yaml")
    if template.exists():
        return template
    shipped = REPO_ROOT / "config" / template.name
    return shipped if shipped.exists() else None


def config_layers(path=None) -> list:
    """Every file that contributes to a config, base first. THE PROVENANCE OF A MEASUREMENT.

    ``config/flygym_rig.local.yaml`` resolves to
    ``[default_config.yaml, flygym_rig.yaml, flygym_rig.local.yaml]`` -- the packaged defaults, the
    shipped rig template, and this machine's own values, in the order they are merged.
    """
    layers = [DEFAULT_CONFIG_PATH]
    if path:
        template = template_for_local(path)
        if template is not None:
            layers.append(template)
        layers.append(Path(path))
    return layers


def ensure_local_config(path=None) -> str:
    """Make sure this machine's own config exists, and return its path.

    Creates it EMPTY (a header comment and nothing else) rather than as a copy of the template:
    an empty override layer means the rig behaves exactly like a fresh clone until somebody
    changes something, and every value still carries the template's own comment explaining it.

    Never raises. A read-only install directory is a real deployment -- a shared rig, a locked-down
    lab machine -- and being unable to create a settings file is not a reason to refuse to start;
    the caller falls back to the template, which is read-only but perfectly usable.
    """
    local = local_config_path(path)
    if local.exists():
        return str(local)
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "w", encoding="utf-8") as f:
            f.write(_LOCAL_HEADER)
    except OSError:
        template = template_for_local(local)
        return str(template if template is not None else DEFAULT_CONFIG_PATH)
    return str(local)

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

    ONE EXTRA LAYER, AND IT IS THE RULE THIS FILE EXISTS TO STATE: a path ending in
    ``.local.yaml`` is loaded ON TOP OF the ``.yaml`` of the same name, if that exists. So
    ``config/flygym_rig.local.yaml`` means "the shipped rig template, plus whatever this machine
    changed" rather than "these values and nothing else".

    That is what lets the app write to a file of its own without the rig losing the template's
    values -- and without the shipped template accumulating one operator's tuning as though it
    were the default everybody gets. `config_layers` reports the resulting chain.

    Raises:
        FileNotFoundError: `path` was given but doesn't exist.
        ValueError: the merged config fails validation.
    """
    merged = _load_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        template = template_for_local(path)
        if template is not None:
            merged = _deep_merge(merged, _load_yaml(template))
        merged = _deep_merge(merged, _load_yaml(path))
    if overrides:
        merged = _deep_merge(merged, overrides)
    _validate(merged)
    return Config(merged)
