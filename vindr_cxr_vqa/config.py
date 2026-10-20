"""Config loading with attribute access.

Uses PyYAML when it is installed.  When it is not -- which is the normal case
on a bare CPU machine that only needs to build and score the dataset -- it
falls back to a small parser covering the subset of YAML the shipped configs
actually use: nested mappings by indentation, scalars, inline ``[a, b]`` lists
and ``#`` comments.  That keeps ``build_dataset`` and ``evaluate`` free of any
third-party dependency.
"""

from __future__ import annotations

import os
import re
from typing import Any


class Config(dict):
    """A dict whose keys are also attributes, nested all the way down."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"no config key {name!r}; available: {sorted(self)}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v
                for k, v in self.items()}


# --------------------------------------------------------------------------
def _scalar(text: str) -> Any:
    t = text.strip()
    if not t or t in ("null", "~", "None"):
        return None
    if t.lower() in ("true", "yes"):
        return True
    if t.lower() in ("false", "no"):
        return False
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        return [_scalar(p) for p in inner.split(",")] if inner else []
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def _mini_yaml(text: str) -> dict:
    """Parse the indentation-based mapping subset used by configs/."""
    root: dict = {}
    # stack of (indent, mapping); the mapping at the top owns the next key
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        m = re.match(r"^\s*([A-Za-z_][\w.-]*)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(value)
    return root


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise SystemExit(f"config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _mini_yaml(text)
    cfg = Config(data)
    _resolve_paths(cfg, os.path.dirname(os.path.abspath(path)))
    return cfg


def _resolve_paths(cfg: Config, config_dir: str) -> None:
    """Make relative paths in ``cfg.paths`` resolve against the repo root."""
    root = os.path.dirname(config_dir)  # configs/ -> repo root
    paths = cfg.get("paths")
    if not isinstance(paths, Config):
        return
    for key, value in list(paths.items()):
        if isinstance(value, str) and not os.path.isabs(value):
            paths[key] = os.path.normpath(os.path.join(root, value))
