"""YAML configuration loading with attribute access and dotted overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __init__(self, mapping: Mapping[str, Any] | None = None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file and apply ``key.subkey=value`` overrides."""
    path = Path(path)
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be of the form key.sub=value")
        dotted, raw = override.split("=", 1)
        _set_dotted(cfg, dotted.strip(), _parse_scalar(raw.strip()))
    return cfg


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], Config):
            node[part] = Config()
        node = node[part]
    node[parts[-1]] = Config._wrap(value)


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def merge_config(base: Config, extra: Mapping[str, Any]) -> Config:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = Config._wrap(value)
    return merged
