"""Config loading: default.toml deep-merged with an optional user file and CLI overrides."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.toml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    if path:
        with open(path, "rb") as fh:
            cfg = _merge(cfg, tomllib.load(fh))
    if overrides:
        cfg = _merge(cfg, overrides)
    weights = cfg["weights"]
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    return cfg


def parse_weight_overrides(spec: str | None) -> dict:
    """'hook=0.3,visual=0.1' -> {'weights': {'hook': 0.3, 'visual': 0.1}}"""
    if not spec:
        return {}
    weights = {}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        weights[key.strip()] = float(val)
    return {"weights": weights}


def load_dotenv(paths: list[Path] | None = None) -> list[str]:
    """Load KEY=VALUE lines from .env files into os.environ (no extra dependency).

    Looks in the project root and the current directory. A variable that is already set to a
    non-empty value wins; empty ones are filled. Returns the names loaded (never the values).
    """
    import os

    root = Path(__file__).resolve().parent.parent
    loaded: list[str] = []
    for p in paths or [root / ".env", Path.cwd() / ".env"]:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip().removeprefix("export ").strip(), val.strip().strip('"').strip("'")
            if key and val and not os.environ.get(key):
                os.environ[key] = val
                loaded.append(key)
    return loaded
