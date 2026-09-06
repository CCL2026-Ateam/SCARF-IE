"""Small .env loader for local experiment scripts.

Values already present in the process environment win over .env values.
"""
from __future__ import annotations

import os
from pathlib import Path


def _iter_env_lines(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name:
            continue
        value = value.strip().strip('"').strip("'")
        yield name, value


def load_env_defaults(*paths: Path) -> None:
    for path in paths:
        for name, value in _iter_env_lines(Path(path)) or ():
            if not os.environ.get(name):
                os.environ[name] = value


def default_env_paths(anchor: Path) -> list[Path]:
    anchor = Path(anchor).resolve()
    paths = []
    for parent in [anchor, *anchor.parents]:
        candidate = parent / ".env"
        if candidate not in paths:
            paths.append(candidate)
    return paths


def api_key_env_for_model(model: str) -> str:
    return "GEMINI_API_KEY" if str(model).lower().startswith("gemini") else "OPUS_API_KEY"
