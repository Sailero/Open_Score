"""Viewer provenance is independent of the scientific compatibility identity.

Legacy recordings stay readable. A saved simulation can continue only when its
behavior identity matches this standalone environment's sources exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
VIEWER_FILES = {"gui.py", "views.py", "export.py", "playback.py", "live.py", "cli.py"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity(root=None):
    root = Path(root or ROOT)
    source, behavior, viewer = {}, {}, {}
    paths = list((root / "had_env").rglob("*.py"))
    if (root / "make_env.py").is_file():
        paths.append(root / "make_env.py")
    for path in sorted(paths):
        name = path.relative_to(root).as_posix()
        content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        source[name] = hashlib.sha256(content.encode()).hexdigest()
        if name.startswith("had_env/core/render/") or (
            "/workbench/" in name and path.name in VIEWER_FILES
        ):
            viewer[name] = source[name]
        else:
            behavior[name] = source[name]
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        # A wheel in another project's venv must not claim that project's HEAD.
        commit = (subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip() if Path(git_root).resolve() == root.resolve() else "unavailable")
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {"git_commit": commit, "source_hash": _digest(source),
            "behavior_hash": _digest(behavior), "viewer_hash": _digest(viewer),
            "source_files": source, "identity_schema": "had-workbench-identity-v1"}


def assert_behavior_compatible(recorded, current=None):
    current = current or source_identity()
    if recorded.get("behavior_hash") != current.get("behavior_hash"):
        raise ValueError("Simulation sources differ; replay is readable, but continuation requires its recorded revision")
