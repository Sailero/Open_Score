"""Protocol, logging, and artifact helpers shared by Stage-3 scripts."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as source:
        values = yaml.safe_load(source)
    if not isinstance(values, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return values


def atomic_write_json(path: Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def atomic_write_text(path: Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, destination)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write a CSV with no rows")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


@dataclass
class LiveProgress:
    """Timestamped console and UTF-8 log output with explicit phase markers."""

    log_path: Path
    phase_total: int = 1
    phase_index: int = 0

    def __post_init__(self) -> None:
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as target:
            target.write(line + "\n")

    def phase(self, name: str, next_name: str | None = None) -> None:
        self.phase_index += 1
        suffix = "" if next_name is None else f"；完成后进入：{next_name}"
        self.write(
            f"[阶段 {self.phase_index}/{self.phase_total}] 开始：{name}{suffix}"
        )

    def progress(
        self,
        label: str,
        completed: int,
        total: int,
        *,
        started_at: float,
    ) -> None:
        elapsed = max(0.0, time.perf_counter() - started_at)
        fraction = completed / max(1, total)
        eta = elapsed * (1.0 - fraction) / max(fraction, 1e-12)
        self.write(
            f"[{label}] {completed:,}/{total:,} ({100.0*fraction:.1f}%)；"
            f"耗时 {elapsed:.1f}s；预计剩余 {eta:.1f}s"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


__all__ = [
    "LiveProgress",
    "atomic_write_json",
    "atomic_write_text",
    "load_yaml",
    "utc_now_iso",
    "write_csv",
]
