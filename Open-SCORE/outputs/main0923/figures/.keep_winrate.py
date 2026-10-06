"""Copy last good win-rate PNGs back if a live scheduler overwrites them."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEMS = ("fig_learning", "fig_learning_smac_appendix")
STAMP = HERE / ".winrate_guard.pid"
CACHE = HERE / ".winrate_cache"
MIN_BYTES = 80_000


def digest(path: Path) -> bytes:
    return hashlib.md5(path.read_bytes()).digest() if path.exists() else b""


def copy(src: Path, dst: Path) -> None:
    pending = dst.with_name(f".{dst.name}.guard")
    pending.write_bytes(src.read_bytes())
    pending.replace(dst)


def main() -> None:
    STAMP.write_text(str(os.getpid()), encoding="utf-8")
    good = {}
    for stem in STEMS:
        src = CACHE / f"{stem}.png"
        if src.exists() and src.stat().st_size >= MIN_BYTES:
            good[stem] = digest(src)
    if len(good) != len(STEMS):
        raise SystemExit("winrate cache missing; draw figures first")
    while STAMP.exists() and STAMP.read_text(encoding="utf-8").strip() == str(os.getpid()):
        for stem in STEMS:
            live = HERE / f"{stem}.png"
            src = CACHE / f"{stem}.png"
            if digest(live) != good[stem]:
                copy(src, live)
        time.sleep(1)


if __name__ == "__main__":
    main()
