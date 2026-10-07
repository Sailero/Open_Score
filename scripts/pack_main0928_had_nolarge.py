#!/usr/bin/env python3
"""Pack HAD analysis + per-method folders (no resume.pt / SMAC) into TP/download.

Usage on the farm:

  python3 /inspire/hdd/project/urbanlowaltitude/fengkairui-25026/shenqili/TP/download/pack_main0928_had_nolarge.py
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TP = Path("/inspire/hdd/project/urbanlowaltitude/fengkairui-25026/shenqili/TP")
ROOT = TP / "Open_Score/Open-SCORE/outputs/main0928"
DEST = TP / "download/main0928_had_nolarge.zip"

PLOT = (
    "regir_sg", "refil", "refil_matched", "b2_qmix_atten", "dcg", "spectra",
    "alma", "transfqmix", "rule_nv1",
    "regir_norefil_sg", "regir_r0_sg", "regir_r1_sg", "regir_nomem",
    "regir_last_sg", "regir_fixed4_sg", "regir_kv0_sg", "regir_untied4_sg",
    "regir_count_sg",
)
DIRS = tuple(method for method in PLOT if method != "rule_nv1")
MAX_BYTES = 32 * 1024 * 1024
SKIP_NAME = {
    "resume.pt", "resume.prev.pt", "latest.pt",
    "stop.request", "eval.stop.request", "trajectories.csv",
}
SKIP_SUFFIX = (".lock", ".pid")
ROOT_CSV = ("episodes.csv", "learning.csv", "progress.csv", "timing.csv")


def is_had(cell):
    return (cell or "").strip().strip('"') == "had"


def skip_file(path: Path):
    name = path.name
    if name in SKIP_NAME or name.startswith("."):
        return True
    if any(name.endswith(suffix) for suffix in SKIP_SUFFIX) or ".pending" in name:
        return True
    if path.parent == ROOT and "smac" in name.lower():
        return True
    return False


def filter_csv(src: Path):
    buf = io.StringIO()
    methods = set()
    with src.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "env" not in reader.fieldnames:
            raise RuntimeError(f"{src.name} has no env column")
        writer = csv.DictWriter(buf, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        n_rows = 0
        for row in reader:
            if not is_had(row.get("env")):
                continue
            writer.writerow(row)
            n_rows += 1
            if row.get("method"):
                methods.add(row["method"].strip())
    return buf.getvalue().encode("utf-8"), n_rows, methods


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"missing {ROOT}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        DEST.unlink()
        print(f"removed old {DEST}")

    n_files = 0
    raw = 0
    csv_methods = {}
    dir_hits = defaultdict(lambda: defaultdict(int))
    to_add = []

    def queue_bytes(arcname, data):
        nonlocal n_files, raw
        to_add.append(("bytes", arcname, data))
        n_files += 1
        raw += len(data)

    def queue_file(path: Path, arcname: str):
        nonlocal n_files, raw
        to_add.append(("file", arcname, path))
        n_files += 1
        raw += path.stat().st_size

    for path in sorted(ROOT.iterdir()):
        if path.is_dir():
            continue
        if path.name == "trajectories.csv":
            print("skip trajectories.csv (not used for tables/figures)")
            continue
        if path.name in ROOT_CSV:
            try:
                data, n_rows, methods = filter_csv(path)
            except Exception as error:
                print("SKIP CSV", path.name, type(error).__name__, error)
                continue
            queue_bytes("main0928/" + path.name, data)
            csv_methods[path.name] = methods
            print(
                "csv %s: %d HAD rows, %.1f MB, methods=%d"
                % (path.name, n_rows, len(data) / 1e6, len(methods))
            )
            continue
        if skip_file(path) or path.stat().st_size >= MAX_BYTES:
            continue
        queue_file(path, "main0928/" + path.name)

    for sub in ("figures", "probe"):
        directory = ROOT / sub
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and not skip_file(path) and path.stat().st_size < MAX_BYTES:
                rel = path.relative_to(ROOT).as_posix()
                queue_file(path, "main0928/" + rel)

    for method in DIRS:
        method_dir = ROOT / method
        if not method_dir.is_dir():
            print("MISSING DIR", method)
            continue
        for path in method_dir.rglob("*"):
            if not path.is_file() or skip_file(path):
                continue
            if path.stat().st_size >= MAX_BYTES:
                print(
                    "SKIP LARGE",
                    path.relative_to(ROOT),
                    round(path.stat().st_size / 1e6, 1),
                    "MB",
                )
                continue
            rel = path.relative_to(ROOT).as_posix()
            queue_file(path, "main0928/" + rel)
            if path.name in ("config.json", "best.pt", "final.pt", "console.log"):
                seed = path.parent.name if path.parent.name.startswith("seed_") else "?"
                dir_hits[method][seed + ":" + path.name] += 1

    print("writing zip ...")
    with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for kind, arcname, payload in to_add:
            if kind == "bytes":
                zf.writestr(arcname, payload)
            else:
                zf.write(payload, arcname=arcname)

    print("packed %d files, %.1f MB uncompressed" % (n_files, raw / 1e6))
    print("wrote", DEST, "%.1f MB" % (DEST.stat().st_size / 1e6))

    print("\n=== CSV methods vs plot list ===")
    for name, methods in csv_methods.items():
        missing = [method for method in PLOT if method not in methods]
        extra = sorted(methods - set(PLOT))
        print("%s: %d methods" % (name, len(methods)))
        if missing:
            print("  MISSING from csv:", missing)
        if extra:
            print("  extra in csv:", extra)

    print("\n=== train folders (config/best/final per seed) ===")
    for method in DIRS:
        hits = dir_hits[method]
        seeds = sorted({key.split(":")[0] for key in hits})
        n_cfg = sum(value for key, value in hits.items() if key.endswith("config.json"))
        n_best = sum(value for key, value in hits.items() if key.endswith("best.pt"))
        n_final = sum(value for key, value in hits.items() if key.endswith("final.pt"))
        print(
            "%-22s seeds=%s  config=%d  best=%d  final=%d"
            % (method, seeds or ["NONE"], n_cfg, n_best, n_final)
        )


if __name__ == "__main__":
    main()
