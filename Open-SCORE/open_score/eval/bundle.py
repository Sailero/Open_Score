"""Live reports and a scp-able zip of main0928 results.

Training already rewrites `实验报告.md` as CSV rows land. This module is the
explicit refresh + the campaign-end (or on-demand) download pack. Runtime
files (`resume.pt`, consoles, cluster claims) stay out of the zip.
"""
from __future__ import annotations

from pathlib import Path
import time
import zipfile

from . import experiment as X
from . import cluster

SKIP_DIR_NAMES = {"imported", "cluster", "__pycache__"}
SKIP_FILE_NAMES = {
    "stop.request", "eval.stop.request", "resume.pt", "resume.prev.pt",
    "console.log", "best.pt",
}
SKIP_SUFFIXES = (".lock", ".pid", ".pending", ".console.log")
SKIP_NAME_PREFIXES = ("queue.", "scheduler.", "run_all.")
ROOT_CSV = ("episodes.csv", "learning.csv", "progress.csv", "timing.csv",
            "trajectories.csv", "verification.csv", "benchmarks.csv")
ROOT_JSON = ("experiment.json", "inventory.json", "aggregates.json", "pipeline.json",
             "control.json")
ZIP_STEM = {
    "analysis": "{profile}_analysis.zip",
    "results": "{profile}_results.zip",
}


def report_path(output):
    return Path(output) / "实验报告.md"


def paper_path(output):
    return Path(output) / "实验报告_论文版.md"


def zip_path(output, kind="results"):
    return Path(output) / ZIP_STEM[kind].format(profile=X.PROFILE)


def refresh_live_report(output):
    """Rewrite the formal markdown and figures from whatever CSV is on disk now."""
    from open_score.eval.report import refresh_report
    output = Path(output)
    path = refresh_report(output, run="train")
    try:
        from open_score.eval import report0928
        report0928.render_paper(output, run="train")
    except Exception as error:
        print(f"paper report skipped: {type(error).__name__}: {error}", flush=True)
    return path


def _skip_file(path, root, *, weights):
    name = path.name
    if name in SKIP_FILE_NAMES:
        return True
    if name.endswith(SKIP_SUFFIXES):
        return True
    if any(name.startswith(prefix) for prefix in SKIP_NAME_PREFIXES):
        return True
    if name.endswith(".zip"):
        return True
    if not weights and path.suffix == ".pt":
        return True
    if weights and path.suffix == ".pt" and name != "final.pt":
        return True
    rel = path.relative_to(root)
    return any(part in SKIP_DIR_NAMES for part in rel.parts[:-1])


def _iter_bundle_files(root, *, weights):
    root = Path(root)
    for name in ("实验计划.md", "实验报告.md", "实验报告_论文版.md") + ROOT_CSV + ROOT_JSON:
        path = root / name
        if path.is_file():
            yield path
    for directory in ("figures", "probe"):
        folder = root / directory
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and not _skip_file(path, root, weights=weights):
                yield path
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.parent == root:
            continue
        if _skip_file(path, root, weights=weights):
            continue
        if path.name in ("config.json", "resource.json", "task_failure.json") or path.name == "final.pt":
            yield path


def _write_zip(root, dest, files):
    root = Path(root)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.name}.{time.time_ns()}.pending")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            readme = (
                f"{X.PROFILE} download bundle\n"
                f"created {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
                "Open 实验报告.md for tables; figures/ for plots; csv for raw cells.\n"
                "final.pt is included only in *_results.zip.\n"
            )
            archive.writestr("下载说明.txt", readme)
            seen = set()
            for path in files:
                rel = path.relative_to(root).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                archive.write(path, arcname=rel)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return dest


def export_results_zip(output, *, weights=True, force=True):
    """Pack current artifacts. `weights=False` is the smaller analysis-only zip."""
    root = Path(output)
    kind = "results" if weights else "analysis"
    dest = zip_path(root, kind)
    files = list(dict.fromkeys(_iter_bundle_files(root, weights=weights)))
    with cluster.nfs_lock(dest, timeout=600):
        if dest.exists() and not force:
            return dest
        _write_zip(root, dest, files)
    payload = dict(profile=X.PROFILE, kind=kind, path=str(dest), bytes=dest.stat().st_size,
                   files=len(files), packed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   host=cluster.hostname())
    X.atomic_json(dest.with_suffix(".manifest.json"), payload)
    return dest


def finish_campaign(output):
    """Final report + both zips. Safe to call from one host while others idle."""
    output = Path(output)
    report = refresh_live_report(output)
    analysis = export_results_zip(output, weights=False, force=True)
    results = export_results_zip(output, weights=True, force=True)
    print(f"campaign pack: report={report} analysis={analysis} results={results} "
          f"({results.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dict(report=str(report), analysis=str(analysis), results=str(results),
                bytes=results.stat().st_size)
