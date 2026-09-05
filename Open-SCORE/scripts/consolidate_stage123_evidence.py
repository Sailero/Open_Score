"""Hash-verified migration of existing evidence; deletion is a separate PS step.

Never rewrites historical metrics. Reports are archived as JSON strings so there
is a single reader-facing Markdown report, with recoverable original wording.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / "outputs"
ROOT = OUTPUTS / "stage123_unknown_upper_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def migrate():
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        for item in manifest["retained_files"]:
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"Migrated evidence changed: {item['path']}")
        print("Existing migration verified; no source deletion performed.")
        return
    inventory = [{"path": p.relative_to(OUTPUTS).as_posix(), "bytes": p.stat().st_size,
                  "sha256": sha(p)} for p in sorted(OUTPUTS.rglob("*")) if p.is_file()]
    retained = []

    def keep(source, target, compress=False):
        source = OUTPUTS / source
        destination = ROOT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha(source)
        if compress:
            with source.open("rb") as stream, destination.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                    shutil.copyfileobj(stream, zipped)
            with gzip.open(destination, "rb") as stream:
                if hashlib.sha256(stream.read()).hexdigest() != source_hash:
                    raise ValueError("Compressed evidence verification failed")
        else:
            shutil.copy2(source, destination)
            if sha(destination) != source_hash:
                raise ValueError("Copy verification failed")
        retained.append({"path": target, "sha256": sha(destination),
                         "bytes": destination.stat().st_size,
                         "source": source.relative_to(OUTPUTS).as_posix(),
                         "source_sha256": source_hash, "gzip": compress})

    for p in sorted((OUTPUTS / "round_01_mvp/stage1").rglob("*")):
        if p.is_file():
            keep(p.relative_to(OUTPUTS), "stage1/" + p.relative_to(OUTPUTS / "round_01_mvp/stage1").as_posix())
    base = "mvp/stage23_identity_blotto_v5/stage2"
    for name in ["data/dataset_manifest.json", "data/dynamic_outcome_time.jsonl",
                 "model/dynamic_outcome_time_model.pt", "model/metrics.json",
                 "model/shape_diagnostics.json", "model/training_history.csv", "pipeline_status.json"]:
        keep(f"{base}/{name}", f"stage2/{name}")
    for name in ["01_training_curves.png", "02_all_scale_win_rates.png", "04_core_calibration.png"]:
        keep(f"{base}/figures/{name}", f"figures/stage2_{name}")
    baselines = {"legacy_idb": "mvp/stage23_identity_blotto_v5/stage3",
                 "saldae": "mvp/stage3_saldae_do_v1",
                 "count_bayesian_v3": "mvp/stage23_bayesian_adaptive_v3/stage3"}
    for label, source in baselines.items():
        for p in sorted((OUTPUTS / source).rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(OUTPUTS / source).as_posix()
            if (rel.startswith(("raw/", "analysis/")) and p.suffix in {".json", ".csv"}) or rel in {"config_snapshot.yaml", "run_manifest.json"}:
                compress = p.stat().st_size > 10_000_000
                keep(p.relative_to(OUTPUTS), f"stage3/baselines/{label}/{rel}" + (".gz" if compress else ""), compress)
    keep("mvp/stage23_bayesian_adaptive_v3/stage2/model/metrics.json", "stage3/baselines/count_bayesian_v3/stage2_metrics.json")
    keep("mvp/stage23_identity_blotto_v5/resolved_stage3_config.yaml", "stage3/baselines/legacy_idb/config_snapshot.yaml")
    archive = []
    for p in sorted(OUTPUTS.rglob("*.md")):
        if p.parts[len(OUTPUTS.parts)] in {"mvp", "round_01_mvp"}:
            archive.append({"source": p.relative_to(OUTPUTS).as_posix(), "sha256": sha(p), "text": p.read_text(encoding="utf8")})
    write_json(ROOT / "stage3/baselines/source_reports.json", archive)
    # This deleted evidence is still available in the pre-snapshot Git revision.
    revision = subprocess.check_output(["git", "rev-parse", "HEAD~"], cwd=PROJECT, text=True).strip()
    recoveries = []
    for source, target in [
        ("evaluation_unseen_scales/summary.json", "stage1/generalization_summary.json"),
        ("evaluation_unseen_scales/episodes.csv", "stage1/generalization_episodes.csv"),
        ("evaluation/win_rate_by_scale.csv", "stage1/win_rate_by_scale.csv"),
        ("stage1/training_curves.csv", "stage1/training_curves.csv"),
    ]:
        original = f"Open-SCORE/outputs/round_01_mvp/{source}"
        data = subprocess.run(["git", "show", f"{revision}:{original}"], cwd=PROJECT, capture_output=True, check=True).stdout
        destination = ROOT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        recoveries.append({"path": target, "source_commit": revision, "source_path": original, "sha256": sha(destination)})
    for p in sorted((ROOT / "stage1").glob("*.json")):
        json.loads(p.read_text(encoding="utf8"))
    write_json(manifest_path, {
        "schema_version": "stage123-unknown-upper-evidence-v1",
        "snapshot_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "historical_evidence_only": True, "unknown_upper_results": "not_run",
        "stage2_retrained": False, "stage2_episodes": 19200,
        "stage2_rows": 101366, "retained_files": retained,
        "recovered_from_git": recoveries, "source_inventory": inventory,
        "source_reports_archive_sha256": sha(ROOT / "stage3/baselines/source_reports.json"),
        "deletion_policy": "Delete obsolete source roots only after all migrated hashes pass. Distinct legacy Stage2 data is obsolete, not a byte duplicate. Raw historical physical evidence is preserved, including negative results.",
    })
    write_json(ROOT / "status.json", {"status": "not_started", "phase": "evidence_consolidated", "completed": 0, "expected": 2400})
    (ROOT / "pipeline.log").write_text("Existing Stage1/2 and historical Stage3 evidence consolidated and hash verified. New protocol has not run.\n", encoding="utf8")
    print(f"Migrated {len(retained)} files; {len(recoveries)} historical artifacts recovered from Git.")


if __name__ == "__main__":
    migrate()
