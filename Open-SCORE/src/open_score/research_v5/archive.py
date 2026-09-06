"""Commit only this completed run's reviewable reports, never model/log volumes."""
from __future__ import annotations

from pathlib import Path
import subprocess

from open_score.research_v4.runner import atomic_json, read_json
from .protocol import ROOT, TASKS


def commit_results(run_dir, config):
    if config.get('smoke'):
        return {'status':'smoke_not_committed'}
    run_dir=Path(run_dir).resolve();repository=ROOT.parent
    candidates=[repository/'实验记录.md',run_dir/'final_report.md',run_dir/'run_result.json',
        run_dir/'calibration_report.json',run_dir/'calibration_report.md',
        run_dir/'shared/budget_manifest.json',run_dir/'shared/config_resolved.json',
        run_dir/'shared/protocol_resolved.json',run_dir/'shared/hardware_profile.json']
    candidates.extend(run_dir/name for name in ('comparison.json','comparison_summary.json','comparison.csv',
        'comparison_cells.csv','comparison_difficulty.csv','diagnostics.csv','reproduction_status.csv',
        'costs.csv','task_analysis.md'))
    for task in TASKS:
        candidates.extend(run_dir/task/name for name in ('task_result.json','summary.json','analysis.md',
            'reproduction_map.md','source_references.json','training_curves.png','success_by_difficulty.png','budget_curves.png'))
    candidates.extend(p for p in (run_dir/'reports').rglob('*') if p.is_file() and
                      p.suffix in ('.md','.csv','.json','.png','.svg','.pdf') and p.name!='event_latency_cache.json')
    paths=sorted({str(p.relative_to(repository)) for p in candidates if p.is_file()})
    try:
        subprocess.run(['git','add','-f','--',*paths],cwd=repository,check=True,capture_output=True,text=True)
        changed=subprocess.run(['git','diff','HEAD','--quiet','--',*paths],cwd=repository).returncode
        if changed:
            # --only leaves unrelated staged user changes out of this commit.
            subprocess.run(['git','commit','--only','-m',f'results: v5.1 completed seed {config["seed"]}',
                            '--',*paths],cwd=repository,check=True,capture_output=True,text=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
        result=dict(status='committed' if changed else 'already_archived',git_commit=commit,files=paths)
    except (OSError,subprocess.SubprocessError) as error:
        result=dict(status='archive_error',error=str(error),files=paths,
                    note='Experiment artifacts remain complete on disk; only result archival needs retry.')
    atomic_json(run_dir/'result_archive.json',result)
    print(f'[result archive] {result["status"]}: {result.get("git_commit",result.get("error"))}',flush=True)
    return result
