"""Explicit v4 defaults and content-addressed experiment identity."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from open_score.grouping.storage import fingerprint
from .environment import RULE_VERSION

ROOT = Path(__file__).resolve().parents[3]
ROUTES = ('r1_ppo', 'r2_teacher_ppo', 'r3_ddqn', 'b1_counts', 'b2_local', 'b3_global')


def configuration(route='b3_global', seed=20260906, smoke=False, device='cpu', **overrides):
    if route not in ROUTES:
        raise ValueError(f'Unknown v4 route: {route}')
    result = dict(schema='known-rule-grouping-v4', route=route, seed=int(seed), smoke=bool(smoke),
                  device=device, executor_version=RULE_VERSION, opponent='reactive',
                  max_steps=50, command_interval=5, group_max_size=None,
                  steps=200 if smoke else 100000, seconds=None,
                  scales=[4, 8] if smoke else [8, 12, 16, 24, 32],
                  train_scales=[4, 8] if smoke else [8, 12, 16, 24, 32],
                  eval_scales=[4, 8] if smoke else [8, 12, 16, 24, 32],
                  eval_episodes=2 if smoke else 100, search_budget=8 if smoke else 64,
                  candidate_budget=8 if smoke else 32, threads=1,
                  model=dict(hidden_dim=32, heads=4, layers=1) if smoke else dict(hidden_dim=128, heads=4, layers=2),
                  reward='native_terminal_success', gamma=1.0,
                  checkpoint_seconds=10 if smoke else 300)
    result.update(overrides)
    return result


def provenance():
    source = {}
    for directory in ('src', 'configs', 'scripts'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.ps1'):
                content = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode()
                source[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    return dict(source_hash=fingerprint(source), source_files=source,
                executor_version=RULE_VERSION, git_commit=commit)


def protocol(config):
    mutable = {'workers', 'steps', 'seconds', 'deadline', 'output', 'resume', 'eval_episodes',
               'evaluation_episodes', 'checkpoint_seconds', 'device', 'threads', 'seeds',
               's2_epochs', 's2_seconds', 'collect_seconds', 's2_collect_seconds', 'data_workers'}
    return {k: v for k, v in config.items() if k not in mutable}
