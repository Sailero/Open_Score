"""Portable provenance and crash-safe experiment files."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # A terminated append may leave only the final line incomplete.
    return rows


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def random_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def provenance():
    manifest = json.loads((ROOT / 'assets/frozen/manifest.json').read_text(encoding='utf-8'))
    for name in ('lcl', 'dlom'):
        actual = sha256(ROOT / 'assets/frozen' / manifest[name]['file'])
        if actual != manifest[name]['sha256']:
            raise ValueError(f'Frozen {name} asset hash mismatch')
    source = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
                  p.read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode('utf-8')).hexdigest()
              for folder in ('src', 'configs')
              for p in sorted((ROOT / folder).rglob('*'))
              if p.is_file() and p.suffix in ('.py', '.yaml')}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    return {'git_commit': commit, 'source_hash': fingerprint(source), 'source_files': source,
            'assets': manifest, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}
