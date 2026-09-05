"""Migrate a stopped run across the specific, audited Windows rename-retry fix.

Only the exact old/new source hashes below are accepted. Original checkpoints
and manifest are copied into recovery/ before metadata changes. Model weights,
optimizer, RNG, counters and configuration must survive a reload unchanged.
This is not a general bypass of experiment provenance checks.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OLD_SOURCE = 'dd4bdae705a542244e32bc78f6b796ce8d404a22f6fd6ff8f30fafa3e6665271'
NEW_SOURCE = 'f91e928e1cbd709f0c1f67842fe8d0a77a4af2f30a6eacc71ba11230b8efad39'


def assert_same(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def repair(output):
    import torch
    from open_score.grouping.storage import atomic_checkpoint, atomic_json, provenance, sha256
    from open_score.grouping.training import contract
    from open_score.grouping.storage import fingerprint
    from run_parallel_comparison import output_lock

    output = output.resolve()
    current = provenance()
    if current['source_hash'] != NEW_SOURCE:
        raise ValueError('This recovery only accepts the reviewed rename-retry source revision')
    manifest_path = output / 'runner_manifest.json'
    prior = json.loads(manifest_path.read_text(encoding='utf-8'))
    if prior['source_hash'] not in (OLD_SOURCE, NEW_SOURCE):
        raise ValueError('Unrecognized experiment source revision')
    if (prior['assets'] != current['assets']
            or prior['runner_sha256'] != sha256(ROOT / 'scripts/run_parallel_comparison.py')
            or prior['plotter_sha256'] != sha256(ROOT / 'scripts/plot_core_results.py')):
        raise ValueError('Assets, runner or plotter changed; this repair cannot migrate them')
    backup = output / 'recovery' / 'windows_checkpoint_io'
    backup.mkdir(parents=True, exist_ok=True)
    folders = [output / prior['settings']['opponent'] / f'seed_{seed}' / method
               for seed in prior['settings']['seeds'] for method in prior['settings']['methods']]
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(output_lock(output))
        for folder in folders:
            stack.enter_context(output_lock(folder))
        if not (backup / 'runner_manifest.json').exists():
            shutil.copyfile(manifest_path, backup / 'runner_manifest.json')
        paths = [path for folder in folders for path in [folder / 'latest.pt', *sorted((folder / 'budgets').glob('*.pt'))]]
        # Validate everything before modifying any checkpoint.
        for path in paths:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            old = payload['provenance']
            if old['source_hash'] not in (OLD_SOURCE, NEW_SOURCE) or old['assets'] != current['assets']:
                raise ValueError(f'Unexpected provenance: {path}')
            if payload['config_hash'] != fingerprint(contract(payload['config'])):
                raise ValueError(f'Invalid configuration hash: {path}')
            changed = {p for p in old['source_files'].keys() | current['source_files'].keys()
                       if old['source_files'].get(p) != current['source_files'].get(p)}
            if changed - {'src/open_score/grouping/storage.py'}:
                raise ValueError(f'This repair cannot migrate model or training code: {changed}')
        records = []
        for path in paths:
            relative = path.relative_to(output)
            archived = backup / relative
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['provenance']['source_hash'] == NEW_SOURCE:
                if not archived.exists():
                    raise ValueError(f'Missing original checkpoint backup: {archived}')
                before = torch.load(archived, map_location='cpu', weights_only=False)
            else:
                before = copy.deepcopy(payload)
                archived.parent.mkdir(parents=True, exist_ok=True)
                if archived.exists():
                    assert_same(before, torch.load(archived, map_location='cpu', weights_only=False))
                else:
                    shutil.copyfile(path, archived)
                    assert sha256(path) == sha256(archived)
                payload['provenance_history'] = [copy.deepcopy(payload['provenance'])]
                payload['provenance'] = copy.deepcopy(current)
                payload['compatibility_migration'] = {
                    'reason': 'Windows atomic rename retries only; all training state preserved',
                    'original_source_hash': OLD_SOURCE, 'compatible_source_hash': NEW_SOURCE}
                atomic_checkpoint(path, payload)
            after = torch.load(path, map_location='cpu', weights_only=False)
            assert_same({k: v for k, v in before.items() if k != 'provenance'},
                        {k: after[k] for k in before if k != 'provenance'})
            records.append({'path': relative.as_posix(), 'original_sha256': sha256(archived),
                            'migrated_sha256': sha256(path), 'counters': after['counters']})
        atomic_json(backup / 'migration.json', {
            'original_source_hash': OLD_SOURCE, 'compatible_source_hash': NEW_SOURCE,
            'changed_source_files': ['src/open_score/grouping/storage.py'],
            'reason': 'Windows rename retry only; original provenance retained in backup/history',
            'checkpoints': records})
        atomic_json(manifest_path, dict(prior, source_hash=NEW_SOURCE))
        print(json.dumps({'migrated_checkpoints': len(records), 'backup': str(backup)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    repair(parser.parse_args().output)
