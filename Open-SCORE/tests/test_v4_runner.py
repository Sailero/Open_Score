"""Runner failure isolation, compatible resume, and paired reporting contracts."""
import json
from pathlib import Path
import sys

import pytest

from open_score.research_v4.evaluation import read_records, seed_for, summary
from open_score.research_v4.reporting import paired_difference, report
from open_score.research_v4.runner import (atomic_json, directory_lock, pid_alive,
                                          run_pool, validate_manifest)


def test_manifest_allows_scheduling_and_step_extensions(tmp_path):
    identity = {'source': 'same', 'executor': 'fixed'}
    validate_manifest(tmp_path, identity, request={'workers': 3, 'steps': 32})
    updated = validate_manifest(tmp_path, identity, resume=True, request={'workers': 6, 'steps': 64})
    assert len(updated['requests']) == 2
    with pytest.raises(ValueError, match='protocol changed'):
        validate_manifest(tmp_path, {'source': 'changed'}, resume=True)


def test_existing_output_requires_resume(tmp_path):
    validate_manifest(tmp_path, {'source': 'same'})
    with pytest.raises(ValueError, match='--resume'):
        validate_manifest(tmp_path, {'source': 'same'})


def test_unmanifested_results_are_not_adopted(tmp_path):
    (tmp_path / 'important.json').write_text('{}')
    with pytest.raises(ValueError, match='without a compatible manifest'):
        validate_manifest(tmp_path, {})


def test_lock_refuses_live_owner_and_releases(tmp_path):
    import os
    assert pid_alive(os.getpid())
    with directory_lock(tmp_path):
        with pytest.raises(RuntimeError, match='in use'):
            with directory_lock(tmp_path):
                pass
    assert not (tmp_path / '.runner.lock').exists()


def test_failed_child_does_not_cancel_healthy_children(tmp_path):
    marker = tmp_path / 'healthy.txt'
    jobs = [
        {'id': 'bad', 'command': [sys.executable, '-c', 'raise SystemExit(7)']},
        {'id': 'healthy', 'command': [sys.executable, '-c',
         'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("ok")', str(marker)]},
    ]
    result = run_pool(jobs, workers=2, output=tmp_path, poll_seconds=.01)
    assert result['completed'] == 1
    assert result['failed'][0]['returncode'] == 7
    assert marker.read_text() == 'ok'
    assert (tmp_path / 'logs' / 'bad.log').exists()


def test_deadline_stops_owned_child(tmp_path):
    import time
    result = run_pool([{'id': 'slow', 'command': [sys.executable, '-c', 'import time; time.sleep(20)']}],
                      workers=1, output=tmp_path, deadline=time.monotonic() + .25, poll_seconds=.01)
    assert result['interrupted']
    status = json.loads((tmp_path / 'scheduler_status.json').read_text())
    assert status['running'] == []
    assert result['jobs'][0]['state'] == 'interrupted'


def test_resume_repairs_only_last_partial_record(tmp_path):
    path = tmp_path / 'evaluation.jsonl'
    path.write_text('{"success": true}\n{"suc', encoding='utf-8')
    assert read_records(path, repair_tail=True) == [{'success': True}]
    assert list(tmp_path.glob('*.incomplete-*'))
    path.write_text('{bad}\n{"success": true}\n', encoding='utf-8')
    with pytest.raises(ValueError):
        read_records(path, repair_tail=True)


def test_evaluation_seeds_are_paired_and_scale_separated():
    assert seed_for(7, 8, 0) == seed_for(7, 8, 0)
    assert len({seed_for(7, s, e) for s in (4, 8, 16) for e in range(500)}) == 1500


def test_missing_evaluations_are_not_zero_success():
    result = summary([], route='b3_global', variants=['policy'], scales=[8], episodes=100)
    assert result['complete'] is False
    assert result['groups'][0]['success_rate'] is None


def test_paired_difference_matches_only_same_openings():
    candidate = [{'scale': 8, 'seed': 1, 'success': True}, {'scale': 8, 'seed': 2, 'success': False}]
    reference = [{'scale': 8, 'seed': 1, 'success': False}, {'scale': 16, 'seed': 2, 'success': False}]
    result = paired_difference(candidate, reference)
    assert result['pairs'] == 1
    assert result['difference'] == 1


def test_empty_report_does_not_claim_completion(tmp_path):
    result = report(tmp_path)
    assert result['all_available_evaluations_complete'] is False
    assert '尚无完成' in (tmp_path / 'comparison_report.md').read_text(encoding='utf-8')


def test_shared_evaluators_are_specific_to_training_seed(tmp_path):
    from open_score.research_v4.cli import parser, shared_folder, _child_command
    args = parser().parse_args(['run', '--output', str(tmp_path), '--seeds', '11', '22'])
    assert shared_folder(args, 11) != shared_folder(args, 22)
    config = {'device': 'cpu', 'steps': 100, 'seconds': 60, 'eval_episodes': 2,
              'search_budget': 8, 'scales': [4, 8]}
    commands = [_child_command(args, config, 'b3_global', seed) for seed in (11, 22)]
    assert commands[0][commands[0].index('--shared-dir') + 1].endswith('seed_11')
    assert commands[1][commands[1].index('--shared-dir') + 1].endswith('seed_22')


def test_foreign_shared_seed_is_rejected_before_loading_models(tmp_path):
    from open_score.research_v4.cli import parser, request_config, shared_folder, shared_assets, _identity
    args = parser().parse_args(['evaluate', '--smoke', '--device', 'cpu', '--output', str(tmp_path), '--seeds', '11'])
    config = request_config(args)
    folder = shared_folder(args, 11)
    atomic_json(folder / 'preparation_request.json', {'training_seed': 22, 'identity': _identity(config)})
    with pytest.raises(ValueError, match='training seed'):
        shared_assets(args, config, prepare=False)


def test_report_identifies_missing_route_even_if_other_routes_complete(tmp_path):
    atomic_json(tmp_path / 'run_manifest.json', {'requests': [{'command': 'run', 'routes': ['b1_counts', 'b3_global'], 'seeds': [11]}]})
    atomic_json(tmp_path / 'b1_counts' / 'seed_11' / 'evaluation_summary.json',
                {'route': 'b1_counts', 'complete': True, 'groups': []})
    result = report(tmp_path)
    assert result['complete'] is False
    assert result['missing_requested_evaluations'] == [{'route': 'b3_global', 'seed': 11}]


def test_benchmark_dry_run_does_not_create_output(tmp_path, capsys):
    from open_score.research_v4.cli import main
    destination = tmp_path / 'dry_only'
    assert main(['benchmark', '--dry-run', '--output', str(destination)]) == 0
    assert not destination.exists()
    assert 'benchmark_workers' in capsys.readouterr().out


def test_evaluation_identity_matches_run_and_separate_evaluate(tmp_path):
    from open_score.research_v4.evaluation import evaluation_identity
    config = {'route': 'r1_ppo', 'seed': 11, 'steps': 1024, 'device': 'cpu'}
    assets = {'global_checkpoint': str(tmp_path / 'unused.pt')}
    through_train = {**config, 'shared_dir': str(tmp_path), 'assets': assets}
    assert evaluation_identity(config, {}, assets) == evaluation_identity(through_train, {}, assets)


def test_changing_worker_counts_preserves_protocol_identity():
    from open_score.research_v4.cli import parser, request_config, _identity
    first = request_config(parser().parse_args(['run', '--smoke', '--device', 'cpu', '--workers', '1']))
    second = request_config(parser().parse_args(['run', '--smoke', '--device', 'cpu', '--workers', '6']))
    assert first['data_workers'] == 1 and second['data_workers'] == 2
    assert _identity(first) == _identity(second)
