"""Cost extrapolation must keep native episodes and learning work distinct."""
import copy

import pytest

from open_score.research_v5.calibration import (
    _normalize, aggregate_samples, choose_budget, estimate_workload,
)
from open_score.research_v5.protocol import defaults


def _statistics():
    # Synthetic throughput numbers are only a deterministic model fixture;
    # production reports can only be made from subprocess benchmark records.
    phase = dict(wall_seconds=1., units=10., episodes=10., real_physical_steps=200.,
                 simulation_physical_steps=1000., worker_seconds=2., terminal_branches=100.,
                 decisions=10., states=10., examples=100., optimizer_steps=10.,
                 events=100., command_events=100., internal_transitions=100.,
                 construction_episodes=10., candidate_rows=100., parallel_episode_pool=True)
    names = {
        'T1': ['plan_decision', 'rule_collection', 'evaluation_main', 'evaluation_rule',
               'evaluation_grand', 'evaluation_singleton', 'evaluation_frozen_b1', 'evaluation_frozen_b3'],
        'T2': ['plan_decision', 'evaluation_main'],
        'T3': ['candidate_sampling', 'candidate_update', 'autoregressive_sampling',
               'autoregressive_update', 'evaluation_candidate', 'evaluation_autoregressive'],
        'T4': ['teacher_label', 'distill_update', 'evaluation_candidate'],
        'T5': ['state_collection', 'potential_simulation', 'optimization', 'evaluation'],
        'T6': ['state_collection', 'label_collection', 'train_bce', 'train_adv', 'evaluation'],
    }
    return {task:{name:copy.deepcopy(phase) for name in keys} for task, keys in names.items()}


def test_formal_test_quota_counts_all_fourteen_arms_and_is_not_scaled():
    config = defaults()
    small = estimate_workload(_statistics(), config, .5, bc_records=4000)
    large = estimate_workload(_statistics(), config, 2., bc_records=4000)
    assert small['complete'] and large['complete']
    assert small['expected_formal_test_executions'] == 21000
    assert large['expected_formal_test_executions'] == 21000
    assert small['components']['T1']['online_test']['units'] == 1500
    assert large['components']['T1']['online_test']['units'] == 1500
    assert small['config']['t3']['steps'] == 500000
    assert large['config']['t3']['steps'] == 2000000
    assert large['task_seconds']['T3'] > small['task_seconds']['T3']


def test_model_does_not_turn_missing_throughput_into_zero_cost():
    stats = _statistics()
    del stats['T6']['label_collection']
    result = estimate_workload(stats, defaults(), .5)
    assert not result['complete']
    assert result['estimated_total_seconds'] is None
    assert result['task_seconds']['T6'] is None
    assert any('T6.label_collection' in x for x in result['missing_measurements'])


def test_highest_fitting_budget_selected_and_smallest_retained_if_overrun():
    estimates = [dict(multiplier=q, complete=True, estimated_total_seconds=t)
                 for q,t in [(.5,36000), (1.,68000), (2.,100000)]]
    selected, reason = choose_budget(estimates)
    assert selected['multiplier'] == 1.
    assert reason == 'highest_measured_budget_within_target'
    selected, reason = choose_budget(estimates, target_seconds=1000)
    assert selected['multiplier'] == .5
    assert reason == 'smallest_budget_exceeds_target_will_still_complete'


def test_unknown_measurements_only_propose_minimum():
    estimates = [dict(multiplier=q, complete=False, estimated_total_seconds=None) for q in (.5,1.,2.)]
    selected, reason = choose_budget(estimates)
    assert selected['multiplier'] == .5
    assert 'insufficient' in reason


def test_separate_sampling_and_update_normalization():
    raw = {kind:dict(sample_seconds=10., physical_steps=200, events=100,
                     update_seconds=2., update_examples=128, optimizer_steps=2)
           for kind in ('candidate', 'autoregressive')}
    phases = _normalize('T3', raw)
    assert phases['candidate_sampling']['real_physical_steps'] == 200
    assert phases['candidate_update']['examples'] == 128
    totals = aggregate_samples({'T3':[dict(phases=phases),dict(phases=phases)]})
    assert totals['T3']['candidate_sampling']['wall_seconds'] == 20.
    assert totals['T3']['candidate_update']['wall_seconds'] == 4.
    assert totals['T3']['candidate_update']['examples'] == 256


def test_makespan_uses_slowest_concurrent_task_and_wait_is_not_added_twice():
    result = estimate_workload(_statistics(), defaults(), 1., prepared_seconds=100.,
                               calibration_seconds=900., bc_records=4000)
    expected = (max(result['task_seconds'].values())+100.+result['exclusive_latency_seconds'])*1.2+900.
    assert result['estimated_total_seconds'] == pytest.approx(expected)
    assert result['estimated_total_seconds'] < sum(result['task_seconds'].values())*1.2+1000


def test_multiplier_changes_only_fixed_work_quotas_not_hyperparameters():
    config = defaults()
    config['t3']['learning_rate'] = 1e-4
    config['t2']['iterations'] = 96
    result = estimate_workload(_statistics(), config, 2.)
    assert result['config']['t3']['learning_rate'] == 1e-4
    assert result['config']['t2']['iterations'] == 96
    assert result['config']['t6']['validation_states'] == config['t6']['validation_states']
    assert result['config']['t5']['episodes'] == config['t5']['episodes']*2


def test_recalibration_after_provisional_half_budget_uses_absolute_multiplier():
    original = defaults()
    previous = estimate_workload(_statistics(), original, .5)['config']
    repeated = estimate_workload(_statistics(), previous, .5)['config']
    restored = estimate_workload(_statistics(), previous, 1.)['config']
    for task, fields in {'t3':['steps'], 't4':['states_per_round'],
                         't5':['states','episodes'], 't6':['train_states']}.items():
        for field in fields:
            assert repeated[task][field] == previous[task][field]
            assert restored[task][field] == original[task][field]


def test_serial_evaluation_speedup_is_explicit_and_bounded_by_quota():
    stats = _statistics()
    stats['T5']['evaluation']['parallel_episode_pool'] = False
    result = estimate_workload(stats, defaults(), 1.)
    cost = result['components']['T5']['online_test']
    assert cost['seconds_per_unit'] == pytest.approx(.1/3)
    assert any('T5/T6 serial' in text for text in result['assumptions'])


def test_quarterly_t5_validation_and_exclusive_latency_are_counted():
    result = estimate_workload(_statistics(), defaults(), .5)
    assert result['components']['T5']['quarterly_validation']['units'] == 600
    assert result['exclusive_latency_decisions'] == 420
    assert len(result['exclusive_latency_components']) == 14
    assert result['exclusive_latency_seconds'] > 0
    assert result['components']['T6']['diagnostics']['units'] == (3*120+30)*10*48
    assert result['components']['T3']['candidate_diagnostics']['units'] == 150*42*48
    assert result['components']['T4']['final_diagnostics']['units'] == 150*10*48


def test_windows_calibration_probe_path_stays_short():
    from pathlib import PureWindowsPath
    root = PureWindowsPath(r'E:\Code\Open_Score\Open-SCORE\outputs\v5_parallel\v5_20260907_main\calibration\20260906T112806')
    path = root/'T3'/'p'/'0'/'evaluations'/'12345678'/'cal'/'c'/'families'/('a'*64+'.jsonl.tmp')
    assert len(str(path)) < 250
