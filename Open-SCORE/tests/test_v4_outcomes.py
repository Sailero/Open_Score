"""Tests of label semantics, actual neural learning and evaluator diagnostics."""
from dataclasses import replace
import json
import pickle
import random

import numpy as np
import pytest
import torch

from open_score.grouping.domain import Group, Grouping, DecisionState
from open_score.grouping.storage import random_state, sha256
from open_score.research_v4.actions import candidate_pool, grand_grouping, rule_grouping
from open_score.research_v4.environment import make_env
from open_score.research_v4.data import (collect_counterfactuals, collect_dataset,
    read_dataset, family_split, paired_terminal)
from open_score.research_v4.outcomes import (GlobalOutcomeNetwork, LocalOutcomeNetwork,
    OutcomeScorer, evaluate_evaluator, train_evaluator, load_evaluator,
    train_count_model, historical_transfer_audit)
from open_score.research_v4.outcomes import cross_evaluator_comparison


@pytest.fixture(autouse=True)
def torch_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_complete_terminal_counterfactual_restores_real_world_and_ambient_rng():
    env = make_env(4, seed=23)
    state = env.reset(seed=23)
    before = pickle.dumps(env.snapshot())
    ambient = random_state()
    pool = candidate_pool(state, 3)
    rows = collect_counterfactuals(env, state, pool, [710, 711])
    assert pickle.dumps(env.snapshot()) == before
    assert random.getstate() == ambient['python']
    assert np.array_equal(np.random.get_state()[1], ambient['numpy'][1])
    assert torch.equal(torch.get_rng_state(), ambient['torch'])
    assert all(row['branch_seeds'] == [710, 711] for row in rows)
    assert all(row['continuation_version'] == 'rule_grouping_v1' for row in rows)
    assert all(0 < d <= 50 for row in rows for d in row['terminal_steps'])
    assert all(row['terminal_steps'] == row['physical_steps'] for row in rows)
    repeated = collect_counterfactuals(env, state, pool, [710, 711])
    assert rows == repeated
    scores, steps = paired_terminal(env, state, pool, branches=2, seed=55)
    assert len(scores) == len(pool) and steps > 0
    assert pickle.dumps(env.snapshot()) == before


def test_first_action_only_then_frozen_continuation_and_exception_restore():
    env = make_env(2, seed=12)
    state = env.reset(seed=12)
    calls = []
    def continuation(s):
        calls.append(s.step)
        return rule_grouping(s)
    rows = collect_counterfactuals(env, state, [Grouping((), state.ids('red'))], [100],
                                  continuation, 'test_frozen_mu')
    assert calls and min(calls) > state.step
    assert rows[0]['continuation_version'] == 'test_frozen_mu'
    before = pickle.dumps(env.snapshot())
    def fail(s):
        raise RuntimeError('test continuation failure')
    with pytest.raises(RuntimeError, match='continuation failure'):
        collect_counterfactuals(env, state, [Grouping((), state.ids('red'))], [100], fail, 'broken')
    assert pickle.dumps(env.snapshot()) == before


def test_global_network_has_unbounded_coalitions_and_responds_to_partition():
    torch.manual_seed(5)
    env = make_env(8, seed=9)
    state = env.reset(seed=9)
    ids, target = state.ids('red'), state.ids('targets')[0]
    whole = Grouping((Group(target, ids),))
    split = Grouping((Group(target, ids[:4]), Group(target, ids[4:])))
    model = GlobalOutcomeNetwork(32, 4, 1).eval()
    scores, _ = model([state], [[whole, split]])
    assert scores.shape == (1, 2) and torch.isfinite(scores).all()
    assert not torch.isclose(scores[0, 0], scores[0, 1], atol=1e-7)
    reversed_state = replace(state, red=tuple(reversed(state.red)), blue=tuple(reversed(state.blue)))
    with torch.no_grad():
        other, _ = model([reversed_state], [[whole, split]])
    assert torch.allclose(scores, other, atol=1e-5)
    # Frozen-controller memories/actions are deliberately absent from v4 input.
    with torch.no_grad():
        changed, _ = model([replace(state, memory={i: (100.,)*64 for i in ids},
                                   last_actions={i: 26 for i in ids})], [[whole, split]])
    assert torch.allclose(scores, changed, atol=1e-6)
    scores.sum().backward()
    assert model.partition_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_local_network_handles_more_than_four_members():
    env = make_env(8, 7, targets=1, seed=30)
    state = env.reset(seed=30)
    action = grand_grouping(state)
    model = LocalOutcomeNetwork(32, 4, 1)
    scores, _ = model([state], [[action]])
    assert scores.shape == (1, 1) and torch.isfinite(scores).all()
    scores.sum().backward()
    assert model.entity[0].weight.grad.abs().sum() > 0


def test_local_encoder_is_target_relative_without_claiming_physics_invariance():
    state = make_env(3, 2, targets=1, seed=22).reset(seed=22)
    action = grand_grouping(state)
    def translated(entities):
        return tuple(replace(e, position=(e.position[0], e.position[1]+650., e.position[2])) for e in entities)
    shifted = replace(state, red=translated(state.red), blue=translated(state.blue),
                      targets=translated(state.targets))
    model = LocalOutcomeNetwork(16, 4, 1).eval()
    original, _ = model([state], [[action]])
    changed, _ = model([shifted], [[action]])
    assert torch.allclose(original, changed, atol=1e-6)


def _row(state, action, outcomes, *, candidate=0, family='f0', state_id='s0'):
    return {'state': state.to_dict(), 'action': action.to_dict(), 'kind': 'global',
            'family_id': family, 'state_id': state_id, 'candidate_id': candidate,
            'outcomes': outcomes, 'y': float(np.mean(outcomes))}


def test_probability_metrics_use_bernoulli_not_sample_mean_and_report_ties():
    state = make_env(2, seed=10).reset(seed=10)
    action = rule_grouping(state)
    class Constant:
        kind = 'global'
        def __call__(self, state, pool):
            return [.5]*len(pool)
    rows = [_row(state, action, [0, 1]), _row(state, action, [0, 1], candidate=1)]
    report = evaluate_evaluator(Constant(), rows)
    assert report['brier'] == pytest.approx(.25)  # not zero against y=.5
    assert sum(b['mass'] for b in report['calibration']) == pytest.approx(1.)
    assert report['pairwise_accuracy'] is None
    assert report['all_equal_states'] == 1 and report['outcome_tied_pairs'] == 1


def test_count_proxy_flags_unseen_rosters_and_undefended_targets():
    env = make_env(8, seed=10)
    state = env.reset(seed=10)
    model = {'cells': {'2:2': {'probability': .7}}}
    scorer = OutcomeScorer(kind='count', counts=model)
    deployed, reserve = grand_grouping(state), Grouping((), state.ids('red'))
    scores = scorer(state, [deployed, reserve])
    assert scores[1] < scores[0]
    assert scorer.coverage['unseen_roster_queries'] > 0
    assert scorer.coverage['undefended_reachability_proxy_queries'] > 0
    scorer.aggregation = 'min'
    assert np.isfinite(scorer(state, [deployed])[0])


def test_batched_local_scoring_matches_individual_queries_and_caches_same_state():
    from open_score.research_v4.environment import responsibilities
    state = make_env(8, seed=31).reset(seed=31)
    pool = candidate_pool(state, 8)
    scorer = OutcomeScorer(LocalOutcomeNetwork(16, 4, 1), kind='local')
    expected = []
    for group, ids in responsibilities(state, pool[-1]):
        expected.append((group, ids, scorer.predict_local(state, group, ids)))
    scorer(state, pool)
    before = scorer.coverage['neural_rows_evaluated']
    cached = scorer(state, pool)
    assert scorer.coverage['neural_rows_evaluated'] == before
    assert scorer.coverage['cached_unique_queries'] > 0
    for group, ids, probability in expected:
        if ids:
            assert scorer._local_cache[(group.target, group.members, tuple(ids))] == pytest.approx(probability, abs=1e-6)
    changed = replace(state, step=state.step+1)
    scorer(changed, pool)
    assert scorer.coverage['neural_rows_evaluated'] > before


@pytest.fixture
def small_datasets(tmp_path):
    global_dir, local_dir = tmp_path/'global', tmp_path/'local'
    collect_dataset(global_dir, families=5, scales=(2,), candidates=2, branches=1,
                    states_per_family=1, seed=8)
    collect_dataset(local_dir, families=5, scales=(2,), candidates=1, branches=1,
                    states_per_family=1, seed=8, kind='local')
    return global_dir, local_dir


def test_family_atomic_resume_and_protocol_rejection(small_datasets):
    global_dir, _ = small_datasets
    before = {p.name: sha256(p) for p in (global_dir/'families').glob('*.json')}
    manifest = collect_dataset(global_dir, families=5, scales=(2,), candidates=2,
                               branches=1, states_per_family=1, seed=8, resume=True)
    assert manifest['status'] == 'complete'
    assert manifest['total_physical_steps'] == manifest['physical_simulation_steps']+manifest['behavior_physical_steps']
    assert manifest['behavior_physical_steps'] > 0
    assert manifest['state_category_family_availability']['initial'] == 5
    assert before == {p.name: sha256(p) for p in (global_dir/'families').glob('*.json')}
    rows = read_dataset(global_dir)
    for index in range(5):
        assert {r['split'] for r in rows if r['family_index'] == index} == {family_split(index)}
    with pytest.raises(ValueError, match='changed'):
        collect_dataset(global_dir, families=5, scales=(2,), candidates=3,
                        branches=1, states_per_family=1, seed=8, resume=True)


@pytest.mark.parametrize('kind', ['global', 'local'])
def test_actual_training_checkpoint_resume_and_independent_test(small_datasets, tmp_path, kind):
    source = small_datasets[0 if kind == 'global' else 1]
    output = tmp_path/f'{kind}_model'
    kwargs = dict(kind=kind, hidden_dim=16, heads=4, layers=1, batch_size=4, seed=18)
    result = train_evaluator(source, output, epochs=1, **kwargs)
    assert result['test']['families'] == 1
    initial = sha256(output/'latest.pt')
    result = train_evaluator(source, output, epochs=2, resume=True, **kwargs)
    assert result['epochs_completed'] == 2 and sha256(output/'latest.pt') != initial
    payload = torch.load(output/'latest.pt', weights_only=False)
    assert payload['optimizer']['state']
    scorer = load_evaluator(output/'best.pt')
    row = read_dataset(source, 'test')[0]
    state, action = DecisionState.from_dict(row['state']), Grouping.from_dict(row['action'])
    assert np.isfinite(scorer(state, [action])[0])


def test_rule_local_labels_are_isolated_and_old_s2_transfer_is_bounded(small_datasets, tmp_path):
    _, local_dir = small_datasets
    rows = read_dataset(local_dir)
    assert all(len(r['state']['targets']) == 1 for r in rows)
    assert all(r['continuation_version'] == 'grand_grouping_v1' for r in rows)
    result = historical_transfer_audit(local_dir)
    assert result['status'] == 'complete' and result['eligible_rows'] > 0
    assert result['original_execution_semantics']['red_executor'] == 'frozen_round01_refil_qmix_single_group'
    count = train_count_model(local_dir, tmp_path/'counts.json')
    assert count['families'] == 1
    assert load_evaluator(tmp_path/'counts.json').kind == 'count'


def test_cross_evaluator_uses_same_global_candidate_pools(small_datasets):
    global_dir, _ = small_datasets
    local = OutcomeScorer(LocalOutcomeNetwork(16, 4, 1), kind='local')
    joint = OutcomeScorer(GlobalOutcomeNetwork(16, 4, 1), kind='global')
    report = cross_evaluator_comparison(global_dir, local, joint)
    common = report['common_old_supported_states']
    assert report['historical_status'] == 'complete'
    assert len({r['rows'] for r in common.values()}) == 1
    assert 'old_local_product_proxy' in common
    assert report['all_states']['new_global']['paired_states'] > 0


def test_dataset_tamper_detected_by_shared_teacher_reader(small_datasets):
    global_dir, _ = small_datasets
    path = next((global_dir/'families').glob('*.json'))
    path.write_text(path.read_text(encoding='utf-8')+' ', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        read_dataset(global_dir, 'train')


def test_parallel_family_collection_matches_sequential_and_can_resume(small_datasets, tmp_path):
    sequential, _ = small_datasets
    parallel = tmp_path/'parallel'
    manifest = collect_dataset(parallel, families=5, scales=(2,), candidates=2,
        branches=1, states_per_family=1, seed=8, workers=2)
    assert manifest['collection_workers'] == 2
    assert read_dataset(parallel) == read_dataset(sequential)
    assert manifest['dataset_hash'] == json.loads((sequential/'manifest.json').read_text(encoding='utf-8'))['dataset_hash']
    resumed = collect_dataset(parallel, families=5, scales=(2,), candidates=2,
        branches=1, states_per_family=1, seed=8, workers=1, resume=True)
    assert resumed['dataset_hash'] == manifest['dataset_hash']


def test_budget_exhaustion_keeps_initial_checkpoint_without_extra_resume_epoch(small_datasets, tmp_path):
    source, _ = small_datasets
    output = tmp_path/'budget_model'
    kwargs = dict(kind='global', hidden_dim=16, heads=4, layers=1, batch_size=4, seed=17,
                  epochs=2, seconds=0.)
    first = train_evaluator(source, output, **kwargs)
    assert first['epochs_completed'] == 0 and not first['trained']
    before = sha256(output/'latest.pt')
    again = train_evaluator(source, output, resume=True, **kwargs)
    assert again['epochs_completed'] == 0 and sha256(output/'latest.pt') == before


def test_custom_continuation_requires_frozen_version():
    env = make_env(2, seed=13)
    state = env.reset(seed=13)
    with pytest.raises(ValueError, match='explicit nondefault'):
        collect_counterfactuals(env, state, [rule_grouping(state)], [14], lambda s: rule_grouping(s))
