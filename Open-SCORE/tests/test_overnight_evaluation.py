"""Paired raw counts, pure-rule controls, split isolation, and resume behavior."""
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from open_score.grouping.domain import DecisionState, Entity, Group, Grouping
from open_score.overnight import evaluation as evaluation
from open_score.overnight.config import configuration
from open_score.overnight.policy import CandidateNetwork


def config():
    result = configuration('ppo_structured', smoke=True)
    result.update(eval_scales=[4, 8], validation_scales=[4, 8], executor_scope='full')
    return result


class FakeEnv:
    calls = []
    closes = 0

    def __init__(self, red, blue, **kwargs):
        self.scale = red
        assert red == blue

    def reset(self, seed):
        self.seed = seed
        self.calls.append((self.scale, seed))
        reds = tuple(Entity(i, (i*100., 0., 0.), (0., 0., 0.), 1.) for i in range(self.scale))
        blues = tuple(Entity(i, (i*100., 500., 0.), (0., 0., 0.), 1.) for i in range(self.scale))
        targets = (Entity(100, (0., 1000., 0.), (0., 0., 0.), 1.2),
                   Entity(101, (1000., 1000., 0.), (0., 0., 0.), 1.2))
        self.state = DecisionState(0, 50, 'reactive', reds, blues, targets,
                                   Grouping((), tuple(range(self.scale))))
        return self.state

    def step(self, action):
        self.state = replace(self.state, step=self.state.step+5, previous=action)
        done = self.state.step >= 10
        success = bool(action.groups) and self.seed % 2 == 0
        return self.state, float(success and done), done, {'delta': 5, 'success': success}

    def close(self):
        type(self).closes += 1


@pytest.fixture
def fake_env(monkeypatch):
    FakeEnv.calls, FakeEnv.closes = [], 0
    monkeypatch.setattr(evaluation, 'make_env', FakeEnv)
    # Numerical/report data is exercised separately from image rendering.
    monkeypatch.setattr(evaluation, 'write_report', lambda output, summary: None)
    return FakeEnv


def test_untrained_baselines_raw_counts_behavior_and_identical_paired_seeds(tmp_path, fake_env, monkeypatch):
    def no_network(*_args, **_kwargs):
        pytest.fail('Pure-rule evaluation constructed a trainable network')
    monkeypatch.setattr(evaluation, 'CandidateNetwork', no_network)
    result = evaluation.evaluate_models(config(), {'static': None, 'dynamic_rule': None, 'all_reserve': None},
                                         tmp_path, episodes=3, wall_seconds=30)
    rows = evaluation.read_jsonl(tmp_path / 'evaluation.jsonl')
    assert result['complete'] and result['complete_pairs'] == result['planned_pairs'] == 6
    assert result['raw_episodes'] == result['complete_episodes'] == len(rows) == 18
    assert len(set(fake_env.calls)) == 6
    assert all(fake_env.calls.count(pair) == 3 for pair in set(fake_env.calls))
    assert fake_env.closes == 2
    assert len(list(tmp_path.glob('trace_*.json'))) == 6
    for row in rows:
        assert row['physical_steps'] == 10 and row['upper_events'] == 2
        assert len(row['decision_seconds']) == 2
        assert row['constraint_violations'] == 0
        if row['method'] == 'all_reserve':
            assert row['mean_reserve_fraction'] == 1
            assert row['mean_group_size'] == row['singleton_fraction'] == 0
            assert not row['success']
    for row in result['table']:
        raw = [item for item in rows if item['method'] == row['method'] and item['scale'] == row['scale']]
        assert row['successes'] == sum(item['success'] for item in raw)
        assert row['episodes'] == 3
        assert row['success_rate'] == row['successes'] / 3
    assert set(result['model_parameters'].values()) == {0}


def test_seed_splits_are_disjoint_and_rng_is_restored(tmp_path, fake_env):
    settings = config()
    training = {settings['seed']+100_000+episode for episode in range(20_000)}
    validation = {evaluation.evaluation_seed(settings, scale, episode, validation=True)
                  for scale in (4, 8, 12, 16, 24, 32) for episode in range(100)}
    test = {evaluation.evaluation_seed(settings, scale, episode)
            for scale in (4, 8, 12, 16, 24, 32) for episode in range(100)}
    assert not training & validation and not training & test and not validation & test
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    before = evaluation.random_state()
    result = evaluation.evaluate_models(settings, {'all_reserve': None}, tmp_path,
                                         episodes=1, wall_seconds=30, validation=True)
    after = evaluation.random_state()
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:]
    assert torch.equal(before['torch'], after['torch'])
    assert result['split'] == 'validation'
    assert {seed for _, seed in fake_env.calls} <= validation


def test_partial_pair_is_excluded_then_resume_only_finishes_missing_member(tmp_path, fake_env, monkeypatch):
    settings = config()
    settings['eval_scales'] = [4]
    original = evaluation.rollout_episode
    attempted = []

    def interrupted(env, method, *args, **kwargs):
        attempted.append(method)
        if method == 'dynamic_rule':
            raise RuntimeError('injected interruption')
        return original(env, method, *args, **kwargs)

    monkeypatch.setattr(evaluation, 'rollout_episode', interrupted)
    checkpoints = {'static': None, 'dynamic_rule': None}
    with pytest.raises(RuntimeError, match='injected'):
        evaluation.evaluate_models(settings, checkpoints, tmp_path, episodes=1, wall_seconds=30)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['complete_pairs'] == 0 and summary['incomplete_pairs'] == 1
    assert summary['raw_episodes'] == 1 and not summary['table']
    preserved = (tmp_path / 'evaluation.jsonl').read_bytes()
    monkeypatch.setattr(evaluation, 'rollout_episode', original)
    result = evaluation.evaluate_models(settings, checkpoints, tmp_path, episodes=1, wall_seconds=30)
    assert result['complete'] and result['complete_pairs'] == 1
    assert len(fake_env.calls) == 2  # static once, resumed dynamic once.
    assert (tmp_path / 'evaluation.jsonl').read_bytes().startswith(preserved)
    assert len(evaluation.read_jsonl(tmp_path / 'evaluation.jsonl')) == 2
    raw_before = (tmp_path / 'evaluation.jsonl').read_bytes()
    evaluation.evaluate_models(settings, checkpoints, tmp_path, episodes=1, wall_seconds=30)
    assert len(fake_env.calls) == 2 and (tmp_path / 'evaluation.jsonl').read_bytes() == raw_before
    settings['executor_scope'] = 'nearest3'
    with pytest.raises(ValueError, match='model/task differs'):
        evaluation.evaluate_models(settings, checkpoints, tmp_path, episodes=1, wall_seconds=30)


def test_zero_budget_records_incomplete_without_running_or_biasing_denominators(tmp_path, fake_env):
    result = evaluation.evaluate_models(config(), {'static': None, 'all_reserve': None}, tmp_path,
                                         episodes=2, wall_seconds=0)
    assert not result['complete'] and result['planned_pairs'] == 4
    assert result['complete_pairs'] == result['raw_episodes'] == 0
    assert result['table'] == [] and fake_env.calls == []


def test_membership_change_excludes_casualty_but_detects_same_target_regrouping():
    previous = Grouping((Group(100, (0, 1, 2)), Group(100, (3, 4))))
    pruned = previous.prune((0, 1, 3, 4))
    assert evaluation.changed_members(previous, pruned, (0, 1, 3, 4)) == ()
    regrouped = Grouping((Group(100, (0, 3)), Group(100, (1, 4))))
    assert set(evaluation.changed_members(previous, regrouped, (0, 1, 3, 4))) == {0, 1, 3, 4}
    change = evaluation.grouping_changes(previous, regrouped, (0, 1, 3, 4))
    assert change['task_change'] == 0 and change['team_change'] > 0


def test_checkpoint_weights_are_used_frozen_and_identity_rejects_weight_change(tmp_path, fake_env):
    settings = config()
    settings['eval_scales'] = [4]
    model = CandidateNetwork(**settings['model'])
    checkpoint = tmp_path / 'policy.pt'
    torch.save({'config': settings, 'model': model.state_dict()}, checkpoint)
    directory = tmp_path / 'evaluation'
    result = evaluation.evaluate_models(settings, {'learned': checkpoint, 'all_reserve': None}, directory,
                                         episodes=1, wall_seconds=30)
    assert result['complete'] and result['model_parameters']['learned'] > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    loaded, _, _ = evaluation._load_models(settings, {'learned': checkpoint})
    assert not loaded['learned'].training and not any(p.requires_grad for p in loaded['learned'].parameters())
    with torch.no_grad():
        next(model.parameters()).add_(.1)
    torch.save({'config': settings, 'model': model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match='model/task differs'):
        evaluation.evaluate_models(settings, {'learned': checkpoint, 'all_reserve': None}, directory,
                                     episodes=1, wall_seconds=30)


def test_summary_rejects_duplicate_raw_rows(tmp_path, fake_env):
    evaluation.evaluate_models(config(), {'all_reserve': None}, tmp_path, episodes=1, wall_seconds=30)
    rows = evaluation.read_jsonl(tmp_path / 'evaluation.jsonl')
    with pytest.raises(ValueError, match='Duplicate raw'):
        evaluation.summarize(rows + [rows[0]], ['all_reserve'])


def test_truncated_last_append_is_backed_up_and_resume_stays_valid_jsonl(tmp_path, fake_env):
    settings = config()
    evaluation.evaluate_models(settings, {'all_reserve': None}, tmp_path, episodes=1, wall_seconds=30)
    path = tmp_path / 'evaluation.jsonl'
    preserved = path.read_bytes()
    incomplete = b'{"scale":4,"method":'
    with path.open('ab') as stream:
        stream.write(incomplete)
    result = evaluation.evaluate_models(settings, {'all_reserve': None}, tmp_path, episodes=2, wall_seconds=30)
    assert result['complete_pairs'] == 4 and len(evaluation.read_jsonl(path)) == 4
    assert path.read_bytes().startswith(preserved)
    assert next(tmp_path.glob('evaluation_truncated_tail_*.bin')).read_bytes() == incomplete


def test_real_environment_all_six_methods_produce_report_and_plot(tmp_path):
    settings = configuration('ppo_structured', smoke=True)
    model = CandidateNetwork(**settings['model'])
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': settings, 'model': model.state_dict()}, checkpoint)
    output = tmp_path / 'evaluation'
    methods = {'best': checkpoint, 'latest': checkpoint, 'static': None,
               'dynamic_rule': None, 'compact_rule': None, 'all_reserve': None}
    summary = evaluation.evaluate_models(settings, methods, output, episodes=1, wall_seconds=90)
    assert summary['complete'] and summary['complete_pairs'] == 2
    assert summary['complete_episodes'] == 12 and len(summary['table']) == 12
    for name in ('comparison.png', 'results.csv', 'report.md', 'summary.json', 'evaluation_manifest.json'):
        assert (output / name).is_file()
    assert len(list(output.glob('trace_*.json'))) == 12
    rows = evaluation.read_jsonl(output / 'evaluation.jsonl')
    assert all(row['physical_steps'] <= 50 and row['constraint_violations'] == 0 for row in rows)
    for scale in settings['eval_scales']:
        paired = {row['method']: row for row in rows if row['scale'] == scale}
        # Identical checkpoint bytes, candidate generator, and initial seed.
        assert paired['best']['success'] == paired['latest']['success']
        assert paired['best']['physical_steps'] == paired['latest']['physical_steps']
    print(f"Real evaluation: {len(rows)} episodes, {summary['elapsed_seconds']:.2f} seconds; output={output}")
