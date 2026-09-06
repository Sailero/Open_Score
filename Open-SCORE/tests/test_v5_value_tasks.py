"""Algorithmic invariants for paired supervision and merge-only DQN."""
import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from open_score.grouping.domain import Group, Grouping
from open_score.research_v4.actions import candidate_pool, partition_key, rule_grouping
from open_score.research_v4.environment import make_env
from open_score.research_v5.paired import (PairedPolicy, group_rows, label_statistics,
    state_balanced_loss, train_model, load_policy)
from open_score.research_v5.tasks.t5_bridge_grouping import (
    MergePolicy, PartitionQNetwork, dqn_update, merge_actions, merge_partition, singleton_partition,
    due_validation_fractions)


@pytest.fixture
def state():
    torch.set_num_threads(1)
    env = make_env(8, 4, seed=20260907)
    result = env.reset(seed=20260907)
    yield result
    env.close()


def paired_groups(state):
    actions = candidate_pool(state, budget=4)
    rule_key = partition_key(rule_grouping(state))
    rows = []
    for index, action in enumerate(actions):
        outcomes = [0, 1] if partition_key(action) == rule_key else [index % 2, index % 2]
        rows.append({'state': state.to_dict(), 'action': action.to_dict(),
            'branch_seeds': [41, 42], 'outcomes': outcomes, 'y': np.mean(outcomes),
            'family_id': 'train:one'})
    return rows, group_rows(rows)


def test_merge_actions_reach_all_bell_partitions_and_never_change_targets():
    initial = Grouping(tuple(Group(1, (i,)) for i in range(4)), (4,))
    pending = [initial]; visited = set()
    while pending:
        partition = pending.pop()
        if partition_key(partition) in visited:
            continue
        visited.add(partition_key(partition))
        assert partition.assignment() == initial.assignment()
        assert merge_actions(partition)[0] is None
        for action in merge_actions(partition)[1:]:
            following = merge_partition(partition, action)
            assert len(following.groups) == len(partition.groups)-1
            pending.append(following)
    assert len(visited) == 15
    with pytest.raises(ValueError):
        merge_partition(Grouping((Group(0, (0,)), Group(1, (1,)))), (0, 1))


def test_paired_dataset_checks_crn_and_keeps_zero_and_negative_labels(state):
    rows, groups = paired_groups(state)
    group = groups[0]
    assert group['advantage'][group['anchor']] == 0.
    assert (group['advantage'] < 0).any()
    assert label_statistics(groups)['zero_difference_branches'] > 0
    broken = copy.deepcopy(rows); broken[-1]['branch_seeds'] = [51, 52]
    with pytest.raises(ValueError, match='share paired seeds'):
        group_rows(broken)
    zeros = copy.deepcopy(rows)
    for row in zeros:
        row['outcomes'] = [0, 0]; row['y'] = 0.
    assert label_statistics(group_rows(zeros))['all_zero_states'] == 1


class ConstantModel(nn.Module):
    def __init__(self, value=0.):
        super().__init__(); self.score = nn.Parameter(torch.tensor(float(value)))
    def forward(self, states, pools):
        return self.score.expand(len(states), max(map(len, pools))), self.score.expand(len(states))


def test_state_equal_loss_and_explicit_advantage_anchor(state):
    _, groups = paired_groups(state)
    group = groups[0]
    model = ConstantModel(.7)
    short = dict(group)
    short.update(actions=[group['actions'][group['anchor']]], y=np.array([.5]),
                 advantage=np.array([0.]), anchor=0)
    actual = state_balanced_loss(model, [group, short], 'bce')
    expected = (F.binary_cross_entropy_with_logits(torch.full((len(group['actions']),), .7),
        torch.tensor(group['y'])) + F.binary_cross_entropy_with_logits(torch.tensor([.7]), torch.tensor([.5])))/2
    assert float(actual.detach()) == pytest.approx(float(expected))
    assert float(state_balanced_loss(model, [short], 'adv').detach()) == 0.


def test_same_tie_falls_back_to_rule_in_both_arms_with_no_extra_query(state):
    rule = rule_grouping(state)
    for kind in ('bce', 'adv'):
        policy = PairedPolicy(ConstantModel(0.), kind, budget=8)
        assert policy.act(state) == rule
        assert policy.last_trace['scored_candidates'] <= 8
        assert policy.last_trace['scored_candidates'] == policy.last_trace['evaluations']
        assert policy.last_trace['rule_included']
        assert not policy.last_trace['accepted_replacement']
        probabilities = policy.predict_candidate_probabilities(state, candidate_pool(state, 4))
        if kind == 'bce':
            assert np.all(probabilities == .5)
        else:
            assert probabilities is None


def test_merge_policy_and_terminal_dqn_bootstrap(state):
    partition = singleton_partition(rule_grouping(state))
    model = PartitionQNetwork(hidden_dim=16)
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    policy = MergePolicy(model)
    assert policy.act(state) == partition  # exact tie STOP
    target = copy.deepcopy(model)
    with torch.no_grad():
        target.action_head[-1].bias.fill_(100.)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch = [{'state': state, 'partition': partition, 'following': partition,
              'action': None, 'reward': 0., 'done': True}]
    metrics = dqn_update(model, target, optimizer, batch)
    assert metrics['td_loss'] == 0.  # STOP cannot bootstrap the 100-valued target
    next_action = merge_actions(partition)[1]
    following = merge_partition(partition, next_action)
    batch = [{'state': state, 'partition': partition, 'following': following,
              'action': next_action, 'reward': .5, 'done': False}]
    metrics = dqn_update(model, target, optimizer, batch)
    assert metrics['td_loss'] == pytest.approx(100.5**2)


def test_paired_epoch_resume_preserves_optimizer_and_model(state, tmp_path):
    _, groups = paired_groups(state)
    config = {'hidden_dim': 16, 'heads': 4, 'layers': 1}
    first, result = train_model(groups, groups, tmp_path, kind='adv', seed=4,
                               epochs=2, batch_states=1, model_config=config)
    before = {key: value.clone() for key, value in first.state_dict().items()}
    resumed, again = train_model(groups, groups, tmp_path, kind='adv', seed=4,
                                epochs=2, batch_states=1, model_config=config)
    assert again['updates'] == result['updates'] == 2
    assert all(torch.equal(before[key], value) for key, value in resumed.state_dict().items())
    loaded = load_policy(tmp_path/'latest.pt', budget=8)
    loaded.act(state).validate(state.ids('red'), state.ids('targets'), max_members=None)
    saved = torch.load(tmp_path/'latest.pt', weights_only=False)
    assert saved['optimizer']['state'] and saved['rng'] and saved['epoch'] == 2


def test_paired_resume_after_interruption_matches_uninterrupted_updates(state, tmp_path):
    _, groups = paired_groups(state)
    config = {'hidden_dim': 16, 'heads': 4, 'layers': 1}
    def interrupt(row):
        if row['epoch'] == 1:
            raise RuntimeError('simulated shutdown after committed epoch')
    with pytest.raises(RuntimeError, match='simulated shutdown'):
        train_model(groups, groups, tmp_path/'interrupted', kind='bce', seed=91,
                    epochs=2, batch_states=1, model_config=config, progress=interrupt)
    resumed, result = train_model(groups, groups, tmp_path/'interrupted', kind='bce', seed=91,
                                 epochs=2, batch_states=1, model_config=config)
    reference, _ = train_model(groups, groups, tmp_path/'reference', kind='bce', seed=91,
                               epochs=2, batch_states=1, model_config=config)
    assert result['updates'] == 2
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in resumed.state_dict().items())


def test_bridge_quarter_validation_is_persisted_and_not_repeated(tmp_path, monkeypatch):
    from open_score.grouping.storage import atomic_json, read_jsonl, sha256
    from open_score.research_v5.protocol import defaults, episode_spec
    from open_score.research_v5.runtime import TaskContext
    from open_score.research_v5.tasks import t5_bridge_grouping as module
    assert due_validation_fractions(599, 2400, []) == []
    assert due_validation_fractions(600, 2400, []) == [.25]
    assert due_validation_fractions(1200, 2400, [.25]) == [.5]
    config = defaults(smoke=True)
    config.update(device='cpu', cells=[[8, 4]])
    config['t5'].update(states=1, episodes=4, branches=1, batch_size=2)
    (tmp_path/'shared').mkdir()
    for split, name in [('validation', 'validation_manifest.json'), ('test', 'evaluation_manifest.json')]:
        atomic_json(tmp_path/'shared'/name, {'episodes': [episode_spec(config['seed'], 0, split, 'shared', config['cells']).to_dict()]})
    empty = {'diagnostic': {'states': 0, 'simulated_physical_steps': 0, 'selected_vs_rule_verification_gain': 0.}}
    monkeypatch.setattr(TaskContext, 'diagnose', lambda *args, **kwargs: empty)
    monkeypatch.setattr(module, 'partition_diagnostic_report', lambda output: {'directions_vs_grand': {
        'learned': {'better_states': 0, 'worse_states': 0, 'tie_states': 0}}})
    first = module.run(TaskContext(tmp_path, 'T5', config))
    assert first['validated_fractions'] == [.25, .5, .75, 1.]
    assert first['validation_episodes'] == 4  # one actual native episode at each node in this test
    before = sha256(tmp_path/'T5/latest.pt')
    resumed = module.run(TaskContext(tmp_path, 'T5', config))
    assert resumed['validation_episodes'] == 4
    assert sha256(tmp_path/'T5/latest.pt') == before
    assert len(read_jsonl(tmp_path/'T5/validation.jsonl')) == 4
    assert len(list((tmp_path/'T5').glob('construction_*.pt'))) == 4
