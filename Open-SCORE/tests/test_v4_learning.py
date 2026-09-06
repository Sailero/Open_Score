"""Checks for independent PPO, true terminal targets and exact recovery."""
from dataclasses import replace
import copy

import numpy as np
import pytest
import torch
from torch import nn

from open_score.grouping.domain import DecisionState, Entity, Group, Grouping
from open_score.grouping.storage import random_state, restore_random_state
from open_score.research_v4.learning import (CandidateNetwork, StateValueNetwork,
    critic_update, ppo_update, double_q_update, imitation_update, vector_gae)
from open_score.research_v4.training import Trainer, configuration, load_policy, train


@pytest.fixture(autouse=True)
def deterministic_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(129)
    np.random.seed(129)
    yield
    torch.set_num_threads(threads)


def state_and_pool(size=6):
    red = tuple(Entity(i, (-500.+i*100, i*80., 100), (20., 0., 0.), 1.) for i in range(size))
    blue = tuple(Entity(100+i, (1700.-i*90, -200.+i*70, 100), (-100., 0., 0.), 1.) for i in range(size))
    targets = (Entity(0, (-2100., -650., 100), (0., 0., 0.), 1.2),
               Entity(1, (-2100., 650., 100), (0., 0., 0.), 1.2))
    grand = Grouping((Group(0, tuple(range(size))),))
    split = Grouping((Group(0, tuple(range(0, size, 2))), Group(1, tuple(range(1, size, 2)))))
    state = DecisionState(0, 50, 'reactive', red, blue, targets, grand)
    return state, [grand, split]


def tiny():
    return CandidateNetwork(hidden_dim=16, heads=4, layers=1)


def test_unbounded_members_permutation_invariance_and_action_sensitivity():
    state, pool = state_and_pool(9)
    model = tiny()
    scores, values = model([state], [pool])
    assert scores.shape == (1, 2) and values.shape == (1,)
    assert abs(float((scores[0, 0]-scores[0, 1]).detach())) > 1e-7
    renamed = replace(state, red=tuple(replace(e, id=e.id+500) for e in reversed(state.red)),
                      blue=tuple(reversed(state.blue)),
                      previous=Grouping(tuple(Group(g.target, tuple(i+500 for i in g.members)) for g in state.previous.groups)))
    renamed_pool = [Grouping(tuple(Group(g.target, tuple(i+500 for i in g.members)) for g in grouping.groups)) for grouping in pool]
    renamed_scores, renamed_values = model([renamed], [renamed_pool])
    torch.testing.assert_close(scores, renamed_scores, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(values, renamed_values, atol=1e-6, rtol=1e-5)
    padded, _ = model([state, state], [pool[:1], pool])
    assert padded[0, 1] == -1e9


def test_critic_updates_cannot_move_actor_logits():
    state, pool = state_and_pool()
    actor, critic = tiny(), StateValueNetwork(hidden_dim=16, heads=4, layers=1)
    optimizer = torch.optim.Adam(critic.parameters(), lr=.01)
    with torch.no_grad():
        before = actor([state], [pool])[0].clone()
        old_value = critic([state], [pool])[1].clone()
    for _ in range(4):
        critic_update(critic, optimizer, [state], [pool], [10.], max_gradient_norm=10.)
    with torch.no_grad():
        after = actor([state], [pool])[0]
        new_value = critic([state], [pool])[1]
    assert torch.equal(before, after)
    assert not torch.equal(old_value, new_value)
    assert all(p.grad is None for p in actor.parameters())


def ppo_rows(actor, critic, wrong_log=False):
    state, pool = state_and_pool()
    states = [state, replace(state, step=10)]
    with torch.no_grad():
        logits, _ = actor(states, [pool]*2)
        _, values = critic(states, [pool]*2)
        logp = torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([0, 1]))
    return [dict(state=states[i], candidates=pool, next_state=replace(states[i], step=50),
        next_candidates=pool, action=i, env=i, done=True, reward=float(i),
        value=float(values[i]), log_prob=float(logp[i])-(2. if wrong_log else 0.)) for i in range(2)]


def test_ppo_logs_rejected_kl_and_continues_independent_value_fit():
    actor, critic = tiny(), StateValueNetwork(hidden_dim=16, heads=4, layers=1)
    before = copy.deepcopy(actor.state_dict())
    metrics = ppo_update(actor, critic, torch.optim.Adam(actor.parameters(), lr=.001),
        torch.optim.Adam(critic.parameters(), lr=.001), ppo_rows(actor, critic, wrong_log=True),
        configuration(dict(smoke=True)))
    assert metrics['actor_updates'] == 0
    assert metrics['critic_updates'] > 0
    assert metrics['rejected_kl'] > .03 and metrics['kl_early_stop']
    assert all(torch.equal(before[k], actor.state_dict()[k]) for k in before)


def test_ppo_real_update_and_terminal_gae():
    actor, critic = tiny(), StateValueNetwork(hidden_dim=16, heads=4, layers=1)
    before = actor.encoder.layers[0].self_attn.in_proj_weight.detach().clone()
    metrics = ppo_update(actor, critic, torch.optim.Adam(actor.parameters(), lr=.001),
        torch.optim.Adam(critic.parameters(), lr=.001), ppo_rows(actor, critic),
        configuration(dict(smoke=True)))
    assert metrics['actor_updates'] > 0
    assert not torch.equal(before, actor.encoder.layers[0].self_attn.in_proj_weight)
    rows = [dict(env=0, reward=0., value=.2, done=False), dict(env=0, reward=1., value=.3, done=True)]
    advantages, returns = vector_gae(rows, [.3, 1000.], lam=1.)
    np.testing.assert_allclose(returns, [1., 1.])


class TableNetwork(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = nn.Parameter(torch.tensor(table, dtype=torch.float32))

    def forward(self, states, pools):
        scores = self.table[torch.tensor(states)]
        return scores, scores.sum(1)*0.


def test_double_q_online_choice_target_value_and_true_terminal():
    online = TableNetwork([[1., 2.], [4., 3.]])
    target = TableNetwork([[0., 0.], [10., 100.]]).requires_grad_(False)
    rows = [dict(state=0, candidates=[0, 1], action=1, reward=1., done=False, next_state=1, next_candidates=[0, 1]),
            dict(state=0, candidates=[0, 1], action=0, reward=-2., done=True, next_state=1, next_candidates=[0, 1])]
    metrics = double_q_update(online, target, torch.optim.SGD(online.parameters(), lr=.1), rows,
                             dict(max_gradient_norm=10.))
    assert metrics['target_mean'] == pytest.approx(4.5)
    assert metrics['loss'] == pytest.approx(5.5)
    assert all(p.grad is None for p in target.parameters())


def test_teacher_all_terminal_ties_create_no_false_label():
    state, pool = state_and_pool()
    actor = tiny()
    before = copy.deepcopy(actor.state_dict())
    optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
    metrics = imitation_update(actor, optimizer, [dict(state=state, candidates=pool, scores=[0., 0.])])
    assert metrics['optimizer_steps'] == 0
    assert all(torch.equal(before[k], actor.state_dict()[k]) for k in before)
    with torch.no_grad():
        old = float(actor([state], [pool])[0][0].softmax(-1)[1])
    for _ in range(5):
        imitation_update(actor, optimizer, [dict(state=state, candidates=pool, scores=[0., 1.])])
    with torch.no_grad():
        new = float(actor([state], [pool])[0][0].softmax(-1)[1])
    assert new > old


def test_checkpoint_restores_physical_world_pending_rollout_and_random_stream(tmp_path, monkeypatch):
    monkeypatch.setattr('open_score.research_v4.training.protocol_fingerprint', lambda: 'fixture-source')
    config = dict(smoke=True, steps=200, seconds=30., envs=1, candidate_budget=4)
    trainer = Trainer(config, tmp_path)
    slot = trainer.new_slot()
    trainer.slots.append(slot)
    state, reward, done, pools = trainer._advance(slot, slot['candidates'][1])
    slot.update(state=state, candidates=pools)
    trainer.rollout.append(dict(marker='pending-unoptimized-experience'))
    trainer.save()
    expected_random = (np.random.random(), torch.rand(2))
    action = slot['candidates'][0]
    expected = slot['env'].step(action)
    restored = Trainer(config, tmp_path, resume=True)
    actual_random = (np.random.random(), torch.rand(2))
    assert expected_random[0] == actual_random[0]
    assert torch.equal(expected_random[1], actual_random[1])
    assert restored.rollout == trainer.rollout
    actual = restored.slots[0]['env'].step(action)
    assert actual[0].to_dict() == expected[0].to_dict()
    assert actual[1:] == expected[1:]
    with pytest.raises(ValueError, match='configuration'):
        Trainer(dict(config, actor_learning_rate=.1), tmp_path, resume=True)
    with pytest.raises(ValueError, match='budgets'):
        Trainer(dict(config, steps=100), tmp_path, resume=True)
    for item in (trainer, restored):
        for slot in item.slots:
            slot['env'].close()


@pytest.mark.parametrize('route', ['r1_ppo', 'r3_ddqn'])
def test_small_real_train_updates_model_and_resumes_completed_run(tmp_path, monkeypatch, route):
    monkeypatch.setattr('open_score.research_v4.training.protocol_fingerprint', lambda: 'fixture-source')
    config = dict(route=route, smoke=True, steps=90, seconds=30., envs=1,
        candidate_budget=4, rollout_events=4, batch_size=4, replay_warmup=4,
        q_update_every=4, validation_episodes=1, validation_scales=[4])
    result = train(config, tmp_path)
    assert result['complete'] and result['counters']['updates'] > 0
    assert result['counters']['physical_steps'] == result['counters']['learner_physical_steps']
    first = torch.load(tmp_path/'initialized.pt', weights_only=False)
    final = torch.load(tmp_path/'latest.pt', weights_only=False)
    assert any(not torch.equal(first['actor'][k], final['actor'][k]) for k in first['actor'])
    policy = load_policy(tmp_path/'best.pt')
    state, _ = state_and_pool()
    policy.act(state).validate(state.ids('red'), state.ids('targets'), max_members=None)
    again = train(config, tmp_path, resume=True)
    assert again == result


def test_terminal_without_targets_never_generates_a_fictitious_decision(tmp_path, monkeypatch):
    monkeypatch.setattr('open_score.research_v4.training.protocol_fingerprint', lambda: 'fixture-source')
    state, pools = state_and_pool()
    terminal = replace(state, step=50, targets=tuple(replace(t, health=0.) for t in state.targets))
    class TerminalEnv:
        def step(self, action):
            return terminal, 0., True, dict(delta=5, success=False)
    trainer = Trainer(dict(smoke=True), tmp_path)
    monkeypatch.setattr('open_score.research_v4.actions.candidate_pool',
                        lambda *args, **kwargs: pytest.fail('terminal action generation'))
    slot = dict(env=TerminalEnv(), state=state, candidates=pools, native_return=0., scale=6, seed=1)
    following, reward, done, candidates = trainer._advance(slot, pools[0])
    assert done and reward == 0.
    candidates[0].validate(following.ids('red'), following.ids('targets'), max_members=None)
    scores, values = trainer.actor([following], [candidates])
    assert torch.isfinite(scores).all() and torch.isfinite(values).all()


def test_shared_teacher_pretrains_only_and_excludes_validation_families(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr('open_score.research_v4.training.protocol_fingerprint', lambda: 'fixture-source')
    state, pool = state_and_pool()
    shared = tmp_path/'shared'/'global_data'/'families'
    shared.mkdir(parents=True)
    rows = [dict(state=state.to_dict(), action=action.to_dict(), state_id='train:0',
                 y=float(index), physical_steps=[50], continuation_version='rule_grouping_v1')
            for index, action in enumerate(pool)]
    (shared/'000000.json').write_text(json.dumps(dict(complete=True, split='train', rows=rows)), encoding='utf-8')
    # A malformed validation row is deliberately unreadable as training data.
    (shared/'000001.json').write_text(json.dumps(dict(complete=True, split='validation', rows=[{'forbidden': True}])), encoding='utf-8')
    config = dict(route='r2_teacher_ppo', smoke=True, steps=100, seconds=30., envs=1,
        candidate_budget=4, rollout_events=4, batch_size=4, teacher_pretrain_updates=3,
        validation_episodes=1, validation_scales=[4], shared_dir=str(tmp_path/'shared'))
    result = train(config, tmp_path/'run')
    assert result['teacher_source'] == 'shared_terminal_data'
    assert result['counters']['teacher_labels'] == 1
    assert result['counters']['teacher_pretrain_updates'] == 3
    assert result['counters']['actor_updates'] > 0
    assert result['counters']['teacher_simulation_steps'] == 0
    assert result['counters']['shared_teacher_simulation_steps'] == 100
    logged = [json.loads(line) for line in (tmp_path/'run'/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all('teacher_loss' not in row for row in logged if row['phase'] == 'training')


def test_budget_extension_preserves_annealing_and_pending_policy_likelihood(tmp_path, monkeypatch):
    monkeypatch.setattr('open_score.research_v4.training.protocol_fingerprint', lambda: 'fixture-source')
    config = dict(smoke=True, steps=100, seconds=60., envs=1)
    original = Trainer(config, tmp_path)
    rows = ppo_rows(original.actor, original.critic)
    original.rollout = rows
    original.counts['physical_steps'] = 75
    original.save()
    resumed = Trainer(dict(config, steps=200, seconds=120.), tmp_path, resume=True)
    assert resumed.annealing_budget == dict(steps=100, seconds=60.)
    assert resumed.fraction() == pytest.approx(.75, abs=.01)
    with torch.no_grad():
        scores, _ = resumed.actor([r['state'] for r in resumed.rollout], [r['candidates'] for r in resumed.rollout])
        logp = torch.distributions.Categorical(logits=scores).log_prob(torch.tensor([r['action'] for r in resumed.rollout]))
    torch.testing.assert_close(logp, torch.tensor([r['log_prob'] for r in rows]))
