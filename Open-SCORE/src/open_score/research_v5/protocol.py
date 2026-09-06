"""Immutable episode families and experiment quotas, independent of scheduling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess

VERSION = 'known-rule-mixed-v5.1'
ROOT = Path(__file__).resolve().parents[3]
CELLS = tuple((n, b) for n in (8, 12, 16, 24, 32) for b in (n//2, n*3//4, n))
TASKS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')


def stable_seed(*parts):
    raw = json.dumps(parts, sort_keys=True, separators=(',', ':'), default=str).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'little') % (2**31-1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class EpisodeSpec:
    family_id: str
    red_count: int
    blue_count: int
    opening_seed: int
    opponent_seed: int
    split: str
    opponent: str = 'reactive'
    protocol_version: str = VERSION

    def __post_init__(self):
        if self.red_count < 1 or self.blue_count < 1:
            raise ValueError('Initial Red and Blue counts must be positive')
        if self.opponent != 'reactive' or self.protocol_version != VERSION:
            raise ValueError('Unsupported frozen opponent/protocol')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def episode_spec(seed, index, split='train', namespace='shared', cells=CELLS):
    red, blue = cells[int(index) % len(cells)]
    replicate = int(index)//len(cells)
    family = f'{VERSION}:{seed}:{namespace}:{split}:{red}v{blue}:{replicate}'
    return EpisodeSpec(family, red, blue, stable_seed(family, 'opening'),
                       stable_seed(family, 'opponent'), split)


def defaults(smoke=False, multiplier=1., seed=20260907):
    q = float(multiplier)
    result = dict(version=VERSION, seed=int(seed), smoke=bool(smoke), multiplier=q,
                  cells=[list(x) for x in CELLS], opponent='reactive', executor='rule_group_v1',
                  max_steps=50, command_interval=5, gamma=1., device='auto', threads=1,
                  cpu_quotas=dict(zip(TASKS, (4, 4, 2, 3, 3, 2))),
                  model=dict(hidden_dim=128, layers=2, heads=4),
                  eval_per_cell=100, validation_per_cell=10, bc_episodes=300, bc_epochs=5,
                  diagnostic_states=120, own_diagnostic_states=30,
                  selection_branches=16, verification_branches=32,
                  latency_states=30, teacher_validation_episodes=30,
                  t1=dict(candidates=8, branches=8, control_episodes=30),
                  t2=dict(iterations=64, depth=3, k_action=1.5, alpha_action=.5,
                          k_state=1., alpha_state=.5, exploration=1.),
                  t3=dict(steps=int(1_000_000*q), rollout=1024, batch_size=128, epochs=4,
                          learning_rate=3e-4, gae_lambda=.95, clip=.2, max_gradient_norm=.5,
                          target_kl=.03, entropy_start=.02, entropy_end=.005, candidates=32,
                          validation_fractions=[0., .1, .25, .5, .75, 1.]),
                  t4=dict(rounds=4, states_per_round=int(300*q), candidates=8, branches=8,
                          epochs=10, batch_size=16, learning_rate=3e-4),
                  t5=dict(states=int(300*q), episodes=int(2400*q), branches=4, batch_size=128,
                          buffer_size=10000, target_interval=100, learning_rate=3e-4,
                          max_gradient_norm=40., epsilon_steps=10000, epsilon_end=.01),
                  t6=dict(train_states=int(1200*q), validation_states=300, candidates=8,
                          branches=16, epochs=40, batch_size=16, learning_rate=3e-4,
                          search_budget=64, thresholds=[0., .005, .01, .02]))
    if smoke:
        result.update(model=dict(hidden_dim=32, layers=1, heads=4),
                      eval_per_cell=1, validation_per_cell=1, bc_episodes=15, bc_epochs=1,
                      diagnostic_states=15, own_diagnostic_states=2, selection_branches=2,
                      verification_branches=2, latency_states=3, teacher_validation_episodes=3,
                      cpu_quotas={t: 1 for t in TASKS})
        result['t1'].update(candidates=4, branches=2, control_episodes=3)
        result['t2'].update(iterations=4, depth=2)
        result['t3'].update(steps=120, rollout=32, batch_size=16, epochs=1, candidates=8,
                            validation_fractions=[0., 1.])
        result['t4'].update(rounds=2, states_per_round=4, candidates=4, branches=2, epochs=1, batch_size=2)
        result['t5'].update(states=4, episodes=12, branches=2, batch_size=4, target_interval=4)
        result['t6'].update(train_states=4, validation_states=3, candidates=4, branches=2,
                            epochs=2, batch_size=2, search_budget=8)
    return result


def source_identity():
    paths = []
    for folder in ('src', 'configs/research_v5', 'scripts'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.ps1'))
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode()).hexdigest() for p in sorted(paths)}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(git_commit=commit, source_hash=digest(files), source_files=files)
