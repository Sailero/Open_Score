"""V6: fixed-rule allocation and grouping, configured by one experiment file."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from open_score.research_v5.protocol import stable_seed

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / 'configs/research_v6.yaml'
METHODS = {
    'G1': 'BLOTTO_Count', 'G2': 'BLOTTO_Group',
    'R1': 'ALMA_Alloc', 'R2': 'MAPPO_Intent',
    'R3': 'ALMA_Group', 'R4': 'ALMA_S2',
}
RL_METHODS = tuple(METHODS[k] for k in ('R1', 'R2', 'R3', 'R4'))


def load_config(path=None):
    import yaml
    config = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding='utf-8'))
    scenes = config['scenarios']
    expected = [('A', 10, 10, 2), ('B', 10, 10, 3), ('C', 20, 20, 2),
                ('D', 20, 20, 3), ('E', 10, 6, 2)]
    if [(s['id'], s['red'], s['blue'], s['targets']) for s in scenes] != expected:
        raise ValueError('V6.4 requires the five declared scenarios A–E')
    if any(abs(float(s['weight'])-.2) > 1e-8 for s in scenes):
        raise ValueError('Training scenarios must be equally weighted by episode')
    if config['seeds']['initializations'] != [20260911, 20260912, 20260913]:
        raise ValueError('Use the three declared initialization seeds')
    if config['reward']['gamma'] != 1 or config['reward']['shaping']['coefficient'] != .5:
        raise ValueError('V6.4 uses gamma=1 and potential coefficient=0.5')
    if config['training']['physical_steps_per_method_seed'] != 5_000_000:
        raise ValueError('Each RL method/seed has 5M physical steps across all scenarios')
    if set(config['methods']) != set(METHODS.values()):
        raise ValueError('The six configured method names must match the V6 methods')
    pairs = config['s2']['data']['single_target']['count_pairs']
    if len(pairs) != 60 or len({tuple(pair) for pair in pairs}) != 60:
        raise ValueError('S2 needs exactly 60 distinct predefined count pairs')
    return config


@dataclass(frozen=True)
class EpisodeSpec:
    family_id: str
    scenario_id: str
    red_count: int
    blue_count: int
    target_count: int
    opening_seed: int
    opponent_seed: int
    policy_seed: int
    split: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def episode_spec(config, index, split='train', seed=20260911, method='shared'):
    """Pair evaluation openings across methods/models, with private policy RNG."""
    index = int(index)
    scene = config['scenarios'][index % len(config['scenarios'])]
    replicate = index // len(config['scenarios'])
    # Formal openings are independent of training initialization and S2 families.
    origin = f'train:{seed}:{method}' if split == 'train' else 'formal'
    family = f'v6.4:{origin}:{split}:{scene["id"]}:{replicate}'
    return EpisodeSpec(family, scene['id'], scene['red'], scene['blue'], scene['targets'],
                       stable_seed(family, 'opening'), stable_seed(family, 'opponent'),
                       stable_seed(family, seed, method, 'policy'), split)


def make_episode_env(config, spec):
    from .environment import V6Env
    environment = config['environment']
    return V6Env(spec.red_count, spec.blue_count, spec.target_count,
                 seed=spec.opening_seed, opponent_seed=spec.opponent_seed,
                 max_steps=environment['max_physical_steps'],
                 command_interval=environment['command_interval'],
                 reward_coefficient=config['reward']['shaping']['coefficient'])


__all__ = ['METHODS', 'RL_METHODS', 'EpisodeSpec', 'load_config', 'episode_spec']
