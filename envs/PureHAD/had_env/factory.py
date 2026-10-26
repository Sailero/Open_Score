"""The single public factory, with MPE-style named scenarios."""
from had_env.environment import HADParallelEnv, MPEEnv
from had_env.scenarios.defense import Scenario


def make_env(scenario_name="defense", *, api="parallel", red_count=4, blue_count=4,
             target_count=2, red_scouts=0, red_disturbers=0, blue_scouts=0,
             blue_disturbers=0, max_cycles=100, continuous=False, render_mode=None,
             reward_weights=None, initialization="random", evaluate=False,
             target_region=None, seed=None, record_events=False,
             task_mode="survival", target_health=None, spatial_dim=2,
             target_initialization="random", target_positions=None):
    """Create a HAD environment without importing any training or GUI stack.

    ``api='parallel'`` returns PettingZoo's simultaneous dictionary interface.
    ``api='mpe'`` returns a legacy-compatible fixed-list adapter. Team counts
    include all roles. ``task_mode='damage'`` uses immortal targets and raw
    zero-sum target damage; ``target_health`` overrides survival target HP.
    ``spatial_dim=2`` uses planar 9-way or 2-vector actions. Select 3 explicitly
    for native 27-way or 3-vector controls and existing spatial policies.
    ``target_initialization='random'`` samples new targets at reset (training
    default); ``'fixed'`` uses a deterministic layout. Explicit target_positions
    selects fixed coordinates. ``initialization`` separately controls agents.
    See HADParallelEnv for exact observation/reward semantics.
    """
    if scenario_name not in ("defense", "defense.py"):
        raise ValueError(f"Unknown scenario {scenario_name!r}; available: defense")
    if api not in ("parallel", "mpe"):
        raise ValueError("api must be 'parallel' or 'mpe'")
    scenario = Scenario(red_count, blue_count, target_count, red_scouts, red_disturbers,
                        blue_scouts, blue_disturbers, initialization, evaluate, target_region,
                        task_mode, target_health, spatial_dim, target_initialization, target_positions)
    env = HADParallelEnv(scenario, max_cycles=max_cycles, continuous=continuous,
                         render_mode=render_mode, reward_weights=reward_weights,
                         seed=seed, record_events=record_events)
    return MPEEnv(env) if api == "mpe" else env


def parallel_env(**kwargs):
    """Conventional PettingZoo-style constructor for the defense scenario."""
    return make_env(api="parallel", **kwargs)
