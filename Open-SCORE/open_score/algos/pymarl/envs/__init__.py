"""ALMA environment registry; SC2 is optional and loaded only on selection."""
from functools import partial


def env_fn(name, **kwargs):
    if name == "had":
        from open_score.envs import make_entity_env
        return make_entity_env(**kwargs)
    if name == "ff":
        from .firefighters import FireFightersEnv
        return FireFightersEnv(**kwargs)
    if name in {"sc2", "sc2custom", "sc2multiarmy"}:
        from .starcraft2 import StarCraft2Env, StarCraft2CustomEnv, StarCraft2MultiArmyEnv
        return {"sc2": StarCraft2Env, "sc2custom": StarCraft2CustomEnv,
                "sc2multiarmy": StarCraft2MultiArmyEnv}[name](**kwargs)
    raise ValueError(name)


REGISTRY = {name: partial(env_fn, name) for name in ("had", "ff", "sc2", "sc2custom", "sc2multiarmy")}
from .firefighters import scenarios as s_REGISTRY
