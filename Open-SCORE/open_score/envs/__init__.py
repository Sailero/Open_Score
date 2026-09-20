"""Public environment and native-rule boundary."""
from .entity_env import HADEntityEnv, NativeEntityEnv
from .scales import Scale, ScaleSampler, SCALE_POOLS, TRAIN_POOL, TEST_POOL, VALIDATION_POOL
from .had_wrapper import (
    HADWrapper, HADEnv, HADStage3Adapter, DecisionState, Entity, Group, Grouping,
    ACCELERATION_PRIMITIVES, PLANAR_NATIVE_IDS, NATIVE_TO_PLANAR,
    had_config, PHYSICS_PROTOCOL, make_env, parallel_env,
    KnownOpponentEnv, RulePolicy, RuleExecutor, make_grouping_env,
    decode_counts, grand_grouping, rule_grouping, sample,
    build_decision_state, EpisodeDiagnostics, trajectory_frame,
)


def make_entity_env(**kwargs):
    return HADEntityEnv(**kwargs)


__all__ = [name for name in globals() if not name.startswith("_")]
