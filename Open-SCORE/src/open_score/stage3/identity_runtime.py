"""HAD bridge for identity-aware coalition-structured Blotto.

This module deliberately keeps the Stage-3 pure action and its physical
execution identical: a coalition stored in slot ``(target, channel)`` is the
exact coalition passed to the frozen Stage-1 controller.  There is no later
count-to-ID matching or anonymous group splitting step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from open_score.envs import HADStage3Adapter
from open_score.stage2 import HADCanonicalizer

from .identity_blotto import (
    EngagementSlot,
    IdentityAction,
    IdentityBlottoGame,
    LocalPayoffRequest,
    make_engagement_slots,
    round_robin_identity_action,
    spatial_candidate_coalitions,
)
from .payoff import FrozenStage2Payoff, survival_probability_to_utility
from .runtime import LocalSubgame


class Stage2IdentityPayoffOracle:
    """Evaluate exact ID coalitions with the frozen permutation-invariant S2."""

    def __init__(
        self,
        adapter: HADStage3Adapter,
        predictor: FrozenStage2Payoff,
        slots: Sequence[EngagementSlot],
        *,
        blue_style: Optional[str] = None,
        batch_size: int = 512,
        utility_mode: str = "joint_survival_log_probability",
        risk_epsilon: float = 1e-6,
    ) -> None:
        self.adapter = adapter
        self.predictor = predictor
        self.slots = tuple(slots)
        self.blue_style = (
            adapter.blue_rule_style if blue_style is None else str(blue_style)
        )
        self.batch_size = int(batch_size)
        self.utility_mode = str(utility_mode)
        self.risk_epsilon = float(risk_epsilon)
        self.canonicalizer = HADCanonicalizer()
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def evaluate(self, requests: Sequence[LocalPayoffRequest]) -> np.ndarray:
        if not requests:
            return np.empty(0, dtype=np.float64)
        states = []
        rosters: list[tuple[int, int]] = []
        for raw_slot, raw_red, raw_blue in requests:
            slot_index = int(raw_slot)
            if not 0 <= slot_index < len(self.slots):
                raise ValueError("local payoff request uses an unknown slot")
            red_ids = tuple(map(int, raw_red))
            blue_ids = tuple(map(int, raw_blue))
            if not 1 <= len(red_ids) <= 4 or not 1 <= len(blue_ids) <= 4:
                raise ValueError("learned identity payoff queries must lie in 1..4v1..4")
            target_id = self.slots[slot_index].target_id
            entities = self.adapter.local_state_entities(
                target_id,
                red_ids,
                blue_ids,
                local_step=self.adapter.step_count,
            )
            states.append(self.canonicalizer.to_entity_set(entities))
            rosters.append((len(red_ids), len(blue_ids)))
        probability = self.predictor.predict_red_win(
            states,
            styles=[self.blue_style] * len(states),
            rosters=rosters,
            batch_size=self.batch_size,
        )
        return survival_probability_to_utility(
            probability,
            utility_mode=self.utility_mode,
            risk_epsilon=self.risk_epsilon,
        )


@dataclass(frozen=True)
class IdentityEventGameBuild:
    game: IdentityBlottoGame
    red_alive_ids: Tuple[int, ...]
    blue_alive_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    slots: Tuple[EngagementSlot, ...]
    initial_red: Tuple[IdentityAction, ...]
    initial_blue: Tuple[IdentityAction, ...]
    full_coalition_domain: bool
    red_candidate_columns: int
    blue_candidate_columns: int


def _alive_ids(adapter: HADStage3Adapter, side: str) -> Tuple[int, ...]:
    return tuple(
        int(agent_id)
        for agent_id, state in adapter.agent_states(side).items()
        if bool(state["alive"])
    )


def _positions(adapter: HADStage3Adapter, side: str) -> Dict[int, np.ndarray]:
    return {
        int(agent_id): np.asarray(state["position"], dtype=np.float64)
        for agent_id, state in adapter.agent_states(side).items()
        if bool(state["alive"])
    }


def _seed_actions(
    ids: Sequence[int], slots: Sequence[EngagementSlot]
) -> Tuple[IdentityAction, ...]:
    targets = tuple(dict.fromkeys(slot.target_id for slot in slots))
    seeds = [
        round_robin_identity_action(ids, slots, offset=offset)
        for offset in range(min(2, len(targets)))
    ]
    seeds.extend(
        round_robin_identity_action(ids, slots, concentrated_target=target)
        for target in (targets[0], targets[-1])
    )
    return tuple(dict.fromkeys(seeds))


def build_identity_event_game(
    adapter: HADStage3Adapter,
    predictor: FrozenStage2Payoff,
    *,
    blue_style: Optional[str] = None,
    max_group_size: int = 4,
    full_domain_agent_threshold: int = 8,
    neighborhood_size: int = 8,
    peer_count: int = 5,
    batch_size: int = 512,
    utility_mode: str = "joint_survival_log_probability",
    risk_epsilon: float = 1e-6,
    allow_unregistered_actions: bool = False,
) -> IdentityEventGameBuild:
    """Construct one simultaneous identity-level game at a command event."""

    if int(max_group_size) != 4:
        raise ValueError("identity Stage3 has a hard 4-agent coalition cap")
    red_ids = _alive_ids(adapter, "Red")
    blue_ids = _alive_ids(adapter, "Blue")
    target_ids = tuple(map(int, adapter.target_ids))
    if not red_ids or not blue_ids or not target_ids:
        raise ValueError("an identity event needs two live sides and fixed targets")
    maximum_side = max(len(red_ids), len(blue_ids))
    full_domain = maximum_side <= int(full_domain_agent_threshold)
    # The mathematical full game has N channels per target.  At 30--50
    # agents we register an explicit active-channel subgame: enough channels
    # for a one-target packing of 4-agent groups and a complete Blue singleton
    # partition across public targets.  This is an admitted candidate-domain
    # restriction, not a hidden local-group-size constraint.
    channels_per_target = (
        maximum_side
        if full_domain
        else max(
            int(np.ceil(maximum_side / 4.0)),
            int(np.ceil(len(blue_ids) / len(target_ids))),
        )
    )
    slots = make_engagement_slots(
        target_ids,
        maximum_side,
        max_group_size=4,
        channels_per_target=channels_per_target,
    )
    red_seed = _seed_actions(red_ids, slots)
    red_all_reserve = IdentityAction(
        tuple(() for _ in slots), reserve_ids=tuple(red_ids)
    )
    blue_seed = _seed_actions(blue_ids, slots)
    target_positions = {
        int(target_id): np.asarray(state["position"], dtype=np.float64)
        for target_id, state in adapter.target_states().items()
    }
    red_candidates = spatial_candidate_coalitions(
        red_ids,
        slots,
        _positions(adapter, "Red"),
        target_positions,
        max_group_size=4,
        neighborhood_size=min(max(4, int(neighborhood_size)), len(red_ids)),
        peer_count=min(max(3, int(peer_count)), max(0, len(red_ids) - 1)),
        full_domain=full_domain,
        seed_actions=red_seed,
    )
    blue_candidates = spatial_candidate_coalitions(
        blue_ids,
        slots,
        _positions(adapter, "Blue"),
        target_positions,
        max_group_size=4,
        neighborhood_size=min(max(4, int(neighborhood_size)), len(blue_ids)),
        peer_count=min(max(3, int(peer_count)), max(0, len(blue_ids) - 1)),
        full_domain=full_domain,
        seed_actions=blue_seed,
    )
    oracle = Stage2IdentityPayoffOracle(
        adapter,
        predictor,
        slots,
        blue_style=blue_style,
        batch_size=batch_size,
        utility_mode=utility_mode,
        risk_epsilon=risk_epsilon,
    )
    breach_penalty = len(red_ids) + len(blue_ids) + 1.0
    game = IdentityBlottoGame(
        red_ids,
        blue_ids,
        slots,
        red_candidates,
        blue_candidates,
        oracle,
        max_group_size=4,
        structural_breach_penalty=breach_penalty,
        # At most one non-empty Blue group per Blue identity can be present,
        # so M*|B| bounds the absolute structural loss.  Dividing by this
        # positive, state-constant number leaves every equilibrium and best
        # response unchanged while making the DO tolerance population-scale
        # invariant and the reported exploitability interpretable in [0, 1].
        payoff_scale=breach_penalty * len(blue_ids),
        full_coalition_domain=full_domain,
        # This action is always legal and does not need a coalition column.
        # It is a solver fallback, not an artificial initial-support strategy;
        # Double Oracle will add it if it is actually a profitable response.
        red_fallback_action=red_all_reserve,
        blue_fallback_action=blue_seed[0],
        allow_red_reserve=True,
        allow_blue_reserve=False,
        allow_unregistered_actions=allow_unregistered_actions,
    )
    return IdentityEventGameBuild(
        game=game,
        red_alive_ids=red_ids,
        blue_alive_ids=blue_ids,
        target_ids=target_ids,
        slots=slots,
        initial_red=red_seed,
        initial_blue=blue_seed,
        full_coalition_domain=full_domain,
        red_candidate_columns=sum(map(len, red_candidates)),
        blue_candidate_columns=sum(map(len, blue_candidates)),
    )


@dataclass(frozen=True)
class GroundedIdentityPlan:
    blue_type_name: str
    red_action: IdentityAction
    blue_action: IdentityAction
    red_assignment: Tuple[Tuple[int, Optional[int]], ...]
    blue_assignment: Tuple[Tuple[int, Optional[int]], ...]
    red_reserve_ids: Tuple[int, ...]
    local_subgames: Tuple[LocalSubgame, ...]


def apply_identity_joint_plan(
    adapter: HADStage3Adapter,
    game: IdentityBlottoGame,
    red_action: IdentityAction,
    blue_action: IdentityAction,
    *,
    blue_type_name: str,
) -> GroundedIdentityPlan:
    """Install exactly the two labelled actions selected by the game."""

    red = game.validate_action(red_action, "Red")
    blue = game.validate_action(blue_action, "Blue")
    red_assignment: Dict[int, Optional[int]] = {
        int(agent_id): None for agent_id in adapter.red_ids
    }
    blue_assignment: Dict[int, Optional[int]] = {
        int(agent_id): None for agent_id in adapter.blue_ids
    }
    subgames: list[LocalSubgame] = []
    for index, slot in enumerate(game.slots):
        red_ids = red.coalitions[index]
        blue_ids = blue.coalitions[index]
        for agent_id in red_ids:
            red_assignment[agent_id] = slot.target_id
        for agent_id in blue_ids:
            blue_assignment[agent_id] = slot.target_id
        if red_ids or blue_ids:
            subgames.append(
                LocalSubgame(
                    target_id=slot.target_id,
                    subgroup_id=slot.channel_id,
                    red_ids=red_ids,
                    blue_ids=blue_ids,
                )
            )
    adapter.set_joint_assignments(red_assignment, blue_assignment)
    live_red = set(_alive_ids(adapter, "Red"))
    live_blue = set(_alive_ids(adapter, "Blue"))
    used_red = [value for subgame in subgames for value in subgame.red_ids]
    used_blue = [value for subgame in subgames for value in subgame.blue_ids]
    reserve_red = list(red.reserve_ids)
    red_cover = used_red + reserve_red
    if set(red_cover) != live_red or len(red_cover) != len(live_red):
        raise RuntimeError(
            "identity plan does not place every live Red agent exactly once "
            "in an active coalition or reserve"
        )
    if any(red_assignment[value] is not None for value in reserve_red):
        raise RuntimeError("Red reserve assignment must remain None")
    if set(used_blue) != live_blue or len(used_blue) != len(live_blue):
        raise RuntimeError("identity plan does not partition every live Blue agent once")
    if any(len(item.red_ids) > 4 or len(item.blue_ids) > 4 for item in subgames):
        raise RuntimeError("identity plan violated the hard 4v4 cap")
    return GroundedIdentityPlan(
        blue_type_name=str(blue_type_name),
        red_action=red,
        blue_action=blue,
        red_assignment=tuple(sorted(red_assignment.items())),
        blue_assignment=tuple(sorted(blue_assignment.items())),
        red_reserve_ids=tuple(reserve_red),
        local_subgames=tuple(subgames),
    )


def roster_override(
    plan: GroundedIdentityPlan,
) -> Mapping[tuple[int, int], tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return the exact coalitions consumed by ``FrozenStage1GroupExecutor``."""

    return {
        (item.target_id, item.subgroup_id): (item.red_ids, item.blue_ids)
        for item in plan.local_subgames
    }


__all__ = [
    "GroundedIdentityPlan",
    "IdentityEventGameBuild",
    "Stage2IdentityPayoffOracle",
    "apply_identity_joint_plan",
    "build_identity_event_game",
    "roster_override",
]
