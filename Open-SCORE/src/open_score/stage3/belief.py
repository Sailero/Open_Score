"""Online Bayesian filtering over a fixed library of Blue strategy types.

The filter in this module is deliberately separate from the one-shot
Harsanyi game in :mod:`open_score.stage3.bayesian`.  The game consumes a prior
for one simultaneous decision event; :class:`BlueTypeBelief` carries the
posterior from completed events into the next event.

The public API enforces the intended information order::

    snapshot = belief.begin_event(event_id)
    # Build/solve the simultaneous game with snapshot.probabilities.
    belief.commit_event_model(event_id, likelihood_model)
    # Execute both players' already selected actions, then observe Blue.
    update = belief.observe_completed_event(event_id, observed_blue_allocation)

Consequently, the action observed in event ``t`` can only change the prior of
event ``t + 1``.  It cannot be used to revise Red's simultaneous decision for
event ``t``.  Likelihood models are event-specific because the legal target
allocations can change with the active battlefield state, while their type
names must remain those of the fixed strategy library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .blotto import Allocation


def _readonly_probabilities(
    values: Sequence[float],
    *,
    expected: int,
    name: str,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (expected,) or not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} must be a finite vector of length {expected}")
    if np.any(probabilities < 0.0) or float(probabilities.sum()) <= 0.0:
        raise ValueError(f"{name} must be non-negative with positive mass")
    probabilities = probabilities / probabilities.sum()
    probabilities.setflags(write=False)
    return probabilities


def _allocation(value: Sequence[int], *, name: str = "allocation") -> Allocation:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite counts")
    integer = array.astype(np.int64)
    if not np.array_equal(array, integer) or np.any(integer < 0):
        raise ValueError(f"{name} must contain non-negative integer counts")
    return tuple(int(item) for item in integer)


def _type_names(values: Sequence[str]) -> Tuple[str, ...]:
    names = tuple(str(item).strip() for item in values)
    if not names or any(not item for item in names):
        raise ValueError("Blue type names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("Blue type names must be unique")
    return names


def _event_id(value: object) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError("event_id must be non-empty")
    return result


@dataclass(frozen=True)
class BlueBeliefSnapshot:
    """The predictive type distribution frozen before one simultaneous event."""

    event_id: str
    type_names: Tuple[str, ...]
    probabilities: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(probability)
            for name, probability in zip(self.type_names, self.probabilities)
        }


@dataclass(frozen=True)
class BlueBeliefUpdate:
    """Audit record produced only after a decision event has completed."""

    event_id: str
    type_names: Tuple[str, ...]
    observed_allocation: Optional[Allocation]
    decision_prior: np.ndarray
    likelihood: Optional[np.ndarray]
    posterior: np.ndarray
    log_evidence: Optional[float]

    def posterior_dict(self) -> dict[str, float]:
        return {
            name: float(probability)
            for name, probability in zip(self.type_names, self.posterior)
        }


class BlueAllocationLikelihoodModel:
    """Type-conditional PMFs for an observed Blue target-count allocation.

    Parameters
    ----------
    type_distributions:
        Mapping ``type name -> {allocation: probability mass}``.  Each inner
        distribution is normalised independently.  The union of all listed
        allocations is the known action alphabet for this event.
    smoothing:
        Non-negative additive pseudo-probability assigned to every known
        action and to one shared ``other allocation`` bucket.  A small positive
        value prevents a single surprising action from irreversibly deleting
        a type.  It is deliberately explicit rather than silently clipping
        likelihoods during the posterior update.
    temperature:
        Positive likelihood temperature.  Values above one flatten each
        type's action PMF; values below one sharpen it.

    The model represents a *fixed, known* type library.  It does not discover
    missing Blue strategies; an out-of-library action is handled by the
    smoothed shared ``other`` bucket.
    """

    def __init__(
        self,
        type_distributions: Mapping[
            str, Mapping[Sequence[int], float]
        ],
        *,
        smoothing: float = 1e-6,
        temperature: float = 1.0,
    ) -> None:
        if not isinstance(type_distributions, Mapping) or not type_distributions:
            raise ValueError("type_distributions must be a non-empty mapping")
        names = _type_names(tuple(type_distributions))
        smoothing = float(smoothing)
        temperature = float(temperature)
        if not np.isfinite(smoothing) or smoothing < 0.0:
            raise ValueError("smoothing must be finite and non-negative")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

        normalised: list[dict[Allocation, float]] = []
        dimensions: set[int] = set()
        universe: set[Allocation] = set()
        for type_name in names:
            distribution = type_distributions[type_name]
            if not isinstance(distribution, Mapping) or not distribution:
                raise ValueError(
                    f"allocation distribution for Blue type {type_name!r} "
                    "must be a non-empty mapping"
                )
            parsed: dict[Allocation, float] = {}
            for raw_action, raw_probability in distribution.items():
                action = _allocation(
                    raw_action,
                    name=f"allocation for Blue type {type_name!r}",
                )
                probability = float(raw_probability)
                if not np.isfinite(probability) or probability < 0.0:
                    raise ValueError(
                        f"allocation probabilities for Blue type {type_name!r} "
                        "must be finite and non-negative"
                    )
                parsed[action] = parsed.get(action, 0.0) + probability
                dimensions.add(len(action))
                universe.add(action)
            total = float(sum(parsed.values()))
            if total <= 0.0:
                raise ValueError(
                    f"allocation distribution for Blue type {type_name!r} "
                    "must have positive mass"
                )
            normalised.append(
                {action: probability / total for action, probability in parsed.items()}
            )
        if len(dimensions) != 1:
            raise ValueError("all allocation likelihoods must have the same dimension")

        actions = tuple(sorted(universe))
        action_index = {action: index for index, action in enumerate(actions)}
        # The final column is a single catch-all bucket for an allocation that
        # was not present in any registered type distribution.
        probabilities = np.zeros((len(names), len(actions) + 1), dtype=np.float64)
        for type_index, distribution in enumerate(normalised):
            for action, probability in distribution.items():
                probabilities[type_index, action_index[action]] = probability
        probabilities += smoothing
        for type_index in range(len(names)):
            row = probabilities[type_index]
            positive = row > 0.0
            log_weights = np.full(row.shape, -np.inf, dtype=np.float64)
            log_weights[positive] = np.log(row[positive]) / temperature
            maximum = float(np.max(log_weights))
            tempered = np.zeros_like(row)
            tempered[positive] = np.exp(log_weights[positive] - maximum)
            probabilities[type_index] = tempered / tempered.sum()
        probabilities.setflags(write=False)

        self._type_names = names
        self._actions = actions
        self._action_index = action_index
        self._probabilities = probabilities
        self._smoothing = smoothing
        self._temperature = temperature
        self._allocation_dimension = dimensions.pop()

    @classmethod
    def from_bayesian_result(
        cls,
        result: Any,
        *,
        smoothing: float = 1e-6,
        temperature: float = 1.0,
    ) -> "BlueAllocationLikelihoodModel":
        """Build ``P(allocation | type, state)`` from a solved typed game.

        The attacker policy mixture and every type-contingent allocation must
        be committed before the realised action is sampled.  This adapter uses
        duck typing intentionally so archived solver-result objects remain
        usable without introducing a circular import.
        """

        try:
            names = _type_names(result.blue_type_names)
            policies = tuple(result.attacker_policies)
            mixture = _readonly_probabilities(
                result.attacker_policy_mixture,
                expected=len(policies),
                name="attacker policy mixture",
            )
        except AttributeError as error:
            raise TypeError("result is not a Bayesian Double Oracle result") from error
        if not policies:
            raise ValueError("Bayesian result contains no attacker policies")
        distributions: dict[str, dict[Allocation, float]] = {
            name: {} for name in names
        }
        for policy_weight, policy in zip(mixture, policies):
            allocations = tuple(policy.allocations)
            if len(allocations) != len(names):
                raise ValueError(
                    "each typed attacker policy must have one allocation per type"
                )
            for type_name, raw_action in zip(names, allocations):
                action = _allocation(raw_action)
                distribution = distributions[type_name]
                distribution[action] = (
                    distribution.get(action, 0.0) + float(policy_weight)
                )
        return cls(
            distributions,
            smoothing=smoothing,
            temperature=temperature,
        )

    @property
    def type_names(self) -> Tuple[str, ...]:
        return self._type_names

    @property
    def action_universe(self) -> Tuple[Allocation, ...]:
        return self._actions

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def probability_matrix(self) -> np.ndarray:
        """Read-only ``type x (known actions + other)`` probability matrix."""

        return self._probabilities

    @property
    def other_probabilities(self) -> np.ndarray:
        result = self._probabilities[:, -1].view()
        result.setflags(write=False)
        return result

    def likelihoods(self, observed_allocation: Sequence[int]) -> np.ndarray:
        """Return a read-only likelihood vector in ``type_names`` order."""

        action = _allocation(observed_allocation, name="observed Blue allocation")
        if len(action) != self._allocation_dimension:
            raise ValueError(
                "observed Blue allocation dimension does not match likelihood model"
            )
        index = self._action_index.get(action, len(self._actions))
        result = self._probabilities[:, index].copy()
        result.setflags(write=False)
        return result


class BlueTypeBelief:
    """Persistent event-to-event posterior over a fixed Blue type library.

    ``switch_hazard`` models a possible type reset between events:

    ``predictive = (1 - hazard) * previous_posterior + hazard * base_prior``.

    The default zero hazard assumes one latent Blue type throughout an episode.
    This class is intentionally mutable: one instance belongs to one episode.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        type_names: Sequence[str],
        prior: Sequence[float],
        *,
        switch_hazard: float = 0.0,
    ) -> None:
        names = _type_names(type_names)
        base_prior = _readonly_probabilities(
            prior, expected=len(names), name="Blue type prior"
        )
        hazard = float(switch_hazard)
        if not np.isfinite(hazard) or not 0.0 <= hazard <= 1.0:
            raise ValueError("switch_hazard must be finite and in [0, 1]")
        self._type_names = names
        self._base_prior = base_prior
        self._posterior = base_prior.copy()
        self._posterior.setflags(write=False)
        self._switch_hazard = hazard
        self._active_snapshot: Optional[BlueBeliefSnapshot] = None
        self._active_model: Optional[BlueAllocationLikelihoodModel] = None
        self._completed_event_ids: list[str] = []
        self._history: list[BlueBeliefUpdate] = []

    @classmethod
    def from_blue_types(
        cls,
        blue_types: Sequence[Any],
        *,
        switch_hazard: float = 0.0,
    ) -> "BlueTypeBelief":
        """Initialise from objects exposing ``name`` and ``prior`` fields."""

        items = tuple(blue_types)
        if not items:
            raise ValueError("blue_types must be non-empty")
        try:
            names = tuple(item.name for item in items)
            prior = tuple(float(item.prior) for item in items)
        except AttributeError as error:
            raise TypeError("each Blue type must expose name and prior") from error
        return cls(names, prior, switch_hazard=switch_hazard)

    @property
    def type_names(self) -> Tuple[str, ...]:
        return self._type_names

    @property
    def base_prior(self) -> np.ndarray:
        return self._base_prior

    @property
    def posterior(self) -> np.ndarray:
        """Posterior after the most recently completed event."""

        return self._posterior

    @property
    def switch_hazard(self) -> float:
        return self._switch_hazard

    @property
    def active_event_id(self) -> Optional[str]:
        return (
            None if self._active_snapshot is None else self._active_snapshot.event_id
        )

    @property
    def history(self) -> Tuple[BlueBeliefUpdate, ...]:
        return tuple(self._history)

    def posterior_dict(self) -> dict[str, float]:
        return {
            name: float(probability)
            for name, probability in zip(self.type_names, self.posterior)
        }

    def begin_event(self, event_id: object) -> BlueBeliefSnapshot:
        """Freeze and return the belief Red is allowed to use this event."""

        key = _event_id(event_id)
        if self._active_snapshot is not None:
            if self._active_snapshot.event_id == key:
                return self._active_snapshot
            raise RuntimeError(
                f"event {self._active_snapshot.event_id!r} is still active; "
                "complete or close it before beginning another event"
            )
        if key in self._completed_event_ids:
            raise ValueError(f"event {key!r} has already been completed")
        predictive = (
            (1.0 - self.switch_hazard) * self.posterior
            + self.switch_hazard * self.base_prior
        )
        predictive = _readonly_probabilities(
            predictive,
            expected=len(self.type_names),
            name="predictive Blue type belief",
        )
        self._active_snapshot = BlueBeliefSnapshot(
            event_id=key,
            type_names=self.type_names,
            probabilities=predictive,
        )
        self._active_model = None
        return self._active_snapshot

    def commit_event_model(
        self,
        event_id: object,
        model: BlueAllocationLikelihoodModel,
    ) -> None:
        """Commit ``P(action | type, state)`` before the action is revealed."""

        snapshot = self._require_active(event_id)
        if not isinstance(model, BlueAllocationLikelihoodModel):
            raise TypeError("model must be a BlueAllocationLikelihoodModel")
        if model.type_names != self.type_names:
            raise ValueError(
                "likelihood-model type names/order must match the belief library"
            )
        if self._active_model is not None:
            raise RuntimeError(
                f"likelihood model for event {snapshot.event_id!r} is already committed"
            )
        self._active_model = model

    def observe_completed_event(
        self,
        event_id: object,
        observed_blue_allocation: Sequence[int],
    ) -> BlueBeliefUpdate:
        """Assimilate a Blue allocation only after the event action completed."""

        snapshot = self._require_active(event_id)
        if self._active_model is None:
            raise RuntimeError(
                "commit the event likelihood model before observing Blue's action"
            )
        observation = _allocation(
            observed_blue_allocation, name="observed Blue allocation"
        )
        likelihood = self._active_model.likelihoods(observation)
        positive = (snapshot.probabilities > 0.0) & (likelihood > 0.0)
        if not np.any(positive):
            raise ValueError(
                "observed allocation has zero probability under every Blue type; "
                "use positive likelihood smoothing or expand the strategy library"
            )
        log_joint = np.full(len(self.type_names), -np.inf, dtype=np.float64)
        log_joint[positive] = (
            np.log(snapshot.probabilities[positive]) + np.log(likelihood[positive])
        )
        maximum = float(np.max(log_joint))
        joint = np.zeros(len(self.type_names), dtype=np.float64)
        joint[positive] = np.exp(log_joint[positive] - maximum)
        posterior = joint / joint.sum()
        log_evidence = maximum + float(np.log(joint.sum()))
        posterior.setflags(write=False)
        update = BlueBeliefUpdate(
            event_id=snapshot.event_id,
            type_names=self.type_names,
            observed_allocation=observation,
            decision_prior=snapshot.probabilities,
            likelihood=likelihood,
            posterior=posterior,
            log_evidence=log_evidence,
        )
        self._finalise_event(update)
        return update

    def close_event_without_observation(self, event_id: object) -> BlueBeliefUpdate:
        """Advance the event state without adding evidence (e.g. no Blue alive)."""

        snapshot = self._require_active(event_id)
        posterior = snapshot.probabilities.copy()
        posterior.setflags(write=False)
        update = BlueBeliefUpdate(
            event_id=snapshot.event_id,
            type_names=self.type_names,
            observed_allocation=None,
            decision_prior=snapshot.probabilities,
            likelihood=None,
            posterior=posterior,
            log_evidence=None,
        )
        self._finalise_event(update)
        return update

    def state_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable closed-event checkpoint.

        Checkpointing an active simultaneous event is rejected so a restored
        filter cannot accidentally reorder model commitment and observation.
        """

        if self._active_snapshot is not None:
            raise RuntimeError("complete or close the active event before checkpointing")
        return {
            "version": self.STATE_VERSION,
            "type_names": list(self.type_names),
            "base_prior": [float(item) for item in self.base_prior],
            "posterior": [float(item) for item in self.posterior],
            "switch_hazard": self.switch_hazard,
            "completed_event_ids": list(self._completed_event_ids),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "BlueTypeBelief":
        """Restore a checkpoint produced by :meth:`state_dict`."""

        if not isinstance(state, Mapping):
            raise TypeError("belief checkpoint must be a mapping")
        if int(state.get("version", -1)) != cls.STATE_VERSION:
            raise ValueError("unsupported Blue belief checkpoint version")
        try:
            belief = cls(
                state["type_names"],  # type: ignore[arg-type]
                state["base_prior"],  # type: ignore[arg-type]
                switch_hazard=float(state["switch_hazard"]),
            )
            posterior = _readonly_probabilities(
                state["posterior"],  # type: ignore[arg-type]
                expected=len(belief.type_names),
                name="checkpoint posterior",
            )
            completed = [_event_id(item) for item in state["completed_event_ids"]]  # type: ignore[union-attr]
        except KeyError as error:
            raise ValueError(f"belief checkpoint is missing {error.args[0]!r}") from error
        if len(set(completed)) != len(completed):
            raise ValueError("checkpoint completed_event_ids must be unique")
        belief._posterior = posterior
        belief._completed_event_ids = completed
        return belief

    def _require_active(self, event_id: object) -> BlueBeliefSnapshot:
        key = _event_id(event_id)
        if self._active_snapshot is None:
            raise RuntimeError("begin_event must be called first")
        if self._active_snapshot.event_id != key:
            raise ValueError(
                f"active event is {self._active_snapshot.event_id!r}, not {key!r}"
            )
        return self._active_snapshot

    def _finalise_event(self, update: BlueBeliefUpdate) -> None:
        self._posterior = update.posterior
        self._completed_event_ids.append(update.event_id)
        self._history.append(update)
        self._active_snapshot = None
        self._active_model = None


__all__ = [
    "BlueAllocationLikelihoodModel",
    "BlueBeliefSnapshot",
    "BlueBeliefUpdate",
    "BlueTypeBelief",
]
