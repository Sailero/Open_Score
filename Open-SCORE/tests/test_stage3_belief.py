from types import SimpleNamespace

import numpy as np
import pytest

from open_score.stage3.bayesian import TypedBluePolicy
from open_score.stage3.belief import (
    BlueAllocationLikelihoodModel,
    BlueTypeBelief,
)


def _opposed_model(*, smoothing=0.0, temperature=1.0):
    return BlueAllocationLikelihoodModel(
        {
            "balanced": {(1, 1): 0.9, (2, 0): 0.1},
            "concentrated": {(1, 1): 0.1, (2, 0): 0.9},
        },
        smoothing=smoothing,
        temperature=temperature,
    )


def test_completed_observation_updates_only_the_next_event_belief():
    belief = BlueTypeBelief(["balanced", "concentrated"], [0.5, 0.5])
    event_one = belief.begin_event("command-1")
    belief.commit_event_model("command-1", _opposed_model())

    update = belief.observe_completed_event("command-1", (2, 0))

    # The immutable snapshot actually used for command 1 remains the ex-ante
    # prior; only command 2 receives the posterior learned after execution.
    np.testing.assert_allclose(event_one.probabilities, [0.5, 0.5])
    np.testing.assert_allclose(update.posterior, [0.1, 0.9])
    event_two = belief.begin_event("command-2")
    np.testing.assert_allclose(event_two.probabilities, [0.1, 0.9])
    assert not event_one.probabilities.flags.writeable
    assert not update.posterior.flags.writeable


def test_temporal_protocol_requires_precommitted_model_and_event_order():
    belief = BlueTypeBelief(["balanced", "concentrated"], [0.5, 0.5])
    belief.begin_event(1)
    with pytest.raises(RuntimeError, match="commit"):
        belief.observe_completed_event(1, (2, 0))
    with pytest.raises(RuntimeError, match="still active"):
        belief.begin_event(2)

    model = _opposed_model()
    belief.commit_event_model(1, model)
    with pytest.raises(RuntimeError, match="already committed"):
        belief.commit_event_model(1, model)
    with pytest.raises(ValueError, match="active event"):
        belief.observe_completed_event(2, (2, 0))
    belief.observe_completed_event(1, (2, 0))
    with pytest.raises(ValueError, match="already been completed"):
        belief.begin_event(1)


def test_switch_hazard_mixes_previous_posterior_with_fixed_library_prior():
    belief = BlueTypeBelief(
        ["balanced", "concentrated"],
        [0.5, 0.5],
        switch_hazard=0.2,
    )
    belief.begin_event("first")
    belief.commit_event_model("first", _opposed_model())
    belief.observe_completed_event("first", (2, 0))

    predictive = belief.begin_event("second")

    # (1 - 0.2) * [0.1, 0.9] + 0.2 * [0.5, 0.5]
    np.testing.assert_allclose(predictive.probabilities, [0.18, 0.82])


def test_smoothing_keeps_out_of_library_observation_finite():
    model = _opposed_model(smoothing=1e-3)
    likelihood = model.likelihoods((0, 2))
    assert np.all(likelihood > 0.0)
    np.testing.assert_allclose(model.probability_matrix.sum(axis=1), 1.0)

    belief = BlueTypeBelief(model.type_names, [0.5, 0.5])
    belief.begin_event("surprise")
    belief.commit_event_model("surprise", model)
    update = belief.observe_completed_event("surprise", (0, 2))
    assert np.all(np.isfinite(update.posterior))
    np.testing.assert_allclose(update.posterior, [0.5, 0.5])


def test_temperature_controls_how_strongly_one_action_separates_types():
    sharp = _opposed_model(temperature=0.5)
    flat = _opposed_model(temperature=2.0)
    sharp_ratio = sharp.likelihoods((2, 0))[1] / sharp.likelihoods((2, 0))[0]
    flat_ratio = flat.likelihoods((2, 0))[1] / flat.likelihoods((2, 0))[0]
    assert sharp_ratio > 9.0
    assert flat_ratio < 9.0
    assert sharp_ratio > flat_ratio


def test_likelihood_adapter_aggregates_typed_equilibrium_policies():
    result = SimpleNamespace(
        blue_type_names=("balanced", "concentrated"),
        attacker_policies=(
            TypedBluePolicy(((1, 1), (2, 0))),
            TypedBluePolicy(((1, 1), (0, 2))),
        ),
        attacker_policy_mixture=np.asarray([0.25, 0.75]),
    )
    model = BlueAllocationLikelihoodModel.from_bayesian_result(
        result, smoothing=0.0
    )

    np.testing.assert_allclose(model.likelihoods((1, 1)), [1.0, 0.0])
    np.testing.assert_allclose(model.likelihoods((2, 0)), [0.0, 0.25])
    np.testing.assert_allclose(model.likelihoods((0, 2)), [0.0, 0.75])


def test_closed_event_checkpoint_round_trip_preserves_next_decision():
    belief = BlueTypeBelief(
        ["balanced", "concentrated"], [0.6, 0.4], switch_hazard=0.1
    )
    belief.begin_event("command-1")
    belief.commit_event_model("command-1", _opposed_model())
    belief.observe_completed_event("command-1", (2, 0))
    checkpoint = belief.state_dict()

    restored = BlueTypeBelief.from_state_dict(checkpoint)

    assert restored.posterior_dict() == pytest.approx(belief.posterior_dict())
    np.testing.assert_allclose(
        restored.begin_event("command-2").probabilities,
        belief.begin_event("command-2").probabilities,
    )


def test_checkpoint_rejects_active_event_to_preserve_information_order():
    belief = BlueTypeBelief(["balanced", "concentrated"], [0.5, 0.5])
    belief.begin_event("active")
    with pytest.raises(RuntimeError, match="active event"):
        belief.state_dict()
