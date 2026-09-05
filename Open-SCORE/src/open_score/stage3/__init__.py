"""Stage 3: robust online reallocation.

The runtime bridge depends on :mod:`open_score.stage1`, while Stage 1's PSRO
utilities depend on the lightweight Stage-3 commander.  Runtime symbols are
therefore exported lazily (PEP 562) so importing either package cannot observe
the other one while it is only partially initialized.
"""

from importlib import import_module

from .blotto import (
    Allocation,
    BestResponse,
    DoubleOracleIteration,
    DoubleOracleResult,
    EventBlottoGame,
    MatrixGameSolution,
    allocation_marginals,
    payoff_from_breach_risk,
    solve_double_oracle,
    solve_restricted_matrix_game,
)
from .bayesian import (
    BayesianDoubleOracleIteration,
    BayesianDoubleOracleResult,
    BayesianEventBlottoGame,
    BayesianProfileSample,
    BlueGroupingType,
    TypedBluePolicy,
    make_blue_grouping_type,
    solve_bayesian_double_oracle,
)
from .belief import (
    BlueAllocationLikelihoodModel,
    BlueBeliefSnapshot,
    BlueBeliefUpdate,
    BlueTypeBelief,
)
from .grouped_blotto import (
    GroupHistogram,
    GroupedAllocation,
    GroupedBlottoGame,
    balanced_grouped_allocation,
    compact_partition,
    concentrated_grouped_allocation,
    enumerate_grouped_allocations,
    expand_group_histogram,
    group_histograms_for_total,
    grouped_allocation_from_target_counts,
    histogram_from_group_sizes,
    histogram_resources,
    pair_group_histograms,
    solve_grouped_double_oracle,
)
from .grouped_bayesian import (
    GroupedBayesianBlottoGame,
    GroupedBayesianDoubleOracleResult,
    GroupedBayesianIteration,
    GroupedBayesianProfileSample,
    GroupedBlueType,
    GroupedTypedBluePolicy,
    make_grouped_blue_type,
    solve_grouped_bayesian_double_oracle,
)
from .identity_blotto import (
    Coalition,
    EngagementSlot,
    IdentityAction,
    IdentityBestResponse,
    IdentityBlottoGame,
    IdentityDoubleOracleResult,
    IdentityLocalPayoffOracle,
    enumerate_coalitions,
    enumerate_identity_actions,
    make_engagement_slots,
    round_robin_identity_action,
    solve_identity_double_oracle,
    spatial_candidate_coalitions,
)
from .saldae import (
    SALDAEConfig,
    SALDAEDoubleOracleDiagnostics,
    SALDAESearchDiagnostics,
    saldae_best_response,
    solve_saldae_double_oracle,
)
from .commander import (
    CommandDecision,
    build_defender_payoff,
    enumerate_count_allocations,
    needs_replan,
    solve_defender_maximin,
)
from .matching import (
    AgentTaskMatching,
    match_agents_to_tasks,
    match_sparse_agents_to_tasks,
)
from .payoff import (
    FrozenStage2Payoff,
    SUPPORTED_LOCAL_UTILITIES,
    StyleCalibration,
    build_local_payoff_tensor,
    fit_train_style_calibration,
    survival_probability_to_utility,
    utility_to_survival_probability,
)

_RUNTIME_EXPORTS = frozenset(
    {
        "EventGameBuild",
        "GroupedEventGameBuild",
        "FrozenStage1GroupExecutor",
        "GroundedJointBlottoPlan",
        "GroundedJointGroupedPlan",
        "GroundedIdentityPlan",
        "IdentityEventGameBuild",
        "LocalSubgame",
        "apply_joint_blotto_plan",
        "apply_joint_grouped_plan",
        "balanced_allocation",
        "build_event_game",
        "build_grouped_event_game",
        "concentrated_allocation",
        "enumerate_feasible_allocations",
        "eta_greedy_defender_allocation",
        "ground_count_allocation",
        "load_round01_stage1_model",
        "moderate_jittered_target_positions",
        "ObservableThreatPatrol",
        "build_observable_threat_patrol",
        "observable_threat_patrol",
        "pure_maximin_allocation",
        "Stage2IdentityPayoffOracle",
        "apply_identity_joint_plan",
        "build_identity_event_game",
        "identity_roster_override",
    }
)


def __getattr__(name: str):
    """Load the S1-dependent runtime bridge only on first public access."""

    if name in _RUNTIME_EXPORTS:
        module = (
            ".identity_runtime"
            if name
            in {
                "GroundedIdentityPlan",
                "IdentityEventGameBuild",
                "Stage2IdentityPayoffOracle",
                "apply_identity_joint_plan",
                "build_identity_event_game",
                "identity_roster_override",
            }
            else ".runtime"
        )
        lookup = "roster_override" if name == "identity_roster_override" else name
        value = getattr(import_module(module, __name__), lookup)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _RUNTIME_EXPORTS)

__all__ = [
    "AgentTaskMatching",
    "Allocation",
    "BayesianDoubleOracleIteration",
    "BayesianDoubleOracleResult",
    "BayesianEventBlottoGame",
    "BayesianProfileSample",
    "BlueAllocationLikelihoodModel",
    "BlueBeliefSnapshot",
    "BlueBeliefUpdate",
    "BestResponse",
    "BlueGroupingType",
    "BlueTypeBelief",
    "CommandDecision",
    "Coalition",
    "DoubleOracleIteration",
    "DoubleOracleResult",
    "EventBlottoGame",
    "EventGameBuild",
    "EngagementSlot",
    "FrozenStage2Payoff",
    "FrozenStage1GroupExecutor",
    "GroundedJointBlottoPlan",
    "GroundedJointGroupedPlan",
    "GroundedIdentityPlan",
    "GroupHistogram",
    "GroupedAllocation",
    "GroupedBayesianBlottoGame",
    "GroupedBayesianDoubleOracleResult",
    "GroupedBayesianIteration",
    "GroupedBayesianProfileSample",
    "GroupedBlottoGame",
    "GroupedBlueType",
    "GroupedEventGameBuild",
    "GroupedTypedBluePolicy",
    "IdentityAction",
    "IdentityBestResponse",
    "IdentityBlottoGame",
    "IdentityDoubleOracleResult",
    "IdentityEventGameBuild",
    "IdentityLocalPayoffOracle",
    "SALDAEConfig",
    "SALDAEDoubleOracleDiagnostics",
    "SALDAESearchDiagnostics",
    "LocalSubgame",
    "MatrixGameSolution",
    "StyleCalibration",
    "Stage2IdentityPayoffOracle",
    "SUPPORTED_LOCAL_UTILITIES",
    "TypedBluePolicy",
    "allocation_marginals",
    "apply_joint_blotto_plan",
    "apply_joint_grouped_plan",
    "apply_identity_joint_plan",
    "balanced_allocation",
    "balanced_grouped_allocation",
    "build_defender_payoff",
    "build_local_payoff_tensor",
    "build_event_game",
    "build_grouped_event_game",
    "build_identity_event_game",
    "compact_partition",
    "concentrated_allocation",
    "concentrated_grouped_allocation",
    "enumerate_count_allocations",
    "enumerate_feasible_allocations",
    "enumerate_grouped_allocations",
    "enumerate_coalitions",
    "enumerate_identity_actions",
    "expand_group_histogram",
    "eta_greedy_defender_allocation",
    "fit_train_style_calibration",
    "ground_count_allocation",
    "group_histograms_for_total",
    "grouped_allocation_from_target_counts",
    "histogram_from_group_sizes",
    "histogram_resources",
    "identity_roster_override",
    "load_round01_stage1_model",
    "moderate_jittered_target_positions",
    "ObservableThreatPatrol",
    "build_observable_threat_patrol",
    "observable_threat_patrol",
    "match_agents_to_tasks",
    "match_sparse_agents_to_tasks",
    "make_grouped_blue_type",
    "make_engagement_slots",
    "make_blue_grouping_type",
    "needs_replan",
    "payoff_from_breach_risk",
    "pair_group_histograms",
    "pure_maximin_allocation",
    "round_robin_identity_action",
    "saldae_best_response",
    "solve_double_oracle",
    "solve_grouped_double_oracle",
    "solve_grouped_bayesian_double_oracle",
    "solve_identity_double_oracle",
    "solve_saldae_double_oracle",
    "solve_bayesian_double_oracle",
    "solve_defender_maximin",
    "solve_restricted_matrix_game",
    "survival_probability_to_utility",
    "utility_to_survival_probability",
    "spatial_candidate_coalitions",
]
