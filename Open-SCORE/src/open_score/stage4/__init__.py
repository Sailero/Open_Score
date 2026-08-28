"""Stage 4: guarded capability-command co-adaptation."""

from .coadaptation import (
    AlternatingUpdateResult,
    FinalOutcomeResidual,
    final_outcome_residual_loss,
    guarded_lower_update,
    stability_regularized_loss,
)

__all__ = [
    "AlternatingUpdateResult",
    "FinalOutcomeResidual",
    "final_outcome_residual_loss",
    "guarded_lower_update",
    "stability_regularized_loss",
]
