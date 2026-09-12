"""Evaluation protocol and the single experiment report."""
from .protocol import (
    VALIDATION_CONFIGS, compute_nds, config_dict, config_key, config_label,
    evaluate_checkpoint, evaluation_thresholds, final_jobs, remaining_jobs, validation_jobs, validation_score,
)


def evaluate(*args, **kwargs):
    return evaluate_checkpoint(*args, **kwargs)


def anchors(*args, **kwargs):
    from .anchors import run_anchors
    return run_anchors(*args, **kwargs)


def refresh_report(*args, **kwargs):
    from .report import refresh_report as refresh
    return refresh(*args, **kwargs)
