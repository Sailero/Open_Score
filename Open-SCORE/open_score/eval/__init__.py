"""Evaluation protocol and the single experiment report."""
from .protocol import (
    CYCLE_SERIES_METHODS, DEPTH_SWEEP_DEPTHS, VALIDATION_CONFIGS, compute_nds, config_dict, config_key, config_label,
    evaluate_checkpoint, evaluate_depth_sweep, evaluation_thresholds, final_jobs, remaining_jobs, validation_jobs, validation_score,
)
from .inventory import migrate_into_main, pending_eval_jobs, pending_train_jobs, scan



def evaluate(*args, **kwargs):
    return evaluate_checkpoint(*args, **kwargs)


def anchors(*args, **kwargs):
    from .anchors import run_anchors
    return run_anchors(*args, **kwargs)


def refresh_report(*args, **kwargs):
    from .report import refresh_report as refresh
    return refresh(*args, **kwargs)
