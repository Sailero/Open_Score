"""Frozen HAD validation and evaluation interfaces."""
from .protocol import (VALIDATION_CONFIGS, FINAL_CONFIGS, DEPTH_SWEEP_DEPTHS,
    config_dict, config_key, config_label, evaluate_checkpoint, evaluate_depth_sweep,
    evaluation_thresholds, final_jobs, remaining_jobs, validation_jobs, validation_score)

def evaluate(*args, **kwargs):
    return evaluate_checkpoint(*args, **kwargs)

def anchors(*args, **kwargs):
    from .anchors import run_anchors
    return run_anchors(*args, **kwargs)
