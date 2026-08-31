"""Auditable transfer from stock SMAClite into the SMAClite-AD models.

The stock adapter and AD protocol deliberately have different raw feature and
action schemas.  A silent ``strict=False`` load would therefore be unsafe: it
could copy an action head whose rows have different meanings.  This module
copies only complete, named latent layers whose tensor shape and semantics are
shared, and records every copied or rejected tensor in a JSON-ready manifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn


TRANSFER_SCHEMA_VERSION = "openscore-stock-to-ad-transfer-v3-selected-update-audit"
ACTION_TARGET_CONTRACT = "explicit-action-entity-index-and-type-v1"
TRANSFERRED_TARGET_TYPE_ROWS = {1: "enemy_damage"}


_DEEPSET_PREFIXES = {
    "qmix": (
        "agent.entity_encoder.element.2.",
        "agent.entity_encoder.output.0.",
        "agent.rnn.",
        "agent.q_head.target_scorer.",
        "mixer.state_encoder.element.2.",
        "mixer.state_encoder.output.0.",
        "mixer.first_bias.",
        "mixer.final_weights.",
        "mixer.final_bias.",
    ),
    "vdn": (
        "agent.entity_encoder.element.2.",
        "agent.entity_encoder.output.0.",
        "agent.rnn.",
        "agent.q_head.target_scorer.",
    ),
    "mappo": (
        "actor.entity_encoder.element.2.",
        "actor.entity_encoder.output.0.",
        "actor.rnn.",
        "actor.policy_head.target_scorer.",
        "critic.encoder.element.2.",
        "critic.encoder.output.0.",
        "critic.value_head.",
    ),
}


_SAQA_ENCODER_PREFIXES = {
    "qmix": (
        "agent.entity_encoder.entity_embedding.2.",
        "agent.entity_encoder.cross_attention.",
        "agent.entity_encoder.norm.",
        "agent.entity_encoder.output.",
    ),
    "vdn": (
        "agent.entity_encoder.entity_embedding.2.",
        "agent.entity_encoder.cross_attention.",
        "agent.entity_encoder.norm.",
        "agent.entity_encoder.output.",
    ),
    "mappo": (
        "actor.entity_encoder.entity_embedding.2.",
        "actor.entity_encoder.cross_attention.",
        "actor.entity_encoder.norm.",
        "actor.entity_encoder.output.",
    ),
}


def _target_encoder_contract(model: nn.Module, algorithm: str) -> tuple[str, int]:
    policy = model.actor if algorithm == "mappo" else model.agent
    kind = str(getattr(policy, "encoder_kind", "deepset"))
    heads = int(getattr(policy, "attention_heads", 4))
    return kind, heads


def _safe_prefixes(algorithm: str, encoder_kind: str) -> Sequence[str]:
    base = _DEEPSET_PREFIXES[algorithm]
    if encoder_kind == "deepset":
        return base
    # Replace only the DeepSets entity-encoder prefixes; recurrent, target,
    # mixer and critic prefixes retain the established allowlist semantics.
    marker = "actor.entity_encoder." if algorithm == "mappo" else "agent.entity_encoder."
    retained = tuple(prefix for prefix in base if not prefix.startswith(marker))
    return _SAQA_ENCODER_PREFIXES[algorithm] + retained


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(tensor).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_source_state(
    checkpoint: Mapping[str, object], algorithm: str
) -> Mapping[str, Tensor]:
    key = "model" if algorithm == "mappo" else "online"
    state = checkpoint.get(key)
    if not isinstance(state, Mapping):
        raise ValueError(
            f"stock checkpoint has no {key!r} state for algorithm {algorithm!r}"
        )
    if not all(isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()):
        raise TypeError("checkpoint model state must map string names to tensors")
    return state  # type: ignore[return-value]


def _source_metadata(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    extra = checkpoint.get("extra", {})
    if not isinstance(extra, Mapping):
        raise TypeError("checkpoint extra metadata must be a mapping")
    return extra


def transfer_stock_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    algorithm: str,
) -> Dict[str, object]:
    """Load semantically compatible latent layers and return an audit manifest.

    The shared per-entity target scorer is transferable under the explicit
    target contract.  The semantically identical enemy-damage type row is
    copied with a row-level audit; new AD-only target types stay freshly
    initialized. Optimizer state, target-network state, raw feature
    projections and fixed-action rows are not transferred.
    """

    if algorithm not in _DEEPSET_PREFIXES:
        raise ValueError(f"unsupported transfer algorithm: {algorithm}")
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"stock checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("stock checkpoint root must be a mapping")
    metadata = _source_metadata(checkpoint)
    if metadata.get("algorithm") != algorithm:
        raise ValueError(
            "stock checkpoint algorithm mismatch: "
            f"expected {algorithm!r}, found {metadata.get('algorithm')!r}"
        )
    if metadata.get("environment_family") != "SMAClite-stock":
        raise ValueError(
            "transfer source must declare environment_family='SMAClite-stock'"
        )
    source_contract = metadata.get("contract")
    if not isinstance(source_contract, Mapping) or source_contract.get(
        "action_target_contract"
    ) != ACTION_TARGET_CONTRACT:
        raise ValueError(
            "stock transfer source lacks the explicit action-to-entity target contract"
        )
    source_target_types = source_contract.get("action_target_types")
    if not isinstance(source_target_types, Mapping) or any(
        source_target_types.get(str(row)) != semantic
        for row, semantic in TRANSFERRED_TARGET_TYPE_ROWS.items()
    ):
        raise ValueError(
            "stock transfer source lacks the required target-type semantic rows"
        )
    target_encoder_kind, target_attention_heads = _target_encoder_contract(
        model, algorithm
    )
    source_encoder_kind = source_contract.get("encoder_kind")
    source_attention_heads = source_contract.get("attention_heads")
    if source_encoder_kind != target_encoder_kind or int(
        source_attention_heads if source_attention_heads is not None else -1
    ) != target_attention_heads:
        raise ValueError(
            "stock and AD encoder contracts differ: "
            f"source=({source_encoder_kind!r}, {source_attention_heads!r}), "
            f"target=({target_encoder_kind!r}, {target_attention_heads!r})"
        )
    if "selected_checkpoint_learner_updates" not in metadata:
        raise ValueError(
            "stock transfer source must explicitly record selected_checkpoint_learner_updates"
        )
    source_selected_updates = int(metadata["selected_checkpoint_learner_updates"])
    source_total_updates = int(
        metadata.get("total_training_learner_updates", source_selected_updates)
    )
    if source_selected_updates < 1:
        raise ValueError(
            "stock transfer source must prove that its validation-selected "
            "checkpoint follows at least one learner update"
        )
    if source_total_updates < source_selected_updates:
        raise ValueError(
            "total training learner updates cannot precede selected-checkpoint updates"
        )
    source_trained_names = metadata.get("source_trained_tensor_names")
    if not isinstance(source_trained_names, (list, tuple)) or not all(
        isinstance(name, str) for name in source_trained_names
    ):
        raise ValueError(
            "stock transfer source must record source_trained_tensor_names"
        )
    trained_target_type_rows = metadata.get("source_trained_target_type_rows")
    if not isinstance(trained_target_type_rows, Mapping) or any(
        trained_target_type_rows.get(str(row)) != semantic
        for row, semantic in TRANSFERRED_TARGET_TYPE_ROWS.items()
    ):
        raise ValueError(
            "stock checkpoint lacks row-level training evidence for transferred target types"
        )
    source_state = _extract_source_state(checkpoint, algorithm)
    target_scorer_prefix = (
        "actor.policy_head.target_scorer."
        if algorithm == "mappo"
        else "agent.q_head.target_scorer."
    )
    scorer_parameter_names = {
        name for name in source_state if name.startswith(target_scorer_prefix)
    }
    trained_name_set = set(source_trained_names)
    missing_trained_scorer = sorted(scorer_parameter_names - trained_name_set)
    if not scorer_parameter_names or missing_trained_scorer:
        raise ValueError(
            "stock checkpoint does not prove that every shared target-scorer "
            f"tensor received training updates: {missing_trained_scorer}"
        )
    source_performance_gate = metadata.get("source_performance_gate")
    if not isinstance(source_performance_gate, Mapping):
        raise ValueError(
            "stock transfer source has not passed its held-out policy-performance gate"
        )
    source_win_rate = float(source_performance_gate.get("win_rate", 0.0))
    source_minimum_win_rate = float(
        source_performance_gate.get("minimum_win_rate", float("inf"))
    )
    source_wins = int(source_performance_gate.get("wins", 0))
    source_heldout_episodes = int(source_performance_gate.get("episodes", 0))
    if not (
        source_performance_gate.get("passed") is True
        and source_minimum_win_rate > 0.0
        and source_win_rate >= source_minimum_win_rate
        and source_wins > 0
        and source_heldout_episodes > 0
    ):
        raise ValueError(
            "stock transfer source has not passed its held-out policy-performance gate"
        )
    source_multi_seed_gate = metadata.get("source_multi_seed_performance_gate")
    if not isinstance(source_multi_seed_gate, Mapping):
        raise ValueError(
            "stock transfer source has not passed the multi-seed performance gate"
        )
    aggregate_seeds = source_multi_seed_gate.get("seeds")
    aggregate_interval = source_multi_seed_gate.get("heldout_win_rate")
    aggregate_minimum_seeds = int(
        source_multi_seed_gate.get("minimum_seed_count", 0)
    )
    aggregate_minimum_win_rate = float(
        source_multi_seed_gate.get("minimum_mean_win_rate", float("inf"))
    )
    if not (
        source_multi_seed_gate.get("passed") is True
        and isinstance(aggregate_seeds, (list, tuple))
        and all(isinstance(seed, int) for seed in aggregate_seeds)
        and len(set(aggregate_seeds)) >= aggregate_minimum_seeds > 0
        and isinstance(aggregate_interval, Mapping)
        and float(aggregate_interval.get("mean", 0.0))
        >= aggregate_minimum_win_rate
        > 0.0
    ):
        raise ValueError(
            "stock transfer source has not passed the multi-seed performance gate"
        )
    destination_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    destination_after: MutableMapping[str, Tensor] = {
        name: tensor.clone() for name, tensor in destination_before.items()
    }
    copied = []
    partial_row_copies = []
    skipped = []
    safe_prefixes = _safe_prefixes(algorithm, target_encoder_kind)
    for name, destination in destination_before.items():
        source = source_state.get(name)
        if source is None:
            skipped.append(
                {"name": name, "reason": "missing_in_source", "target_shape": list(destination.shape)}
            )
            continue
        if not name.startswith(tuple(safe_prefixes)):
            skipped.append(
                {
                    "name": name,
                    "reason": "excluded_semantic_boundary",
                    "source_shape": list(source.shape),
                    "target_shape": list(destination.shape),
                }
            )
            continue
        if source.shape != destination.shape:
            skipped.append(
                {
                    "name": name,
                    "reason": "shape_mismatch",
                    "source_shape": list(source.shape),
                    "target_shape": list(destination.shape),
                }
            )
            continue
        converted = source.detach().cpu().to(dtype=destination.dtype)
        destination_after[name] = converted.clone()
        copied.append(
            {
                "name": name,
                "shape": list(destination.shape),
                "numel": destination.numel(),
                "source_sha256": _tensor_sha256(source),
                "target_before_sha256": _tensor_sha256(destination),
                "target_after_sha256": _tensor_sha256(converted),
            }
        )
    target_embedding_name = (
        "actor.policy_head.target_type_embedding.weight"
        if algorithm == "mappo"
        else "agent.q_head.target_type_embedding.weight"
    )
    source_embedding = source_state.get(target_embedding_name)
    destination_embedding = destination_before.get(target_embedding_name)
    if source_embedding is None or destination_embedding is None:
        raise ValueError("target-type embedding is missing from source or destination")
    if source_embedding.shape != destination_embedding.shape or source_embedding.ndim != 2:
        raise ValueError("target-type embedding shape mismatch")
    if target_embedding_name not in trained_name_set:
        raise ValueError(
            "stock checkpoint does not prove that the target-type embedding was trained"
        )
    updated_embedding = destination_after[target_embedding_name].clone()
    for row, semantic in TRANSFERRED_TARGET_TYPE_ROWS.items():
        source_row = source_embedding[row].detach().cpu().to(updated_embedding.dtype)
        before_row = updated_embedding[row].clone()
        updated_embedding[row] = source_row
        partial_row_copies.append(
            {
                "name": target_embedding_name,
                "row": row,
                "semantic": semantic,
                "numel": source_row.numel(),
                "source_sha256": _tensor_sha256(source_row),
                "target_before_sha256": _tensor_sha256(before_row),
                "target_after_sha256": _tensor_sha256(source_row),
            }
        )
    destination_after[target_embedding_name] = updated_embedding
    if not copied:
        raise ValueError("no semantically safe, shape-compatible tensors were found")
    changed_copies = [
        record
        for record in copied
        if record["source_sha256"] != record["target_before_sha256"]
    ]
    changed_partial_rows = [
        record
        for record in partial_row_copies
        if record["source_sha256"] != record["target_before_sha256"]
    ]
    if not changed_copies and not changed_partial_rows:
        raise ValueError("compatible tensors were found but transfer changed no weights")
    trained_copies = [
        record for record in copied if record["name"] in trained_name_set
    ]
    if not trained_copies:
        raise ValueError(
            "no copied latent tensor is recorded as changed by stock training"
        )
    model.load_state_dict(destination_after, strict=True)
    total_parameters = sum(tensor.numel() for tensor in destination_before.values())
    copied_parameters = sum(int(record["numel"]) for record in copied) + sum(
        int(record["numel"]) for record in partial_row_copies
    )
    return {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "algorithm": algorithm,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_environment_family": metadata.get("environment_family"),
        "source_environment_id": metadata.get("environment_id"),
        "source_upstream_commit": metadata.get("upstream_commit"),
        "source_use_cpp_rvo2": bool(metadata.get("use_cpp_rvo2", False)),
        "source_contract": source_contract,
        "action_target_contract": ACTION_TARGET_CONTRACT,
        "encoder_kind": target_encoder_kind,
        "attention_heads": target_attention_heads,
        "source_training_hyperparameters": metadata.get(
            "training_hyperparameters"
        ),
        # The legacy name now deliberately aliases the selected-checkpoint
        # count.  Explicit fields remove the previous ambiguity with final
        # training progress.
        "source_training_learner_updates": source_selected_updates,
        "source_selected_checkpoint_learner_updates": source_selected_updates,
        "source_total_training_learner_updates": source_total_updates,
        "source_performance_gate": dict(source_performance_gate),
        "source_multi_seed_performance_gate": dict(source_multi_seed_gate),
        "source_trained_tensor_count": len(source_trained_names),
        "source_target_scorer_trained": True,
        "source_target_scorer_tensor_names": sorted(scorer_parameter_names),
        "source_trained_target_type_rows": dict(trained_target_type_rows),
        "target_type_partial_transfer": {
            "authorized_rows": {
                str(row): semantic
                for row, semantic in TRANSFERRED_TARGET_TYPE_ROWS.items()
            },
            "fresh_rows": [0, 2, 3],
            "reason": (
                "enemy_damage has identical stock/AD semantics; non-target, "
                "ally-heal and protected-asset rows remain destination initialized"
            ),
        },
        "target_environment_id": "OpenSCORE/SMACliteAD-Asset-v0",
        "policy": (
            "exact-name/exact-shape transfer restricted to registered latent "
            "layers plus the shared per-target scorer; raw feature projections, "
            "fixed-action rows, optimizer and target network are excluded; "
            "only explicitly shared target-type embedding rows are copied"
        ),
        "safe_prefixes": list(safe_prefixes),
        "model_sha256_before": model_state_sha256(destination_before),
        "model_sha256_after": model_state_sha256(model.state_dict()),
        "copied_tensor_count": len(copied),
        "partial_row_copy_count": len(partial_row_copies),
        "changed_copied_tensor_count": len(changed_copies),
        "changed_partial_row_count": len(changed_partial_rows),
        "copied_source_trained_tensor_count": len(trained_copies),
        "copied_parameter_count": copied_parameters,
        "target_parameter_count": total_parameters,
        "copied_parameter_fraction": copied_parameters / max(1, total_parameters),
        "copied": copied,
        "partial_row_copies": partial_row_copies,
        "skipped": skipped,
        "fresh_optimizer_required": True,
        "source_target_network_transferred": False,
    }


__all__ = [
    "TRANSFER_SCHEMA_VERSION",
    "ACTION_TARGET_CONTRACT",
    "file_sha256",
    "model_state_sha256",
    "transfer_stock_checkpoint",
]
