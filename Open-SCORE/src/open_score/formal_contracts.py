"""Hash-pinned validation for preregistered Stage-1 experiment contracts.

The Stage-1 command-line entry points deliberately keep their ordinary defaults
small enough for engineering checks.  A ``--formal-evidence`` run must therefore
prove that every registered scientific argument matches the tracked YAML rather
than merely checking a few lower bounds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml


FORMAL_CONTRACT_SCHEMA_VERSION = "openscore-stage1-formal-contract-v1"


class FormalContractMismatch(ValueError):
    """Raised when a formal CLI invocation differs from its registered YAML."""


def _normalise(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalise(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalise(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _read_path(document: Mapping[str, Any], path: str) -> Any:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise FormalContractMismatch(
                f"registered formal contract is missing YAML path {path!r}"
            )
        value = value[component]
    return value


def _normalise_project_path(value: Any, project_root: Path) -> str:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(project_root.resolve())
        except ValueError:
            return path.resolve().as_posix()
    return path.as_posix()


def validate_registered_formal_contract(
    *,
    project_root: Path,
    config_path: Path,
    contract_name: str,
    expected_protocol_version: str,
    actual_values: Mapping[str, Any],
    yaml_paths: Mapping[str, str],
    unordered_fields: Sequence[str] = (),
    project_path_fields: Sequence[str] = (),
) -> Dict[str, object]:
    """Load, hash and exactly validate a tracked formal YAML contract.

    ``actual_values`` uses stable audit labels.  ``yaml_paths`` maps those labels
    to dotted paths in the YAML document.  Exact comparison is the default;
    fields whose order is scientifically irrelevant can be declared unordered.
    Path fields are normalised relative to the repository before comparison.
    """

    root = Path(project_root).resolve()
    path = Path(config_path).resolve()
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise FormalContractMismatch(
            "formal contract must be stored inside the project repository"
        ) from exc
    raw = path.read_bytes()
    loaded = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(loaded, Mapping):
        raise FormalContractMismatch("formal contract YAML must contain a mapping")
    protocol_version = loaded.get("protocol_version")
    if protocol_version != expected_protocol_version:
        raise FormalContractMismatch(
            "formal contract protocol_version mismatch: "
            f"expected {expected_protocol_version!r}, found {protocol_version!r}"
        )
    if set(actual_values) != set(yaml_paths):
        missing_actual = sorted(set(yaml_paths) - set(actual_values))
        missing_path = sorted(set(actual_values) - set(yaml_paths))
        raise FormalContractMismatch(
            "formal validator field registry is inconsistent: "
            f"missing_actual={missing_actual}, missing_yaml_path={missing_path}"
        )

    unordered = set(unordered_fields)
    path_fields = set(project_path_fields)
    unknown_modes = (unordered | path_fields) - set(actual_values)
    if unknown_modes:
        raise FormalContractMismatch(
            f"formal validator declares unknown special fields: {sorted(unknown_modes)}"
        )

    mismatches = []
    validated: Dict[str, object] = {}
    for label, actual in actual_values.items():
        expected = _read_path(loaded, yaml_paths[label])
        if label in path_fields:
            actual_normalised: Any = _normalise_project_path(actual, root)
            expected_normalised: Any = _normalise_project_path(expected, root)
        else:
            actual_normalised = _normalise(actual)
            expected_normalised = _normalise(expected)
        if label in unordered:
            if not isinstance(actual_normalised, list) or not isinstance(
                expected_normalised, list
            ):
                raise FormalContractMismatch(
                    f"unordered formal field {label!r} must be list-like"
                )
            actual_normalised = sorted(actual_normalised, key=_canonical_json)
            expected_normalised = sorted(expected_normalised, key=_canonical_json)
        if _canonical_json(actual_normalised) != _canonical_json(expected_normalised):
            mismatches.append(
                {
                    "field": label,
                    "yaml_path": yaml_paths[label],
                    "expected": expected_normalised,
                    "actual": actual_normalised,
                }
            )
        validated[label] = actual_normalised

    if mismatches:
        details = "; ".join(
            f"{item['field']} ({item['yaml_path']}): expected "
            f"{item['expected']!r}, got {item['actual']!r}"
            for item in mismatches
        )
        raise FormalContractMismatch(
            f"{contract_name} formal CLI does not match registered YAML: {details}"
        )

    canonical_document = _canonical_json(loaded).encode("utf-8")
    return {
        "schema_version": FORMAL_CONTRACT_SCHEMA_VERSION,
        "contract_name": contract_name,
        "config_path": relative_path,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_config_sha256": hashlib.sha256(canonical_document).hexdigest(),
        "protocol_version": str(protocol_version),
        "validation_status": "exact_match",
        "validated_fields": validated,
        "validated_yaml_paths": dict(yaml_paths),
    }


__all__ = [
    "FORMAL_CONTRACT_SCHEMA_VERSION",
    "FormalContractMismatch",
    "validate_registered_formal_contract",
]
