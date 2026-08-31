"""Bind the validation-selected Stage-1 QMIX policy into both Stage-2 configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

import yaml


PROJECT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH_PLACEHOLDER = (
    "REPLACE_WITH_STAGE1_QMIX_VALIDATION_BEST_CHECKPOINT_PATH.pt"
)
CHECKPOINT_SHA_PLACEHOLDER = "REPLACE_WITH_64_HEX_STAGE1_CHECKPOINT_SHA256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(path: Path, project_root: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = project_root / value
    return value.resolve()


def _project_path(path: Path, project_root: Path = PROJECT) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("policy-lock paths must remain inside the project") from error


def _git_executable() -> str:
    configured = os.environ.get("OPEN_SCORE_GIT")
    if configured:
        return configured
    fixed = Path(r"D:\Software\Git\cmd\git.exe")
    return str(fixed) if fixed.is_file() else "git"


def _git_state(
    require_clean: bool,
    *,
    project_root: Path = PROJECT,
    git_executable: Optional[str] = None,
    runner: Callable[..., object] = subprocess.run,
) -> Dict[str, object]:
    """Audit the entire containing repository, not only the inner project."""

    project_root = project_root.resolve()
    git = git_executable or _git_executable()
    top_level = runner(
        [git, "rev-parse", "--show-toplevel"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repository_root = Path(top_level).resolve()
    try:
        project_root.relative_to(repository_root)
    except ValueError as error:
        raise RuntimeError("project is not inside the reported Git worktree") from error
    commit = runner(
        [git, "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = runner(
        [git, "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean and status:
        raise RuntimeError("policy binding requires a clean committed Git worktree")
    return {
        "commit": commit,
        "repository_root": str(repository_root),
        "dirty_before_binding": bool(status),
    }


def _load_mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _candidate(summary: Mapping[str, object]) -> Mapping[str, object]:
    if summary.get("schema_version") != "stage1-had-reproduction-v5":
        raise ValueError("Stage-1 summary schema must be stage1-had-reproduction-v5")
    if summary.get("formal_convergence_claim") is not True:
        raise ValueError("Stage-1 formal convergence gate did not pass")
    primary = summary.get("primary_strategy_gate")
    if not isinstance(primary, Mapping) or primary.get("passed") is not True:
        raise ValueError("Stage-1 primary QMIX strategy gate did not pass")
    try:
        value = summary["multi_seed_heldout"]["algorithms"]["qmix"][
            "deployment_candidate"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("Stage-1 summary lacks the QMIX deployment candidate") from error
    if not isinstance(value, Mapping):
        raise ValueError("QMIX deployment candidate must be an object")
    if value.get("selection_split") != "validation_only":
        raise ValueError("deployment candidate was not selected on Validation only")
    if value.get("heldout_was_not_used_for_selection") is not True:
        raise ValueError("deployment candidate selection used Held-out information")
    for field in ("checkpoint", "checkpoint_sha256", "seed"):
        if field not in value:
            raise ValueError(f"QMIX deployment candidate lacks {field}")
    return value


def _replace_once(raw: str, old: str, new: str, label: str) -> str:
    count = raw.count(old)
    if count != 1:
        raise ValueError(f"{label} expected exactly one placeholder, found {count}")
    return raw.replace(old, new, 1)


def _checkpoint_spec(config: object) -> Mapping[str, object]:
    if not isinstance(config, Mapping):
        raise ValueError("Stage-2 collection YAML must contain a mapping")
    try:
        candidates = config["data"]["defender_candidates"]
    except (KeyError, TypeError) as error:
        raise ValueError("collection YAML lacks defender_candidates") from error
    if not isinstance(candidates, list):
        raise ValueError("collection defender_candidates must be a list")
    checkpoints = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("kind") == "checkpoint"
    ]
    if len(checkpoints) != 1:
        raise ValueError("collection YAML must contain exactly one checkpoint candidate")
    return checkpoints[0]


def _binding_values(collect_raw: str, train_raw: str) -> Dict[str, str]:
    collect_yaml = yaml.safe_load(collect_raw)
    train_yaml = yaml.safe_load(train_raw)
    if not isinstance(train_yaml, Mapping):
        raise ValueError("Stage-2 training YAML must contain a mapping")
    checkpoint_spec = _checkpoint_spec(collect_yaml)
    formal_contract = train_yaml.get("formal_data_contract")
    if not isinstance(formal_contract, Mapping):
        raise ValueError("training YAML lacks formal_data_contract")
    return {
        "path": str(checkpoint_spec.get("path", "")).replace("\\", "/"),
        "collect_sha256": str(checkpoint_spec.get("expected_sha256", "")).lower(),
        "train_sha256": str(
            formal_contract.get("expected_stage1_checkpoint_sha256", "")
        ).lower(),
    }


def bind_policy(
    summary_path: Path,
    collect_config_path: Path,
    train_config_path: Path,
    output_path: Path,
    *,
    protocol_tag: str,
    apply: bool,
    require_clean: bool = True,
    project_root: Path = PROJECT,
    git_state_reader: Optional[Callable[..., Dict[str, object]]] = None,
) -> Dict[str, object]:
    project_root = project_root.resolve()
    summary_path = _resolve_project_path(summary_path, project_root)
    collect_config_path = _resolve_project_path(collect_config_path, project_root)
    train_config_path = _resolve_project_path(train_config_path, project_root)
    output_path = _resolve_project_path(output_path, project_root)
    for path in (
        summary_path,
        collect_config_path,
        train_config_path,
        output_path,
    ):
        _project_path(path, project_root)
    if len({collect_config_path, train_config_path, output_path}) != 3:
        raise ValueError("collection config, training config and lock output must differ")
    if not protocol_tag.strip():
        raise ValueError("protocol_tag must be non-empty")
    summary = _load_mapping(summary_path)
    candidate = _candidate(summary)
    checkpoint = Path(str(candidate["checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    registered_sha = str(candidate.get("checkpoint_sha256", "")).lower()
    if len(registered_sha) != 64 or any(
        character not in "0123456789abcdef" for character in registered_sha
    ):
        raise ValueError("Stage-1 summary checkpoint_sha256 is not a SHA-256 digest")
    if checkpoint_sha256 != registered_sha:
        raise ValueError("Stage-1 checkpoint SHA-256 differs from its summary")
    checkpoint_project_path = _project_path(checkpoint, project_root)

    collect_raw = collect_config_path.read_text(encoding="utf-8")
    train_raw = train_config_path.read_text(encoding="utf-8")
    git = None
    if apply:
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite an existing policy lock: {output_path}")
        collect_bound = _replace_once(
            collect_raw,
            CHECKPOINT_PATH_PLACEHOLDER,
            json.dumps(checkpoint_project_path),
            "collection checkpoint path",
        )
        collect_bound = _replace_once(
            collect_bound,
            CHECKPOINT_SHA_PLACEHOLDER,
            checkpoint_sha256,
            "collection checkpoint SHA",
        )
        train_bound = _replace_once(
            train_raw,
            CHECKPOINT_SHA_PLACEHOLDER,
            checkpoint_sha256,
            "training checkpoint SHA",
        )
        bound_values = _binding_values(collect_bound, train_bound)
        expected_values = {
            "path": checkpoint_project_path,
            "collect_sha256": checkpoint_sha256,
            "train_sha256": checkpoint_sha256,
        }
        if bound_values != expected_values:
            raise ValueError("rendered Stage-2 policy binding failed semantic validation")
        state_reader = git_state_reader or _git_state
        git = state_reader(
            require_clean=require_clean,
            project_root=project_root,
        )
        if require_clean and bool(git.get("dirty_before_binding")):
            raise RuntimeError("policy binding requires a clean committed Git worktree")
        collect_config_path.write_text(collect_bound, encoding="utf-8", newline="\n")
        train_config_path.write_text(train_bound, encoding="utf-8", newline="\n")
    else:
        current = _binding_values(collect_raw, train_raw)
        placeholder_flags = (
            current["path"] == CHECKPOINT_PATH_PLACEHOLDER,
            current["collect_sha256"] == CHECKPOINT_SHA_PLACEHOLDER.lower(),
            current["train_sha256"] == CHECKPOINT_SHA_PLACEHOLDER.lower(),
        )
        if any(placeholder_flags) and not all(placeholder_flags):
            raise ValueError("Stage-2 configs contain a partial policy binding")
        if not any(placeholder_flags) and current != {
            "path": checkpoint_project_path,
            "collect_sha256": checkpoint_sha256,
            "train_sha256": checkpoint_sha256,
        }:
            raise ValueError("existing Stage-2 policy binding differs from Stage-1 summary")

    lock = {
        "schema_version": "stage1-policy-lock-v1",
        "algorithm": "qmix",
        "seed": int(candidate["seed"]),
        "selection": "validation_only_lexicographic_worst_cell_then_mean",
        "validation_score": candidate.get("validation_score"),
        "heldout_was_not_used_for_selection": True,
        "checkpoint": checkpoint_project_path,
        "checkpoint_sha256": checkpoint_sha256,
        "stage1_summary": _project_path(summary_path, project_root),
        "stage1_summary_sha256": _sha256(summary_path),
        "protocol_tag": protocol_tag,
        "binding_applied": bool(apply),
        "git_before_binding": git,
    }
    if apply:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lock["collect_config_sha256"] = _sha256(collect_config_path)
        lock["train_config_sha256"] = _sha256(train_config_path)
        output_path.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT / "outputs" / "stage1_had_formal" / "summary.json",
    )
    parser.add_argument(
        "--collect-config",
        type=Path,
        default=PROJECT / "configs" / "stage2_had_formal_collect.yaml",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=PROJECT / "configs" / "stage2_had_formal.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs" / "stage1_had_formal" / "stage1_policy_lock.json",
    )
    parser.add_argument("--protocol-tag", default="s1s2-protocol-v1")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the two registered config bindings and the policy-lock JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = bind_policy(
        args.summary,
        args.collect_config,
        args.train_config,
        args.output,
        protocol_tag=args.protocol_tag,
        apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
