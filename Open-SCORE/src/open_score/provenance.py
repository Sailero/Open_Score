"""Git provenance shared by auditable Open-SCORE experiment entry points."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


PROVENANCE_SCHEMA_VERSION = "openscore-git-provenance-v1"


def _find_git_executable() -> Optional[Path]:
    configured = os.environ.get("OPEN_SCORE_GIT")
    candidates = [
        configured,
        shutil.which("git"),
        r"D:\Software\Git\cmd\git.exe",
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    return None


def _git(
    executable: Path,
    cwd: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(executable), "-C", str(cwd), *arguments],
        capture_output=True,
        text=text,
        check=False,
        timeout=30,
    )


def _worktree_fingerprint(executable: Path, root: Path, head: str) -> str:
    digest = hashlib.sha256()
    digest.update(head.encode("ascii"))
    diff = _git(executable, root, ["diff", "--binary", "HEAD", "--"], text=False)
    if diff.returncode:
        raise RuntimeError("git diff failed while computing worktree fingerprint")
    digest.update(diff.stdout)
    untracked = _git(
        executable,
        root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        text=False,
    )
    if untracked.returncode:
        raise RuntimeError("git ls-files failed while computing worktree fingerprint")
    for raw_name in sorted(name for name in untracked.stdout.split(b"\0") if name):
        digest.update(b"\0untracked\0")
        digest.update(raw_name)
        relative = raw_name.decode("utf-8", errors="surrogateescape")
        path = root / relative
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Git returned an untracked path outside its root") from exc
        if path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def collect_git_provenance(start: Path) -> Dict[str, object]:
    """Discover the containing (including outer) repository and inspect it."""

    location = Path(start).resolve()
    if location.is_file():
        location = location.parent
    executable = _find_git_executable()
    unavailable: Dict[str, object] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "git_available": executable is not None,
        "git_executable": None if executable is None else str(executable),
        "repository_detected": False,
        "repository_root": None,
        "head_commit": None,
        "branch": None,
        "detached_head": None,
        "clean": None,
        "status_porcelain": [],
        "status_sha256": None,
        "worktree_fingerprint_sha256": None,
        "formal_eligible": False,
    }
    if executable is None:
        unavailable["error"] = "git_executable_not_found"
        return unavailable
    root_result = _git(executable, location, ["rev-parse", "--show-toplevel"])
    if root_result.returncode:
        unavailable["error"] = "not_inside_git_repository"
        return unavailable
    root = Path(root_result.stdout.strip()).resolve()
    head_result = _git(executable, root, ["rev-parse", "--verify", "HEAD^{commit}"])
    if head_result.returncode:
        unavailable.update(
            {
                "repository_detected": True,
                "repository_root": str(root),
                "error": "repository_has_no_committed_head",
            }
        )
        return unavailable
    head = head_result.stdout.strip()
    branch_result = _git(executable, root, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    status_result = _git(
        executable, root, ["status", "--porcelain=v1", "--untracked-files=all"]
    )
    if status_result.returncode:
        unavailable.update(
            {
                "repository_detected": True,
                "repository_root": str(root),
                "head_commit": head,
                "error": "git_status_failed",
            }
        )
        return unavailable
    status_text = status_result.stdout
    status = status_text.splitlines()
    clean = not status
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "git_available": True,
        "git_executable": str(executable),
        "repository_detected": True,
        "repository_root": str(root),
        "head_commit": head,
        "branch": branch,
        "detached_head": branch is None,
        "clean": clean,
        "status_porcelain": status,
        "status_sha256": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
        "worktree_fingerprint_sha256": _worktree_fingerprint(
            executable, root, head
        ),
        "formal_eligible": clean,
    }


def require_formal_git_provenance(
    provenance: Mapping[str, object], formal: bool
) -> None:
    """Reject a formal run unless it starts from a clean committed checkout."""

    if not formal:
        return
    failures = []
    if provenance.get("git_available") is not True:
        failures.append("Git is unavailable")
    if provenance.get("repository_detected") is not True:
        failures.append("no containing Git repository was detected")
    if not provenance.get("head_commit"):
        failures.append("the repository has no committed HEAD")
    if provenance.get("clean") is not True:
        failures.append("the worktree contains tracked or untracked changes")
    if failures:
        raise RuntimeError(
            "formal evidence requires a clean committed Git worktree: "
            + "; ".join(failures)
        )


def collect_and_require_git_provenance(
    start: Path, *, formal: bool
) -> Dict[str, object]:
    provenance = collect_git_provenance(start)
    require_formal_git_provenance(provenance, formal)
    return provenance


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "collect_and_require_git_provenance",
    "collect_git_provenance",
    "require_formal_git_provenance",
]
