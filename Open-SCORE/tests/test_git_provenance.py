"""Git provenance and formal-run admission tests."""

import subprocess
from pathlib import Path

import pytest

from open_score import provenance


def _git() -> Path:
    executable = provenance._find_git_executable()
    if executable is None:
        pytest.skip("Git is unavailable for repository-fixture tests")
    return executable


def _run(executable: Path, root: Path, *arguments: str) -> None:
    subprocess.run(
        [str(executable), "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _committed_repository(tmp_path: Path) -> tuple[Path, Path]:
    executable = _git()
    root = tmp_path / "outer"
    nested = root / "Open-SCORE" / "src"
    nested.mkdir(parents=True)
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _run(executable, root, "init")
    _run(executable, root, "add", "tracked.txt")
    _run(
        executable,
        root,
        "-c",
        "user.name=OpenSCORE Test",
        "-c",
        "user.email=openscore@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return root, nested


def test_clean_outer_repository_is_discovered_from_nested_project(tmp_path):
    root, nested = _committed_repository(tmp_path)
    result = provenance.collect_git_provenance(nested)
    assert result["git_available"]
    assert result["repository_detected"]
    assert Path(result["repository_root"]) == root.resolve()
    assert len(result["head_commit"]) == 40
    assert result["branch"]
    assert result["clean"] is True
    assert result["status_porcelain"] == []
    assert len(result["status_sha256"]) == 64
    assert len(result["worktree_fingerprint_sha256"]) == 64
    assert result["formal_eligible"] is True
    provenance.require_formal_git_provenance(result, formal=True)


def test_dirty_content_and_untracked_files_change_fingerprint_and_fail_formal(tmp_path):
    root, nested = _committed_repository(tmp_path)
    clean = provenance.collect_git_provenance(nested)
    (root / "tracked.txt").write_text("modified\n", encoding="utf-8")
    (root / "untracked.txt").write_text("new evidence\n", encoding="utf-8")
    dirty = provenance.collect_git_provenance(nested)
    assert dirty["clean"] is False
    assert dirty["formal_eligible"] is False
    assert any("tracked.txt" in row for row in dirty["status_porcelain"])
    assert any("untracked.txt" in row for row in dirty["status_porcelain"])
    assert dirty["worktree_fingerprint_sha256"] != clean[
        "worktree_fingerprint_sha256"
    ]
    with pytest.raises(RuntimeError, match="clean committed Git worktree"):
        provenance.require_formal_git_provenance(dirty, formal=True)
    provenance.require_formal_git_provenance(dirty, formal=False)


def test_no_git_is_recorded_and_only_blocks_formal(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_find_git_executable", lambda: None)
    result = provenance.collect_git_provenance(tmp_path)
    assert result["git_available"] is False
    assert result["repository_detected"] is False
    assert result["error"] == "git_executable_not_found"
    provenance.require_formal_git_provenance(result, formal=False)
    with pytest.raises(RuntimeError, match="Git is unavailable"):
        provenance.require_formal_git_provenance(result, formal=True)


def test_uncommitted_repository_fails_formal_gate(tmp_path):
    executable = _git()
    root = tmp_path / "empty-repository"
    root.mkdir()
    _run(executable, root, "init")
    result = provenance.collect_git_provenance(root)
    assert result["repository_detected"] is True
    assert result["head_commit"] is None
    with pytest.raises(RuntimeError, match="no committed HEAD"):
        provenance.require_formal_git_provenance(result, formal=True)


@pytest.mark.parametrize(
    "script_name, minimum_metadata_occurrences",
    [
        ("train_smaclite_stock_entity_baselines.py", 2),
        ("train_smaclite_ad_baselines.py", 3),
        ("train_stage1_baselines.py", 2),
    ],
)
def test_stage1_entry_points_gate_before_writes_and_embed_provenance(
    script_name, minimum_metadata_occurrences
):
    project = Path(__file__).resolve().parents[1]
    text = (project / "scripts" / script_name).read_text(encoding="utf-8")
    gate = text.index("collect_and_require_git_provenance(", text.index("def main()"))
    first_output_write = text.index(".mkdir(", text.index("def main()"))
    assert gate < first_output_write
    assert text.count('"git_provenance": args._git_provenance') >= (
        minimum_metadata_occurrences
    )
