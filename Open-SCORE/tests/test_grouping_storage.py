"""Windows checkpoint lock regressions: retry rename, preserve valid files."""
import errno
import json

import pytest
import torch

from open_score.grouping import storage


@pytest.mark.parametrize('kind', ['checkpoint', 'json'])
def test_transient_windows_lock_retries_without_removing_old_file(tmp_path, monkeypatch, kind):
    path = tmp_path / ('latest.pt' if kind == 'checkpoint' else 'status.json')
    save = storage.atomic_checkpoint if kind == 'checkpoint' else storage.atomic_json
    save(path, {'step': 10})
    old_bytes = path.read_bytes()
    original = storage.os.replace
    calls, sleeps = [], []

    def locked(source, target):
        calls.append((source, target))
        assert path.read_bytes() == old_bytes
        if len(calls) < 3:
            error = PermissionError(errno.EACCES, 'temporary Windows lock')
            error.winerror = 5 if len(calls) == 1 else 32
            raise error
        return original(source, target)

    monkeypatch.setattr(storage.os, 'replace', locked)
    monkeypatch.setattr(storage.time, 'sleep', sleeps.append)
    save(path, {'step': 20})
    loaded = torch.load(path, weights_only=False) if kind == 'checkpoint' else json.loads(path.read_text())
    assert loaded == {'step': 20}
    assert len(calls) == 3 and sleeps == [.05, .1]


@pytest.mark.parametrize('winerror', [5, 112, None])
def test_persistent_or_unrelated_error_preserves_old_checkpoint(tmp_path, monkeypatch, winerror):
    path = tmp_path / 'latest.pt'
    storage.atomic_checkpoint(path, {'step': 10})
    original_bytes = path.read_bytes()
    calls = []

    def fail(*args):
        calls.append(args)
        error = OSError(errno.EACCES, 'cannot replace')
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(storage.os, 'replace', fail)
    monkeypatch.setattr(storage.time, 'sleep', lambda _: None)
    with pytest.raises(OSError):
        storage.atomic_checkpoint(path, {'step': 20})
    assert path.read_bytes() == original_bytes
    assert torch.load(path, weights_only=False) == {'step': 10}
    assert torch.load(path.with_suffix('.tmp'), weights_only=False) == {'step': 20}
    assert len(calls) == (9 if winerror == 5 else 1)
