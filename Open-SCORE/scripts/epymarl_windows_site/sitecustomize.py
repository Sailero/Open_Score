"""Path-safe EPyMARL timestamp overlay loaded by Python's site machinery."""

from __future__ import annotations

import datetime
import importlib.abc
import importlib.machinery
import os
import sys
from importlib.util import spec_from_loader


class _WindowsSafeDateTime(datetime.datetime):
    def __str__(self) -> str:
        return self.strftime("%Y-%m-%d_%H-%M-%S-%f")


class _RunLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):  # noqa: ANN001 - mirrors importlib protocol
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:  # noqa: ANN001 - mirrors importlib protocol
        self._wrapped.exec_module(module)
        module.datetime.datetime = _WindowsSafeDateTime


class _RunFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001 - importlib protocol
        if fullname != "run":
            return None
        try:
            sys.meta_path.remove(self)
            original = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if original is None or original.loader is None:
            return None
        return spec_from_loader(
            fullname,
            _RunLoader(original.loader),
            origin=original.origin,
            is_package=original.submodule_search_locations is not None,
        )


if os.environ.get("OPEN_SCORE_EPYMARL_WINDOWS_SAFE") == "1":
    sys.meta_path.insert(0, _RunFinder())
