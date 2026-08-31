"""Emit the CPU/thread runtime facts used by the stock reproduction launcher."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import gymnasium
import numpy
import sacred
import smaclite
import torch
import yaml


def windows_process_affinity_mask() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    ok = kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(process_mask.value)


def main() -> None:
    variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    smaclite_module = Path(smaclite.__file__).resolve()
    checkout_value = os.environ.get("OPEN_SCORE_SMACLITE_CHECKOUT")
    checkout = Path(checkout_value).resolve() if checkout_value else None
    imported_from_checkout = False
    if checkout is not None:
        try:
            smaclite_module.relative_to(checkout)
            imported_from_checkout = True
        except ValueError:
            pass
    scenario_root = smaclite_module.parent / "env" / "maps" / "smaclite_maps"
    scenario_digest = hashlib.sha256()
    scenario_files = sorted(scenario_root.glob("*.json"))
    for path in scenario_files:
        scenario_digest.update(path.name.encode("utf-8"))
        scenario_digest.update(b"\0")
        scenario_digest.update(path.read_bytes())
        scenario_digest.update(b"\n")
    payload = {
        "platform": platform.platform(),
        "pid": os.getpid(),
        "process_affinity_mask": windows_process_affinity_mask(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_version": platform.python_version(),
        "dependency_versions": {
            "gymnasium": gymnasium.__version__,
            "numpy": numpy.__version__,
            "sacred": sacred.__version__,
            "smaclite": importlib.metadata.version("smaclite"),
            "torch": torch.__version__,
            "pyyaml": yaml.__version__,
        },
        "smaclite_module_path": str(smaclite_module),
        "smaclite_expected_checkout": None if checkout is None else str(checkout),
        "smaclite_imported_from_expected_checkout": imported_from_checkout,
        "stock_scenario_file_count": len(scenario_files),
        "stock_scenario_combined_sha256": scenario_digest.hexdigest(),
        "thread_environment": {name: os.environ.get(name) for name in variables},
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
