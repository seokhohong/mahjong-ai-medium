#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict

import torch  # type: ignore


def _detect_os_from_env(os_hint: str | None) -> str:
    if os_hint in ("windows", "mac", "linux"):
        return os_hint  # type: ignore[return-value]
    plat = os.sys.platform
    if plat.startswith("win"):
        return "windows"
    if plat == "darwin":
        return "mac"
    return "linux"


def _apply_runtime_config(*, os_hint: str | None, start_method: str | None, torch_threads: int | None, interop_threads: int | None) -> None:
    # Set multiprocessing start method if specified or if platform needs it
    resolved_os = _detect_os_from_env(os_hint)
    try:
        import torch.multiprocessing as mp
        if start_method is None:
            # Defaults: mac/windows -> spawn; linux -> leave default (often fork)
            if resolved_os in ("mac", "windows"):
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method("spawn", force=True)
        else:
            if mp.get_start_method(allow_none=True) != start_method:
                mp.set_start_method(start_method, force=True)
    except Exception:
        pass

    # Torch intra/inter-op threads
    try:
        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(int(torch_threads))
        if interop_threads is not None and interop_threads > 0:
            torch.set_num_interop_threads(int(interop_threads))
    except Exception:
        pass


def _resolve_loader_defaults(*, os_hint: str | None, dl_workers: int | None, pin_memory: bool | None, prefetch_factor: int | None, persistent_workers: bool | None) -> Dict[str, Any]:
    resolved_os = _detect_os_from_env(os_hint)
    cpu_count = max(1, (os.cpu_count() or 1))
    # Reasonable defaults per-OS
    if dl_workers is None:
        if resolved_os == "mac":
            # macOS often benefits from a few workers; avoid oversubscription
            dl_workers = min(8, max(0, cpu_count // 2))
        elif resolved_os == "windows":
            dl_workers = min(8, max(0, cpu_count // 2))
        else:  # linux
            dl_workers = min(16, max(0, cpu_count - 2))
    if pin_memory is None:
        pin_memory = (resolved_os != "mac")  # Pinned memory less impactful for MPS
    if prefetch_factor is None:
        prefetch_factor = 2
    if persistent_workers is None:
        persistent_workers = True
    return {
        "dl_workers": int(max(0, dl_workers)),
        "pin_memory": bool(pin_memory),
        "prefetch_factor": int(max(1, prefetch_factor)),
        "persistent_workers": bool(persistent_workers),
    }
