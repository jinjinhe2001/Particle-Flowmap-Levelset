"""Warp runtime helpers shared by the example runners and the tests."""

from __future__ import annotations

import logging
import platform
import sys

import warp as wp

__all__ = ["check_device", "quiet_warp", "runtime_info"]


def quiet_warp() -> None:
    """Silence Warp's initialisation banner and module-load messages (call before ``wp.init()``).

    Recent Warp versions report these through their logger (``wp.config.log_level``) and ignore
    ``wp.config.quiet``; older versions only have ``wp.config.quiet``. Both are set.
    """
    if hasattr(wp.config, "log_level"):
        wp.config.log_level = logging.WARNING
    wp.config.quiet = True


def runtime_info(device=None) -> dict:
    """Versions and the device a run used, for ``summary.json`` files and headers.

    Keys: ``device`` (e.g. ``"cuda:0"``), ``device_name`` (the GPU or CPU name), ``warp``, ``pfmls``, ``python``,
    ``platform`` and, on CUDA, ``cuda_driver`` / ``cuda_toolkit`` (``"major.minor"``).
    """
    from .. import __version__

    wp.init()
    dev = wp.get_device(device)
    info = {"device": str(dev), "device_name": getattr(dev, "name", str(dev)), "warp": wp.config.version,
            "pfmls": __version__, "python": platform.python_version(), "platform": platform.platform(terse=True)}
    if dev.is_cuda:
        for key, fn in (("cuda_driver", "get_cuda_driver_version"), ("cuda_toolkit", "get_cuda_toolkit_version")):
            f = getattr(wp, fn, None)
            if f is not None:
                try:
                    v = f()
                    info[key] = f"{v[0]}.{v[1]}" if isinstance(v, tuple) else str(v)
                except Exception:  # older Warp builds
                    pass
    return info


def check_device(device, requested=None, hint: str = "", log=None, slow: str | None = None) -> None:
    """Warn when a run falls back to the CPU.

    ``device`` is the Warp device in use, ``requested`` the device the user asked for (None: Warp's preferred
    device).  If the CPU was not requested, a warning goes to stderr (no usable CUDA device, or a driver too old for
    this Warp build), followed by ``hint`` (e.g. flags for a quick CPU run).  If it was requested and ``slow`` is
    given, ``log(slow)`` notes that the chosen size is slow on the CPU.
    """
    dev = wp.get_device(device)
    if not dev.is_cpu:
        return
    if requested is None:
        print("[runner] warning: no usable CUDA device, running on the CPU (slow). If this machine has an NVIDIA "
              "GPU, its driver may be too old for the installed warp-lang build (see the README, Install)."
              + (f" {hint}" if hint else ""), file=sys.stderr, flush=True)
    elif slow and log is not None:
        log(slow)
