"""Test configuration.

Every test runs on the CPU device; tests that need CUDA are marked ``gpu`` automatically (fixture ``cuda_device``,
a ``[cuda:0]`` parameter of the ``device`` fixture, or a ``skipif`` mark that requires CUDA) and run when a GPU is
available.  CPU-only runs: ``pytest --cpu-only`` or the environment variable ``PFMLS_TEST_DEVICE=cpu`` (both also
make the CPU Warp's default device, so tests without an explicit device stay off the GPU); ``pytest -m "not gpu"``
deselects the GPU tests.
"""

import os

import pytest
import warp as wp

from pfmls.core import quiet_warp

quiet_warp()
wp.init()

_ENV_CPU = os.environ.get("PFMLS_TEST_DEVICE", "").lower() == "cpu"


def pytest_addoption(parser):
    parser.addoption("--cpu-only", action="store_true", default=False,
                     help="run on the CPU device only (skip GPU tests; also PFMLS_TEST_DEVICE=cpu)")


def _cpu_only(config) -> bool:
    return _ENV_CPU or config.getoption("--cpu-only")


def pytest_configure(config):
    if _cpu_only(config):
        wp.set_device("cpu")


def _devices():
    devs = ["cpu"]
    if wp.is_cuda_available() and not _ENV_CPU:
        devs.append("cuda:0")
    return devs


def _needs_gpu(item) -> bool:
    if "cuda_device" in getattr(item, "fixturenames", ()):
        return True
    callspec = getattr(item, "callspec", None)
    if callspec is not None and callspec.params.get("device") == "cuda:0":
        return True
    return any("CUDA" in str(m.kwargs.get("reason", "")) for m in item.iter_markers("skipif"))


def pytest_collection_modifyitems(config, items):
    cpu_only = _cpu_only(config)
    skip = pytest.mark.skip(reason="CPU-only run (--cpu-only / PFMLS_TEST_DEVICE=cpu)")
    for item in items:
        if _needs_gpu(item):
            item.add_marker(pytest.mark.gpu)
            if cpu_only:
                item.add_marker(skip)


@pytest.fixture(params=_devices())
def device(request):
    """Every available Warp device (CPU always, CUDA when present)."""
    return request.param


@pytest.fixture
def cuda_device(request):
    if not wp.is_cuda_available() or _cpu_only(request.config):
        pytest.skip("CUDA device required")
    return "cuda:0"
