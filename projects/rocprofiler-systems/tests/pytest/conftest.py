# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Pytest configuration and fixtures for rocprofiler-systems tests.

This module provides shared fixtures and configuration for all test modules.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Generator, Any

# Add the pytest directory to Python path for rocprofsys package
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    discover_build_config,
    GPUInfo,
    detect_gpu,
)


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring a GPU"
    )
    config.addinivalue_line(
        "markers", "mpi: mark test as requiring MPI"
    )
    config.addinivalue_line(
        "markers", "rocm: mark test as requiring ROCm"
    )
    config.addinivalue_line(
        "markers", "rocpd: mark test as requiring ROCpd support"
    )
    config.addinivalue_line(
        "markers", "rocprofiler: mark test as using ROCProfiler counters"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "loops: mark test as testing loop instrumentation"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests based on markers and available resources."""
    gpu_info = detect_gpu()

    skip_gpu = pytest.mark.skip(reason="No valid GPU available")
    skip_mpi = pytest.mark.skip(reason="MPI not available")
    skip_rocpd = pytest.mark.skip(reason="ROCpd not available (requires ROCm >= 7.0)")

    mpi_available = shutil.which("mpiexec") is not None or shutil.which("mpirun") is not None

    rocpd_available = gpu_info.available  # Simplified check

    for item in items:
        if "gpu" in item.keywords and not gpu_info.available:
            item.add_marker(skip_gpu)

        if "mpi" in item.keywords and not mpi_available:
            item.add_marker(skip_mpi)

        if "rocpd" in item.keywords and not rocpd_available:
            item.add_marker(skip_rocpd)


# ============================================================================
# Session-scoped Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def rocprof_config() -> RocprofsysConfig:
    """Session-wide rocprofiler-systems configuration.

    Discovers build directory and creates configuration object.
    Can be overridden with ROCPROFSYS_BUILD_DIR environment variable.
    """
    return discover_build_config()


@pytest.fixture(scope="session")
def gpu_info() -> GPUInfo:
    """Session-wide GPU information.

    Detects available GPUs and their capabilities.
    """
    return detect_gpu()


@pytest.fixture(scope="session")
def source_dir(rocprof_config: RocprofsysConfig) -> Path:
    """Path to rocprofiler-systems source directory."""
    return rocprof_config.source_dir


@pytest.fixture(scope="session")
def build_dir(rocprof_config: RocprofsysConfig) -> Path:
    """Path to rocprofiler-systems build directory."""
    return rocprof_config.build_dir


@pytest.fixture(scope="session")
def tests_dir(source_dir: Path) -> Path:
    """Path to tests directory."""
    return source_dir / "tests"


@pytest.fixture(scope="session")
def validation_rules_dir(tests_dir: Path) -> Path:
    """Path to validation rules directory."""
    return tests_dir / "rocpd-validation-rules"


# ============================================================================
# Module-scoped Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def test_output_base(rocprof_config: RocprofsysConfig) -> Path:
    """Base directory for test outputs (module-scoped).

    All test outputs for a module are stored under this directory.
    """
    output_dir = rocprof_config.build_dir / "rocprof-sys-pytest-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Function-scoped Fixtures
# ============================================================================


@pytest.fixture
def test_output_dir(
    test_output_base: Path,
    request: pytest.FixtureRequest,
) -> Generator[Path, None, None]:
    """Unique output directory for each test.

    Creates a directory named after the test and cleans up on success.
    On failure, the directory is preserved for debugging.
    """
    test_name = request.node.name
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in test_name)
    output_dir = test_output_base / safe_name

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    yield output_dir

    # Cleanup on success unless ROCPROFSYS_KEEP_TEST_OUTPUT is set
    keep_output = os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1"
    test_failed = hasattr(request.node, "rep_call") and request.node.rep_call.failed

    if not keep_output and not test_failed and output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def base_env(rocprof_config: RocprofsysConfig) -> dict[str, str]:
    """Base environment variables for test execution."""
    return rocprof_config.get_base_environment()


@pytest.fixture
def transpose_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for transpose tests."""
    env = base_env.copy()
    env.update({
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation,hsa_api",
    })
    return env


@pytest.fixture
def rocpd_env(transpose_env: dict[str, str], gpu_info: GPUInfo) -> dict[str, str]:
    """Environment variables for ROCpd-enabled tests."""
    env = transpose_env.copy()
    if gpu_info.available:
        env["ROCPROFSYS_USE_ROCPD"] = "ON"
    return env


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_files(rocprof_config: RocprofsysConfig):
    """Session-scoped cleanup fixture that runs after all tests complete.

    Cleans up:
    - Temporary buffered storage files (/tmp/buffered_storage*.bin)
    - Temporary metadata files (/tmp/metadata*.json)
    - Empty pytest output directories
    """
    yield

    if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
        return

    import glob

    temp_patterns = [
        "/tmp/buffered_storage*.bin",
        "/tmp/metadata*.json",
    ]

    for pattern in temp_patterns:
        for filepath in glob.glob(pattern):
            try:
                Path(filepath).unlink()
            except OSError:
                pass

    output_base = rocprof_config.build_dir / "rocprof-sys-pytest-output"
    if output_base.exists():
        for child in output_base.iterdir():
            if child.is_dir() and not any(child.iterdir()):
                try:
                    child.rmdir()
                except OSError:
                    pass


# ============================================================================
# Pytest Hooks for Result Tracking
# ============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Track test results for cleanup decisions."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
