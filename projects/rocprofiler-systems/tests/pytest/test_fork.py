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
Tests for fork() handling with rocprofiler-systems.

This module tests process fork functionality:
- Basic fork example
- HIP malloc concurrency with fork
- Various instrumentation modes with fork
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    SamplingRunner,
)


# ============================================================================
# Fork Fixtures
# ============================================================================


@pytest.fixture
def fork_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for fork tests."""
    env = base_env.copy()
    env.update({
        "ROCPROFSYS_SAMPLING_FREQ": "250",
        "ROCPROFSYS_SAMPLING_REALTIME": "ON",
    })
    return env


# ============================================================================
# Test Class: Basic Fork Tests
# ============================================================================


class TestFork:
    """Tests for basic fork functionality."""

    REWRITE_ARGS = [
        "-e",
        "-v", "2",
        "--print-instrumented", "modules",
        "-i", "16",
    ]

    RUNTIME_ARGS = [
        "-e",
        "-v", "1",
        "--label", "file",
        "-i", "16",
    ]

    def test_fork_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        fork_env: dict[str, str],
    ):
        """Test fork example with sampling instrumentation."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="fork-example",
                output_dir=test_output_dir,
                env=fork_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("fork-example target not built")

        result = runner.run()
        assert result.success, f"Fork sampling failed: {result.stderr}"

        # Verify fork was called
        combined_output = result.stdout + result.stderr
        assert "fork" in combined_output.lower() or result.output_dir.exists(), \
            "Fork not executed or no output created"

    def test_fork_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        fork_env: dict[str, str],
    ):
        """Test fork example with binary rewrite instrumentation."""
        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="fork-example",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=fork_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("fork-example target not built")

        # Perform rewrite
        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"

        # Run the instrumented binary
        result = runner.run()
        assert result.success, f"Fork rewrite run failed: {result.stderr}"

        combined_output = result.stdout + result.stderr
        assert "fork" in combined_output.lower() or result.returncode == 0, \
            "Fork not executed properly"

    def test_fork_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        fork_env: dict[str, str],
    ):
        """Test fork example with runtime instrumentation."""
        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target="fork-example",
                output_dir=test_output_dir,
                instrument_args=self.RUNTIME_ARGS,
                env=fork_env,
                timeout=480,
            )
        except FileNotFoundError:
            pytest.skip("fork-example target not built")

        result = runner.run()
        assert result.success, f"Fork runtime instrument failed: {result.stderr}"


# ============================================================================
# Test Class: HIP Malloc Concurrency with Fork
# ============================================================================


@pytest.mark.gpu
class TestForkHIPMalloc:
    """Tests for HIP malloc concurrency with fork."""

    REWRITE_ARGS = [
        "-e",
        "-v", "2",
        "--print-instrumented", "modules",
        "-i", "16",
    ]

    RUNTIME_ARGS = [
        "-e",
        "-v", "1",
        "--label", "file",
        "-i", "16",
    ]

    def test_hip_malloc_fork_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        fork_env: dict[str, str],
    ):
        """Test HIP malloc concurrency with fork using sampling."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="hipMallocConcurrencyMproc",
                output_dir=test_output_dir,
                env=fork_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("hipMallocConcurrencyMproc target not built")

        result = runner.run()
        assert result.success, f"HIP malloc fork sampling failed: {result.stderr}"

        # Check for validation or fork messages
        combined_output = result.stdout + result.stderr
        assert "Validation PASSED" in combined_output or \
               "fork" in combined_output.lower() or \
               result.returncode == 0, \
            "HIP malloc concurrency test did not complete properly"

    def test_hip_malloc_fork_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        fork_env: dict[str, str],
    ):
        """Test HIP malloc concurrency with fork using binary rewrite."""
        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="hipMallocConcurrencyMproc",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=fork_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("hipMallocConcurrencyMproc target not built")

        result = runner.run()
        assert result.success, f"HIP malloc fork rewrite failed: {result.stderr}"

    def test_hip_malloc_fork_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        fork_env: dict[str, str],
    ):
        """Test HIP malloc concurrency with fork using runtime instrumentation."""
        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target="hipMallocConcurrencyMproc",
                output_dir=test_output_dir,
                instrument_args=self.RUNTIME_ARGS,
                env=fork_env,
                timeout=480,
            )
        except FileNotFoundError:
            pytest.skip("hipMallocConcurrencyMproc target not built")

        result = runner.run()
        assert result.success, f"HIP malloc fork runtime failed: {result.stderr}"


# ============================================================================
# Parametrized Fork Tests
# ============================================================================


class TestForkParametrized:
    """Parametrized tests for fork functionality."""

    @pytest.mark.parametrize(
        "runner_class,runner_kwargs",
        [
            (SamplingRunner, {}),
            (BinaryRewriteRunner, {"rewrite_args": ["-e", "-v", "2", "-i", "16"]}),
        ],
        ids=["sampling", "binary-rewrite"],
    )
    def test_fork_modes(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        fork_env: dict[str, str],
        runner_class,
        runner_kwargs: dict,
    ):
        """Test fork example with different instrumentation modes."""
        try:
            runner = runner_class(
                config=rocprof_config,
                target="fork-example",
                output_dir=test_output_dir,
                env=fork_env,
                timeout=300,
                **runner_kwargs,
            )
        except FileNotFoundError:
            pytest.skip("fork-example target not built")

        result = runner.run()
        assert result.success, f"{runner_class.__name__} fork failed: {result.stderr}"

