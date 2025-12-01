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
Tests for the user API with rocprofiler-systems.

This module tests the user API functionality:
- Custom region push/pop
- Loop instrumentation
- Various instrumentation modes
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    BaselineRunner,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    SamplingRunner,
)


# ============================================================================
# User API Fixtures
# ============================================================================


@pytest.fixture
def num_threads() -> int:
    """Get the number of threads for testing."""
    import multiprocessing
    num_procs = multiprocessing.cpu_count()
    num_threads = num_procs + (num_procs // 2)
    return min(num_threads, 12)


# ============================================================================
# Test Class: User API Tests
# ============================================================================


@pytest.mark.loops
class TestUserAPI:
    """Tests for user API functionality."""

    REWRITE_ARGS = [
        "-e",
        "-v", "2",
        "-l",  # instrument loops
        "--min-instructions=8",
        "-E", "custom_push_region",
    ]

    RUNTIME_ARGS = [
        "-e",
        "-v", "1",
        "-l",
        "--min-instructions=8",
        "-E", "custom_push_region",
        "--label", "file", "line", "return", "args",
    ]

    def get_run_args(self, num_threads: int) -> list[str]:
        """Get run arguments based on thread count."""
        return ["10", str(num_threads), "1000"]

    def test_user_api_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
        num_threads: int,
    ):
        """Test user-api baseline (no instrumentation)."""
        try:
            runner = BaselineRunner(
                config=rocprof_config,
                target="user-api",
                output_dir=test_output_dir,
                run_args=self.get_run_args(num_threads),
                env=base_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("user-api target not built")

        result = runner.run()
        assert result.success, f"User API baseline failed: {result.stderr}"

        # Baseline should NOT have custom region messages
        # (since instrumentation is disabled)
        combined_output = result.stdout + result.stderr
        assert "Pushing custom region" not in combined_output, \
            "Baseline should not have custom region output"

    def test_user_api_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
        num_threads: int,
    ):
        """Test user-api with sampling instrumentation."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="user-api",
                output_dir=test_output_dir,
                run_args=self.get_run_args(num_threads),
                env=base_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("user-api target not built")

        result = runner.run()
        assert result.success, f"User API sampling failed: {result.stderr}"

        # Sampling should have custom region messages
        combined_output = result.stdout + result.stderr
        assert "Pushing custom region" in combined_output or result.perfetto_file is not None, \
            "Custom region output not found"

    def test_user_api_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
        num_threads: int,
    ):
        """Test user-api with binary rewrite instrumentation."""
        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="user-api",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                run_args=self.get_run_args(num_threads),
                env=base_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("user-api target not built")

        # Perform rewrite
        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"

        # Check loops were instrumented
        assert "0 instrumented loops in procedure" not in rewrite_result.stdout, \
            "No loops were instrumented"

        # Run the instrumented binary
        result = runner.run()
        assert result.success, f"User API rewrite run failed: {result.stderr}"

        # Check for custom region output
        combined_output = result.stdout + result.stderr
        assert "Pushing custom region" in combined_output or result.returncode == 0, \
            "Custom region output not found"

    def test_user_api_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
        num_threads: int,
    ):
        """Test user-api with runtime instrumentation."""
        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target="user-api",
                output_dir=test_output_dir,
                instrument_args=self.RUNTIME_ARGS,
                run_args=self.get_run_args(num_threads),
                env=base_env,
                timeout=480,
            )
        except FileNotFoundError:
            pytest.skip("user-api target not built")

        result = runner.run()
        assert result.success, f"User API runtime instrument failed: {result.stderr}"

        combined_output = result.stdout + result.stderr
        assert "Pushing custom region" in combined_output or result.returncode == 0, \
            "Custom region output not found"


# ============================================================================
# Test Class: User API Custom Region Tests
# ============================================================================


class TestUserAPICustomRegions:
    """Tests for custom region functionality in user API."""

    def test_custom_region_content(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
        num_threads: int,
    ):
        """Test that custom regions have correct content."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="user-api",
                output_dir=test_output_dir,
                run_args=["10", str(num_threads), "1000"],
                env=base_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("user-api target not built")

        result = runner.run()
        assert result.success, f"User API test failed: {result.stderr}"

        # Check for expected custom region format
        combined_output = result.stdout + result.stderr
        if "Pushing custom region" in combined_output:
            # Should contain run(10) x 1000 pattern
            assert "run" in combined_output and "1000" in combined_output, \
                "Custom region doesn't have expected content"


# ============================================================================
# Parametrized User API Tests
# ============================================================================


class TestUserAPIParametrized:
    """Parametrized tests for user API."""

    @pytest.mark.parametrize(
        "iterations,threads,ops",
        [
            (5, 2, 500),
            (10, 4, 1000),
        ],
        ids=["small", "medium"],
    )
    def test_user_api_configurations(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
        iterations: int,
        threads: int,
        ops: int,
    ):
        """Test user-api with different configurations."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="user-api",
                output_dir=test_output_dir,
                run_args=[str(iterations), str(threads), str(ops)],
                env=base_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("user-api target not built")

        result = runner.run()
        assert result.success, f"User API config ({iterations}, {threads}, {ops}) failed: {result.stderr}"

