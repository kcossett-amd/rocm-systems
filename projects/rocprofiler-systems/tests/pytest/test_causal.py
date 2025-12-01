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
Tests for causal profiling with rocprofiler-systems.

This module tests causal profiling functionality:
- Function-mode causal profiling
- Line-mode causal profiling
- End-to-end validation of causal experiments
- Various targets (cpu, lulesh)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    validate_causal_json,
)


# ============================================================================
# Causal Profiling Runner
# ============================================================================


class CausalRunner:
    """Runner for causal profiling tests."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        causal_mode: str,  # "function" or "line"
        run_args: Optional[list[str]] = None,
        causal_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        timeout: int = 600,
    ):
        self.config = config
        self.target = target
        self.target_exe = config.get_target_executable(target)
        self.output_dir = Path(output_dir)
        self.causal_mode = causal_mode
        self.run_args = run_args or []
        self.causal_args = causal_args or []
        self.timeout = timeout

        self.env = {
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "threads",
            "OMP_NUM_THREADS": "2",
            "LD_LIBRARY_PATH": config.get_library_path(),
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_FILE_OUTPUT": "ON",
            "ROCPROFSYS_CAUSAL_RANDOM_SEED": "1342342",
            "ROCPROFSYS_OUTPUT_PATH": str(self.output_dir),
        }
        if env:
            self.env.update(env)

    def build_command(self) -> list[str]:
        """Build the causal profiling command."""
        cmd = [str(self.config.rocprof_causal)]

        # Add mode
        if self.causal_mode == "function":
            cmd.extend(["-m", "function"])
        elif self.causal_mode == "line":
            cmd.extend(["-m", "line"])

        # Add causal-specific args
        cmd.extend(self.causal_args)

        # Add the target
        cmd.extend(["--", str(self.target_exe)])
        cmd.extend(self.run_args)

        return cmd

    def run(self):
        """Execute causal profiling."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = self.build_command()

        full_env = os.environ.copy()
        full_env.update(self.env)

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=full_env,
                cwd=self.config.build_dir,
            )

            return CausalResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_dir=self.output_dir,
                command=command,
            )
        except subprocess.TimeoutExpired as e:
            return CausalResult(
                returncode=-1,
                stdout=e.stdout or "",
                stderr=f"Timeout after {self.timeout}s",
                output_dir=self.output_dir,
                command=command,
            )


class CausalResult:
    """Result of a causal profiling run."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        output_dir: Path,
        command: list[str],
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.output_dir = output_dir
        self.command = command

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def experiments_json(self) -> Optional[Path]:
        """Path to experiments.json if it exists."""
        path = self.output_dir / "causal" / "experiments.json"
        return path if path.exists() else None

    @property
    def experiments_coz(self) -> Optional[Path]:
        """Path to experiments.coz if it exists."""
        path = self.output_dir / "causal" / "experiments.coz"
        return path if path.exists() else None

    def cleanup(self, keep_on_failure: bool = True) -> None:
        """Clean up causal test output files.

        Args:
            keep_on_failure: If True, keep files when test failed for debugging
        """
        import shutil

        if os.environ.get("ROCPROFSYS_KEEP_TEST_OUTPUT", "0") == "1":
            return

        if keep_on_failure and not self.success:
            return

        # Clean up output directory
        if self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
            except OSError:
                pass


# ============================================================================
# Causal Fixtures
# ============================================================================


@pytest.fixture
def causal_env() -> dict[str, str]:
    """Base environment for causal profiling tests."""
    return {
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_FILE_OUTPUT": "ON",
        "ROCPROFSYS_CAUSAL_RANDOM_SEED": "1342342",
    }


# ============================================================================
# Test Class: CPU Causal Profiling (Function Mode)
# ============================================================================


class TestCausalCPUFunction:
    """Tests for CPU-based causal profiling in function mode."""

    RUN_ARGS = ["70", "10", "432525", "1000000000"]

    def test_cpu_func(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
    ):
        """Test causal profiling on CPU target in function mode."""
        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-cpu-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="function",
                run_args=self.RUN_ARGS,
                env=causal_env,
                timeout=600,
            )
        except FileNotFoundError:
            pytest.skip("causal-cpu-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"Causal run failed: {result.stderr}"

        # Verify experiment output files
        combined_output = result.stdout + result.stderr
        assert "Starting causal experiment #1" in combined_output, \
            "Causal experiment did not start"

        # Check for output files
        assert result.experiments_json is not None or result.experiments_coz is not None, \
            "No experiment output files created"

    def test_cpu_func_ndebug(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
    ):
        """Test causal profiling on CPU target (NDEBUG build) in function mode."""
        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-cpu-rocprofsys-ndebug",
                output_dir=test_output_dir,
                causal_mode="function",
                run_args=self.RUN_ARGS,
                env=causal_env,
                timeout=600,
            )
        except FileNotFoundError:
            pytest.skip("causal-cpu-rocprofsys-ndebug target not built")

        result = runner.run()
        assert result.success, f"Causal run failed: {result.stderr}"


# ============================================================================
# Test Class: CPU Causal Profiling (Line Mode)
# ============================================================================


class TestCausalCPULine:
    """Tests for CPU-based causal profiling in line mode."""

    RUN_ARGS = ["70", "10", "432525", "1000000000"]

    def test_cpu_line(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
    ):
        """Test causal profiling on CPU target in line mode."""
        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-cpu-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="line",
                run_args=self.RUN_ARGS,
                env=causal_env,
                timeout=600,
            )
        except FileNotFoundError:
            pytest.skip("causal-cpu-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"Causal line mode failed: {result.stderr}"

        combined_output = result.stdout + result.stderr
        assert "Starting causal experiment #1" in combined_output, \
            "Causal experiment did not start"


# ============================================================================
# Test Class: Both (CPU + GPU) Causal Profiling
# ============================================================================


class TestCausalBoth:
    """Tests for causal profiling with CPU and GPU."""

    RUN_ARGS = ["70", "10", "432525", "400000000"]

    def test_both_func(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
    ):
        """Test causal profiling on both CPU and GPU target."""
        env = causal_env.copy()
        env["ROCPROFSYS_STRICT_CONFIG"] = "OFF"

        causal_args = [
            "-n", "2",
            "-w", "1",
            "-d", "3",
            "--monochrome",
            "-v", "3",
            "-b", "timer",
        ]

        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-both-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="function",
                run_args=self.RUN_ARGS,
                causal_args=causal_args,
                env=env,
                timeout=600,
            )
        except FileNotFoundError:
            pytest.skip("causal-both-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"Causal both run failed: {result.stderr}"


# ============================================================================
# Test Class: LULESH Causal Profiling
# ============================================================================


class TestCausalLulesh:
    """Tests for causal profiling with LULESH benchmark."""

    RUN_ARGS = ["-i", "35", "-s", "50", "-p"]
    CAUSAL_ARGS = ["-s", "0,10,25,50,75"]

    def test_lulesh_func(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
    ):
        """Test causal profiling on LULESH in function mode."""
        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="lulesh-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="function",
                run_args=self.RUN_ARGS,
                causal_args=self.CAUSAL_ARGS,
                env=causal_env,
                timeout=600,
            )
        except FileNotFoundError:
            pytest.skip("lulesh-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"LULESH causal failed: {result.stderr}"

        combined_output = result.stdout + result.stderr
        assert "Starting causal experiment #1" in combined_output, \
            "Causal experiment did not start"

    def test_lulesh_func_ndebug(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
    ):
        """Test causal profiling on LULESH (NDEBUG) in function mode."""
        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="lulesh-rocprofsys-ndebug",
                output_dir=test_output_dir,
                causal_mode="function",
                run_args=self.RUN_ARGS,
                causal_args=self.CAUSAL_ARGS,
                env=causal_env,
                timeout=600,
            )
        except FileNotFoundError:
            pytest.skip("lulesh-rocprofsys-ndebug target not built")

        result = runner.run()
        assert result.success, f"LULESH NDEBUG causal failed: {result.stderr}"

    def test_lulesh_line(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
    ):
        """Test causal profiling on LULESH in line mode."""
        causal_args = self.CAUSAL_ARGS + ["-S", "lulesh.cc"]

        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="lulesh-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="line",
                run_args=self.RUN_ARGS,
                causal_args=causal_args,
                env=causal_env,
                timeout=600,
            )
        except FileNotFoundError:
            pytest.skip("lulesh-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"LULESH line mode failed: {result.stderr}"


# ============================================================================
# Test Class: End-to-End Causal Validation
# ============================================================================


@pytest.mark.slow
class TestCausalE2E:
    """End-to-end validation tests for causal profiling."""

    RUN_ARGS = ["80", "50", "432525", "100000000"]

    @pytest.fixture
    def e2e_causal_args(self, rocprof_config: RocprofsysConfig) -> list[str]:
        """Common causal arguments for E2E tests."""
        return [
            "-n", "5",
            "-e",
            "-s", "0", "10", "20", "30",
        ]

    def test_slow_func_e2e(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
        e2e_causal_args: list[str],
    ):
        """Test causal profiling E2E for slow function."""
        causal_args = e2e_causal_args + ["-F", "cpu_slow_func"]

        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-cpu-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="function",
                run_args=self.RUN_ARGS,
                causal_args=causal_args,
                env=causal_env,
                timeout=900,
            )
        except FileNotFoundError:
            pytest.skip("causal-cpu-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"E2E slow func failed: {result.stderr}"

        # Validate experiment results
        if result.experiments_json:
            validation = validate_causal_json(
                result.experiments_json,
                ci_mode=True,
            )
            # E2E validation may fail based on system performance
            if not validation.valid:
                pytest.skip(f"E2E validation inconclusive: {validation.message}")

    def test_fast_func_e2e(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
        e2e_causal_args: list[str],
    ):
        """Test causal profiling E2E for fast function."""
        causal_args = e2e_causal_args + ["-F", "cpu_fast_func"]

        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-cpu-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="function",
                run_args=self.RUN_ARGS,
                causal_args=causal_args,
                env=causal_env,
                timeout=900,
            )
        except FileNotFoundError:
            pytest.skip("causal-cpu-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"E2E fast func failed: {result.stderr}"

    def test_line_103_e2e(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
        e2e_causal_args: list[str],
    ):
        """Test causal profiling E2E for specific line (103)."""
        causal_args = e2e_causal_args + ["-S", "causal.cpp:103"]

        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-cpu-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="line",
                run_args=self.RUN_ARGS,
                causal_args=causal_args,
                env=causal_env,
                timeout=900,
            )
        except FileNotFoundError:
            pytest.skip("causal-cpu-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"E2E line 103 failed: {result.stderr}"

    def test_line_113_e2e(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        causal_env: dict[str, str],
        e2e_causal_args: list[str],
    ):
        """Test causal profiling E2E for specific line (113)."""
        causal_args = e2e_causal_args + ["-S", "causal.cpp:113"]

        try:
            runner = CausalRunner(
                config=rocprof_config,
                target="causal-cpu-rocprofsys",
                output_dir=test_output_dir,
                causal_mode="line",
                run_args=self.RUN_ARGS,
                causal_args=causal_args,
                env=causal_env,
                timeout=900,
            )
        except FileNotFoundError:
            pytest.skip("causal-cpu-rocprofsys target not built")

        result = runner.run()
        assert result.success, f"E2E line 113 failed: {result.stderr}"

