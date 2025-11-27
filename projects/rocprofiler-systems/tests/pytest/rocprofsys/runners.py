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
Test runners for different rocprofiler-systems instrumentation modes.

Provides classes for running tests with:
- Baseline execution (no instrumentation)
- Sampling instrumentation
- Binary rewrite instrumentation
- Runtime instrumentation
- rocprof-sys-run wrapper
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import RocprofsysConfig


@dataclass
class TestResult:
    """Result of a test execution.

    Attributes:
        returncode: Process exit code
        stdout: Standard output content
        stderr: Standard error content
        output_dir: Directory containing output files
        command: The command that was executed
        env: Environment variables used
        duration: Execution time in seconds (if measured)
    """

    returncode: int
    stdout: str
    stderr: str
    output_dir: Path
    command: list[str]
    env: dict[str, str]
    duration: Optional[float] = None

    @property
    def success(self) -> bool:
        """Whether the test completed successfully."""
        return self.returncode == 0

    @property
    def perfetto_file(self) -> Optional[Path]:
        """Path to perfetto trace file if it exists."""
        candidates = [
            self.output_dir / "perfetto-trace.proto",
            self.output_dir / "perfetto-trace-0.proto",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        protos = list(self.output_dir.glob("perfetto-trace*.proto"))
        return protos[0] if protos else None

    @property
    def rocpd_file(self) -> Optional[Path]:
        """Path to rocpd database file if it exists."""
        candidate = self.output_dir / "rocpd.db"
        if candidate.exists():
            return candidate
        # Try globbing
        dbs = list(self.output_dir.glob("*.db"))
        return dbs[0] if dbs else None

    @property
    def timemory_files(self) -> list[Path]:
        """List of timemory output files."""
        return list(self.output_dir.glob("*.json")) + list(
            self.output_dir.glob("*.txt")
        )

    def get_output_file(self, pattern: str) -> Optional[Path]:
        """Get an output file matching the given pattern.

        Args:
            pattern: Glob pattern to match

        Returns:
            First matching file or None
        """
        matches = list(self.output_dir.glob(pattern))
        return matches[0] if matches else None

    def assert_file_exists(self, filename: str) -> Path:
        """Assert that an output file exists and return its path.

        Args:
            filename: Name of the file to check

        Returns:
            Path to the file

        Raises:
            AssertionError: If file doesn't exist
        """
        path = self.output_dir / filename
        assert path.exists(), f"Expected output file not found: {path}"
        return path


class BaseRunner(ABC):
    """Abstract base class for test runners."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        run_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        timeout: int = 300,
        mpi_ranks: int = 0,
    ):
        """Initialize runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            run_args: Arguments to pass to the target
            env: Additional environment variables
            timeout: Execution timeout in seconds
            mpi_ranks: Number of MPI ranks (0 = no MPI)
        """
        self.config = config
        self.target = target
        self.target_exe = config.get_target_executable(target)
        self.output_dir = Path(output_dir)
        self.run_args = run_args or []
        self.timeout = timeout
        self.mpi_ranks = mpi_ranks

        self.env = config.get_base_environment()
        self.env["ROCPROFSYS_OUTPUT_PATH"] = str(self.output_dir)
        if env:
            self.env.update(env)

    @abstractmethod
    def build_command(self) -> list[str]:
        """Build the command to execute.

        Returns:
            List of command components
        """
        pass

    def _wrap_with_mpi(self, command: list[str]) -> list[str]:
        """Wrap command with MPI launcher if needed.

        Args:
            command: Base command

        Returns:
            Command wrapped with mpiexec if MPI is enabled
        """
        if self.mpi_ranks > 0 and self.config.mpiexec:
            mpi_cmd = [
                str(self.config.mpiexec),
                "-n",
                str(self.mpi_ranks),
            ]

            try:
                result = subprocess.run(
                    [str(self.config.mpiexec), "--oversubscribe", "-n", "1", "true"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    mpi_cmd.insert(1, "--oversubscribe")
            except (subprocess.TimeoutExpired, OSError):
                pass

            return mpi_cmd + command

        return command

    def run(self) -> TestResult:
        """Execute the test.

        Returns:
            TestResult with execution results
        """
        import time

        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = self.build_command()
        command = self._wrap_with_mpi(command)

        full_env = os.environ.copy()
        full_env.update(self.env)

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=full_env,
                cwd=self.config.build_dir,
            )
            duration = time.time() - start_time

            return TestResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_dir=self.output_dir,
                command=command,
                env=self.env,
                duration=duration,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return TestResult(
                returncode=-1,
                stdout=e.stdout or "",
                stderr=f"Timeout after {self.timeout}s\n{e.stderr or ''}",
                output_dir=self.output_dir,
                command=command,
                env=self.env,
                duration=duration,
            )


class BaselineRunner(BaseRunner):
    """Run target without any instrumentation."""

    def build_command(self) -> list[str]:
        return [str(self.target_exe)] + self.run_args


class SamplingRunner(BaseRunner):
    """Run target with sampling instrumentation."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        sample_args: Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize sampling runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            sample_args: Arguments for rocprof-sys-sample
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(config, target, output_dir, **kwargs)
        self.sample_args = sample_args or []

    def build_command(self) -> list[str]:
        return (
            [str(self.config.rocprof_sample)]
            + self.sample_args
            + ["--", str(self.target_exe)]
            + self.run_args
        )


class BinaryRewriteRunner(BaseRunner):
    """Run binary rewrite instrumentation (two-phase: rewrite then run)."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        rewrite_args: Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize binary rewrite runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            rewrite_args: Arguments for rocprof-sys-instrument
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(config, target, output_dir, **kwargs)
        self.rewrite_args = rewrite_args or []
        self.instrumented_exe = output_dir / f"{target}.inst"

    def rewrite(self) -> TestResult:
        """Perform binary rewrite phase.

        Returns:
            TestResult from rewrite operation
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        command = (
            [str(self.config.rocprof_instrument)]
            + ["-o", str(self.instrumented_exe)]
            + self.rewrite_args
            + ["--print-instrumented", "functions"]
            + ["--", str(self.target_exe)]
        )

        full_env = os.environ.copy()
        full_env.update(self.env)

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=full_env,
            cwd=self.config.build_dir,
        )

        return TestResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            output_dir=self.output_dir,
            command=command,
            env=self.env,
        )

    def build_command(self) -> list[str]:
        """Build command to run the instrumented binary."""
        return (
            [str(self.config.rocprof_run), "--", str(self.instrumented_exe)]
            + self.run_args
        )

    def run(self) -> TestResult:
        """Execute full rewrite + run sequence.

        Returns:
            TestResult from run phase (rewrite must succeed first)
        """
        # First, perform rewrite
        rewrite_result = self.rewrite()
        if not rewrite_result.success:
            return rewrite_result

        # Then run the instrumented binary
        return super().run()


class RuntimeInstrumentRunner(BaseRunner):
    """Run target with runtime instrumentation."""

    def __init__(
        self,
        config: RocprofsysConfig,
        target: str,
        output_dir: Path,
        instrument_args: Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize runtime instrument runner.

        Args:
            config: rocprofiler-systems configuration
            target: Name of target executable
            output_dir: Directory for output files
            instrument_args: Arguments for rocprof-sys-instrument
            **kwargs: Additional arguments passed to BaseRunner
        """
        super().__init__(config, target, output_dir, **kwargs)
        self.instrument_args = instrument_args or []

    def build_command(self) -> list[str]:
        return (
            [str(self.config.rocprof_instrument)]
            + self.instrument_args
            + ["--print-instrumented", "functions"]
            + ["--", str(self.target_exe)]
            + self.run_args
        )


class SysRunRunner(BaseRunner):
    """Run target with rocprof-sys-run wrapper."""

    def build_command(self) -> list[str]:
        return (
            [str(self.config.rocprof_run), "--", str(self.target_exe)]
            + self.run_args
        )

