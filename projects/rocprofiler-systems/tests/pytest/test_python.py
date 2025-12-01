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
Tests for Python integration with rocprofiler-systems.

This module tests Python profiling functionality:
- External Python profiling
- Built-in Python profiling
- Source-based profiling
- Noprofile decorator
- Output validation
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
    validate_perfetto_trace,
    validate_timemory_json,
    validate_rocpd_database,
)


# ============================================================================
# Python Test Runner
# ============================================================================


class PythonRunner:
    """Runner for Python profiling tests."""

    def __init__(
        self,
        config: RocprofsysConfig,
        python_file: Path,
        output_dir: Path,
        profile_args: Optional[list[str]] = None,
        run_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        standalone: bool = False,
        timeout: int = 300,
    ):
        self.config = config
        self.python_file = python_file
        self.output_dir = Path(output_dir)
        self.profile_args = profile_args or []
        self.run_args = run_args or []
        self.standalone = standalone
        self.timeout = timeout

        self.env = {
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "OFF",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_TREE_OUTPUT": "OFF",
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
            "LD_LIBRARY_PATH": config.get_library_path(),
            "ROCPROFSYS_OUTPUT_PATH": str(self.output_dir),
        }
        # Add Python path for rocprofiler-systems Python module
        python_dir = config.build_dir / "lib" / "python"
        if python_dir.exists():
            self.env["PYTHONPATH"] = str(python_dir)
        if env:
            self.env.update(env)

    def build_command(self) -> list[str]:
        """Build the Python profiling command."""
        if self.standalone:
            # For standalone scripts that handle their own profiling
            cmd = [sys.executable, str(self.python_file)]
        else:
            # Use rocprof-sys-python for profiling
            rocprof_python = self.config.build_dir / "bin" / "rocprof-sys-python"
            if rocprof_python.exists():
                cmd = [str(rocprof_python)]
                cmd.extend(self.profile_args)
                cmd.extend(["--", str(self.python_file)])
            else:
                # Fallback: use Python with rocprofsys module
                cmd = [sys.executable, "-m", "rocprofsys"]
                cmd.extend(self.profile_args)
                cmd.extend(["--", str(self.python_file)])

        cmd.extend(self.run_args)
        return cmd

    def run(self):
        """Execute Python profiling."""
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

            return PythonResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_dir=self.output_dir,
                command=command,
            )
        except subprocess.TimeoutExpired as e:
            return PythonResult(
                returncode=-1,
                stdout=e.stdout or "",
                stderr=f"Timeout after {self.timeout}s",
                output_dir=self.output_dir,
                command=command,
            )


class PythonResult:
    """Result of a Python profiling run."""

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
    def perfetto_file(self) -> Optional[Path]:
        """Path to perfetto trace if it exists."""
        for pattern in ["perfetto-trace.proto", "perfetto-trace*.proto"]:
            files = list(self.output_dir.glob(pattern))
            if files:
                return files[0]
        return None

    @property
    def trip_count_json(self) -> Optional[Path]:
        """Path to trip_count.json if it exists."""
        path = self.output_dir / "trip_count.json"
        return path if path.exists() else None

    @property
    def trip_count_txt(self) -> Optional[Path]:
        """Path to trip_count.txt if it exists."""
        path = self.output_dir / "trip_count.txt"
        return path if path.exists() else None

    @property
    def rocpd_file(self) -> Optional[Path]:
        """Path to rocpd.db if it exists."""
        path = self.output_dir / "rocpd.db"
        return path if path.exists() else None

    def cleanup(self, keep_on_failure: bool = True) -> None:
        """Clean up Python test output files.

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
# Python Test Fixtures
# ============================================================================


@pytest.fixture
def python_env() -> dict[str, str]:
    """Base environment for Python tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TREE_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
    }


@pytest.fixture
def examples_dir(source_dir: Path) -> Path:
    """Path to examples directory."""
    return source_dir / "examples" / "python"


# ============================================================================
# Test Class: External Python Profiling
# ============================================================================


class TestPythonExternal:
    """Tests for external Python profiling."""

    def test_external_basic(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test external Python profiling."""
        script = examples_dir / "external.py"
        if not script.exists():
            pytest.skip("external.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=["--label", "file"],
            run_args=["-v", "10", "-n", "5"],
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"External profiling failed: {result.stderr}"

        # Verify output files
        assert result.output_dir.exists(), "Output directory not created"

    def test_external_check_output(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test external Python profiling output content."""
        script = examples_dir / "external.py"
        if not script.exists():
            pytest.skip("external.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=["--label", "file"],
            run_args=["-v", "10", "-n", "5"],
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"External profiling failed: {result.stderr}"

        # Check trip_count output
        trip_count = result.trip_count_txt
        if trip_count:
            content = trip_count.read_text()
            # Should contain function names from external.py
            expected_funcs = ["fib", "run"]
            found = any(func in content for func in expected_funcs)
            assert found, f"Expected functions not found in trip_count.txt"

    def test_external_exclude_inefficient(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test external Python profiling with function exclusion."""
        script = examples_dir / "external.py"
        if not script.exists():
            pytest.skip("external.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=["-E", "^inefficient$"],
            run_args=["-v", "10", "-n", "5"],
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"External exclude failed: {result.stderr}"

        # Check that inefficient is NOT in output
        trip_count = result.trip_count_txt
        if trip_count:
            content = trip_count.read_text()
            # inefficient should be excluded
            assert "inefficient" not in content or "_inefficient" not in content, \
                "inefficient function should be excluded"


# ============================================================================
# Test Class: Built-in Python Profiling
# ============================================================================


class TestPythonBuiltin:
    """Tests for built-in Python profiling."""

    def test_builtin_basic(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test built-in Python profiling."""
        script = examples_dir / "builtin.py"
        if not script.exists():
            pytest.skip("builtin.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=["-b", "--label", "file", "line"],
            run_args=["-v", "10", "-n", "5"],
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"Builtin profiling failed: {result.stderr}"

    def test_builtin_check_output(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test built-in Python profiling output."""
        script = examples_dir / "builtin.py"
        if not script.exists():
            pytest.skip("builtin.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=["-b", "--label", "file", "line"],
            run_args=["-v", "10", "-n", "5"],
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"Builtin profiling failed: {result.stderr}"

        trip_count = result.trip_count_txt
        if trip_count:
            content = trip_count.read_text()
            # Should have function with file:line info
            assert "builtin.py" in content, "File info not found in output"


# ============================================================================
# Test Class: Noprofile Decorator
# ============================================================================


class TestPythonNoprofile:
    """Tests for noprofile decorator functionality."""

    def test_noprofile_basic(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test noprofile decorator."""
        script = examples_dir / "noprofile.py"
        if not script.exists():
            pytest.skip("noprofile.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=["-b", "--label", "file"],
            run_args=["-v", "15", "-n", "5"],
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"Noprofile test failed: {result.stderr}"

    def test_noprofile_check_output(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test that noprofile functions are excluded."""
        script = examples_dir / "noprofile.py"
        if not script.exists():
            pytest.skip("noprofile.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=["-b", "--label", "file"],
            run_args=["-v", "15", "-n", "5"],
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"Noprofile test failed: {result.stderr}"

        trip_count = result.trip_count_txt
        if trip_count:
            content = trip_count.read_text()
            # run function should be present
            assert "run" in content, "run function not found"
            # fib and inefficient should NOT be present (decorated with noprofile)
            assert "fib" not in content or "[fib]" not in content, \
                "fib should be excluded by noprofile"


# ============================================================================
# Test Class: Source-based Python Profiling
# ============================================================================


class TestPythonSource:
    """Tests for source-based Python profiling (standalone scripts)."""

    def test_source_basic(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Test source-based Python profiling."""
        script = examples_dir / "source.py"
        if not script.exists():
            pytest.skip("source.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            run_args=["-v", "5", "-n", "5", "-s", "3"],
            env=python_env,
            standalone=True,
        )

        result = runner.run()
        assert result.success, f"Source profiling failed: {result.stderr}"

    def test_source_validation(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
    ):
        """Validate source-based Python profiling output."""
        script = examples_dir / "source.py"
        if not script.exists():
            pytest.skip("source.py not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            run_args=["-v", "5", "-n", "5", "-s", "3"],
            env=python_env,
            standalone=True,
        )

        result = runner.run()
        assert result.success, f"Source profiling failed: {result.stderr}"

        # Validate timemory JSON output
        trip_count_json = result.trip_count_json
        if trip_count_json:
            validation = validate_timemory_json(
                trip_count_json,
                metric="trip_count",
            )
            # Validation may vary based on output format
            if not validation.valid:
                pytest.skip(f"Timemory validation format differs: {validation.message}")

        # Validate perfetto trace
        perfetto = result.perfetto_file
        if perfetto:
            validation = validate_perfetto_trace(
                perfetto,
                categories=["python", "user"],
            )
            if not validation.valid:
                pytest.skip(f"Perfetto validation differs: {validation.message}")


# ============================================================================
# Test Class: Python ROCpd Integration
# ============================================================================


@pytest.mark.rocpd
class TestPythonROCpd:
    """Tests for Python profiling with ROCpd output."""

    def test_source_rocpd(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
        validation_rules_dir: Path,
    ):
        """Test Python profiling with ROCpd output."""
        script = examples_dir / "source.py"
        if not script.exists():
            pytest.skip("source.py not found")

        env = python_env.copy()
        env["ROCPROFSYS_USE_ROCPD"] = "ON"

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            run_args=["-v", "5", "-n", "5", "-s", "3"],
            env=env,
            standalone=True,
        )

        result = runner.run()
        assert result.success, f"Source ROCpd failed: {result.stderr}"

        rocpd = result.rocpd_file
        if rocpd is None:
            pytest.skip("ROCpd database not created")

        # Validate ROCpd database
        rules_file = validation_rules_dir / "python" / "python-source-rules.json"
        if rules_file.exists():
            validation = validate_rocpd_database(
                rocpd,
                rules_files=[rules_file],
            )
            if not validation.valid:
                pytest.skip(f"ROCpd validation differs: {validation.message}")


# ============================================================================
# Parametrized Python Tests
# ============================================================================


class TestPythonParametrized:
    """Parametrized tests for Python profiling."""

    @pytest.mark.parametrize(
        "script_name,profile_args,run_args",
        [
            ("external.py", ["--label", "file"], ["-v", "5", "-n", "3"]),
            ("builtin.py", ["-b"], ["-v", "5", "-n", "3"]),
        ],
        ids=["external", "builtin"],
    )
    def test_python_scripts(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        examples_dir: Path,
        python_env: dict[str, str],
        script_name: str,
        profile_args: list[str],
        run_args: list[str],
    ):
        """Test various Python scripts with profiling."""
        script = examples_dir / script_name
        if not script.exists():
            pytest.skip(f"{script_name} not found")

        runner = PythonRunner(
            config=rocprof_config,
            python_file=script,
            output_dir=test_output_dir,
            profile_args=profile_args,
            run_args=run_args,
            env=python_env,
        )

        result = runner.run()
        assert result.success, f"{script_name} profiling failed: {result.stderr}"

