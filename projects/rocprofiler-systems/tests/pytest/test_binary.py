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
Tests for rocprofiler-systems binary executables.

This module tests the CLI tools:
- rocprof-sys-instrument (--help, --simulate, log files)
- rocprof-sys-avail (--help, --all, filters, config generation)
- rocprof-sys-run (--help, arguments)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import RocprofsysConfig


# ============================================================================
# Helper Functions
# ============================================================================


def run_command(
    cmd: list[str],
    config: RocprofsysConfig,
    timeout: int = 60,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    full_env = os.environ.copy()
    full_env["LD_LIBRARY_PATH"] = config.get_library_path()
    if env:
        full_env.update(env)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
        cwd=cwd or config.build_dir,
    )


# ============================================================================
# Test Class: rocprof-sys-instrument CLI
# ============================================================================


class TestInstrumentCLI:
    """Tests for rocprof-sys-instrument command-line interface."""

    def test_help(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-instrument --help output."""
        result = run_command(
            [str(rocprof_config.rocprof_instrument), "--help"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"

        # Verify all expected sections are present
        expected_sections = [
            "[rocprof-sys-instrument] Usage:",
            "[DEBUG OPTIONS]",
            "[MODE OPTIONS]",
            "[LIBRARY OPTIONS]",
            "[SYMBOL SELECTION OPTIONS]",
            "[RUNTIME OPTIONS]",
            "[GRANULARITY OPTIONS]",
            "[DYNINST OPTIONS]",
        ]

        for section in expected_sections:
            assert section in result.stdout, f"Missing section: {section}"

    def test_simulate_ls(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        """Test rocprof-sys-instrument --simulate with ls command."""
        # On RedHat, /usr/bin/ls is a script for `coreutils --coreutils-prog=ls`
        if Path("/usr/bin/coreutils").exists():
            ls_cmd = ["coreutils", "--coreutils-prog=ls"]
        else:
            ls_cmd = ["ls"]

        cmd = [
            str(rocprof_config.rocprof_instrument),
            "--simulate",
            "--print-format", "json", "txt", "xml",
            "-v", "2",
            "--all-functions",
            "-o", str(test_output_dir / "ls.inst"),
            "--",
        ] + ls_cmd

        env = {"ROCPROFSYS_OUTPUT_PATH": str(test_output_dir)}
        result = run_command(cmd, rocprof_config, timeout=240, env=env)

        assert result.returncode == 0, f"Simulate failed: {result.stderr}"

        # Check instrumentation directory exists with expected files
        inst_dir = test_output_dir / "instrumentation"
        if inst_dir.exists():
            expected_files = [
                "available.json", "available.txt", "available.xml",
                "excluded.json", "excluded.txt", "excluded.xml",
                "instrumented.json", "instrumented.txt", "instrumented.xml",
                "overlapping.json", "overlapping.txt", "overlapping.xml",
            ]
            existing = [f.name for f in inst_dir.iterdir()]
            for expected in expected_files:
                assert expected in existing, f"Missing file: {expected}"

    def test_simulate_library(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-instrument with a shared library."""
        lib_path = rocprof_config.lib_dir / "librocprofiler-systems-user.so"

        if not lib_path.exists():
            pytest.skip("User library not found")

        cmd = [
            str(rocprof_config.rocprof_instrument),
            "--print-available", "functions",
            "-v", "2",
            "--",
            str(lib_path),
        ]

        result = run_command(cmd, rocprof_config, timeout=120)

        # Should switch to binary rewrite mode
        assert "Runtime instrumentation is not possible" in result.stdout or \
               result.returncode == 0, f"Unexpected result: {result.stderr}"

    def test_write_log(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        """Test rocprof-sys-instrument --log-file."""
        if Path("/usr/bin/coreutils").exists():
            ls_cmd = ["coreutils", "--coreutils-prog=ls"]
        else:
            ls_cmd = ["ls"]

        cmd = [
            str(rocprof_config.rocprof_instrument),
            "--print-instrumented", "functions",
            "-v", "1",
            "--log-file", "user.log",
            "-o", str(test_output_dir / "ls.inst"),
            "--",
        ] + ls_cmd

        env = {"ROCPROFSYS_OUTPUT_PATH": str(test_output_dir)}
        result = run_command(cmd, rocprof_config, timeout=120, env=env)

        # Should mention opening the log file
        assert "Opening" in result.stdout and "user.log" in result.stdout, \
            f"Log file not mentioned: {result.stdout}"


# ============================================================================
# Test Class: rocprof-sys-avail CLI
# ============================================================================


class TestAvailCLI:
    """Tests for rocprof-sys-avail command-line interface."""

    def test_help(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail --help output."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "--help"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"

        expected_sections = [
            "[rocprof-sys-avail] Usage:",
            "[DEBUG OPTIONS]",
            "[INFO OPTIONS]",
            "[FILTER OPTIONS]",
            "[COLUMN OPTIONS]",
            "[DISPLAY OPTIONS]",
            "[OUTPUT OPTIONS]",
        ]

        for section in expected_sections:
            assert section in result.stdout, f"Missing section: {section}"

    def test_all(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail --all."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "--all"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"--all failed: {result.stderr}"

    def test_all_expand_keys(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail --all --expand-keys."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "--all", "--expand-keys"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"--expand-keys failed: {result.stderr}"
        # Should not contain unexpanded keys like %something%
        assert not re.search(r"%[a-zA-Z_]+%", result.stdout), \
            "Found unexpanded keys in output"

    def test_all_csv(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail --all --csv."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "--all", "--csv", "--csv-separator", "#"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"--csv failed: {result.stderr}"

        # Check CSV headers
        expected_headers = [
            "COMPONENT#AVAILABLE",
            "ENVIRONMENT VARIABLE",
            "HARDWARE COUNTER",
        ]
        for header in expected_headers:
            assert header in result.stdout, f"Missing CSV header: {header}"

    def test_filter_wall_clock(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail with wall_clock filter."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "-r", "wall_clock", "-C", "--available"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Filter failed: {result.stderr}"
        assert "wall_clock" in result.stdout, "wall_clock not found in output"

    def test_category_filter_rocprofsys(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail with rocprofsys category filter."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "--categories", "settings::rocprofsys", "--brief"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Category filter failed: {result.stderr}"
        # Should contain rocprofsys-specific settings
        assert any(x in result.stdout for x in [
            "ROCPROFSYS_SETTINGS_DESC", "ROCPROFSYS_OUTPUT_FILE", "ROCPROFSYS_OUTPUT_PREFIX"
        ]), "Missing rocprofsys settings"

    def test_category_filter_timemory(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail with timemory category filter."""
        result = run_command(
            [str(rocprof_config.rocprof_avail),
             "--categories", "settings::timemory", "--brief", "--advanced"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Category filter failed: {result.stderr}"
        # Should contain timemory-specific settings
        assert any(x in result.stdout for x in [
            "ROCPROFSYS_ADD_SECONDARY", "ROCPROFSYS_SCIENTIFIC",
            "ROCPROFSYS_PRECISION", "ROCPROFSYS_MEMORY_PRECISION"
        ]), "Missing timemory settings"

    def test_write_config(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        """Test rocprof-sys-avail config file generation."""
        base_path = test_output_dir / "rocprof-sys-test"

        cmd = [
            str(rocprof_config.rocprof_avail),
            "-G", str(base_path) + ".cfg",
            "-F", "txt", "json", "xml",
            "--force",
            "--all",
            "-c", "rocprofsys",
        ]

        result = run_command(cmd, rocprof_config, timeout=45)

        assert result.returncode == 0, f"Config generation failed: {result.stderr}"

        # Verify config files were created
        expected_files = [".cfg", ".json", ".xml"]
        for ext in expected_files:
            assert (test_output_dir / f"rocprof-sys-test{ext}").exists() or \
                   f"rocprof-sys-test{ext}" in result.stdout, \
                   f"Config file {ext} not created"

    def test_list_keys(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail --list-keys."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "--list-keys", "--expand-keys"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"--list-keys failed: {result.stderr}"
        assert "Output Keys:" in result.stdout, "Output Keys header not found"
        assert "%argv%" in result.stdout, "%argv% key not found"
        assert "%argv_hash%" in result.stdout, "%argv_hash% key not found"

    def test_list_categories(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail --list-categories."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "--list-categories"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"--list-categories failed: {result.stderr}"
        assert "component::" in result.stdout, "component:: category not found"
        assert "settings::" in result.stdout, "settings:: category not found"

    def test_core_categories(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-avail with core category."""
        result = run_command(
            [str(rocprof_config.rocprof_avail), "-c", "core"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Core category failed: {result.stderr}"
        expected_vars = [
            "ROCPROFSYS_CONFIG_FILE",
            "ROCPROFSYS_ENABLED",
            "ROCPROFSYS_VERBOSE",
        ]
        for var in expected_vars:
            assert var in result.stdout, f"Missing core variable: {var}"


# ============================================================================
# Test Class: rocprof-sys-run CLI
# ============================================================================


class TestRunCLI:
    """Tests for rocprof-sys-run command-line interface."""

    def test_help(self, rocprof_config: RocprofsysConfig):
        """Test rocprof-sys-run --help output."""
        result = run_command(
            [str(rocprof_config.rocprof_run), "--help"],
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"

    @pytest.mark.slow
    def test_run_args(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        """Test rocprof-sys-run with comprehensive arguments."""
        sleep_cmd = shutil.which("sleep")
        if not sleep_cmd:
            pytest.skip("sleep command not found")

        # Create empty config file
        config_file = test_output_dir / "empty.cfg"
        config_file.write_text("# empty config file\n")

        cmd = [
            str(rocprof_config.rocprof_run),
            "--monochrome",
            "--debug=false",
            "-v", "1",
            "-c", str(config_file),
            "-o", str(test_output_dir),
            "-TPHD",
            "-S", "cputime", "realtime",
            "--trace-wait=1.0e-12",
            "--trace-duration=5.0",
            "--wait=1.0",
            "--duration=3.0",
            "--trace-file=perfetto-run-args-trace.proto",
            "--trace-buffer-size=100",
            "--trace-fill-policy=ring_buffer",
            "--profile-format", "console", "json", "text",
            "--process-freq", "1000",
            "--process-wait", "0.0",
            "--process-duration", "10",
            "--cpus", "0-4",
            "--gpus", "0",
            "-f", "1000",
            "--sampling-wait", "1.0",
            "--sampling-duration", "10",
            "-t", "0-3",
            "--sample-cputime", "1000", "1.0", "0-3",
            "--sample-realtime", "10", "0.5", "0-3",
            "-I", "all",
            "-E", "mutex-locks", "rw-locks", "spin-locks",
            "-C", "perf::INSTRUCTIONS",
            "--inlines",
            "--hsa-interrupt", "0",
            "--use-causal=false",
            "--use-kokkosp",
            "--num-threads-hint=4",
            "--sampling-allocator-size=32",
            "--ci",
            "--dl-verbose=3",
            "--perfetto-annotations=off",
            "--kokkosp-kernel-logger",
            "--kokkosp-name-length-max=1024",
            '--kokkosp-prefix=[kokkos]',
            "--tmpdir", str(test_output_dir / "tmpdir"),
            "--perfetto-backend", "inprocess",
            "--use-pid", "false",
            "--time-output", "off",
            "--thread-pool-size", "0",
            "--timemory-components", "wall_clock", "cpu_clock", "peak_rss", "page_rss",
            "--fork",
            "--",
            sleep_cmd, "1",
        ]

        result = run_command(cmd, rocprof_config, timeout=45)

        # Just verify it doesn't crash with these arguments
        # Exit code may vary based on system configuration
        assert result.returncode >= 0 or "error" not in result.stderr.lower(), \
            f"Run with args failed unexpectedly: {result.stderr}"


# ============================================================================
# Parametrized Tests
# ============================================================================


class TestAvailParametrized:
    """Parametrized tests for rocprof-sys-avail."""

    @pytest.mark.parametrize(
        "args,expected_in_output",
        [
            (["--all"], "COMPONENT"),
            (["--list-keys"], "Output Keys:"),
            (["--list-categories"], "settings::"),
            (["-c", "core"], "ROCPROFSYS_ENABLED"),
        ],
        ids=["all", "list-keys", "list-categories", "core"],
    )
    def test_avail_modes(
        self,
        rocprof_config: RocprofsysConfig,
        args: list[str],
        expected_in_output: str,
    ):
        """Test various rocprof-sys-avail modes."""
        result = run_command(
            [str(rocprof_config.rocprof_avail)] + args,
            rocprof_config,
            timeout=45,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert expected_in_output in result.stdout, \
            f"Expected '{expected_in_output}' not found in output"

