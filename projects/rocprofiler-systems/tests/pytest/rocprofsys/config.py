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
Configuration management for rocprofiler-systems tests.

Handles discovery and management of build directories, executables,
and environment variables needed for testing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RocprofsysConfig:
    """Configuration for rocprofiler-systems test execution.

    Contains paths to build artifacts, executables, and default environment
    variables for test execution.

    Attributes:
        build_dir: Path to the build directory
        source_dir: Path to the source directory
        rocprof_instrument: Path to rocprof-sys-instrument executable
        rocprof_sample: Path to rocprof-sys-sample executable
        rocprof_run: Path to rocprof-sys-run executable
        rocprof_causal: Path to rocprof-sys-causal executable
        rocprof_avail: Path to rocprof-sys-avail executable
        lib_dir: Path to library directory
        rocm_path: Optional path to ROCm installation
    """

    build_dir: Path
    source_dir: Path
    rocprof_instrument: Path
    rocprof_sample: Path
    rocprof_run: Path
    rocprof_causal: Path
    rocprof_avail: Path
    lib_dir: Path
    rocm_path: Optional[Path] = None
    mpiexec: Optional[Path] = None

    def get_library_path(self) -> str:
        """Get LD_LIBRARY_PATH including rocprofiler-systems libraries."""
        paths = [str(self.lib_dir)]

        # Add ROCm LLVM lib if available
        if self.rocm_path:
            llvm_lib = self.rocm_path / "lib" / "llvm" / "lib"
            if llvm_lib.exists():
                paths.append(str(llvm_lib))

        # Append existing LD_LIBRARY_PATH
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if existing:
            paths.append(existing)

        return ":".join(paths)

    def get_target_executable(self, name: str) -> Path:
        """Get path to a test target executable.

        Args:
            name: Name of the target (e.g., 'transpose', 'lulesh')

        Returns:
            Path to the executable

        Raises:
            FileNotFoundError: If executable doesn't exist
        """
        exe = self.build_dir / name
        if exe.exists():
            return exe

        exe = self.build_dir / "examples" / name / name
        if exe.exists():
            return exe

        raise FileNotFoundError(
            f"Target executable '{name}' not found in {self.build_dir}"
        )

    def get_base_environment(self) -> dict[str, str]:
        """Get base environment variables for test execution."""
        return {
            "ROCPROFSYS_CI": "ON",
            "ROCPROFSYS_TRACE": "ON",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_FILE_OUTPUT": "ON",
            "ROCPROFSYS_VERBOSE": "1",
            "ROCPROFSYS_SAMPLING_FREQ": "300",
            "ROCPROFSYS_SAMPLING_DELAY": "0.05",
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "threads",
            "OMP_NUM_THREADS": "2",
            "LD_LIBRARY_PATH": self.get_library_path(),
        }


def discover_build_config(
    build_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
) -> RocprofsysConfig:
    """Discover rocprofiler-systems build configuration.

    Attempts to find the build directory and source directory automatically
    if not provided, checking common locations and environment variables.

    Args:
        build_dir: Explicit build directory path
        source_dir: Explicit source directory path

    Returns:
        RocprofsysConfig with discovered paths

    Raises:
        FileNotFoundError: If build directory cannot be found
    """
    if build_dir is None:
        env_build = os.environ.get("ROCPROFSYS_BUILD_DIR")
        if env_build:
            build_dir = Path(env_build)
        else:
            test_dir = Path(__file__).parent.parent.parent.parent
            for candidate in [
                test_dir / "build" / "debug",
                test_dir / "build" / "release",
                test_dir / "build",
                Path.cwd() / "build" / "debug",
                Path.cwd() / "build" / "release",
                Path.cwd() / "build",
            ]:
                if candidate.exists() and (candidate / "bin").exists():
                    build_dir = candidate
                    break

    if build_dir is None or not build_dir.exists():
        raise FileNotFoundError(
            "Could not find build directory. Set ROCPROFSYS_BUILD_DIR environment "
            "variable or pass build_dir explicitly."
        )

    build_dir = build_dir.resolve()

    if source_dir is None:
        env_source = os.environ.get("ROCPROFSYS_SOURCE_DIR")
        if env_source:
            source_dir = Path(env_source)
        else:
            cmake_cache = build_dir / "CMakeCache.txt"
            if cmake_cache.exists():
                content = cmake_cache.read_text()
                match = re.search(
                    r"CMAKE_HOME_DIRECTORY:INTERNAL=(.+)", content
                )
                if match:
                    source_dir = Path(match.group(1))

            if source_dir is None:
                source_dir = build_dir.parent.parent

    source_dir = source_dir.resolve()

    # Find ROCm path
    rocm_path = None
    for candidate in [
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
    ]:
        if candidate and Path(candidate).exists():
            rocm_path = Path(candidate)
            break

    # Find mpiexec
    mpiexec = None
    for candidate in ["mpiexec", "mpirun"]:
        import shutil
        path = shutil.which(candidate)
        if path:
            mpiexec = Path(path)
            break

    bin_dir = build_dir / "bin"
    lib_dir = build_dir / "lib"

    return RocprofsysConfig(
        build_dir=build_dir,
        source_dir=source_dir,
        rocprof_instrument=bin_dir / "rocprof-sys-instrument",
        rocprof_sample=bin_dir / "rocprof-sys-sample",
        rocprof_run=bin_dir / "rocprof-sys-run",
        rocprof_causal=bin_dir / "rocprof-sys-causal",
        rocprof_avail=bin_dir / "rocprof-sys-avail",
        lib_dir=lib_dir,
        rocm_path=rocm_path,
        mpiexec=mpiexec,
    )

