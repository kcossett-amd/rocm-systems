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

Handles discovery and management of build/install directories, executables,
and environment variables needed for testing.

Supports two modes:
1. Build directory mode: Tests run against a build directory
2. Install directory mode: Tests run against installed binaries

Environment variables:
- ROCPROFSYS_BUILD_DIR: Path to build directory (takes precedence)
- ROCPROFSYS_INSTALL_DIR: Path to installation prefix (e.g., /opt/rocm)
- ROCPROFSYS_SOURCE_DIR: Path to source directory (for validation rules)
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RocprofsysConfig:
    """Configuration for rocprofiler-systems test execution.

    Contains paths to build/install artifacts, executables, and default
    environment variables for test execution.

    Attributes:
        build_dir: Path to the build or install directory (used for output)
        source_dir: Path to the source directory (for validation rules)
        rocprof_instrument: Path to rocprof-sys-instrument executable
        rocprof_sample: Path to rocprof-sys-sample executable
        rocprof_run: Path to rocprof-sys-run executable
        rocprof_causal: Path to rocprof-sys-causal executable
        rocprof_avail: Path to rocprof-sys-avail executable
        lib_dir: Path to library directory
        bin_dir: Path to binary directory
        examples_dir: Optional path to examples directory
        rocm_path: Optional path to ROCm installation
        mpiexec: Optional path to MPI launcher
        is_installed: Whether this is an installed configuration
    """

    build_dir: Path
    source_dir: Path
    rocprof_instrument: Path
    rocprof_sample: Path
    rocprof_run: Path
    rocprof_causal: Path
    rocprof_avail: Path
    lib_dir: Path
    bin_dir: Path
    examples_dir: Optional[Path] = None
    rocm_path: Optional[Path] = None
    mpiexec: Optional[Path] = None
    is_installed: bool = False

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

        Searches in the following order:
        1. build_dir/name (build directory layout)
        2. build_dir/examples/name/name (build directory layout)
        3. examples_dir/name (installed examples)
        4. bin_dir/name (installed binaries)
        5. PATH lookup via shutil.which (system-installed)

        Args:
            name: Name of the target (e.g., 'transpose', 'lulesh')

        Returns:
            Path to the executable

        Raises:
            FileNotFoundError: If executable doesn't exist
        """
        # Build directory layout
        exe = self.build_dir / name
        if exe.exists() and exe.is_file():
            return exe

        exe = self.build_dir / "examples" / name / name
        if exe.exists() and exe.is_file():
            return exe

        # Installed examples directory
        if self.examples_dir:
            exe = self.examples_dir / name
            if exe.exists() and exe.is_file():
                return exe

            exe = self.examples_dir / name / name
            if exe.exists() and exe.is_file():
                return exe

        # Installed bin directory
        exe = self.bin_dir / name
        if exe.exists() and exe.is_file():
            return exe

        # PATH lookup for system-installed binaries
        path_exe = shutil.which(name)
        if path_exe:
            return Path(path_exe)

        raise FileNotFoundError(
            f"Target executable '{name}' not found. Searched in:\n"
            f"  - {self.build_dir}/{name}\n"
            f"  - {self.build_dir}/examples/{name}/{name}\n"
            f"  - {self.examples_dir}/{name} (if set)\n"
            f"  - {self.bin_dir}/{name}\n"
            f"  - PATH lookup"
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
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_VERBOSE": "1",
            "ROCPROFSYS_SAMPLING_FREQ": "300",
            "ROCPROFSYS_SAMPLING_DELAY": "0.05",
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "threads",
            "OMP_NUM_THREADS": "2",
            "LD_LIBRARY_PATH": self.get_library_path(),
        }


def _find_rocm_path() -> Optional[Path]:
    """Find ROCm installation path."""
    for candidate in [
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
        "/usr/local/rocm",
    ]:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _find_mpiexec() -> Optional[Path]:
    """Find MPI launcher executable."""
    for candidate in ["mpiexec", "mpirun"]:
        path = shutil.which(candidate)
        if path:
            return Path(path)
    return None


def _find_executable(name: str, search_paths: list[Path]) -> Optional[Path]:
    """Find an executable in search paths or via PATH."""
    for search_dir in search_paths:
        exe = search_dir / name
        if exe.exists() and exe.is_file():
            return exe

    # Fallback to PATH
    path_exe = shutil.which(name)
    if path_exe:
        return Path(path_exe)

    return None


def discover_install_config(
    install_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
) -> RocprofsysConfig:
    """Discover rocprofiler-systems installation configuration.

    Creates configuration for testing against installed binaries.

    Args:
        install_dir: Installation prefix (e.g., /opt/rocm or /usr/local)
        source_dir: Source directory (for validation rules, optional)

    Returns:
        RocprofsysConfig configured for installed binaries

    Raises:
        FileNotFoundError: If installation cannot be found
    """
    # Find installation directory
    if install_dir is None:
        env_install = os.environ.get("ROCPROFSYS_INSTALL_DIR")
        if env_install:
            install_dir = Path(env_install)
        else:
            # Try common installation locations
            for candidate in [
                _find_rocm_path(),
                Path("/usr/local"),
                Path("/usr"),
            ]:
                if candidate and (candidate / "bin" / "rocprof-sys-instrument").exists():
                    install_dir = candidate
                    break

    if install_dir is None:
        # check in PATH for rocprof-sys-* (instrument in this case, but any tool will do)
        rocprof_instrument = shutil.which("rocprof-sys-instrument")
        if rocprof_instrument:
            install_dir = Path(rocprof_instrument).parent.parent
        else:
            raise FileNotFoundError(
                "Could not find rocprofiler-systems installation. Set ROCPROFSYS_INSTALL_DIR "
                "environment variable or ensure rocprof-sys-instrument is in PATH."
            )

    install_dir = install_dir.resolve()

    # Determine directory layout
    bin_dir = install_dir / "bin"
    lib_dir = install_dir / "lib"

    # For lib64 systems
    if not lib_dir.exists() and (install_dir / "lib64").exists():
        lib_dir = install_dir / "lib64"

    # Examples may be in share/rocprofiler-systems/examples or similar
    examples_dir = None
    for candidate in [
        install_dir / "share" / "rocprofiler-systems" / "examples",
        install_dir / "share" / "rocprofsys" / "examples",
        install_dir / "examples",
    ]:
        if candidate.exists():
            examples_dir = candidate
            break

    # Find source directory for validation rules
    if source_dir is None:
        env_source = os.environ.get("ROCPROFSYS_SOURCE_DIR")
        if env_source:
            source_dir = Path(env_source)
        else:
            # Try to find source from current directory
            test_dir = Path(__file__).parent.parent.parent.parent
            if (test_dir / "tests" / "rocpd-validation-rules").exists():
                source_dir = test_dir
            else:
                # Use install dir as fallback (validation rules may not be available)
                source_dir = install_dir

    source_dir = source_dir.resolve()

    # Create a temporary directory for test outputs if needed
    output_dir = Path(tempfile.gettempdir()) / "rocprof-sys-pytest-output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rocm_path = _find_rocm_path()
    mpiexec = _find_mpiexec()

    # Find executables
    search_paths = [bin_dir]
    if rocm_path:
        search_paths.append(rocm_path / "bin")

    rocprof_instrument = _find_executable("rocprof-sys-instrument", search_paths)
    rocprof_sample = _find_executable("rocprof-sys-sample", search_paths)
    rocprof_run = _find_executable("rocprof-sys-run", search_paths)
    rocprof_causal = _find_executable("rocprof-sys-causal", search_paths)
    rocprof_avail = _find_executable("rocprof-sys-avail", search_paths)

    if not rocprof_instrument:
        raise FileNotFoundError(
            f"rocprof-sys-instrument not found in {bin_dir} or PATH"
        )

    return RocprofsysConfig(
        build_dir=output_dir,  # Use temp dir for outputs
        source_dir=source_dir,
        rocprof_instrument=rocprof_instrument,
        rocprof_sample=rocprof_sample or bin_dir / "rocprof-sys-sample",
        rocprof_run=rocprof_run or bin_dir / "rocprof-sys-run",
        rocprof_causal=rocprof_causal or bin_dir / "rocprof-sys-causal",
        rocprof_avail=rocprof_avail or bin_dir / "rocprof-sys-avail",
        lib_dir=lib_dir,
        bin_dir=bin_dir,
        examples_dir=examples_dir,
        rocm_path=rocm_path,
        mpiexec=mpiexec,
        is_installed=True,
    )


def discover_build_config(
    build_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
) -> RocprofsysConfig:
    """Discover rocprofiler-systems build configuration.

    Attempts to find the build directory and source directory automatically
    if not provided, checking common locations and environment variables.

    If no build directory is found but an installation is available,
    falls back to discover_install_config().

    Args:
        build_dir: Explicit build directory path
        source_dir: Explicit source directory path

    Returns:
        RocprofsysConfig with discovered paths

    Raises:
        FileNotFoundError: If neither build directory nor installation found
    """
    # Check for explicit install mode
    if os.environ.get("ROCPROFSYS_INSTALL_DIR"):
        return discover_install_config(source_dir=source_dir)

    if build_dir is None:
        env_build = os.environ.get("ROCPROFSYS_BUILD_DIR")
        if env_build:
            build_dir = Path(env_build)
        else:
            test_dir = Path(__file__).parent.parent.parent.parent
            for candidate in [
                test_dir / "rocprof-sys-build",
                test_dir / "build" / "debug",
                test_dir / "build" / "release",
                test_dir / "build",
                Path.cwd() / "rocprof-sys-build",
                Path.cwd() / "build" / "debug",
                Path.cwd() / "build" / "release",
                Path.cwd() / "build",
            ]:
                if candidate.exists() and (candidate / "bin").exists():
                    build_dir = candidate
                    break

    # If no build directory found, try install configuration
    if build_dir is None or not build_dir.exists():
        try:
            return discover_install_config(source_dir=source_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find build directory or installation. Set one of:\n"
                "  - ROCPROFSYS_BUILD_DIR: Path to build directory\n"
                "  - ROCPROFSYS_INSTALL_DIR: Path to installation prefix\n"
                "Or ensure rocprof-sys-instrument is in PATH."
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

    rocm_path = _find_rocm_path()
    mpiexec = _find_mpiexec()

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
        bin_dir=bin_dir,
        examples_dir=build_dir / "examples",
        rocm_path=rocm_path,
        mpiexec=mpiexec,
        is_installed=False,
    )

