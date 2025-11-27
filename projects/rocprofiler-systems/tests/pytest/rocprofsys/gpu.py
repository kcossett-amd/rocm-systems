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
GPU detection and information utilities.

Provides functionality for detecting AMD GPUs, querying their capabilities,
and determining architecture-specific test configurations.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


# Need to import os mofule for get_hip_visible_devices
import os


@dataclass
class GPUInfo:
    """Information about detected GPU(s).

    Attributes:
        available: Whether any valid GPU is available
        architectures: List of detected GPU architectures (e.g., ['gfx90a', 'gfx1100'])
        device_count: Number of GPU devices detected
        is_navi: Whether any detected GPU is NAVI architecture (gfx10xx, gfx11xx, gfx12xx)
        amd_smi_available: Whether amd-smi tool is available and functional
    """

    available: bool
    architectures: list[str]
    device_count: int
    is_navi: bool
    amd_smi_available: bool

    @property
    def rocm_events_for_test(self) -> str:
        """Get appropriate ROCm events for testing based on architecture."""
        if self.is_navi:
            return "SQ_WAVES"
        return "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0"

    @property
    def counter_names(self) -> list[str]:
        """Get counter names for validation based on architecture."""
        if self.is_navi:
            return ["SQ_WAVES"]
        return ["GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU", "TA_TA_BUSY"]

    @property
    def expected_counter_files(self) -> list[str]:
        """Get expected counter output files based on architecture."""
        return [f"rocprof-device-0-{name}.txt" for name in self.counter_names]


@lru_cache(maxsize=1)
def detect_gpu() -> GPUInfo:
    """Detect available AMD GPUs and their capabilities.

    Uses rocminfo and amd-smi to gather GPU information. Results are cached
    for the lifetime of the process.

    Returns:
        GPUInfo with detected GPU information
    """
    architectures: list[str] = []
    device_count = 0
    amd_smi_available = False

    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        try:
            result = subprocess.run(
                [rocminfo],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                matches = re.findall(r"gfx([0-9A-Fa-f]+)", result.stdout)
                architectures = list(set(f"gfx{m}" for m in matches))
                device_count = len(architectures)
        except (subprocess.TimeoutExpired, OSError):
            pass

    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi],
                capture_output=True,
                text=True,
                timeout=10,
            )
            amd_smi_available = (
                result.returncode == 0
                and "ERROR" not in result.stdout
                and "ERROR" not in result.stderr
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    is_navi = any(is_navi_architecture(arch) for arch in architectures)

    return GPUInfo(
        available=device_count > 0 and amd_smi_available,
        architectures=sorted(architectures),
        device_count=device_count,
        is_navi=is_navi,
        amd_smi_available=amd_smi_available,
    )


def is_navi_architecture(arch: str) -> bool:
    """Check if an architecture string represents NAVI GPU.

    NAVI includes gfx10xx, gfx11xx, and gfx12xx architectures.

    Args:
        arch: Architecture string (e.g., 'gfx1100', 'gfx90a')

    Returns:
        True if NAVI architecture
    """
    match = re.match(r"gfx(10|11|12)[0-9A-Fa-f]{2}", arch)
    return match is not None


def get_hip_visible_devices() -> Optional[str]:
    """Get HIP_VISIBLE_DEVICES environment variable if set."""
    return os.environ.get("HIP_VISIBLE_DEVICES")



