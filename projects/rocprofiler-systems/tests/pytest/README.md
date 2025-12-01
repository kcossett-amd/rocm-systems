# rocprofiler-systems pytest Test Suite

This directory contains the pytest-based test framework for rocprofiler-systems.

## Overview

The pytest framework provides a more maintainable, debuggable, and readable alternative to the CMake/CTest-based testing. It offers:

- **Clear test definitions**: Tests are easy to read and understand
- **Better debugging**: Use `pytest --pdb` or add breakpoints
- **Parametrization**: Test multiple configurations cleanly
- **Rich fixtures**: Reusable setup/teardown with dependency injection
- **Output validation**: Built-in validators for perfetto, rocpd, and timemory output

## Directory Structure

```
tests/pytest/
├── conftest.py              # Shared fixtures and pytest configuration
├── requirements.txt         # Python dependencies
├── build_standalone.sh      # Script to build portable test packages
├── rocprofsys/             # Test utilities package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── gpu.py              # GPU detection utilities
│   ├── runners.py          # Test execution runners
│   └── validators.py       # Wrappers for existing validation scripts
├── test_binary.py          # CLI tool tests (instrument, avail, run)
├── test_causal.py          # Causal profiling tests
├── test_fork.py            # Fork-related tests
├── test_mpi.py             # MPI integration tests
├── test_openmp.py          # OpenMP tests (cg, lu, target)
├── test_python.py          # Python integration tests
├── test_transpose.py       # Transpose example tests (GPU)
├── test_user_api.py        # User API tests
└── README.md               # This file
```

**Note:** The validators module wraps the existing validation scripts from
`tests/` directory (`validate-perfetto-proto.py`, `validate-rocpd.py`, etc.)
to ensure consistency between pytest and CMake/CTest validation.

## Installation

1. Install test dependencies:

```bash
pip install -r tests/pytest/requirements.txt
```

2. (Optional) Install perfetto for trace validation:

```bash
pip install perfetto
```

## Running Tests

### Basic Usage

Run all tests:

```bash
# From project root
pytest tests/pytest/

# Or using the configured testpaths
pytest
```

Run specific test file:

```bash
pytest tests/pytest/test_transpose.py
```

Run specific test class:

```bash
pytest tests/pytest/test_transpose.py::TestTranspose
```

Run specific test:

```bash
pytest tests/pytest/test_transpose.py::TestTranspose::test_sampling
```

### Configuration

The test framework supports two modes: **build directory** and **installed binaries**.

#### Build Directory Mode (default)

Set the build directory (if not auto-detected):

```bash
export ROCPROFSYS_BUILD_DIR=/path/to/build/debug
pytest
```

#### Installed Binaries Mode

Run tests against installed rocprofiler-systems (e.g., from ROCm):

```bash
# Option 1: Set installation prefix
export ROCPROFSYS_INSTALL_DIR=/opt/rocm
pytest

# Option 2: If rocprof-sys-instrument is in PATH, it will be auto-detected
pytest

# For validation rules, also set source directory:
export ROCPROFSYS_SOURCE_DIR=/path/to/rocprofiler-systems
export ROCPROFSYS_INSTALL_DIR=/opt/rocm
pytest
```

#### Other Options

Keep test output directories for debugging:

```bash
export ROCPROFSYS_KEEP_TEST_OUTPUT=1
pytest
```

## Standalone Test Packages

The test suite can be packaged into a standalone executable for running on remote machines
where rocprofiler-systems is installed but the source code is not available.

### Building Standalone Packages

Use the `build_standalone.sh` script to create portable test packages:

```bash
cd tests/pytest

# Build a Python zipapp (recommended - most portable)
./build_standalone.sh --shiv

# Build a PyInstaller binary (no Python needed on target)
./build_standalone.sh --pyinstaller

# Build PyInstaller binary in Docker (for glibc compatibility)
./build_standalone.sh --pyinstaller-docker

# Build both zipapp and PyInstaller
./build_standalone.sh --all

# See all options
./build_standalone.sh --help
```

### Package Types

| Package | Size | Python on Target | glibc Compatibility |
|---------|------|------------------|---------------------|
| Zipapp (`.pyz`) | ~72KB | Required + pytest | Any (uses system Python) |
| PyInstaller | ~50-100MB | Not needed | Matches build machine |
| PyInstaller+Docker | ~50-100MB | Not needed | glibc 2.17+ (RHEL 7+) |

**Recommendation**: Use the **zipapp** (`.pyz`) for maximum portability. It uses the target
system's Python interpreter, avoiding glibc version mismatch issues.

### Running on Target Machine

**Zipapp** (requires `pip install pytest` on target):

```bash
# Copy to target
scp dist/rocprofsys-tests.pyz target-machine:/path/to/

# On target machine
pip install pytest
export ROCPROFSYS_INSTALL_DIR=/opt/rocm  # if not in PATH
python3 rocprofsys-tests.pyz --collect-only   # List available tests
python3 rocprofsys-tests.pyz -v               # Run all tests
python3 rocprofsys-tests.pyz -k transpose -v  # Run specific tests
python3 rocprofsys-tests.pyz -x               # Stop on first failure
```

**PyInstaller binary** (no Python needed):

```bash
# Copy to target
scp dist/rocprofsys-tests target-machine:/path/to/

# On target machine
export ROCPROFSYS_INSTALL_DIR=/opt/rocm
./rocprofsys-tests --collect-only
./rocprofsys-tests -v
```

### Troubleshooting

**glibc version error** (PyInstaller only):
```
GLIBC_2.38 not found
```
Solution: Use `--pyinstaller-docker` to build with manylinux, or use `--shiv` instead.

**pytest not found** (Zipapp only):
```
ERROR: pytest is not installed
```
Solution: `pip install pytest` on the target machine.

## Cleanup Behavior

The framework includes comprehensive automatic cleanup at multiple levels:

### Per-Test Cleanup
- Output directories are cleaned up after each passing test
- Instrumented binaries (`.inst` files) are cleaned up automatically
- Failed test outputs are preserved for debugging

### Module-Level Cleanup
- Instrumented binaries in the build directory are cleaned up after each test module
- Intermediate temp files are cleaned between modules

### Session-Level Cleanup
After all tests complete, the following are cleaned up:
- Temporary buffered storage files (`/tmp/buffered_storage*.bin`)
- Temporary metadata files (`/tmp/metadata*.json`)
- Perfetto temp files (`/tmp/perfetto-*.proto`)
- HSA/ROCm temp files (`/tmp/hsa-*.tmp`, `/tmp/rocm-*.tmp`, `/tmp/hip-*.tmp`)
- Instrumented binaries (`/tmp/*.inst`)
- Causal profiling temp files (`/tmp/causal-*.json`, `/tmp/experiments-*.coz`)
- Empty output directories

### Controlling Cleanup

To keep all test outputs (even from passing tests):

```bash
export ROCPROFSYS_KEEP_TEST_OUTPUT=1
```

### Cleanup Methods in Test Results

All test result classes (`TestResult`, `CausalResult`, `PythonResult`) include:
- `cleanup()`: Clean up all output files (respects `keep_on_failure` flag)
- `cleanup_instrumented_binaries()`: Clean up only instrumented binary files

### Running Tests by Marker

Run only GPU tests:

```bash
pytest -m gpu
```

Run tests excluding slow tests:

```bash
pytest -m "not slow"
```

Run only ROCpd tests:

```bash
pytest -m rocpd
```

### Parallel Execution

Run tests in parallel:

```bash
pytest -n auto  # Uses all available CPUs
pytest -n 4     # Uses 4 workers
```

### Verbose Output and Debugging

Verbose output:

```bash
pytest -v
pytest -vv  # Extra verbose
```

Stop on first failure:

```bash
pytest -x
```

Drop into debugger on failure:

```bash
pytest --pdb
```

Show print statements:

```bash
pytest -s
```

### Coverage Reporting

```bash
pytest --cov=tests/pytest/rocprofsys --cov-report=html
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from rocprofsys import (
    RocprofsysConfig,
    SamplingRunner,
    validate_perfetto_trace,
)

@pytest.mark.gpu
class TestMyFeature:
    """Test class for my feature."""

    def test_basic_execution(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test that basic execution works."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=transpose_env,
        )

        result = runner.run()

        assert result.success, f"Test failed: {result.stderr}"
        assert result.perfetto_file is not None
```

### Available Fixtures

- `rocprof_config`: Session-wide configuration
- `gpu_info`: GPU detection information
- `test_output_dir`: Unique directory for test outputs
- `base_env`: Base environment variables
- `transpose_env`: Environment for transpose tests
- `rocpd_env`: Environment with ROCpd enabled
- `validation_rules_dir`: Path to validation rules

### Available Markers

- `@pytest.mark.gpu`: Requires GPU
- `@pytest.mark.mpi`: Requires MPI
- `@pytest.mark.rocpd`: Requires ROCpd support
- `@pytest.mark.rocprofiler`: Uses ROCProfiler counters
- `@pytest.mark.slow`: Slow-running test
- `@pytest.mark.loops`: Loop instrumentation test

### Available Runners

- `BaselineRunner`: Run without instrumentation
- `SamplingRunner`: Run with sampling instrumentation
- `BinaryRewriteRunner`: Binary rewrite + run
- `RuntimeInstrumentRunner`: Runtime instrumentation
- `SysRunRunner`: Run with rocprof-sys-run wrapper

## Test Modules

| Module | Description | Markers |
|--------|-------------|---------|
| `test_binary.py` | CLI tool tests (instrument, avail, run) | - |
| `test_causal.py` | Causal profiling tests | `slow` |
| `test_fork.py` | Fork-related tests | `gpu` |
| `test_mpi.py` | MPI integration tests | `mpi` |
| `test_openmp.py` | OpenMP tests (cg, lu, target) | `gpu`, `rocpd` |
| `test_python.py` | Python integration tests | `rocpd` |
| `test_transpose.py` | Transpose example tests | `gpu`, `rocpd`, `rocprofiler`, `loops` |
| `test_user_api.py` | User API tests | `loops` |

### Available Validators

These validators wrap the existing validation scripts from the `tests/` directory:

- `validate_perfetto_trace()`: Wraps `validate-perfetto-proto.py`
- `validate_rocpd_database()`: Wraps `validate-rocpd.py`
- `validate_timemory_json()`: Wraps `validate-timemory-json.py`
- `validate_causal_json()`: Wraps `validate-causal-json.py`
- `validate_file_exists()`: Check file exists and is non-empty

## Comparison with CMake Tests

| Aspect | CMake CTest | Pytest |
|--------|-------------|--------|
| Lines per test | 50+ | 10-20 |
| Debugging | Regex matching | `--pdb`, rich tracebacks |
| Parametrization | foreach loops | `@pytest.mark.parametrize` |
| Fixtures | Manual env setup | Declarative fixtures |
| Skip logic | CMake conditionals | `@pytest.mark.skipif` |
| Parallel | CTest parallel | `pytest-xdist` |