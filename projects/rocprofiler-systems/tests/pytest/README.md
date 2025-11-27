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
├── rocprofsys/             # Test utilities package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── gpu.py              # GPU detection utilities
│   ├── runners.py          # Test execution runners
│   └── validators.py       # Wrappers for existing validation scripts
├── test_transpose.py       # Transpose example tests
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

Set the build directory (if not auto-detected):

```bash
export ROCPROFSYS_BUILD_DIR=/path/to/build/debug
pytest
```

Keep test output directories for debugging:

```bash
export ROCPROFSYS_KEEP_TEST_OUTPUT=1
pytest
```

## Cleanup Behavior

The framework includes automatic cleanup:

- **Per-test cleanup**: Output directories are cleaned up after each passing test
- **Session cleanup**: Temporary files (`/tmp/buffered_storage*.bin`, `/tmp/metadata*.json`)
  are cleaned up after all tests complete
- **Failed test preservation**: Output directories from failed tests are preserved for debugging

To keep all test outputs (even from passing tests):

```bash
export ROCPROFSYS_KEEP_TEST_OUTPUT=1
```

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