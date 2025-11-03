# Quick Test Execution Guide

## Run All Tests

```bash
# Run all tests with summary
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with detailed output and show print statements
pytest tests/ -vv -s
```

## Run Specific Test Files

```bash
# Classification module tests (15 tests)
pytest tests/test_classification_module.py -v

# Regression module tests (25 tests)
pytest tests/test_regression_module.py -v

# Tools module tests
pytest tests/test_tools.py -v
```

## Run Tests with Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Show only missing lines
pytest tests/ --cov=src --cov-report=term-missing:skip-covered
```

## Run Specific Test Categories

```bash
# Test preprocessing functions only
pytest tests/test_classification_module.py::TestPreprocessCsvToWav -v
pytest tests/test_regression_module.py::TestPreprocessCsvData -v

# Test model building only
pytest tests/test_classification_module.py::TestBuildAndCompileModel -v
pytest tests/test_regression_module.py::TestBuildAndCompileModel -v

# Test training functions only
pytest tests/test_classification_module.py::TestTrainAndEvaluateModel -v
pytest tests/test_regression_module.py::TestTrainAndEvaluateModel -v

# Test export functions only
pytest tests/test_classification_module.py::TestCreateExportModel -v
pytest tests/test_regression_module.py::TestCreateExportModel -v

# Test registry management only
pytest tests/test_classification_module.py::TestUpdateModelsRegistry -v
pytest tests/test_regression_module.py::TestUpdateModelsRegistry -v
```

## Run Single Tests

```bash
# Run a single test method
pytest tests/test_classification_module.py::TestBuildAndCompileModel::test_build_and_compile_model_creates_model -v

# Run with extra debugging
pytest tests/test_regression_module.py::TestTrainAndEvaluateModel::test_train_and_evaluate_model_creates_plots -vv -s
```

## Run Tests with Different Options

```bash
# Stop on first failure
pytest tests/ -x

# Show test execution times (slowest tests)
pytest tests/ --durations=10

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Suppress warnings
pytest tests/ --disable-warnings

# Run only failed tests from last run
pytest tests/ --lf

# Run failed tests first, then all others
pytest tests/ --ff
```

## Continuous Testing (Watch Mode)

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file changes
ptw tests/
```

## Quick Validation

```bash
# Just check if tests can be collected (doesn't run them)
pytest tests/ --collect-only

# Dry run (show what would run)
pytest tests/ --collect-only -v
```

## Test Output Examples

### Successful Test Run
```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-8.4.2, pluggy-1.6.0
collected 40 items

tests/test_classification_module.py ...............                      [ 37%]
tests/test_regression_module.py .........................                [100%]

============================== 40 passed in 12.34s ==============================
```

### Failed Test Example
```
FAILED tests/test_classification_module.py::TestBuildAndCompileModel::test_build_and_compile_model_creates_model
```

### Coverage Report Example
```
---------- coverage: platform linux, python 3.10.12 -----------
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/classification_module.py    180     12    93%   45-48, 102-105
src/regression_module.py         165      8    95%   89-92, 156-159
-----------------------------------------------------------
TOTAL                            345     20    94%
```

## Environment Setup

Before running tests, ensure dependencies are installed:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install test dependencies (if not in requirements.txt)
pip install pytest pytest-cov pytest-xdist
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd /home/arisenthil/Onion_Sensor

# Run tests
pytest tests/
```

### TensorFlow GPU Issues
```bash
# Run with CPU only
CUDA_VISIBLE_DEVICES="" pytest tests/

# Or set environment variable
export CUDA_VISIBLE_DEVICES=""
pytest tests/
```

### Cleanup After Tests
```bash
# Remove test artifacts
rm -rf models/test_*
rm -f models.json

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Statistics

- **Total Tests**: 40+
- **Classification Module**: 15 tests
- **Regression Module**: 25 tests
- **Expected Runtime**: < 5 minutes
- **Expected Coverage**: > 90%

## Next Steps

1. Run tests before committing changes
2. Add new tests for new features
3. Maintain test coverage above 90%
4. Review test failures carefully
5. Update tests when refactoring code

For detailed documentation, see [tests/README.md](tests/README.md)
