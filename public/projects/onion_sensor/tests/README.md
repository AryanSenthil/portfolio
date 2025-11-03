# Test Suite Documentation

This directory contains comprehensive test suites for the Onion_Sensor project.

## Test Files

### test_classification_module.py
Tests for the audio classification pipeline (`src/classification_module.py`):

- **TestPreprocessCsvToWav**: Tests CSV to WAV conversion
  - Directory structure creation
  - Metadata saving
  - Invalid input handling

- **TestLoadAndPreprocessData**: Tests dataset loading and preprocessing
  - TensorFlow dataset creation
  - Train/val/test splits
  - Metadata validation

- **TestBuildAndCompileModel**: Tests model building
  - Model architecture verification
  - Compilation settings
  - Custom learning rates

- **TestTrainAndEvaluateModel**: Tests training pipeline
  - Output file generation
  - Visualization plots
  - Training metrics

- **TestCreateExportModel**: Tests model export
  - SavedModel creation
  - Export metadata

- **TestUpdateModelsRegistry**: Tests registry management
  - Registry creation and updates
  - Model information storage

**Total Tests**: 15

### test_regression_module.py
Tests for the Wide & Deep regression pipeline (`src/regression_module.py`):

- **TestPreprocessCsvData**: Tests data preprocessing
  - Directory creation
  - Train/val/test splits
  - Metadata generation
  - Raw file copying

- **TestBuildAndCompileModel**: Tests Wide & Deep model building
  - Multi-input architecture
  - Custom hidden layers
  - Architecture metadata

- **TestTrainAndEvaluateModel**: Tests regression training
  - History tracking
  - Test metrics (MSE, RMSE)
  - Residuals analysis
  - Visualization plots

- **TestCreateExportModel**: Tests model export
  - SavedModel generation
  - Export metadata

- **TestUpdateModelsRegistry**: Tests registry management
  - Architecture information
  - Performance metrics
  - Registry updates

**Total Tests**: 25

### test_tools.py
Tests for utility functions (`src/tools.py`):
- CSV processing
- Signal interpolation
- Audio generation
- TensorFlow helpers
- DataProcessor class

## Running Tests

### Run All Tests
```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Files
```bash
# Classification module tests only
pytest tests/test_classification_module.py

# Regression module tests only
pytest tests/test_regression_module.py

# Tools tests only
pytest tests/test_tools.py
```

### Run Specific Test Classes
```bash
# Test only preprocessing
pytest tests/test_classification_module.py::TestPreprocessCsvToWav

# Test only model building
pytest tests/test_regression_module.py::TestBuildAndCompileModel
```

### Run Specific Test Methods
```bash
# Single test
pytest tests/test_classification_module.py::TestBuildAndCompileModel::test_build_and_compile_model_creates_model

# With verbose output
pytest tests/test_regression_module.py::TestTrainAndEvaluateModel::test_train_and_evaluate_model_creates_plots -v
```

## Test Structure

All tests follow the standard unittest pattern:

```python
class TestClassName(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create temporary directories, mock data, etc.

    def tearDown(self):
        """Clean up after each test"""
        # Remove temporary files, directories, etc.

    def test_something(self):
        """Test description"""
        # Test implementation
        self.assertTrue(condition)
```

## Mocking Strategy

Tests use `unittest.mock` to avoid dependencies on:
- External data files
- Heavy computations
- File I/O operations
- TensorFlow dataset creation

Example:
```python
@patch('classification_module.read_csv_files')
@patch('classification_module.wav_generator')
def test_function(self, mock_wav, mock_read):
    mock_read.return_value = []
    mock_wav.return_value = []
    # Test implementation
```

## Test Coverage Areas

### Unit Tests
- Individual function behavior
- Input validation
- Error handling
- Edge cases

### Integration Tests
- Multi-step pipelines
- File I/O operations
- Model persistence
- Registry management

### End-to-End Tests
- Complete classification pipeline
- Complete regression pipeline
- Model training and export

## Test Data

Tests use:
- **Temporary directories** (`tempfile.mkdtemp()`) for file operations
- **Mock CSV data** created with pandas
- **Synthetic datasets** using NumPy/TensorFlow
- **Small models** with minimal layers for fast execution

All test data is cleaned up in `tearDown()` methods.

## Continuous Integration

These tests are designed to run in CI/CD environments:
- No external dependencies required
- Self-contained test data
- Deterministic behavior (seeded random operations)
- Fast execution (< 5 minutes for full suite)

## Common Issues

### Import Errors
If you see import errors, ensure you're running tests from the project root:
```bash
cd /path/to/Onion_Sensor
pytest tests/
```

### TensorFlow Warnings
TensorFlow may emit warnings during tests. These are normal and don't indicate test failures:
```bash
# Suppress warnings
pytest tests/ --disable-warnings
```

### Cleanup Issues
If tests fail and leave artifacts:
```bash
# Clean up models directory
rm -rf models/test_*

# Clean up test registry
rm -f models.json
```

## Writing New Tests

When adding new tests:

1. **Import the module under test**:
   ```python
   import sys
   sys.path.append(str(Path(__file__).parent.parent / 'src'))
   from your_module import your_function
   ```

2. **Create test class**:
   ```python
   class TestYourFunction(unittest.TestCase):
       def setUp(self):
           # Setup code

       def tearDown(self):
           # Cleanup code

       def test_basic_behavior(self):
           # Test code
   ```

3. **Follow naming conventions**:
   - Test files: `test_<module_name>.py`
   - Test classes: `Test<ClassName>`
   - Test methods: `test_<what_it_tests>`

4. **Add docstrings**:
   ```python
   def test_something(self):
       """Test that something does X when Y happens."""
       # Test code
   ```

## Test Metrics

Current test coverage by module:

- **classification_module.py**: 15 tests
  - All major functions covered
  - Error cases tested
  - Integration paths verified

- **regression_module.py**: 25 tests
  - Comprehensive coverage
  - Architecture validation
  - Performance metrics verified

- **tools.py**: Extensive coverage in test_tools.py

Total: **40+ tests** across all modules

## Contributing

When contributing new features:
1. Write tests first (TDD approach)
2. Ensure tests pass locally
3. Add test documentation
4. Update this README if needed
