"""
Test suite for regression_module.py

Tests all functions in the Wide & Deep regression pipeline including:
- CSV data preprocessing
- Model building and compilation
- Training and evaluation
- Model export and registry management
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import tensorflow as tf

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from regression_module import (
    preprocess_csv_data,
    build_and_compile_model,
    train_and_evaluate_model,
    create_export_model,
    _update_models_registry
)


class TestPreprocessCsvData(unittest.TestCase):
    """Test CSV data preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_dir = Path(self.temp_dir) / 'csv_data'
        self.csv_dir.mkdir()
        self.model_name = 'test_regression_model'

        # Create test CSV files
        for i in range(3):
            csv_path = self.csv_dir / f'test_{i}.csv'
            test_data = [
                [1.5, 'Point'],  # deformation, load_type
                [0.0, 0.1],      # time, voltage
                [0.1, 0.2],
                [0.2, 0.15],
                [0.3, 0.1],
                [0.4, 0.05]
            ]
            df = pd.DataFrame(test_data)
            df.to_csv(csv_path, header=False, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        # Clean up models directory if created
        models_dir = Path('models') / self.model_name
        if models_dir.exists():
            shutil.rmtree(models_dir)

    @patch('regression_module.load_data')
    @patch('regression_module.DataProcessor')
    def test_preprocess_csv_data_creates_directories(self, mock_dp_class, mock_load):
        """Test that preprocessing creates necessary directory structure."""
        # Mock DataProcessor
        mock_dp = MagicMock()
        mock_dp.combined_data = np.array([
            [0.5, 1.0, 1.5],  # peaks, load_type, deformation
            [0.6, 1.0, 1.6],
            [0.7, 2.0, 1.7],
            [0.8, 2.0, 1.8],
        ])
        mock_dp_class.return_value = mock_dp
        mock_load.return_value = []

        preprocess_csv_data(str(self.csv_dir), self.model_name)

        # Verify directories were created
        model_dir = Path('models') / self.model_name
        self.assertTrue(model_dir.exists())
        self.assertTrue((model_dir / 'data' / 'raw').exists())
        self.assertTrue((model_dir / 'datasets').exists())
        self.assertTrue((model_dir / 'metadata').exists())

    @patch('regression_module.load_data')
    @patch('regression_module.DataProcessor')
    def test_preprocess_csv_data_creates_splits(self, mock_dp_class, mock_load):
        """Test that preprocessing creates train/val/test splits."""
        # Create more data for proper splitting
        mock_dp = MagicMock()
        data_points = []
        for i in range(100):
            data_points.append([0.5 + i*0.01, float(i % 2), 1.5 + i*0.02])
        mock_dp.combined_data = np.array(data_points)
        mock_dp_class.return_value = mock_dp
        mock_load.return_value = []

        preprocess_csv_data(str(self.csv_dir), self.model_name, seed=42)

        # Verify dataset files were created
        datasets_dir = Path('models') / self.model_name / 'datasets'
        self.assertTrue((datasets_dir / 'train.npz').exists())
        self.assertTrue((datasets_dir / 'val.npz').exists())
        self.assertTrue((datasets_dir / 'test.npz').exists())

        # Verify dataset content
        train_data = np.load(datasets_dir / 'train.npz')
        self.assertIn('X_wide', train_data)
        self.assertIn('X_deep', train_data)
        self.assertIn('y', train_data)

    @patch('regression_module.load_data')
    @patch('regression_module.DataProcessor')
    def test_preprocess_csv_data_saves_metadata(self, mock_dp_class, mock_load):
        """Test that preprocessing saves correct metadata."""
        mock_dp = MagicMock()
        mock_dp.combined_data = np.array([
            [0.5, 1.0, 1.5],
            [0.6, 1.0, 1.6],
            [0.7, 2.0, 1.7],
            [0.8, 2.0, 1.8],
        ])
        mock_dp_class.return_value = mock_dp
        mock_load.return_value = []

        preprocess_csv_data(str(self.csv_dir), self.model_name)

        # Verify metadata file
        metadata_file = Path('models') / self.model_name / 'metadata' / 'dataset_info.json'
        self.assertTrue(metadata_file.exists())

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self.assertEqual(metadata['model_type'], 'wide_and_deep_regression')
        self.assertIn('total_samples', metadata)
        self.assertIn('train_samples', metadata)
        self.assertIn('val_samples', metadata)
        self.assertIn('test_samples', metadata)
        self.assertIn('features', metadata)

    def test_preprocess_csv_data_invalid_directory(self):
        """Test that preprocessing raises error for non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            preprocess_csv_data('/nonexistent/path', self.model_name)

    @patch('regression_module.load_data')
    @patch('regression_module.DataProcessor')
    def test_preprocess_csv_data_copies_raw_files(self, mock_dp_class, mock_load):
        """Test that original CSV files are copied to raw data directory."""
        mock_dp = MagicMock()
        # Need at least 10 samples for proper splitting
        data_points = []
        for i in range(20):
            data_points.append([0.5 + i*0.01, float(i % 2), 1.5 + i*0.02])
        mock_dp.combined_data = np.array(data_points)
        mock_dp_class.return_value = mock_dp
        mock_load.return_value = []

        preprocess_csv_data(str(self.csv_dir), self.model_name)

        # Verify CSV files were copied
        raw_data_dir = Path('models') / self.model_name / 'data' / 'raw'
        csv_files = list(raw_data_dir.glob('*.csv'))
        self.assertEqual(len(csv_files), 3)


class TestBuildAndCompileModel(unittest.TestCase):
    """Test Wide & Deep model building and compilation."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_regression_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)
        self.metadata_dir = self.model_dir / 'metadata'
        self.metadata_dir.mkdir()

        # Create mock dataset info
        dataset_info = {
            'model_type': 'wide_and_deep_regression',
            'total_samples': 100,
            'train_samples': 70,
            'val_samples': 20,
            'test_samples': 10,
            'features': {
                'wide_feature': 'load_type',
                'deep_feature': 'peaks',
                'target': 'deformation'
            }
        }
        with open(self.metadata_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

    def test_build_and_compile_model_creates_model(self):
        """Test that model building creates and saves a model."""
        build_and_compile_model(self.model_name)

        # Verify model was saved
        model_path = self.model_dir / 'models' / 'untrained' / 'model.keras'
        self.assertTrue(model_path.exists())

        # Verify model can be loaded
        model = tf.keras.models.load_model(model_path)
        self.assertIsInstance(model, tf.keras.Model)

    def test_build_and_compile_model_correct_architecture(self):
        """Test that model has correct Wide & Deep architecture."""
        build_and_compile_model(self.model_name)

        model_path = self.model_dir / 'models' / 'untrained' / 'model.keras'
        model = tf.keras.models.load_model(model_path)

        # Verify model has two inputs (wide and deep)
        self.assertEqual(len(model.inputs), 2)
        self.assertIn('wide_input', model.inputs[0].name)
        self.assertIn('deep_input', model.inputs[1].name)

        # Verify single output for regression
        self.assertEqual(len(model.outputs), 1)
        self.assertEqual(model.output_shape[-1], 1)

    def test_build_and_compile_model_custom_hidden_layers(self):
        """Test that custom hidden layers are applied."""
        custom_layers = [32, 16, 8]
        build_and_compile_model(self.model_name, hidden_layers=custom_layers)

        model_path = self.model_dir / 'models' / 'untrained' / 'model.keras'
        model = tf.keras.models.load_model(model_path)

        # Model should be successfully built
        self.assertTrue(model.built)

    def test_build_and_compile_model_saves_architecture_info(self):
        """Test that architecture information is saved."""
        hidden_layers = [64, 32]
        build_and_compile_model(self.model_name, hidden_layers=hidden_layers)

        arch_file = self.model_dir / 'models' / 'untrained' / 'model_architecture.json'
        self.assertTrue(arch_file.exists())

        with open(arch_file, 'r') as f:
            arch_info = json.load(f)

        self.assertEqual(arch_info['model_type'], 'wide_and_deep_regression')
        self.assertEqual(arch_info['hidden_layers'], hidden_layers)
        self.assertIn('total_params', arch_info)

    def test_build_and_compile_model_saves_summary(self):
        """Test that model summary is saved to text file."""
        build_and_compile_model(self.model_name)

        summary_file = self.model_dir / 'models' / 'untrained' / 'model_summary.txt'
        self.assertTrue(summary_file.exists())

        with open(summary_file, 'r') as f:
            summary_text = f.read()

        self.assertIn('Model:', summary_text)

    def test_build_and_compile_model_missing_metadata(self):
        """Test error when dataset metadata is missing."""
        (self.metadata_dir / 'dataset_info.json').unlink()

        with self.assertRaises(FileNotFoundError):
            build_and_compile_model(self.model_name)


class TestTrainAndEvaluateModel(unittest.TestCase):
    """Test model training and evaluation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_regression_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)

        # Create necessary directories
        datasets_dir = self.model_dir / 'datasets'
        datasets_dir.mkdir()
        (self.model_dir / 'metadata').mkdir()
        (self.model_dir / 'models' / 'untrained').mkdir(parents=True)

        # Create test datasets
        n_train, n_val, n_test = 50, 20, 20
        X_train_wide = np.random.randn(n_train, 1)
        X_train_deep = np.random.randn(n_train, 1)
        y_train = 2 * X_train_wide.flatten() + 3 * X_train_deep.flatten() + np.random.randn(n_train) * 0.1

        X_val_wide = np.random.randn(n_val, 1)
        X_val_deep = np.random.randn(n_val, 1)
        y_val = 2 * X_val_wide.flatten() + 3 * X_val_deep.flatten() + np.random.randn(n_val) * 0.1

        X_test_wide = np.random.randn(n_test, 1)
        X_test_deep = np.random.randn(n_test, 1)
        y_test = 2 * X_test_wide.flatten() + 3 * X_test_deep.flatten() + np.random.randn(n_test) * 0.1

        np.savez(datasets_dir / 'train.npz', X_wide=X_train_wide, X_deep=X_train_deep, y=y_train)
        np.savez(datasets_dir / 'val.npz', X_wide=X_val_wide, X_deep=X_val_deep, y=y_val)
        np.savez(datasets_dir / 'test.npz', X_wide=X_test_wide, X_deep=X_test_deep, y=y_test)

        # Create and save a test model
        input_wide = tf.keras.layers.Input(shape=[1], name='wide_input')
        input_deep = tf.keras.layers.Input(shape=[1], name='deep_input')
        norm_wide = tf.keras.layers.Normalization(name='wide_normalization')(input_wide)
        norm_deep = tf.keras.layers.Normalization(name='deep_normalization')(input_deep)
        deep = tf.keras.layers.Dense(32, activation='relu')(norm_deep)
        deep = tf.keras.layers.Dense(16, activation='relu')(deep)
        concat = tf.keras.layers.concatenate([norm_wide, deep], name='wide_deep_concat')
        output = tf.keras.layers.Dense(1, name='output')(concat)

        model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])
        model.compile(loss='mse', optimizer='adam', metrics=['RootMeanSquaredError'])
        model.save(self.model_dir / 'models' / 'untrained' / 'model.keras')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

    def test_train_and_evaluate_model_creates_outputs(self):
        """Test that training creates all expected output files."""
        train_and_evaluate_model(self.model_name, epochs=5, patience=2)

        # Verify outputs exist
        self.assertTrue((self.model_dir / 'models' / 'trained' / 'model.keras').exists())
        self.assertTrue((self.model_dir / 'training' / 'history.json').exists())
        self.assertTrue((self.model_dir / 'training' / 'test_results.json').exists())
        self.assertTrue((self.model_dir / 'training' / 'training_summary.json').exists())

    def test_train_and_evaluate_model_history_content(self):
        """Test that training history contains expected keys."""
        train_and_evaluate_model(self.model_name, epochs=5, patience=2)

        history_file = self.model_dir / 'training' / 'history.json'
        with open(history_file, 'r') as f:
            history = json.load(f)

        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('epochs', history)
        self.assertIn('epoch_times_seconds', history)
        self.assertIn('cumulative_times_seconds', history)

    def test_train_and_evaluate_model_test_results(self):
        """Test that test results contain predictions and metrics."""
        train_and_evaluate_model(self.model_name, epochs=5, patience=2)

        test_results_file = self.model_dir / 'training' / 'test_results.json'
        with open(test_results_file, 'r') as f:
            test_results = json.load(f)

        self.assertIn('test_loss_mse', test_results)
        self.assertIn('test_rmse', test_results)
        self.assertIn('predictions', test_results)
        self.assertIn('actuals', test_results)
        self.assertIn('residuals', test_results)

    def test_train_and_evaluate_model_creates_plots(self):
        """Test that training creates all visualization plots."""
        train_and_evaluate_model(self.model_name, epochs=5, patience=2)

        plots_dir = self.model_dir / 'training' / 'plots'
        self.assertTrue((plots_dir / 'loss_vs_epoch.png').exists())
        self.assertTrue((plots_dir / 'rmse_vs_epoch.png').exists())
        self.assertTrue((plots_dir / 'loss_vs_time.png').exists())
        self.assertTrue((plots_dir / 'predictions_vs_actual.png').exists())
        self.assertTrue((plots_dir / 'residuals_analysis.png').exists())

    def test_train_and_evaluate_model_plots_metadata(self):
        """Test that plots metadata is saved correctly."""
        train_and_evaluate_model(self.model_name, epochs=5, patience=2)

        plots_metadata_file = self.model_dir / 'training' / 'plots_metadata.json'
        self.assertTrue(plots_metadata_file.exists())

        with open(plots_metadata_file, 'r') as f:
            plots_metadata = json.load(f)

        self.assertIn('plots_directory_absolute', plots_metadata)
        self.assertIn('plots', plots_metadata)
        self.assertIn('loss_vs_epoch', plots_metadata['plots'])
        self.assertIn('residuals_analysis', plots_metadata['plots'])

    def test_train_and_evaluate_model_training_summary(self):
        """Test that training summary contains key metrics."""
        train_and_evaluate_model(self.model_name, epochs=5, patience=2)

        summary_file = self.model_dir / 'training' / 'training_summary.json'
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        self.assertIn('total_epochs', summary)
        self.assertIn('test_loss_mse', summary)
        self.assertIn('test_rmse', summary)
        self.assertIn('best_val_loss', summary)
        self.assertIn('residuals_mean', summary)
        self.assertIn('residuals_std', summary)


class TestCreateExportModel(unittest.TestCase):
    """Test model export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_regression_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)

        # Create necessary directories
        (self.model_dir / 'metadata').mkdir()
        (self.model_dir / 'models' / 'trained').mkdir(parents=True)

        # Create and save a test model
        input_wide = tf.keras.layers.Input(shape=[1], name='wide_input')
        input_deep = tf.keras.layers.Input(shape=[1], name='deep_input')
        concat = tf.keras.layers.concatenate([input_wide, input_deep])
        output = tf.keras.layers.Dense(1)(concat)

        model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])
        model.compile(optimizer='adam', loss='mse')
        model.save(self.model_dir / 'models' / 'trained' / 'model.keras')

        # Create dataset info
        dataset_info = {
            'total_samples': 100,
            'features': {
                'wide_feature': 'load_type',
                'deep_feature': 'peaks',
                'target': 'deformation'
            }
        }
        with open(self.model_dir / 'metadata' / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
        # Clean up models.json in project root
        project_root = Path(__file__).resolve().parent.parent
        models_json = project_root / 'models.json'
        if models_json.exists():
            models_json.unlink()

    def test_create_export_model_saves_export(self):
        """Test that export model creates saved_model directory."""
        create_export_model(self.model_name)

        export_dir = self.model_dir / 'export' / 'saved_model'
        self.assertTrue(export_dir.exists())
        self.assertTrue((export_dir / 'saved_model.pb').exists())

    def test_create_export_model_saves_metadata(self):
        """Test that export model saves metadata."""
        create_export_model(self.model_name)

        metadata_file = self.model_dir / 'export' / 'export_metadata.json'
        self.assertTrue(metadata_file.exists())

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self.assertEqual(metadata['model_type'], 'wide_and_deep_regression')
        self.assertIn('export_path_absolute', metadata)
        self.assertIn('inputs', metadata)
        self.assertIn('output', metadata)

    def test_create_export_model_missing_trained_model(self):
        """Test error when trained model doesn't exist."""
        # Remove trained model
        (self.model_dir / 'models' / 'trained' / 'model.keras').unlink()

        with self.assertRaises(FileNotFoundError):
            create_export_model(self.model_name)


class TestUpdateModelsRegistry(unittest.TestCase):
    """Test models registry management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_regression_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)

        # Create necessary directory structure
        (self.model_dir / 'metadata').mkdir()
        (self.model_dir / 'models' / 'trained').mkdir(parents=True)
        (self.model_dir / 'models' / 'untrained').mkdir(parents=True)
        (self.model_dir / 'export' / 'saved_model').mkdir(parents=True)
        (self.model_dir / 'training' / 'plots').mkdir(parents=True)

        # Create test metadata files
        dataset_info = {
            'features': {
                'wide_feature': 'load_type',
                'deep_feature': 'peaks',
                'target': 'deformation'
            },
            'total_samples': 100
        }
        with open(self.model_dir / 'metadata' / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f)

        training_summary = {
            'test_loss_mse': 0.05,
            'test_rmse': 0.22
        }
        with open(self.model_dir / 'training' / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f)

        architecture_info = {
            'hidden_layers': [64, 32, 16],
            'total_params': 1234
        }
        with open(self.model_dir / 'models' / 'untrained' / 'model_architecture.json', 'w') as f:
            json.dump(architecture_info, f)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

        # Clean up models.json in project root
        project_root = Path(__file__).resolve().parent.parent
        models_json = project_root / 'models.json'
        if models_json.exists():
            models_json.unlink()

    def test_update_models_registry_creates_registry(self):
        """Test that registry is created if it doesn't exist."""
        _update_models_registry(self.model_name)

        project_root = Path(__file__).resolve().parent.parent
        registry_path = project_root / 'models.json'
        self.assertTrue(registry_path.exists())

    def test_update_models_registry_adds_model_info(self):
        """Test that model information is added to registry."""
        _update_models_registry(self.model_name)

        project_root = Path(__file__).resolve().parent.parent
        registry_path = project_root / 'models.json'

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        self.assertIn('models', registry)
        self.assertIn(self.model_name, registry['models'])

        model_info = registry['models'][self.model_name]
        self.assertEqual(model_info['model_type'], 'wide_and_deep_regression')
        self.assertEqual(model_info['status'], 'exported')
        self.assertIn('features', model_info)
        self.assertIn('performance', model_info)
        self.assertIn('architecture', model_info)

    def test_update_models_registry_includes_architecture(self):
        """Test that architecture information is included in registry."""
        _update_models_registry(self.model_name)

        project_root = Path(__file__).resolve().parent.parent
        registry_path = project_root / 'models.json'

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        architecture = registry['models'][self.model_name]['architecture']
        self.assertEqual(architecture['type'], 'Wide & Deep')
        self.assertEqual(architecture['hidden_layers'], [64, 32, 16])
        self.assertEqual(architecture['total_params'], 1234)

    def test_update_models_registry_includes_performance(self):
        """Test that performance metrics are included in registry."""
        _update_models_registry(self.model_name)

        project_root = Path(__file__).resolve().parent.parent
        registry_path = project_root / 'models.json'

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        performance = registry['models'][self.model_name]['performance']
        self.assertEqual(performance['test_loss_mse'], 0.05)
        self.assertEqual(performance['test_rmse'], 0.22)

    def test_update_models_registry_updates_existing(self):
        """Test that registry updates existing model entries."""
        # Create initial registry
        _update_models_registry(self.model_name)

        # Update training summary
        training_summary = {
            'test_loss_mse': 0.02,
            'test_rmse': 0.14
        }
        with open(self.model_dir / 'training' / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f)

        # Update registry again
        _update_models_registry(self.model_name)

        project_root = Path(__file__).resolve().parent.parent
        registry_path = project_root / 'models.json'

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        # Verify updated values
        self.assertEqual(
            registry['models'][self.model_name]['performance']['test_rmse'],
            0.14
        )


if __name__ == '__main__':
    unittest.main()
