"""
Test suite for classification_module.py

Tests all functions in the audio classification pipeline including:
- CSV to WAV preprocessing
- Dataset loading and preparation
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
from unittest.mock import MagicMock, Mock, patch, call

import numpy as np
import pandas as pd
import tensorflow as tf

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from classification_module import (
    preprocess_csv_to_wav,
    load_and_preprocess_data,
    build_and_compile_model,
    train_and_evaluate_model,
    create_export_model,
    _update_models_registry
)


class TestPreprocessCsvToWav(unittest.TestCase):
    """Test CSV to WAV preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_dir = Path(self.temp_dir) / 'csv_data'
        self.csv_dir.mkdir()
        self.model_name = 'test_model'

        # Create test CSV files
        for class_name in ['class_a', 'class_b']:
            for i in range(2):
                csv_path = self.csv_dir / f'{class_name}_{i}.csv'
                test_data = [
                    [1.5, class_name],  # deformation, load_type
                    [0.0, 0.1],         # time, voltage
                    [0.1, 0.2],
                    [0.2, 0.15],
                    [0.3, 0.1]
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

    @patch('classification_module.read_csv_files')
    @patch('classification_module.wav_generator')
    @patch('classification_module.save_wav_files')
    def test_preprocess_csv_to_wav_creates_directories(self, mock_save, mock_gen, mock_read):
        """Test that preprocessing creates necessary directory structure."""
        mock_read.return_value = []
        mock_gen.return_value = []
        wave_dir = Path(self.temp_dir) / 'wave_files'
        wave_dir.mkdir()
        (wave_dir / 'class_a').mkdir()
        (wave_dir / 'class_b').mkdir()
        mock_save.return_value = str(wave_dir)

        preprocess_csv_to_wav(str(self.csv_dir), self.model_name)

        # Verify directories were created
        model_dir = Path('models') / self.model_name
        self.assertTrue(model_dir.exists())
        self.assertTrue((model_dir / 'data' / 'raw').exists())
        self.assertTrue((model_dir / 'metadata').exists())

    @patch('classification_module.read_csv_files')
    @patch('classification_module.wav_generator')
    @patch('classification_module.save_wav_files')
    def test_preprocess_csv_to_wav_saves_metadata(self, mock_save, mock_gen, mock_read):
        """Test that preprocessing saves metadata correctly."""
        mock_read.return_value = []
        mock_gen.return_value = []
        wave_dir = Path(self.temp_dir) / 'wave_files'
        wave_dir.mkdir()
        (wave_dir / 'class_a').mkdir()
        (wave_dir / 'class_b').mkdir()
        mock_save.return_value = str(wave_dir)

        preprocess_csv_to_wav(str(self.csv_dir), self.model_name)

        # Check metadata file exists
        metadata_file = Path('models') / self.model_name / 'metadata' / 'preprocessing_info.json'
        self.assertTrue(metadata_file.exists())

        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        self.assertIn('wave_files_path', metadata)
        self.assertIn('class_distribution', metadata)

    def test_preprocess_csv_to_wav_invalid_directory(self):
        """Test that preprocessing raises error for non-existent directory."""
        with self.assertRaises(FileNotFoundError):
            preprocess_csv_to_wav('/nonexistent/path', self.model_name)


class TestLoadAndPreprocessData(unittest.TestCase):
    """Test data loading and preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_name = 'test_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)
        self.metadata_dir = self.model_dir / 'metadata'
        self.metadata_dir.mkdir()

        # Create mock preprocessing info
        wave_dir = Path(self.temp_dir) / 'wave_files'
        wave_dir.mkdir()
        preprocessing_info = {
            'wave_files_path': str(wave_dir),
            'wave_files_path_absolute': str(wave_dir.absolute()),
            'num_wav_files': 10,
            'class_distribution': {'class_a': 5, 'class_b': 5}
        }
        with open(self.metadata_dir / 'preprocessing_info.json', 'w') as f:
            json.dump(preprocessing_info, f)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

    @patch('classification_module.tf.keras.utils.audio_dataset_from_directory')
    @patch('classification_module.get_audio_length')
    @patch('classification_module.count_audio_files')
    def test_load_and_preprocess_data_creates_datasets(self, mock_count, mock_length, mock_dataset):
        """Test that data loading creates train/val/test datasets."""
        mock_count.return_value = 10
        mock_length.return_value = (16000, 16000)

        # Create mock datasets
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_train_ds.class_names = ['class_a', 'class_b']

        # Mock the map and dataset operations
        mapped_train = MagicMock()
        mapped_val = MagicMock()
        mock_train_ds.map.return_value = mapped_train
        mock_val_ds.map.return_value = mapped_val

        mock_dataset.return_value = (mock_train_ds, mock_val_ds)

        # Create a mock batch of data for iteration
        mock_batch = (tf.zeros((1, 124, 129, 1)), tf.zeros((1,), dtype=tf.int32))

        with patch('classification_module.preprocess_dataset') as mock_preprocess:
            mock_preprocess_train = MagicMock()
            mock_preprocess_train.__iter__ = Mock(return_value=iter([mock_batch]))
            mock_preprocess.return_value = mock_preprocess_train

            with patch('classification_module.tf.data.Dataset.save'):
                load_and_preprocess_data(self.model_name)

        # Verify dataset directories were created
        self.assertTrue((self.model_dir / 'datasets').exists())

    def test_load_and_preprocess_data_missing_preprocessing_info(self):
        """Test error when preprocessing info is missing."""
        # Remove preprocessing info
        (self.metadata_dir / 'preprocessing_info.json').unlink()

        with self.assertRaises(FileNotFoundError):
            load_and_preprocess_data(self.model_name)


class TestBuildAndCompileModel(unittest.TestCase):
    """Test model building and compilation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)
        self.metadata_dir = self.model_dir / 'metadata'
        self.metadata_dir.mkdir()

        # Create mock dataset info
        dataset_info = {
            'input_shape': [124, 129, 1],
            'num_labels': 3,
            'label_names': ['class_a', 'class_b', 'class_c']
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
        """Test that model has correct architecture."""
        build_and_compile_model(self.model_name)

        model_path = self.model_dir / 'models' / 'untrained' / 'model.keras'
        model = tf.keras.models.load_model(model_path)

        # Verify output shape matches number of labels
        self.assertEqual(model.output_shape[-1], 3)

    def test_build_and_compile_model_custom_learning_rate(self):
        """Test that custom learning rate is applied."""
        custom_lr = 0.0001
        build_and_compile_model(self.model_name, learning_rate=custom_lr)

        model_path = self.model_dir / 'models' / 'untrained' / 'model.keras'
        model = tf.keras.models.load_model(model_path)

        # Verify model is compiled
        self.assertTrue(model.built)


class TestTrainAndEvaluateModel(unittest.TestCase):
    """Test model training and evaluation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)

        # Create necessary directories
        (self.model_dir / 'datasets').mkdir()
        (self.model_dir / 'metadata').mkdir()
        (self.model_dir / 'models' / 'untrained').mkdir(parents=True)

        # Create a simple test model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[124, 129]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        model.save(self.model_dir / 'models' / 'untrained' / 'model.keras')

        # Save label names
        label_names = np.array(['class_a', 'class_b', 'class_c'])
        np.save(self.model_dir / 'metadata' / 'label_names.npy', label_names)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

    @patch('classification_module.tf.data.Dataset.load')
    def test_train_and_evaluate_model_creates_outputs(self, mock_load):
        """Test that training creates all expected output files."""
        # Create mock datasets
        X = tf.random.normal((10, 124, 129))
        y = tf.random.uniform((10,), maxval=3, dtype=tf.int32)
        mock_ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(2)
        mock_load.return_value = mock_ds

        train_and_evaluate_model(self.model_name, epochs=2, patience=1)

        # Verify outputs exist
        self.assertTrue((self.model_dir / 'models' / 'trained' / 'model.keras').exists())
        self.assertTrue((self.model_dir / 'training' / 'history.json').exists())
        self.assertTrue((self.model_dir / 'training' / 'test_results.json').exists())
        self.assertTrue((self.model_dir / 'training' / 'training_summary.json').exists())

    @patch('classification_module.tf.data.Dataset.load')
    def test_train_and_evaluate_model_creates_plots(self, mock_load):
        """Test that training creates visualization plots."""
        X = tf.random.normal((10, 124, 129))
        y = tf.random.uniform((10,), maxval=3, dtype=tf.int32)
        mock_ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(2)
        mock_load.return_value = mock_ds

        train_and_evaluate_model(self.model_name, epochs=2, patience=1)

        plots_dir = self.model_dir / 'training' / 'plots'
        self.assertTrue((plots_dir / 'loss_vs_epoch.png').exists())
        self.assertTrue((plots_dir / 'accuracy_vs_epoch.png').exists())
        self.assertTrue((plots_dir / 'loss_vs_time.png').exists())
        self.assertTrue((plots_dir / 'confusion_matrix.png').exists())


class TestCreateExportModel(unittest.TestCase):
    """Test model export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)

        # Create necessary directories
        (self.model_dir / 'metadata').mkdir()
        (self.model_dir / 'models' / 'trained').mkdir(parents=True)

        # Create and save a test model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[124, 129]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.save(self.model_dir / 'models' / 'trained' / 'model.keras')

        # Save label names
        label_names = np.array(['class_a', 'class_b', 'class_c'])
        np.save(self.model_dir / 'metadata' / 'label_names.npy', label_names)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
        # Clean up models.json
        models_json = Path('models.json')
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
        self.assertIn('export_path_absolute', metadata)
        self.assertIn('classes', metadata)
        self.assertEqual(len(metadata['classes']), 3)


class TestUpdateModelsRegistry(unittest.TestCase):
    """Test models registry management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = 'test_model'
        self.model_dir = Path('models') / self.model_name
        self.model_dir.mkdir(parents=True)

        # Create necessary directory structure
        (self.model_dir / 'metadata').mkdir()
        (self.model_dir / 'models' / 'trained').mkdir(parents=True)
        (self.model_dir / 'export' / 'saved_model').mkdir(parents=True)
        (self.model_dir / 'training' / 'plots').mkdir(parents=True)

        # Create test metadata files
        dataset_info = {
            'label_names': ['class_a', 'class_b'],
            'num_labels': 2
        }
        with open(self.model_dir / 'metadata' / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f)

        training_summary = {
            'test_accuracy': 0.95,
            'test_loss': 0.05
        }
        with open(self.model_dir / 'training' / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

        # Clean up models/models.json
        models_json = Path('models') / 'models.json'
        if models_json.exists():
            models_json.unlink()

    def test_update_models_registry_creates_registry(self):
        """Test that registry is created if it doesn't exist."""
        _update_models_registry(self.model_name)

        registry_path = Path('models') / 'models.json'
        self.assertTrue(registry_path.exists())

    def test_update_models_registry_adds_model_info(self):
        """Test that model information is added to registry."""
        _update_models_registry(self.model_name)

        registry_path = Path('models') / 'models.json'

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        self.assertIn('models', registry)
        self.assertIn(self.model_name, registry['models'])

        model_info = registry['models'][self.model_name]
        self.assertEqual(model_info['status'], 'exported')
        self.assertIn('classes', model_info)
        self.assertIn('performance', model_info)

    def test_update_models_registry_updates_existing(self):
        """Test that registry updates existing model entries."""
        # Create initial registry
        _update_models_registry(self.model_name)

        # Update training summary
        training_summary = {
            'test_accuracy': 0.98,
            'test_loss': 0.02
        }
        with open(self.model_dir / 'training' / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f)

        # Update registry again
        _update_models_registry(self.model_name)

        registry_path = Path('models') / 'models.json'

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        # Verify updated values
        self.assertEqual(
            registry['models'][self.model_name]['performance']['test_accuracy'],
            0.98
        )


if __name__ == '__main__':
    unittest.main()
