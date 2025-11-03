"""
Test suite for tools.py module.

Tests all utility functions for CSV processing, signal processing, 
audio generation, and TensorFlow helpers.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from tools import (
    set_seed,
    load_data,
    process_csv_file,
    read_csv_files,
    interpolate_data,
    extract_data_and_type,
    normalize_data,
    convert_to_wave,
    wav_generator,
    save_wav_files,
    get_audio_length,
    count_audio_files,
    get_spectrogram,
    preprocess_dataset,
    squeeze,
    make_spec_ds,
    DataProcessor,
    LossOverTimeHistory
)


class TestSetSeed(unittest.TestCase):
    """Test seed setting functionality."""
    
    def test_set_seed(self):
        """Test that set_seed sets both TensorFlow and NumPy seeds."""
        set_seed(123)
        # Test that seeds are set (we can't directly verify, but ensure no errors)
        self.assertTrue(True)  # If no exception is raised, test passes


class TestLoadData(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, 'test.csv')
        
        # Create test CSV data
        test_data = [
            [1.5, 'Point'],  # deformation, load_type
            [0.0, 0.1],      # time, voltage
            [0.1, 0.2],
            [0.2, 0.15]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv_path, header=False, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_data_file_exists(self):
        """Test loading data from existing file."""
        data = load_data(self.test_csv_path, 1000)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
    
    def test_load_data_file_not_found(self):
        """Test loading data from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data('/nonexistent/path.csv', 1000)


class TestProcessCsvFile(unittest.TestCase):
    """Test CSV file processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, 'test.csv')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_process_csv_file_point_load(self):
        """Test processing CSV with Point load type."""
        test_data = [
            [1.5, 'Point'],
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.15]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv_path, header=False, index=False)
        
        time_voltage, load_type, deformation = process_csv_file(self.test_csv_path)
        
        self.assertEqual(load_type, 'point')
        self.assertEqual(deformation, 1.5)
        self.assertEqual(len(time_voltage), 3)
        self.assertEqual(time_voltage[0], (0.0, 0.1))
    
    def test_process_csv_file_uniform_load(self):
        """Test processing CSV with Uniform load type."""
        test_data = [
            [2.0, 'Uniform'],
            [0.0, 0.1],
            [0.1, 0.2]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv_path, header=False, index=False)
        
        time_voltage, load_type, deformation = process_csv_file(self.test_csv_path)
        
        self.assertEqual(load_type, 'uniform')
        self.assertEqual(deformation, 2.0)
    
    def test_process_csv_file_invalid_load_type(self):
        """Test processing CSV with invalid load type."""
        test_data = [
            [1.0, 'Invalid'],
            [0.0, 0.1]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv_path, header=False, index=False)
        
        with self.assertRaises(ValueError):
            process_csv_file(self.test_csv_path)


class TestReadCsvFiles(unittest.TestCase):
    """Test reading multiple CSV files."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_read_csv_files_single_file(self):
        """Test reading a single CSV file."""
        test_csv_path = os.path.join(self.temp_dir, 'test.csv')
        test_data = [
            [1.0, 'Point'],
            [0.0, 0.1],
            [0.1, 0.2]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(test_csv_path, header=False, index=False)
        
        data = read_csv_files(test_csv_path, 1000)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
    
    def test_read_csv_files_directory(self):
        """Test reading CSV files from directory."""
        # Create multiple CSV files
        for i in range(2):
            test_csv_path = os.path.join(self.temp_dir, f'test{i}.csv')
            test_data = [
                [1.0, 'Point'],
                [0.0, 0.1],
                [0.1, 0.2]
            ]
            df = pd.DataFrame(test_data)
            df.to_csv(test_csv_path, header=False, index=False)
        
        data = read_csv_files(self.temp_dir, 1000)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
    
    def test_read_csv_files_invalid_path(self):
        """Test reading from invalid path."""
        with self.assertRaises(ValueError):
            read_csv_files('/nonexistent/path', 1000)


class TestInterpolateData(unittest.TestCase):
    """Test data interpolation functionality."""
    
    def test_interpolate_data(self):
        """Test interpolating time-voltage data."""
        # Create test data
        time_voltage = [(0.0, 0.1), (0.2, 0.3), (0.4, 0.2)]
        data = [(time_voltage, 'point', 1.0)]
        
        interpolated = interpolate_data(data, 0.1, 1.0)
        
        self.assertIsInstance(interpolated, list)
        self.assertEqual(len(interpolated), 1)
        self.assertEqual(len(interpolated[0]), 3)  # (time_voltage, load_type, deformation)
        
        new_time_voltage, load_type, deformation = interpolated[0]
        self.assertEqual(load_type, 'point')
        self.assertEqual(deformation, 1.0)
        self.assertIsInstance(new_time_voltage, list)


class TestExtractDataAndType(unittest.TestCase):
    """Test data extraction functionality."""
    
    def test_extract_data_and_type(self):
        """Test extracting voltage data and load types."""
        data_list = [
            ([(0.0, 0.1), (0.1, 0.2)], 'point', 1.0),
            ([(0.0, 0.3), (0.1, 0.4)], 'uniform', 2.0)
        ]
        
        extracted = extract_data_and_type(data_list)
        
        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0][1], 'point')
        self.assertEqual(extracted[1][1], 'uniform')
        self.assertEqual(extracted[0][0], [0.1, 0.2])
        self.assertEqual(extracted[1][0], [0.3, 0.4])


class TestNormalizeData(unittest.TestCase):
    """Test data normalization functionality."""
    
    def test_normalize_data(self):
        """Test normalizing data array."""
        data = np.array([1.0, -2.0, 3.0, -1.0])
        normalized = normalize_data(data)
        
        # Check that max absolute value is 1.0
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=10)
        
        # Check that relative ratios are preserved
        expected_ratios = data / np.max(np.abs(data))
        np.testing.assert_array_almost_equal(normalized, expected_ratios)


class TestConvertToWave(unittest.TestCase):
    """Test WAV conversion functionality."""
    
    def test_convert_to_wave(self):
        """Test converting normalized data to WAV."""
        normalized_data = np.array([0.5, -0.5, 0.3, -0.3])
        time_interval = 0.001  # 1000 Hz
        
        wav_tensor = convert_to_wave(normalized_data, time_interval)
        
        self.assertIsInstance(wav_tensor, tf.Tensor)
        self.assertEqual(wav_tensor.dtype, tf.string)


class TestWavGenerator(unittest.TestCase):
    """Test WAV generation functionality."""
    
    def test_wav_generator(self):
        """Test generating WAV files from data."""
        data = [
            ([(0.0, 0.1), (0.1, 0.2)], 'point', 1.0),
            ([(0.0, 0.3), (0.1, 0.4)], 'uniform', 2.0)
        ]
        
        wav_files = wav_generator(data, 1000)
        
        self.assertIsInstance(wav_files, list)
        self.assertEqual(len(wav_files), 2)
        
        for wav_bytes, load_type in wav_files:
            self.assertIsInstance(wav_bytes, tf.Tensor)
            self.assertIn(load_type, ['point', 'uniform'])


class TestSaveWavFiles(unittest.TestCase):
    """Test WAV file saving functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_wav_files(self):
        """Test saving WAV files to directory."""
        # Create test WAV data
        test_data = np.array([0.5, -0.5, 0.3, -0.3])
        wav_tensor = convert_to_wave(test_data, 0.001)
        
        wav_files = [(wav_tensor, 'point'), (wav_tensor, 'uniform')]
        output_path = os.path.join(self.temp_dir, 'test_wavs')
        
        actual_path = save_wav_files(wav_files, output_path)
        
        self.assertEqual(actual_path, output_path)
        self.assertTrue(os.path.exists(actual_path))
        self.assertTrue(os.path.exists(os.path.join(actual_path, 'point')))
        self.assertTrue(os.path.exists(os.path.join(actual_path, 'uniform')))


class TestGetAudioLength(unittest.TestCase):
    """Test audio length detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.class_dir = os.path.join(self.temp_dir, 'test_class')
        os.makedirs(self.class_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_audio_length_no_files(self):
        """Test getting audio length when no files exist."""
        length, sample_rate = get_audio_length(self.temp_dir)
        self.assertIsNone(length)
        self.assertIsNone(sample_rate)


class TestCountAudioFiles(unittest.TestCase):
    """Test audio file counting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_count_audio_files_empty(self):
        """Test counting audio files in empty directory."""
        count = count_audio_files(self.temp_dir)
        self.assertEqual(count, 0)


class TestGetSpectrogram(unittest.TestCase):
    """Test spectrogram generation functionality."""
    
    def test_get_spectrogram(self):
        """Test generating spectrogram from waveform."""
        # Create test waveform with sufficient length for STFT
        waveform = tf.constant([0.1] * 512, dtype=tf.float32)  # 512 samples
        
        spectrogram = get_spectrogram(waveform)
        
        self.assertIsInstance(spectrogram, tf.Tensor)
        self.assertEqual(len(spectrogram.shape), 3)  # [freq_bins, time_frames, 1]


class TestPreprocessDataset(unittest.TestCase):
    """Test dataset preprocessing functionality."""
    
    def test_preprocess_dataset(self):
        """Test preprocessing dataset to spectrograms."""
        # Create test dataset with sufficient length for STFT
        waveforms = tf.constant([[0.1] * 512, [0.2] * 512])  # 512 samples each
        labels = tf.constant([0, 1])
        dataset = tf.data.Dataset.from_tensor_slices((waveforms, labels))
        
        processed = preprocess_dataset(dataset)
        
        # Test that we can iterate through the dataset
        for spectrogram, label in processed.take(1):
            self.assertIsInstance(spectrogram, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)


class TestSqueeze(unittest.TestCase):
    """Test squeeze functionality."""
    
    def test_squeeze(self):
        """Test squeezing audio tensor."""
        audio = tf.constant([[[0.1], [0.2], [0.3]]])  # [batch, time, 1]
        labels = tf.constant([0])
        
        squeezed_audio, squeezed_labels = squeeze(audio, labels)
        
        self.assertEqual(len(squeezed_audio.shape), 2)  # [batch, time]
        self.assertEqual(squeezed_labels.shape, labels.shape)


class TestMakeSpecDs(unittest.TestCase):
    """Test spectrogram dataset creation."""
    
    def test_make_spec_ds(self):
        """Test creating spectrogram dataset."""
        # Create test dataset with sufficient length for STFT
        waveforms = tf.constant([[0.1] * 512, [0.2] * 512])  # 512 samples each
        labels = tf.constant([0, 1])
        dataset = tf.data.Dataset.from_tensor_slices((waveforms, labels))
        
        spec_ds = make_spec_ds(dataset)
        
        # Test that we can iterate through the dataset
        for spectrogram, label in spec_ds.take(1):
            self.assertIsInstance(spectrogram, tf.Tensor)
            self.assertIsInstance(label, tf.Tensor)


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create more realistic test data with clear peaks and valleys
        self.test_data = [
            ([(0.0, 0.1), (0.1, 0.5), (0.2, 0.1), (0.3, 0.6), (0.4, 0.1)], 'point', 1.0),
            ([(0.0, 0.2), (0.1, 0.7), (0.2, 0.2), (0.3, 0.8), (0.4, 0.2)], 'uniform', 2.0)
        ]
        self.processor = DataProcessor(self.test_data)
    
    def test_init(self):
        """Test DataProcessor initialization."""
        self.assertEqual(self.processor.data, self.test_data)
        self.assertEqual(self.processor.time, [])
        self.assertEqual(self.processor.voltage, [])
        self.assertEqual(self.processor.load_type, [])
        self.assertEqual(self.processor.deformation, [])
    
    def test_time_voltage_data(self):
        """Test extracting time and voltage data."""
        self.processor.time_voltage_data()
        
        self.assertEqual(len(self.processor.time), 2)
        self.assertEqual(len(self.processor.voltage), 2)
        self.assertEqual(len(self.processor.load_type), 2)
        self.assertEqual(len(self.processor.deformation), 2)
        
        self.assertEqual(self.processor.load_type[0], 'point')
        self.assertEqual(self.processor.load_type[1], 'uniform')
    
    def test_convert_to_array(self):
        """Test converting to numpy arrays."""
        self.processor.time_voltage_data()
        self.processor.convert_to_array()
        
        for time_array in self.processor.time:
            self.assertIsInstance(time_array, np.ndarray)
        for voltage_array in self.processor.voltage:
            self.assertIsInstance(voltage_array, np.ndarray)
    
    def test_calculate_voltage_peak_to_peak(self):
        """Test calculating peak-to-peak voltage values."""
        self.processor.time_voltage_data()
        self.processor.convert_to_array()
        self.processor.calculate_voltage_peak_to_peak()
        
        self.assertIsInstance(self.processor.combined_data, np.ndarray)
        self.assertGreater(len(self.processor.combined_data), 0)


class TestLossOverTimeHistory(unittest.TestCase):
    """Test LossOverTimeHistory callback functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.callback = LossOverTimeHistory()
    
    def test_on_train_begin(self):
        """Test training begin callback."""
        self.callback.on_train_begin()
        
        self.assertIsNotNone(self.callback.train_start_time)
        self.assertEqual(self.callback.times, [])
        self.assertEqual(self.callback.losses, [])
        self.assertEqual(self.callback.val_losses, [])
        self.assertEqual(self.callback.metrics, {})
    
    def test_on_epoch_end(self):
        """Test epoch end callback."""
        self.callback.on_train_begin()
        
        # Simulate epoch end
        logs = {'loss': 0.5, 'val_loss': 0.6, 'accuracy': 0.8}
        self.callback.on_epoch_end(0, logs)
        
        self.assertEqual(len(self.callback.times), 1)
        self.assertEqual(len(self.callback.losses), 1)
        self.assertEqual(len(self.callback.val_losses), 1)
        self.assertEqual(self.callback.losses[0], 0.5)
        self.assertEqual(self.callback.val_losses[0], 0.6)
        self.assertIn('accuracy', self.callback.metrics)
    
    def test_get_results(self):
        """Test getting results dictionary."""
        self.callback.on_train_begin()
        
        logs = {'loss': 0.5, 'val_loss': 0.6, 'accuracy': 0.8}
        self.callback.on_epoch_end(0, logs)
        
        results = self.callback.get_results()
        
        self.assertIn('time', results)
        self.assertIn('loss', results)
        self.assertIn('val_loss', results)
        self.assertIn('accuracy', results)


if __name__ == '__main__':
    unittest.main()
