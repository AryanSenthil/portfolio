"""
Test suite for test_model.py module.

Tests all functions and classes for CSV processing, data interpolation,
WAV generation, and the WaveAnalysis class for model testing.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
import tensorflow as tf

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from test_model import (
    process_csv_file,
    read_csv_files,
    interpolate_data,
    extract_data_and_type,
    normalize_data,
    convert_to_wave,
    wav_generator,
    WaveAnalysis,
    display_model_results
)


class TestProcessCsvFile(unittest.TestCase):
    """Test CSV file processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, 'test.csv')

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_process_csv_file_basic(self):
        """Test basic CSV file processing."""
        test_data = [
            ['header1', 'header2'],  # Header row
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.15]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv_path, header=False, index=False)

        time_voltage_pairs, type_of_load, deformation = process_csv_file(self.test_csv_path)

        self.assertEqual(len(time_voltage_pairs), 3)
        self.assertIsInstance(time_voltage_pairs, list)
        self.assertEqual(time_voltage_pairs[0], (0.0, 0.1))
        self.assertEqual(time_voltage_pairs[1], (0.1, 0.2))
        self.assertEqual(time_voltage_pairs[2], (0.2, 0.15))
        self.assertIsNone(type_of_load)
        self.assertIsNone(deformation)

    def test_process_csv_file_returns_tuples(self):
        """Test that process_csv_file returns proper tuple structure."""
        test_data = [
            ['header1', 'header2'],
            [0.5, 1.5],
            [1.0, 2.0]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_csv_path, header=False, index=False)

        result = process_csv_file(self.test_csv_path)

        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], list)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])


class TestReadCsvFiles(unittest.TestCase):
    """Test reading and processing CSV files."""

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
            ['header1', 'header2'],
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.3]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(test_csv_path, header=False, index=False)

        sampling_rate = 10
        period = 0.3

        data = read_csv_files(test_csv_path, sampling_rate, period)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertEqual(len(data[0]), 3)  # (time_voltage, load_type, deformation)

    def test_read_csv_files_directory(self):
        """Test reading multiple CSV files from directory."""
        for i in range(2):
            test_csv_path = os.path.join(self.temp_dir, f'test{i}.csv')
            test_data = [
                ['header1', 'header2'],
                [0.0, 0.1],
                [0.1, 0.2]
            ]
            df = pd.DataFrame(test_data)
            df.to_csv(test_csv_path, header=False, index=False)

        sampling_rate = 10
        period = 0.2

        data = read_csv_files(self.temp_dir, sampling_rate, period)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

    def test_read_csv_files_invalid_path(self):
        """Test reading from invalid path raises ValueError."""
        with self.assertRaises(ValueError):
            read_csv_files('/nonexistent/path', 1000, 10)

    def test_read_csv_files_non_csv_file(self):
        """Test reading non-CSV file raises ValueError."""
        txt_file = os.path.join(self.temp_dir, 'test.txt')
        with open(txt_file, 'w') as f:
            f.write('test')

        with self.assertRaises(ValueError):
            read_csv_files(txt_file, 1000, 10)


class TestInterpolateData(unittest.TestCase):
    """Test data interpolation functionality."""

    def test_interpolate_data_basic(self):
        """Test basic data interpolation."""
        time_voltage = [(0.0, 0.0), (0.1, 1.0), (0.2, 0.0)]
        data = [(time_voltage, 'point', 1.5)]

        interval = 0.05
        period = 0.2

        interpolated = interpolate_data(data, interval, period)

        self.assertIsInstance(interpolated, list)
        self.assertEqual(len(interpolated), 1)

        new_time_voltage, load_type, deformation = interpolated[0]
        self.assertEqual(load_type, 'point')
        self.assertEqual(deformation, 1.5)
        self.assertIsInstance(new_time_voltage, list)

    def test_interpolate_data_multiple_entries(self):
        """Test interpolation with multiple data entries."""
        data = [
            ([(0.0, 0.0), (0.2, 1.0)], 'point', 1.0),
            ([(0.0, 0.5), (0.2, 0.5)], 'uniform', 2.0)
        ]

        interpolated = interpolate_data(data, 0.1, 0.2)

        self.assertEqual(len(interpolated), 2)
        for new_time_voltage, load_type, deformation in interpolated:
            self.assertIsInstance(new_time_voltage, list)
            self.assertIn(load_type, ['point', 'uniform'])
            self.assertIn(deformation, [1.0, 2.0])

    def test_interpolate_data_preserves_metadata(self):
        """Test that interpolation preserves load type and deformation."""
        data = [([(0.0, 0.0), (1.0, 1.0)], 'uniform', 3.5)]

        interpolated = interpolate_data(data, 0.25, 1.0)

        _, load_type, deformation = interpolated[0]
        self.assertEqual(load_type, 'uniform')
        self.assertEqual(deformation, 3.5)


class TestExtractDataAndType(unittest.TestCase):
    """Test data extraction functionality."""

    def test_extract_data_and_type_basic(self):
        """Test extracting voltage data and load types."""
        data_list = [
            ([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)], 'point', 1.0),
            ([(0.0, 0.5), (0.1, 0.6)], 'uniform', 2.0)
        ]

        extracted = extract_data_and_type(data_list)

        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0][1], 'point')
        self.assertEqual(extracted[1][1], 'uniform')
        self.assertEqual(extracted[0][0], [0.1, 0.2, 0.3])
        self.assertEqual(extracted[1][0], [0.5, 0.6])

    def test_extract_data_and_type_ignores_deformation(self):
        """Test that deformation values are ignored during extraction."""
        data_list = [
            ([(0.0, 1.0), (0.1, 2.0)], 'point', 99.9)
        ]

        extracted = extract_data_and_type(data_list)

        self.assertEqual(len(extracted), 1)
        voltage_data, load_type = extracted[0]
        self.assertEqual(load_type, 'point')
        self.assertEqual(voltage_data, [1.0, 2.0])


class TestNormalizeData(unittest.TestCase):
    """Test data normalization functionality."""

    def test_normalize_data_basic(self):
        """Test normalizing data array."""
        data = np.array([1.0, -2.0, 3.0, -1.5])

        normalized = normalize_data(data)

        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=10)
        expected = data / 3.0
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_normalize_data_all_positive(self):
        """Test normalizing all positive values."""
        data = np.array([1.0, 2.0, 3.0, 4.0])

        normalized = normalize_data(data)

        self.assertAlmostEqual(np.max(normalized), 1.0, places=10)
        self.assertEqual(normalized[-1], 1.0)  # Max value should be 1.0

    def test_normalize_data_all_negative(self):
        """Test normalizing all negative values."""
        data = np.array([-1.0, -2.0, -3.0, -4.0])

        normalized = normalize_data(data)

        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=10)
        self.assertEqual(normalized[3], -1.0)  # Max absolute value should be 1.0

    def test_normalize_data_zero_values(self):
        """Test normalizing data with zero values."""
        data = np.array([0.0, 1.0, 0.0, -1.0])

        normalized = normalize_data(data)

        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[2], 0.0)


class TestConvertToWave(unittest.TestCase):
    """Test WAV conversion functionality."""

    def test_convert_to_wave_basic(self):
        """Test converting normalized data to WAV format."""
        normalized_data = np.array([0.5, -0.5, 0.3, -0.3, 0.0])
        time_interval = 0.001  # 1000 Hz

        wav_tensor = convert_to_wave(normalized_data, time_interval)

        self.assertIsInstance(wav_tensor, tf.Tensor)
        self.assertEqual(wav_tensor.dtype, tf.string)

    def test_convert_to_wave_different_sample_rates(self):
        """Test conversion with different sample rates."""
        normalized_data = np.array([0.1, 0.2, 0.3])

        for sample_rate in [8000, 16000, 44100]:
            time_interval = 1.0 / sample_rate
            wav_tensor = convert_to_wave(normalized_data, time_interval)
            self.assertIsInstance(wav_tensor, tf.Tensor)

    def test_convert_to_wave_data_range(self):
        """Test conversion with data in valid range [-1, 1]."""
        normalized_data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        time_interval = 0.0001

        wav_tensor = convert_to_wave(normalized_data, time_interval)

        self.assertIsInstance(wav_tensor, tf.Tensor)


class TestWavGenerator(unittest.TestCase):
    """Test WAV generation functionality."""

    def test_wav_generator_basic(self):
        """Test generating WAV files from data."""
        data = [
            ([(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)], 'point', 1.0),
            ([(0.0, 0.5), (0.1, 0.6), (0.2, 0.7)], 'uniform', 2.0)
        ]

        sampling_rate = 1000
        wav_files = wav_generator(data, sampling_rate)

        self.assertIsInstance(wav_files, list)
        self.assertEqual(len(wav_files), 2)

        for wav_bytes, load_type in wav_files:
            self.assertIsInstance(wav_bytes, tf.Tensor)
            self.assertIn(load_type, ['point', 'uniform'])

    def test_wav_generator_single_entry(self):
        """Test WAV generation with single data entry."""
        data = [
            ([(0.0, 1.0), (0.5, 2.0), (1.0, 1.5)], 'point', 1.5)
        ]

        wav_files = wav_generator(data, 2000)

        self.assertEqual(len(wav_files), 1)
        wav_bytes, load_type = wav_files[0]
        self.assertEqual(load_type, 'point')

    def test_wav_generator_preserves_load_types(self):
        """Test that WAV generator preserves load type information."""
        data = [
            ([(0.0, 0.1)], 'point', 1.0),
            ([(0.0, 0.2)], 'uniform', 2.0),
            ([(0.0, 0.3)], 'point', 3.0)
        ]

        wav_files = wav_generator(data, 1000)

        load_types = [lt for _, lt in wav_files]
        self.assertEqual(load_types, ['point', 'uniform', 'point'])


class TestWaveAnalysis(unittest.TestCase):
    """Test WaveAnalysis class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock models
        self.mock_classification_model = MagicMock()
        self.mock_regression_model = MagicMock()

        self.wave_analysis = WaveAnalysis(
            self.mock_classification_model,
            self.mock_regression_model
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test WaveAnalysis initialization."""
        self.assertEqual(self.wave_analysis.classification_model, self.mock_classification_model)
        self.assertEqual(self.wave_analysis.regression_model, self.mock_regression_model)
        self.assertEqual(self.wave_analysis.sampling_rate, 1600)
        self.assertEqual(self.wave_analysis.time_period, 10)

    def test_class_test_csv_file(self):
        """Test CSV file reading for classification."""
        test_csv_path = os.path.join(self.temp_dir, 'test.csv')
        test_data = [
            ['header1', 'header2'],
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.3]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(test_csv_path, header=False, index=False)

        data = self.wave_analysis.class_test_csv_file(test_csv_path)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_generate_wav_files(self):
        """Test WAV file generation."""
        test_data = [
            ([(0.0, 0.1), (0.1, 0.2)], 'point', 1.0)
        ]

        wav_files = self.wave_analysis.generate_wav_files(test_data)

        self.assertIsInstance(wav_files, list)
        self.assertEqual(len(wav_files), 1)

    def test_save_wav_files(self):
        """Test saving WAV files."""
        test_data = np.array([0.5, -0.5, 0.3, -0.3])
        wav_tensor = convert_to_wave(test_data, 0.001)
        wav_file_list = [(wav_tensor, 'point')]

        # Mock the base folder path to use temp directory
        with patch.object(self.wave_analysis, 'save_wav_files') as mock_save:
            mock_save.return_value = os.path.join(self.temp_dir, 'test')

            result_path = self.wave_analysis.save_wav_files(wav_file_list)

            self.assertIsInstance(result_path, str)
            mock_save.assert_called_once()

    def test_voltage_test_files(self):
        """Test reading voltage data from CSV file."""
        test_csv_path = os.path.join(self.temp_dir, 'voltage_test.csv')
        test_data = [
            ['header1', 'header2'],
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.15]
        ]
        df = pd.DataFrame(test_data)
        df.to_csv(test_csv_path, header=False, index=False)

        time_voltage_pairs = self.wave_analysis.voltage_test_files(test_csv_path)

        self.assertIsInstance(time_voltage_pairs, list)
        self.assertEqual(len(time_voltage_pairs), 3)
        self.assertEqual(time_voltage_pairs[0], (0.0, 0.1))

    def test_calculate_voltage_peak_to_peak(self):
        """Test peak-to-peak voltage calculation."""
        # Create time-voltage pairs with clear peaks
        time_voltage_pairs = [
            (0.0, 0.0),
            (0.1, 1.0),   # peak
            (0.2, 0.0),
            (0.3, -1.0),  # valley
            (0.4, 0.0),
            (0.5, 1.0),   # peak
            (0.6, 0.0),
            (0.7, -1.0),  # valley
            (0.8, 0.0)
        ]

        peak_values = self.wave_analysis.calculate_voltage_peak_to_peak(time_voltage_pairs)

        self.assertIsInstance(peak_values, np.ndarray)
        self.assertGreater(len(peak_values), 0)

    def test_calculate_voltage_peak_to_peak_no_peaks(self):
        """Test peak calculation with flat signal."""
        time_voltage_pairs = [(i * 0.1, 0.5) for i in range(10)]

        peak_values = self.wave_analysis.calculate_voltage_peak_to_peak(time_voltage_pairs)

        self.assertIsInstance(peak_values, np.ndarray)

    @patch('test_model.WaveAnalysis.classify_directory')
    @patch('test_model.WaveAnalysis.save_wav_files')
    @patch('test_model.WaveAnalysis.generate_wav_files')
    @patch('test_model.WaveAnalysis.class_test_csv_file')
    @patch('test_model.WaveAnalysis.calculate_voltage_peak_to_peak')
    @patch('test_model.WaveAnalysis.voltage_test_files')
    def test_process_file_and_predict_deformation(
        self,
        mock_voltage_test,
        mock_calculate_peak,
        mock_class_test,
        mock_generate_wav,
        mock_save_wav,
        mock_classify
    ):
        """Test complete processing and prediction pipeline."""
        # Set up mocks
        test_csv_path = os.path.join(self.temp_dir, 'test.csv')

        mock_voltage_test.return_value = [(0.0, 0.1), (0.1, 0.2)]
        mock_calculate_peak.return_value = np.array([0.5, 0.6, 0.7])
        mock_class_test.return_value = [([], 'point', 1.0)]
        mock_generate_wav.return_value = [(MagicMock(), 'point')]
        mock_save_wav.return_value = '/test/directory'
        mock_classify.return_value = (0, 'point')

        self.mock_regression_model.predict.return_value = np.array([[1.5], [1.6], [1.4]])

        deformation, class_name = self.wave_analysis.process_file_and_predict_deformation(test_csv_path)

        self.assertIsInstance(deformation, (float, np.floating))
        self.assertEqual(class_name, 'point')
        self.mock_regression_model.predict.assert_called_once()


class TestDisplayModelResults(unittest.TestCase):
    """Test display_model_results functionality."""

    @patch('test_model.Console')
    @patch('test_model.display')
    def test_display_model_results_basic(self, mock_display, mock_console):
        """Test displaying model results."""
        deformation = 1.234
        class_name = 'point'

        # This should not raise any exceptions
        display_model_results(deformation, class_name)

        # Verify display was called (for HTML CSS)
        self.assertTrue(mock_display.called)

    @patch('test_model.Console')
    @patch('test_model.display')
    def test_display_model_results_custom_font_size(self, mock_display, mock_console):
        """Test displaying results with custom font size."""
        deformation = 2.567
        class_name = 'uniform'
        font_size = 20

        display_model_results(deformation, class_name, font_size=font_size)

        self.assertTrue(mock_display.called)

    @patch('test_model.Console')
    @patch('test_model.display')
    def test_display_model_results_formats_deformation(self, mock_display, mock_console):
        """Test that deformation value formatting works."""
        deformation = 1.23456789
        class_name = 'point'

        # Should format to 3 decimal places (1.235 mm)
        display_model_results(deformation, class_name)

        self.assertTrue(mock_display.called)


if __name__ == '__main__':
    unittest.main()
