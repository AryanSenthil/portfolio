"""
tools.py — utilities for CSV → interpolated signals → WAV, plus TF helpers.

"""

# ===========
# Standard lib
# ===========
import os
import time
from pathlib import Path
from typing import Optional, Tuple

# ===========
# Third-party
# ===========
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


# ======================
# Reproducibility helpers
# ======================
def set_seed(seed: int = 42) -> None:
    """
    Set seeds for TensorFlow and NumPy to improve reproducibility.

    Note: This does not guarantee full determinism across all hardware/ops.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


# ==================
# Filesystem / CSV IO
# ==================
def load_data(data_path, sampling_rate):
    """Load CSV data from the specified path (file or directory)."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"{path.resolve()} not found")

    data = read_csv_files(str(path), sampling_rate)
    return data


def process_csv_file(file_path):
    """
    Process a single CSV file containing deformation and time–voltage data.

    Expected format:
      - Row 1: deformation (float) in col 0; load type ('Point'|'Uniform') in col 1
      - Rows 2+: time, voltage pairs (floats)

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: (time_voltage_pairs, type_of_load, deformation) where
            - time_voltage_pairs: list[(time, voltage)]
            - type_of_load: 'point' or 'uniform'
            - deformation: float
    """
    df = pd.read_csv(file_path, header=None, encoding="utf-8")

    deformation = float(df.iloc[0, 0])
    type_of_load_raw = df.iloc[0, 1]
    if type_of_load_raw == "Point":
        type_of_load = "point"
    elif type_of_load_raw == "Uniform":
        type_of_load = "uniform"
    else:
        raise ValueError(f"Unrecognized type {type_of_load_raw} in file {file_path}")

    time_voltage_data = df.iloc[1:].astype(float).values
    time_voltage_pairs = [tuple(row) for row in time_voltage_data]

    return (time_voltage_pairs, type_of_load, deformation)


def read_csv_files(path, sampling_rate):
    """
    Read CSV(s), process/interpolate, and return data ready for modeling.

    Args:
        path (str): Path to a CSV file or directory containing CSV files.
        sampling_rate (int): Sampling rate for interpolation step size.

    Returns:
        list: Interpolated data in the form
              [(new_time_voltage, load_type, deformation), ...]
    """
    data_for_model = []
    all_data = []  # ((time, voltage), type, deformation) for each file

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                file_path = os.path.join(path, filename)
                all_data.append(process_csv_file(file_path))
    elif os.path.isfile(path) and path.endswith(".csv"):
        all_data.append(process_csv_file(path))
    else:
        raise ValueError(
            f"The path provided is neither a CSV file nor a directory: {path}"
        )

    period = 10
    interval = 1 / sampling_rate
    data_for_model = interpolate_data(all_data, interval, period)
    return data_for_model


# =======================
# Signal processing helpers
# =======================
def interpolate_data(data, interval, period):
    """
    Interpolate time–voltage data at regular intervals using linear interpolation.

    Args:
        data (list): [(time_voltage, load_type, deformation), ...] where
            time_voltage is list[(time, voltage)].
        interval (float): Time interval between interpolated points.
        period (float): Total time period for interpolation.

    Returns:
        list: [(new_time_voltage, load_type, deformation), ...] with regular spacing.
    """
    interpolated_data = []

    for time_voltage, load_type, deformation in data:
        # Extract the time and voltage values
        time_vals, voltage_vals = zip(*time_voltage)

        interpolation_func = interp1d(
            time_vals, voltage_vals, kind="linear", fill_value="extrapolate"
        )
        max_time = interval * (round(period / interval))
        new_time = np.arange(0, max_time, interval)
        new_voltage = interpolation_func(new_time)

        new_time_voltage = list(zip(new_time, new_voltage))
        interpolated_data.append((new_time_voltage, load_type, deformation))
    return interpolated_data


def extract_data_and_type(data_list):
    """
    Extract voltage sequences and load type from the data list.

    Args:
        data_list (list): [(time_voltage, load_type, _), ...]

    Returns:
        list: [(voltage_data, load_type), ...]
    """
    new_data = []
    for time_voltage, load_type, _ in data_list:
        voltage_data = [v for _, v in time_voltage]
        new_data.append((voltage_data, load_type))
    return new_data


def normalize_data(data):
    """
    Normalize input array by maximum absolute value (range ~ [-1, 1]).

    Args:
        data (np.ndarray): Array to normalize.

    Returns:
        np.ndarray: Normalized array.
    """
    max_val = np.max((np.abs(data)))
    return data / max_val


def convert_to_wave(normalized_data, time_interval):
    """
    Convert normalized data into a WAV-encoded audio tensor.

    Args:
        normalized_data: Sequence of floats in [-1, 1].
        time_interval: Sampling interval; 1 / interval = sample rate (Hz).

    Returns:
        tf.Tensor: WAV-encoded bytes tensor.
    """
    sample_rate = tf.cast(int(1 / time_interval), tf.int32)

    audio_tensor = tf.convert_to_tensor(normalized_data, dtype=tf.float32)
    audio_tensor = tf.reshape(audio_tensor, (-1, 1))  # [samples, channels]
    return tf.audio.encode_wav(audio_tensor, sample_rate)


def wav_generator(data, sampling_rate):
    """
    Generate WAV files from voltage sequence data.

    Steps:
      - extract voltage sequences + load types
      - normalize
      - encode to WAV bytes using sampling_rate

    Args:
        data: Input data containing voltage sequences and associated load types.
        sampling_rate (float): WAV sample rate (Hz).

    Returns:
        list[(tf.Tensor, str)]: (wav_bytes, load_type) tuples.
    """
    classification_data = extract_data_and_type(data)
    normalized_data = [
        (normalize_data(voltage_sequence), load_type)
        for voltage_sequence, load_type in classification_data
    ]
    time_interval = 1 / sampling_rate
    wav_files = [
        (convert_to_wave(voltage_sequence, time_interval), load_type)
        for voltage_sequence, load_type in normalized_data
    ]
    return wav_files


def save_wav_files(wav_files, base_folder_path):
    """
    Save WAV files to a structured directory per class.

    Directory structure:
        base/
          class_a/
            class_a_audio_1.wav
            ...
          class_b/
            class_b_audio_1.wav
            ...

    Args:
        wav_files (list[tuple]): [(audio_binary, load_type), ...]
        base_folder_path (str): Base directory path. If it exists, create a
            new versioned directory.

    Returns:
        str: The actual path where the WAV files were saved.

    Raises:
        OSError: On directory or file write failures.
    """
    # Create a new version if it already exists
    version = 1
    while os.path.exists(base_folder_path):
        base_folder_path = base_folder_path.rstrip("/") + f"_v{version}"
        version += 1

    # Create class subdirectories and save WAV files
    for audio_binary, load_type in wav_files:
        class_directory = os.path.join(base_folder_path, load_type)

        os.makedirs(class_directory, exist_ok=True)

        existing_files = os.listdir(class_directory)
        existing_indices = [
            int(f.split("_")[-1].split(".")[0])
            for f in existing_files
            if f.endswith(".wav")
        ]
        file_index = max(existing_indices, default=0) + 1

        wav_filename = f"{load_type}_audio_{file_index}.wav"
        wav_filepath = os.path.join(class_directory, wav_filename)

        tf.io.write_file(wav_filepath, audio_binary)
    return base_folder_path


# =========================
# Audio dataset / TF helpers
# =========================
def get_audio_length(directory: str) -> Tuple[Optional[int], Optional[tf.Tensor]]:
    """
    Return the number of samples and sample rate of the first .wav found.

    Searches one level deep (immediate subdirectories).
    Returns (None, None) if no .wav file is found.

    Args:
        directory: Root directory containing class subdirectories with .wav files.

    Returns:
        (num_samples, sample_rate): sample_rate is a scalar tf.Tensor.
    """
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(subdir_path, filename)
                    audio_binary = tf.io.read_file(file_path)
                    waveform, sample_rate = tf.audio.decode_wav(audio_binary)
                    return waveform.shape[0], sample_rate
    return None, None


def count_audio_files(directory: str) -> int:
    """
    Count .wav files across immediate subdirectories.

    Args:
        directory: Root directory containing class subdirectories.

    Returns:
        Total number of .wav files found.
    """
    total_files = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            audio_files = [f for f in os.listdir(subdir_path) if f.endswith(".wav")]
            total_files += len(audio_files)
    return total_files


def get_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    """
    Convert a 1D waveform tensor to a magnitude spectrogram.

    Args:
        waveform: Tensor of shape [samples] or [samples, channels].

    Returns:
        Spectrogram tensor of shape [..., freq_bins, time_frames, 1].
    """
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram[..., tf.newaxis]


def preprocess_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Map a dataset of (audio, label) to (spectrogram, label).

    Args:
        dataset: tf.data.Dataset yielding (waveform, label).

    Returns:
        tf.data.Dataset yielding (spectrogram, label) with AUTOTUNE mapping.
    """
    return dataset.map(
        lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def squeeze(audio, labels):
    """Remove last axis from audio: (batch, time, 1) → (batch, time)."""
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def make_spec_ds(ds):
    """Map a dataset to spectrogram space using get_spectrogram."""
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ======================
# Analysis / feature class
# ======================
class DataProcessor:
    """
    Process time–voltage data and compute peak-to-peak voltage values.

    Usage:
        processor = DataProcessor(raw_data)
        processor.time_voltage_data()
        processor.convert_to_array()
        processor.calculate_voltage_peak_to_peak()
    """

    def __init__(self, data):
        self.data = data
        self.time = []
        self.voltage = []
        self.voltage_peak_values = []
        self.load_type = []
        self.deformation = []
        self.combined_data = ()

    def time_voltage_data(self):
        for time_voltage, load, deformation in self.data:
            time_vals = [t for t, _ in time_voltage]
            voltage_vals = [v for _, v in time_voltage]
            self.time.append(time_vals)
            self.voltage.append(voltage_vals)
            self.load_type.append(load)
            self.deformation.append(deformation)

    def convert_to_array(self):
        self.time = [np.array(t) for t in self.time]
        self.voltage = [np.array(v) for v in self.voltage]

    def calculate_voltage_peak_to_peak(self):
        combined_data_list = []
        for voltage, load, deformation in zip(
            self.voltage, self.load_type, self.deformation
        ):
            if load == "point":
                load_value = 0.0
            elif load == "uniform":
                load_value = 1.0
            prominence = 0
            low_peaks, _ = find_peaks(-voltage, prominence=prominence)
            high_peaks, _ = find_peaks(voltage, prominence=prominence)
            min_length = min(len(low_peaks), len(high_peaks))

            low_y = voltage[low_peaks][:min_length]
            high_y = voltage[high_peaks][:min_length]

            peak_to_peak_values = abs(low_y - high_y)

            combined_data_list.extend(
                [(peak, load_value, deformation) for peak in peak_to_peak_values]
            )

        # Convert the list to a numpy array after the loop
        self.combined_data = np.array(combined_data_list)


# ==================
# Training monitoring
# ==================
class LossOverTimeHistory(tf.keras.callbacks.Callback):
    """Callback to track loss and metrics over actual training time."""

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.times = []  # Cumulative time at each epoch end
        self.losses = []
        self.val_losses = []
        self.metrics = {}  # Store any additional metrics

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Calculate cumulative time since training started
        elapsed_time = time.time() - self.train_start_time
        self.times.append(elapsed_time)

        # Track losses
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))

        # Track any additional metrics
        for key, value in logs.items():
            if key not in ["loss", "val_loss"]:
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)

    def get_results(self):
        """Return results as a dictionary."""
        results = {
            "time": self.times,
            "loss": self.losses,
            "val_loss": self.val_losses,
        }
        results.update(self.metrics)
        return results
