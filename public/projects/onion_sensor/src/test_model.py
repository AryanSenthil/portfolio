import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def process_csv_file(file_path):
    # Function to process a single csv file
    df = pd.read_csv(file_path, header=None, encoding='utf-8')

    deformation = None
    type_of_load= None


    time_voltage_data = df.iloc[1:].astype(float).values
    time_voltage_pairs = [tuple(row) for row in time_voltage_data]

    return (time_voltage_pairs, type_of_load, deformation)

def read_csv_files(path, sampling_rate, period):
    data_for_model = []
    all_data = [] # ((time,voltage), type, deformation) for each file

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.csv'):
                file_path = os.path.join(path, filename)
                all_data.append(process_csv_file(file_path))
    elif os.path.isfile(path) and path.endswith('.csv'):
        all_data.append(process_csv_file(path))
    else:
        raise ValueError(f"The path provided is neither a CSV file nor a directory: {path}")

    interval = 1/sampling_rate
    data_for_model = interpolate_data(all_data, interval, period)
    return data_for_model

def interpolate_data(data, interval, period):
    interpolated_data = []

    for time_voltage, load_type, deformation in data:
        # Extract the time and voltage values
        time, voltage = zip(*time_voltage)

        interpolation_func = interp1d(time, voltage, kind='linear', fill_value='extrapolate')
        max_time = interval * (round(period/interval))
        new_time = np.arange(0, max_time, interval)
        new_voltage = interpolation_func(new_time)

        new_time_voltage = list(zip(new_time, new_voltage))
        interpolated_data.append((new_time_voltage, load_type, deformation))
    return interpolated_data
#%%

import os
import pandas as pd
import numpy as np
import tensorflow as tf

def extract_data_and_type(data_list):
    new_data = []
    for time_voltage, load_type, _ in data_list:
        voltage_data = [v for _,v in time_voltage]
        new_data.append((voltage_data, load_type))
    return new_data

# Normalize the Data
def normalize_data(data):
    max_val = np.max((np.abs(data)))
    return data / max_val

# Convert to wav files
def convert_to_wave(normalized_data, time_interval):
    sample_rate = tf.cast(int(1/time_interval), tf.int32)

    audio_tensor = tf.convert_to_tensor(normalized_data, dtype=tf.float32)
    audio_tensor = tf.reshape(audio_tensor, (-1,1)) # Ensure correct shape for audio encoding
    return tf.audio.encode_wav(audio_tensor, sample_rate)

# I took this from
# tf.audio.encode_wav(
#     audio: _atypes.TensorFuzzingAnnotation[_atypes.Float32],
# sample_rate: _atypes.TensorFuzzingAnnotation[_atypes.Int32],
# name=None
# ) -> _atypes.TensorFuzzingAnnotation[_atypes.String]

def wav_generator(data, sampling_rate):
    classification_data = extract_data_and_type(data)
    # Normalize the data
    normalized_data = [(normalize_data(voltage_sequence), load_type) for voltage_sequence, load_type in classification_data]
    # Generate WAV files
    time_interval = 1 / sampling_rate
    wav_files = [(convert_to_wave(voltage_sequence, time_interval), load_type) for voltage_sequence, load_type in normalized_data]
    return wav_files

#%%

#%%

class WaveAnalysis:
    def __init__(self, classification_model, regression_model):
        self.classification_model = classification_model
        self.regression_model = regression_model
        self.sampling_rate = 1600
        self.time_period = 10

    def class_test_csv_file(self, file_path):
        data = read_csv_files(file_path, self.sampling_rate, self.time_period)
        return data

    def generate_wav_files(self, test_data):
        test_wav_files = wav_generator(test_data, self.sampling_rate)
        return test_wav_files

    def save_wav_files(self, wav_file_list):
        base_folder_path = r"/home/ari/Documents/Onion_Sensor/test_data/test"
        version = 1
        original_path = base_folder_path
        while os.path.exists(base_folder_path):
            base_folder_path = f"{original_path}_v{version}"
            version += 1
        os.makedirs(base_folder_path, exist_ok=True)
        for index, audio_tensor in enumerate(wav_file_list, start=1):
            wav_filename = f"audio_{index}.wav"
            wav_filepath = os.path.join(base_folder_path, wav_filename)
            tf.io.write_file(wav_filepath, audio_tensor[0])
        return base_folder_path

    def classify_directory(self, directory_path):
        wav_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
        classification_results = {}
        for wav_file in wav_files:
            file_path = os.path.join(directory_path, wav_file)
            prediction = self.classification_model(tf.constant(str(file_path)))
            class_name = prediction['class_names'].numpy()[0].decode('utf-8')
            class_num = 0 if class_name == 'point' else 1
            classification_results = class_num
        return classification_results, class_name

    def voltage_test_files(self, file_path):
        df = pd.read_csv(file_path, header=None, encoding='utf-8')
        time_voltage_data = df.iloc[1:].astype(float).values
        return [tuple(row) for row in time_voltage_data]

    def calculate_voltage_peak_to_peak(self, time_voltage_pairs):
        voltage_data = [v for _, v in time_voltage_pairs]
        prominence = 0
        low_peaks, _ = find_peaks(-np.array(voltage_data), prominence=prominence)
        high_peaks, _ = find_peaks(np.array(voltage_data), prominence=prominence)
        min_length = min(len(low_peaks), len(high_peaks))
        low_y = np.array(voltage_data)[low_peaks][:min_length]
        high_y = np.array(voltage_data)[high_peaks][:min_length]
        peak_to_peak_values = abs(low_y - high_y)
        return peak_to_peak_values

    def process_file_and_predict_deformation(self, file_path):
        # regression 
        time_voltage_pairs = self.voltage_test_files(file_path)
        peak_values = self.calculate_voltage_peak_to_peak(time_voltage_pairs)
        # classification 
        data_class = self.class_test_csv_file(file_path)
        test_wav_files = self.generate_wav_files(data_class)
        test_directory = self.save_wav_files(test_wav_files)
        classification_results, class_name = self.classify_directory(test_directory)
        # Models
        combined_test_data = np.array([(peak, classification_results) for i, peak in enumerate(peak_values)])
        y_predict = self.regression_model.predict([combined_test_data[:, 1], combined_test_data[:, 0]])
        deformation_value = np.average(y_predict)
        return deformation_value, class_name
# %%
# save this as model_output.py

from IPython.display import display, HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align

def display_model_results(deformation, class_name, font_size=16):
    """
    Display model prediction results in a styled, centered table in Jupyter.
    
    Parameters:
    -----------
    deformation : float
        Predicted deformation value in mm
    class_name : str
        Predicted type of load
    font_size : int, optional
        Font size for the output (default: 16)
    """
    # Increase Jupyter cell output font size with CSS
    display(HTML(f"""
    <style>
    .jp-OutputArea-output pre {{
        font-size: {font_size}px !important;
        line-height: 1.8 !important;
    }}
    </style>
    """))

    # For Jupyter, set a specific width
    console = Console(width=100, force_terminal=True, force_jupyter=True)

    # Create a table for results with more padding
    table = Table(show_header=True, header_style="bold magenta", padding=(2, 4), border_style="blue", expand=True)
    table.add_column("Metric", style="cyan bold", width=40, justify="left")
    table.add_column("Value", style="green bold", width=40, justify="center")
    table.add_row("Predicted deformation", f"{deformation:.3f} mm")
    table.add_row("Type of load", class_name)

    # Create panel with more padding
    panel = Panel(
        table,
        title="[bold yellow]MODEL PREDICTION RESULTS[/bold yellow]",
        border_style="blue",
        padding=(2, 3),
        expand=False
    )

    console.print(Align.center(panel))
    console.print("\n" * 2)