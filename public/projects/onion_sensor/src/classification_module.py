"""
classification_module.py - Audio classification pipeline optimized for agent orchestration
All functions are side-effect based (save to disk, no return values)
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers, models

from tools import (
    count_audio_files, extract_data_and_type, get_audio_length,
    get_spectrogram, preprocess_dataset, read_csv_files,
    save_wav_files, set_seed, wav_generator
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def preprocess_csv_to_wav(csv_dir: str, model_name: str, sampling_rate: int = 16000) -> None:
    """Convert CSV to WAV files
    
    Args:
        csv_dir: Directory containing CSV files
        model_name: Name of the model
        sampling_rate: Audio sampling rate (default: 16000)
    
    Outputs:
        wave_files/: Directory containing converted WAV files
        preprocessing_info.json: Metadata about preprocessing
    
    Requires: []
    """
    start_time = time.time()
    csv_dir = Path(csv_dir)
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    wave_output_dir = model_dir / 'wave_files' / 'wave_files'
    raw_data_dir = model_dir / 'data' / 'raw'
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    
    import shutil
    csv_files = list(csv_dir.glob('*.csv'))
    for csv_file in csv_files:
        shutil.copy2(csv_file, raw_data_dir / csv_file.name)
    
    data = read_csv_files(str(csv_dir), sampling_rate)
    wav_files = wav_generator(data, sampling_rate)
    actual_wave_dir = Path(save_wav_files(wav_files, str(wave_output_dir)))
    
    class_counts = {}
    total_wav_files = 0
    for class_dir in actual_wave_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.wav')))
            class_counts[class_dir.name] = count
            total_wav_files += count
    
    dir_size_mb = sum(f.stat().st_size for f in actual_wave_dir.rglob('*.wav')) / (1024 * 1024)
    
    metadata_dir = model_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    preprocess_info = {
        'wave_files_path': str(actual_wave_dir),
        'wave_files_path_absolute': str(actual_wave_dir.absolute()),
        'num_wav_files': total_wav_files,
        'class_distribution': class_counts
    }
    
    with open(metadata_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocess_info, f, indent=2)
    
    logger.info(f"✓ WAV files: {actual_wave_dir.absolute()}")
    logger.info(f"✓ Preprocessing complete: {time.time() - start_time:.1f}s")


def load_and_preprocess_data(model_name: str, validation_split: float = 0.2, seed: int = 42) -> None:
    """Create train/val/test datasets from WAV files
    
    Args:
        model_name: Name of the model
        validation_split: Fraction of data for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Outputs:
        datasets/: Directory containing train/val/test datasets
        dataset_info.json: Dataset metadata and configuration
        label_names.npy: Array of class labels
    
    Requires: [preprocessing_info.json]
    """
    start_time = time.time()
    set_seed(seed)
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    metadata_dir = model_dir / 'metadata'
    
    with open(metadata_dir / 'preprocessing_info.json', 'r') as f:
        preprocess_info = json.load(f)
    
    data_dir = Path(preprocess_info['wave_files_path'])
    dataset_dir = model_dir / 'datasets'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Wave files directory {data_dir} does not exist")
    
    audio_length, sample_rate = get_audio_length(str(data_dir))
    total_files = count_audio_files(str(data_dir))
    
    if audio_length is None or sample_rate is None or total_files == 0:
        raise FileNotFoundError(f"No audio files found in {data_dir}")
    
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        str(data_dir),
        batch_size=1,
        validation_split=validation_split,
        seed=seed,
        output_sequence_length=audio_length,
        subset='both'
    )
    
    label_names = np.array(train_ds.class_names)
    
    train_ds = preprocess_dataset(
        train_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    )
    val_ds = preprocess_dataset(
        val_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    )
    
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)
    
    input_shape = next(iter(train_ds))[0].shape[1:]
    
    train_path = dataset_dir / 'train'
    val_path = dataset_dir / 'val'
    test_path = dataset_dir / 'test'
    
    tf.data.Dataset.save(train_ds, str(train_path))
    tf.data.Dataset.save(val_ds, str(val_path))
    tf.data.Dataset.save(test_ds, str(test_path))
    
    label_names_path = metadata_dir / 'label_names.npy'
    np.save(label_names_path, label_names)
    
    metadata = {
        'input_shape': list(input_shape),
        'num_labels': len(label_names),
        'label_names': label_names.tolist(),
        'dataset_paths_absolute': {
            'train': str(train_path.absolute()),
            'val': str(val_path.absolute()),
            'test': str(test_path.absolute())
        }
    }
    
    dataset_info_path = metadata_dir / 'dataset_info.json'
    with open(dataset_info_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Dataset metadata: {dataset_info_path.absolute()}")
    logger.info(
        f"✓ Dataset details:\n"
        f" - Input shape: {list(input_shape)}\n"
        f" - Number of labels: {len(label_names)}\n"
        f" - Label names: {label_names.tolist()}"
    )
    logger.info(f"✓ Data preprocessing complete: {time.time() - start_time:.1f}s")


def build_and_compile_model(model_name: str, learning_rate: float = 0.001) -> None:
    """Build CNN architecture
    
    Args:
        model_name: Name of the model
        learning_rate: Learning rate for optimizer (default: 0.001)
    
    Outputs:
        models/untrained/model.keras: Compiled untrained model
    
    Requires: [dataset_info.json]
    """
    start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    metadata_dir = model_dir / 'metadata'
    untrained_model_dir = model_dir / 'models' / 'untrained'
    untrained_model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_dir / 'dataset_info.json', 'r') as f:
        metadata = json.load(f)
    
    input_shape = tuple(metadata['input_shape'])
    num_labels = metadata['num_labels']
    
    norm_layer = layers.Normalization()
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()
    
    model_path = untrained_model_dir / 'model.keras'
    model.save(model_path)
    
    logger.info(f"✓ Model built: {model_path.absolute()}")
    logger.info(f"✓ Model build complete: {time.time() - start_time:.1f}s")


def train_and_evaluate_model(model_name: str, epochs: int = 10, patience: int = 2) -> None:
    """Train model and generate plots
    
    Args:
        model_name: Name of the model
        epochs: Number of training epochs (default: 10)
        patience: Early stopping patience (default: 2)
    
    Outputs:
        models/trained/model.keras: Trained model with weights
        training/*.json: Training history and results
        training/plots/*.png: Training visualization plots
    
    Requires: [datasets/, models/untrained/model.keras]
    """
    overall_start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    dataset_dir = model_dir / 'datasets'
    models_dir = model_dir / 'models'
    training_dir = model_dir / 'training'
    plots_dir = training_dir / 'plots'
    metadata_dir = model_dir / 'metadata'
    
    training_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    label_names = np.load(metadata_dir / 'label_names.npy')
    
    train_ds = tf.data.Dataset.load(str(dataset_dir / 'train'))
    val_ds = tf.data.Dataset.load(str(dataset_dir / 'val'))
    test_ds = tf.data.Dataset.load(str(dataset_dir / 'test'))
    
    model = tf.keras.models.load_model(models_dir / 'untrained' / 'model.keras')
    
    train_ds_optimized = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds_optimized = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.times = []
            self.epoch_times = []
            self.train_start = time.time()
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.time()
            
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start
            cumulative_time = time.time() - self.train_start
            self.epoch_times.append(epoch_time)
            self.times.append(cumulative_time)
    
    time_callback = TimeHistory()
    early_stopping = tf.keras.callbacks.EarlyStopping(verbose=0, patience=patience)
    
    history = model.fit(
        train_ds_optimized,
        validation_data=val_ds_optimized,
        epochs=epochs,
        callbacks=[early_stopping, time_callback],
        verbose=0
    )
    
    test_results = model.evaluate(test_ds, return_dict=True, verbose=0)
    
    y_pred = model.predict(test_ds, verbose=0)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_ds.map(lambda s, lab: lab)), axis=0)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    
    trained_model_dir = models_dir / 'trained'
    trained_model_dir.mkdir(parents=True, exist_ok=True)
    trained_model_path = trained_model_dir / 'model.keras'
    model.save(trained_model_path)
    
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    history_dict['epochs'] = [int(e) for e in history.epoch]
    history_dict['epoch_times_seconds'] = [float(t) for t in time_callback.epoch_times]
    history_dict['cumulative_times_seconds'] = [float(t) for t in time_callback.times]
    
    with open(training_dir / 'history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    test_results_serializable = {k: float(v) for k, v in test_results.items()}
    with open(training_dir / 'test_results.json', 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.epoch, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(history.epoch, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss vs Epoch')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    loss_epoch_path = plots_dir / 'loss_vs_epoch.png'
    plt.savefig(loss_epoch_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.epoch, 100 * np.array(history.history['accuracy']), 'b-', label='Training', linewidth=2)
    plt.plot(history.epoch, 100 * np.array(history.history['val_accuracy']), 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy [%]'); plt.title('Accuracy vs Epoch')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    accuracy_epoch_path = plots_dir / 'accuracy_vs_epoch.png'
    plt.savefig(accuracy_epoch_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    times_in_minutes = [t / 60 for t in time_callback.times]
    plt.figure(figsize=(10, 6))
    plt.plot(times_in_minutes, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(times_in_minutes, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Time (minutes)'); plt.ylabel('Loss'); plt.title('Loss vs Time')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    loss_time_path = plots_dir / 'loss_vs_time.png'
    plt.savefig(loss_time_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=label_names, yticklabels=label_names,
                annot=True, fmt='g', cmap='Blues', linewidths=0.5)
    plt.xlabel('Prediction'); plt.ylabel('Label'); plt.title('Confusion Matrix')
    plt.tight_layout()
    confusion_matrix_path = plots_dir / 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    plots_metadata = {
        'plots_directory_absolute': str(plots_dir.absolute()),
        'plots': {
            'loss_vs_epoch': {'absolute_path': str(loss_epoch_path.absolute())},
            'accuracy_vs_epoch': {'absolute_path': str(accuracy_epoch_path.absolute())},
            'loss_vs_time': {'absolute_path': str(loss_time_path.absolute())},
            'confusion_matrix': {'absolute_path': str(confusion_matrix_path.absolute())}
        }
    }
    
    with open(training_dir / 'plots_metadata.json', 'w') as f:
        json.dump(plots_metadata, f, indent=2)
    
    training_summary = {
        'total_epochs': len(history.epoch),
        'test_loss': float(test_results['loss']),
        'test_accuracy': float(test_results['accuracy']),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    }
    
    with open(training_dir / 'training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    logger.info(f"✓ Trained model: {trained_model_path.absolute()}")
    logger.info(f"✓ Training plots: {plots_dir.absolute()}")
    logger.info(f"✓ Training summary: {(training_dir / 'training_summary.json').absolute()}")
    logger.info(f"✓ Test accuracy: {test_results['accuracy']*100:.1f}%")
    logger.info(f"✓ Training complete: {time.time() - overall_start_time:.1f}s")


def create_export_model(model_name: str) -> None:
    """Create production SavedModel
    
    Args:
        model_name: Name of the model
    
    Outputs:
        export/saved_model/: TensorFlow SavedModel directory
        models.json: Model registry with metadata
    
    Requires: [models/trained/model.keras, label_names.npy]
    """
    start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    trained_model_path = model_dir / 'models' / 'trained' / 'model.keras'
    metadata_dir = model_dir / 'metadata'
    export_dir = model_dir / 'export' / 'saved_model'
    
    model = tf.keras.models.load_model(trained_model_path)
    label_names = np.load(metadata_dir / 'label_names.npy')
    
    class ExportModel(tf.Module):
        def __init__(self, model, label_names):
            self.model = model
            self.label_names = label_names
            self.__call__.get_concrete_function(x=tf.TensorSpec(shape=(), dtype=tf.string))
            self.__call__.get_concrete_function(x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))
        
        @tf.function
        def __call__(self, x):
            if x.dtype == tf.string:
                x = tf.io.read_file(x)
                x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
                x = tf.squeeze(x, axis=-1)
                x = x[tf.newaxis, :]
            x = get_spectrogram(x)
            result = self.model(x, training=False)
            class_ids = tf.argmax(result, axis=-1)
            class_names = tf.gather(self.label_names, class_ids)
            return {'predictions': result, 'class_ids': class_ids, 'class_names': class_names}
    
    export_model = ExportModel(model, label_names)
    tf.saved_model.save(export_model, str(export_dir))
    
    export_metadata = {
        'export_path_absolute': str(export_dir.absolute()),
        'classes': label_names.tolist(),
        'num_classes': len(label_names)
    }
    
    with open(model_dir / 'export' / 'export_metadata.json', 'w') as f:
        json.dump(export_metadata, f, indent=2)
    
    _update_models_registry(model_name)
    
    logger.info(f"✓ Export model: {export_dir.absolute()}")
    logger.info(f"✓ Export complete: {time.time() - start_time:.1f}s")


def _update_models_registry(model_name: str) -> None:
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    project_root = Path(__file__).resolve().parent.parent
    registry_path = project_root / 'models.json'
    
    if registry_path.exists():
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except (json.JSONDecodeError, ValueError):
            registry = {'models': {}}
    else:
        registry = {'models': {}}
    
    model_info = {
        'model_directory': str(model_dir.absolute()),
        'status': 'exported',
        'absolute_paths': {
            'trained_model': str((model_dir / 'models' / 'trained' / 'model.keras').absolute()),
            'export_model': str((model_dir / 'export' / 'saved_model').absolute()),
            'training_plots': str((model_dir / 'training' / 'plots').absolute()),
            'plots_metadata': str((model_dir / 'training' / 'plots_metadata.json').absolute()),
            'dataset_info': str((model_dir / 'metadata' / 'dataset_info.json').absolute()),
            'training_summary': str((model_dir / 'training' / 'training_summary.json').absolute()),
            'export_metadata': str((model_dir / 'export' / 'export_metadata.json').absolute())
        }
    }
    
    dataset_info_path = model_dir / 'metadata' / 'dataset_info.json'
    if dataset_info_path.exists():
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        model_info['classes'] = dataset_info['label_names']
        model_info['num_classes'] = dataset_info['num_labels']
    
    training_summary_path = model_dir / 'training' / 'training_summary.json'
    if training_summary_path.exists():
        with open(training_summary_path, 'r') as f:
            training_summary = json.load(f)
        model_info['performance'] = {
            'test_accuracy': training_summary['test_accuracy'],
            'test_loss': training_summary['test_loss']
        }
    
    registry['models'][model_name] = model_info
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"✓ Registry updated: {registry_path.absolute()}")

"""
│
└── models/                               # Base models directory
    │
    │
    └── {MODEL_NAME}/                     # Individual model directory (e.g., "deformation_regressor_v1")
        │
        ├── data/
        │   └── raw/                      # Backup of original CSV files
        │       ├── file1.csv
        │       ├── file2.csv
        │       └── ...
        │
        ├── datasets/                     # NumPy arrays (not TensorFlow datasets)
        │   ├── train.npz                 # Training data
        │   │                             # Contains: X_wide, X_deep, y
        │   ├── val.npz                   # Validation data
        │   │                             # Contains: X_wide, X_deep, y
        │   └── test.npz                  # Test data
        │                                 # Contains: X_wide, X_deep, y
        │
        ├── metadata/                     # Metadata and configuration files
        │   └── dataset_info.json         # Dataset structure metadata
        │                                 # {features: {wide_feature, deep_feature, target},
        │                                 #  feature_stats, train/val/test samples,
        │                                 #  splits, seed, etc.}
        │
        ├── models/
        │   ├── untrained/                # Initial compiled model
        │   │   ├── model.keras           # Keras model file (Wide & Deep architecture)
        │   │   ├── model_summary.txt     # Human-readable architecture
        │   │   │                         # (Input layers, normalization, hidden layers, etc.)
        │   │   └── model_architecture.json  # Structured layer information
        │   │                             # {model_type: 'wide_and_deep_regression',
        │   │                             #  hidden_layers: [64, 64, 32, 16],
        │   │                             #  inputs: {wide, deep}, output, etc.}
        │   └── trained/                  # Trained model weights
        │       └── model.keras           # Keras model with trained weights
        │
        ├── training/                     # Training results and metrics
        │   ├── history.json              # Complete training history
        │   │                             # {loss, val_loss, root_mean_squared_error,
        │   │                             #  val_root_mean_squared_error per epoch,
        │   │                             #  epoch_times_seconds, cumulative_times_seconds}
        │   │
        │   ├── test_results.json         # Test set performance
        │   │                             # {test_loss_mse, test_rmse,
        │   │                             #  predictions: [...], actuals: [...],
        │   │                             #  residuals: [...]}
        │   │
        │   ├── training_summary.json     # Key metrics summary
        │   │                             # {total_epochs, training_time,
        │   │                             #  final_train_loss, final_val_loss,
        │   │                             #  final_train_rmse, final_val_rmse,
        │   │                             #  test_loss_mse, test_rmse,
        │   │                             #  residuals_mean, residuals_std}
        │   │
        │   ├── plots_metadata.json       # Structured plot information
        │   │                             # {plots_directory, total_plots: 5,
        │   │                             #  plots: {loss_vs_epoch, rmse_vs_epoch,
        │   │                             #          loss_vs_time, predictions_vs_actual,
        │   │                             #          residuals_analysis}}
        │   │
        │   └── plots/                    # Training visualizations (PNG images)
        │       ├── loss_vs_epoch.png     # MSE loss curves over epochs
        │       │                         # (Training vs Validation)
        │       │
        │       ├── rmse_vs_epoch.png     # RMSE curves over epochs
        │       │                         # (Training vs Validation)
        │       │
        │       ├── loss_vs_time.png      # Loss curves over real training time (minutes)
        │       │                         # (Training vs Validation)
        │       │
        │       ├── predictions_vs_actual.png  # Scatter plot with diagonal line
        │       │                         # Shows how well predictions match actuals
        │       │                         # Perfect prediction = points on red diagonal line
        │       │
        │       └── residuals_analysis.png     # Combined plot (2 subplots)
        │                                 # Left: Histogram of residuals distribution
        │                                 # Right: Residuals vs predicted values
        │
        └── export/                       # Production-ready model
            ├── export_metadata.json      # Export configuration
            │                             # {model_type: 'wide_and_deep_regression',
            │                             #  inputs: {wide_input, deep_input},
            │                             #  output: {shape, dtype, target},
            │                             #  usage: {python_example}}
            │
            └── saved_model/              # TensorFlow SavedModel format
                ├── saved_model.pb        # Model graph definition
                ├── variables/            # Model weights
                │   ├── variables.data-00000-of-00001
                │   └── variables.index
                ├── assets/               # Additional assets (if any)
                └── fingerprint.pb        # Model fingerprint

"""