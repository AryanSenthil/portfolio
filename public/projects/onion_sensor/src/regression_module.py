"""
regression_module.py - Wide & Deep regression pipeline optimized for agent orchestration
All functions are side-effect based (save to disk, no return values)
"""

import json
import logging
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

from tools import DataProcessor, load_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def preprocess_csv_data(
    csv_dir: str,
    model_name: str,
    sampling_rate: int = 16000,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42
) -> None:
    """Extract features from CSV, create train/val/test splits
    
    Args:
        csv_dir: Directory containing CSV files
        model_name: Name of the model
        sampling_rate: Audio sampling rate (default: 16000)
        validation_split: Fraction for validation (default: 0.2)
        test_split: Fraction for test (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    
    Outputs:
        datasets/*.npz: NumPy arrays for train/val/test splits
        dataset_info.json: Dataset metadata and configuration
    
    Data Format:
        X_wide (load_type), X_deep (peaks), y (deformation)
    
    Requires: []
    """
    
    start_time = time.time()
    csv_dir = Path(csv_dir)
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data_dir = model_dir / 'data' / 'raw'
    datasets_dir = model_dir / 'datasets'
    metadata_dir = model_dir / 'metadata'
    
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    
    csv_files = list(csv_dir.glob('*.csv'))
    num_csv_files = len(csv_files)
    
    for csv_file in csv_files:
        shutil.copy2(csv_file, raw_data_dir / csv_file.name)
    
    data = load_data(str(csv_dir), sampling_rate)
    
    dp = DataProcessor(data)
    dp.time_voltage_data()
    dp.convert_to_array()
    dp.calculate_voltage_peak_to_peak()
    
    combined_data = dp.combined_data
    peaks = combined_data[:, 0]
    load_type = combined_data[:, 1]
    deformation = combined_data[:, 2]
    
    total_samples = len(deformation)
    
    valid_size_adjusted = validation_split / (1 - test_split)
    combined_features = np.column_stack((load_type, peaks))
    
    combined_train_valid, combined_test, y_train_valid, y_test = train_test_split(
        combined_features, deformation, test_size=test_split, random_state=seed
    )
    
    combined_train, combined_valid, y_train, y_valid = train_test_split(
        combined_train_valid, y_train_valid, test_size=valid_size_adjusted, random_state=seed
    )
    
    X_train_wide = combined_train[:, 0].reshape(-1, 1)
    X_train_deep = combined_train[:, 1].reshape(-1, 1)
    X_valid_wide = combined_valid[:, 0].reshape(-1, 1)
    X_valid_deep = combined_valid[:, 1].reshape(-1, 1)
    X_test_wide = combined_test[:, 0].reshape(-1, 1)
    X_test_deep = combined_test[:, 1].reshape(-1, 1)
    
    train_samples = len(y_train)
    val_samples = len(y_valid)
    test_samples = len(y_test)
    
    train_path = datasets_dir / 'train.npz'
    np.savez(train_path, X_wide=X_train_wide, X_deep=X_train_deep, y=y_train)
    
    val_path = datasets_dir / 'val.npz'
    np.savez(val_path, X_wide=X_valid_wide, X_deep=X_valid_deep, y=y_valid)
    
    test_path = datasets_dir / 'test.npz'
    np.savez(test_path, X_wide=X_test_wide, X_deep=X_test_deep, y=y_test)
    
    metadata = {
        'model_type': 'wide_and_deep_regression',
        'total_samples': int(total_samples),
        'train_samples': int(train_samples),
        'val_samples': int(val_samples),
        'test_samples': int(test_samples),
        'features': {
            'wide_feature': 'load_type',
            'deep_feature': 'peaks',
            'target': 'deformation'
        },
        'dataset_paths_absolute': {
            'train': str(train_path.absolute()),
            'val': str(val_path.absolute()),
            'test': str(test_path.absolute())
        }
    }
    
    metadata_path = metadata_dir / 'dataset_info.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"OK Dataset metadata: {metadata_path.absolute()}")
    logger.info(f"OK Preprocessing complete: {time.time() - start_time:.1f}s")


def build_and_compile_model(
    model_name: str,
    hidden_layers: list = [64, 64, 32, 16],
    learning_rate: float = 0.001
) -> None:
    """Build Wide & Deep regression model
    
    Args:
        model_name: Name of the model
        hidden_layers: List of hidden layer sizes (default: [64, 64, 32, 16])
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
    
    metadata_path = metadata_dir / 'dataset_info.json'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found at {metadata_path}. "
            f"Run preprocess_csv_data() first."
        )
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    input_wide = layers.Input(shape=[1], name='wide_input')
    input_deep = layers.Input(shape=[1], name='deep_input')
    
    norm_layer_wide = layers.Normalization(name='wide_normalization')
    norm_layer_deep = layers.Normalization(name='deep_normalization')
    norm_wide = norm_layer_wide(input_wide)
    norm_deep = norm_layer_deep(input_deep)
    
    deep = norm_deep
    for i, units in enumerate(hidden_layers):
        deep = layers.Dense(units, activation='relu', name=f'deep_hidden_{i+1}')(deep)
    
    concat = layers.concatenate([norm_wide, deep], name='wide_deep_concat')
    output = layers.Dense(1, name='output')(concat)
    
    model = models.Model(inputs=[input_wide, input_deep], outputs=[output])
    
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['RootMeanSquaredError']
    )
    
    model_path = untrained_model_dir / 'model.keras'
    model.save(model_path)

    model.summary()
    
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    with open(untrained_model_dir / 'model_summary.txt', 'w') as f:
        f.write("\n".join(summary_lines))
    
    architecture_info = {
        'model_type': 'wide_and_deep_regression',
        'hidden_layers': hidden_layers,
        'learning_rate': learning_rate,
        'total_params': int(model.count_params()),
        'inputs': {
            'wide': {'feature': metadata['features']['wide_feature']},
            'deep': {'feature': metadata['features']['deep_feature']}
        },
        'output': {'target': metadata['features']['target']}
    }
    
    with open(untrained_model_dir / 'model_architecture.json', 'w') as f:
        json.dump(architecture_info, f, indent=2)
    
    logger.info(f"OK Model built: {model_path.absolute()}")
    logger.info(f"OK Model build complete: {time.time() - start_time:.1f}s")


def train_and_evaluate_model(
    model_name: str,
    epochs: int = 3000,
    patience: int = 5
) -> None:
    """Train model, evaluate RMSE, generate plots
    
    Args:
        model_name: Name of the model
        epochs: Number of training epochs (default: 3000)
        patience: Early stopping patience (default: 5)
    
    Outputs:
        models/trained/model.keras: Trained model with weights
        training/*.json: Training history and results
        training/plots/*.png: Training visualization plots
    
    Metrics: RMSE, MSE
    
    Requires: [datasets/*.npz, models/untrained/model.keras]
    """
    overall_start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    datasets_dir = model_dir / 'datasets'
    models_dir = model_dir / 'models'
    training_dir = model_dir / 'training'
    plots_dir = training_dir / 'plots'
    
    training_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    train_data = np.load(datasets_dir / 'train.npz')
    val_data = np.load(datasets_dir / 'val.npz')
    test_data = np.load(datasets_dir / 'test.npz')
    
    X_train_wide, X_train_deep, y_train = train_data['X_wide'], train_data['X_deep'], train_data['y']
    X_val_wide, X_val_deep, y_val = val_data['X_wide'], val_data['X_deep'], val_data['y']
    X_test_wide, X_test_deep, y_test = test_data['X_wide'], test_data['X_deep'], test_data['y']
    
    model = tf.keras.models.load_model(models_dir / 'untrained' / 'model.keras')
    
    norm_layer_wide = model.get_layer('wide_normalization')
    norm_layer_deep = model.get_layer('deep_normalization')
    norm_layer_wide.adapt(X_train_wide)
    norm_layer_deep.adapt(X_train_deep)
    
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
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        (X_train_wide, X_train_deep),
        y_train,
        epochs=epochs,
        validation_data=((X_val_wide, X_val_deep), y_val),
        callbacks=[early_stopping, time_callback],
        verbose=0
    )
    
    epochs_trained = len(history.epoch)
    
    test_results = model.evaluate((X_test_wide, X_test_deep), y_test, return_dict=True, verbose=0)
    
    y_pred_test = model.predict((X_test_wide, X_test_deep), verbose=0).flatten()
    residuals = y_test - y_pred_test
    
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
    
    test_results_serializable = {
        'test_loss_mse': float(test_results['loss']),
        'test_rmse': float(np.sqrt(test_results['loss'])),
        'predictions': y_pred_test.tolist(),
        'actuals': y_test.tolist(),
        'residuals': residuals.tolist()
    }
    
    with open(training_dir / 'test_results.json', 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    train_rmse = np.sqrt(history.history['loss'])
    val_rmse = np.sqrt(history.history['val_loss'])
    times_in_minutes = [t / 60 for t in time_callback.times]
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.epoch, history.history['loss'], 'b-', label='Training', linewidth=2)
    plt.plot(history.epoch, history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.title('Loss vs Epoch')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    loss_epoch_path = plots_dir / 'loss_vs_epoch.png'
    plt.savefig(loss_epoch_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.epoch, train_rmse, 'b-', label='Training', linewidth=2)
    plt.plot(history.epoch, val_rmse, 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.title('RMSE vs Epoch')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    rmse_epoch_path = plots_dir / 'rmse_vs_epoch.png'
    plt.savefig(rmse_epoch_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(times_in_minutes, history.history['loss'], 'b-', label='Training', linewidth=2)
    plt.plot(times_in_minutes, history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    plt.xlabel('Time (minutes)'); plt.ylabel('Loss (MSE)'); plt.title('Loss vs Time')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    loss_time_path = plots_dir / 'loss_vs_time.png'
    plt.savefig(loss_time_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, s=50)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('Predictions vs Actual')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    pred_actual_path = plots_dir / 'predictions_vs_actual.png'
    plt.savefig(pred_actual_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Residuals'); ax1.set_ylabel('Frequency')
    ax1.set_title('Residuals Distribution'); ax1.grid(True, alpha=0.3)
    ax2.scatter(y_pred_test, residuals, alpha=0.5, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    residuals_path = plots_dir / 'residuals_analysis.png'
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plots_metadata = {
        'plots_directory_absolute': str(plots_dir.absolute()),
        'plots': {
            'loss_vs_epoch': {'absolute_path': str(loss_epoch_path.absolute())},
            'rmse_vs_epoch': {'absolute_path': str(rmse_epoch_path.absolute())},
            'loss_vs_time': {'absolute_path': str(loss_time_path.absolute())},
            'predictions_vs_actual': {'absolute_path': str(pred_actual_path.absolute())},
            'residuals_analysis': {'absolute_path': str(residuals_path.absolute())}
        }
    }
    
    with open(training_dir / 'plots_metadata.json', 'w') as f:
        json.dump(plots_metadata, f, indent=2)
    
    training_summary = {
        'total_epochs': epochs_trained,
        'test_loss_mse': float(test_results['loss']),
        'test_rmse': float(np.sqrt(test_results['loss'])),
        'best_val_loss': float(min(history.history['val_loss'])),
        'residuals_mean': float(residuals.mean()),
        'residuals_std': float(residuals.std())
    }
    
    with open(training_dir / 'training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    logger.info(f"OK Trained model: {trained_model_path.absolute()}")
    logger.info(f"OK Training plots: {plots_dir.absolute()}")
    logger.info(f"OK Training summary: {(training_dir / 'training_summary.json').absolute()}")
    logger.info(f"OK Test RMSE: {np.sqrt(test_results['loss']):.6f}")
    logger.info(f"OK Training complete: {time.time() - overall_start_time:.1f}s")


def create_export_model(model_name: str) -> None:
    """Create production SavedModel
    
    Args:
        model_name: Name of the model
    
    Outputs:
        export/saved_model/: TensorFlow SavedModel directory
        models.json: Model registry with metadata
    
    Requires: [models/trained/model.keras, dataset_info.json]
    """
    start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / 'models' / model_name
    trained_model_path = model_dir / 'models' / 'trained' / 'model.keras'
    export_dir = model_dir / 'export' / 'saved_model'
    metadata_dir = model_dir / 'metadata'
    
    if not trained_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {trained_model_path}. "
            f"Run train_and_evaluate_model() first."
        )
    
    model = tf.keras.models.load_model(trained_model_path)
    tf.saved_model.save(model, str(export_dir))
    
    with open(metadata_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    export_metadata = {
        'model_type': 'wide_and_deep_regression',
        'export_path_absolute': str(export_dir.absolute()),
        'inputs': {
            'wide_input': {'feature': dataset_info['features']['wide_feature']},
            'deep_input': {'feature': dataset_info['features']['deep_feature']}
        },
        'output': {'target': dataset_info['features']['target']}
    }
    
    with open(model_dir / 'export' / 'export_metadata.json', 'w') as f:
        json.dump(export_metadata, f, indent=2)
    
    _update_models_registry(model_name)
    
    logger.info(f"OK Export model: {export_dir.absolute()}")
    logger.info(f"OK Export complete: {time.time() - start_time:.1f}s")


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
        'model_type': 'wide_and_deep_regression',
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
        model_info['features'] = dataset_info['features']
        model_info['total_samples'] = dataset_info['total_samples']
    
    training_summary_path = model_dir / 'training' / 'training_summary.json'
    if training_summary_path.exists():
        with open(training_summary_path, 'r') as f:
            training_summary = json.load(f)
        model_info['performance'] = {
            'test_loss_mse': training_summary['test_loss_mse'],
            'test_rmse': training_summary['test_rmse']
        }
    
    arch_path = model_dir / 'models' / 'untrained' / 'model_architecture.json'
    if arch_path.exists():
        with open(arch_path, 'r') as f:
            arch_info = json.load(f)
        model_info['architecture'] = {
            'type': 'Wide & Deep',
            'hidden_layers': arch_info['hidden_layers'],
            'total_params': arch_info['total_params']
        }
    
    registry['models'][model_name] = model_info
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"OK Registry updated: {registry_path.absolute()}")

"""
│
└── models/                               # Base models directory
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