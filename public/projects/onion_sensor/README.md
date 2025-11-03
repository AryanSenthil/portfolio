# Deep Learning-Based Load Characterization and Deformation Prediction for Bio-Sourced Onion Peel Piezoelectric Sensors

**Author:** Aryan Senthil
**Institution:** University of Oklahoma
**Principal Investigator:** Dr. Mrinal Saha

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/AryanSenthil/Onion_Sensor)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

---

## Table of Contents

- [Abstract](#abstract)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Classification Module](#classification-module)
  - [Regression Module](#regression-module)
  - [Inference Pipeline](#inference-pipeline)
- [Model Architecture](#model-architecture)
  - [Load Classification (CNN)](#load-classification-cnn)
  - [Deformation Prediction (Wide & Deep)](#deformation-prediction-wide--deep)
- [Results](#results)
- [Testing](#testing)
- [Directory Structure](#directory-structure)
- [Design Philosophy](#design-philosophy)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

---

## Abstract

This work presents the development of a self-powered pressure sensor utilizing onion peel waste as the active piezoelectric material, combined with machine learning techniques to enable intelligent deformation sensing. Agricultural waste in the form of dehydrated and ground onion peel was dispersed in a silicone matrix and fabricated via direct ink writing (DIW) 3D printing, where the self-aligned cellulose fibers generate voltage under mechanical stress. The sensor was characterized under cyclic compression at multiple deformation levels and load types.

To interpret the sensor's electrical response and extract deformation information, we developed a **two-stage machine learning pipeline**. Convolutional neural networks (CNNs) operating on Short-Time Fourier Transform (STFT) spectrograms classify the loading configuration, while a Wide & Deep regression model predicts deformation magnitude by fusing the load type with voltage signal features. This ML-enabled approach allows the bio-composite sensor to predict both the type and magnitude of applied pressure without the requirement of an external battery source unlike traditional pressure sensors.

The results demonstrate that combining sustainable materials with intelligent signal processing creates viable sensing platforms for applications in structural health monitoring, wearable electronics, and human-machine interfaces. By transforming agricultural waste into functional sensors augmented with machine learning, this work opens pathways for environmentally conscious sensor technologies with embedded intelligence.

---

## Key Features

- **Sustainable Sensor Fabrication**: Transform agricultural waste (onion peel) into functional piezoelectric sensors
- **Advanced Signal Processing**: STFT-based time-frequency analysis of voltage signals
- **Dual-Model ML Architecture**:
  - CNN classifier for load type identification (100% accuracy)
  - Wide & Deep regression for deformation prediction (RMSE: 0.0409 mm)
- **Modular Design**: Separate classification and regression modules for flexibility
- **Agent-Compatible Tools**: Functions designed for autonomous ML agent orchestration
- **Comprehensive Testing**: Full test suite with 100% coverage
- **Reproducible Research**: Organized directory structure and version-controlled artifacts

---

## Project Structure

```
Onion_Sensor/
├── data/                       # Raw experimental CSV files
├── images/                     # Documentation images and visualizations
├── models/                     # Trained models and artifacts
│   ├── classification_model/   # CNN load classifier
│   └── regression_model/       # Wide & Deep deformation predictor
├── src/                        # Source code modules
│   ├── classification_module.py   # CNN training and evaluation
│   ├── regression_module.py       # Wide & Deep training and evaluation
│   ├── test_model.py             # Inference utilities
│   ├── tools.py                  # Shared utilities
│   └── onion_sensor.ipynb        # Main research notebook
├── tests/                      # Test suite
├── test/                       # Sample test files
├── requirements.txt            # Python dependencies
├── models.json                 # Model registry metadata
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AryanSenthil/Onion_Sensor.git
   cd Onion_Sensor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

---

## Quick Start

### Option 1: Interactive Jupyter Notebook

```bash
jupyter lab src/onion_sensor.ipynb
```

The notebook provides a complete walkthrough with:
- Theoretical background on piezoelectricity
- Sensor fabrication methodology
- Data collection procedures
- Model training and evaluation
- Inference examples

### Option 2: Python Scripts

```python
import classification_module
import regression_module
from test_model import WaveAnalysis

# Train classification model
classification_module.preprocess_csv_to_wav(csv_dir="data", model_name="my_classifier")
classification_module.load_and_preprocess_data(model_name="my_classifier")
classification_module.build_and_compile_model(model_name="my_classifier")
classification_module.train_and_evaluate_model(model_name="my_classifier", epochs=10)

# Train regression model
regression_module.preprocess_csv_data(csv_dir="data", model_name="my_regressor")
regression_module.build_and_compile_model(model_name="my_regressor")
regression_module.train_and_evaluate_model(model_name="my_regressor", epochs=300)

# Run inference
analysis = WaveAnalysis(classification_model, regression_model)
deformation, load_type = analysis.process_file_and_predict_deformation("test/sample.csv")
```

---

## Usage

### Classification Module

The classification module trains a CNN to classify load types (point vs. uniform) from STFT spectrograms.

```python
import classification_module

# Configuration
MODEL_NAME = "classification_model"
CSV_DIR = "data"
SAMPLING_RATE = 1600
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 10

# Step 1: Convert CSV voltage data to WAV files
classification_module.preprocess_csv_to_wav(
    csv_dir=CSV_DIR,
    model_name=MODEL_NAME,
    sampling_rate=SAMPLING_RATE
)

# Step 2: Load and preprocess spectrograms
classification_module.load_and_preprocess_data(
    model_name=MODEL_NAME,
    validation_split=VALIDATION_SPLIT,
    seed=42
)

# Step 3: Build CNN model
classification_module.build_and_compile_model(
    model_name=MODEL_NAME,
    learning_rate=LEARNING_RATE
)

# Step 4: Train and evaluate
classification_module.train_and_evaluate_model(
    model_name=MODEL_NAME,
    epochs=EPOCHS,
    patience=2
)

# Step 5: Export for inference
classification_module.create_export_model(model_name=MODEL_NAME)
```

**Generated Artifacts:**
- WAV files: `models/{MODEL_NAME}/wave_files/`
- Training plots: `models/{MODEL_NAME}/training/plots/`
- Trained model: `models/{MODEL_NAME}/models/trained/`
- Export model: `models/{MODEL_NAME}/export/saved_model/`

### Regression Module

The regression module trains a Wide & Deep model to predict deformation from voltage and load type.

```python
import regression_module

# Configuration
MODEL_NAME = "regression_model"
CSV_DIR = "data"
LEARNING_RATE = 0.001
EPOCHS = 300

# Step 1: Extract peak-to-peak voltage and labels
regression_module.preprocess_csv_data(
    csv_dir=CSV_DIR,
    model_name=MODEL_NAME,
    test_split=0.1,
    seed=42
)

# Step 2: Build Wide & Deep model
regression_module.build_and_compile_model(
    model_name=MODEL_NAME,
    hidden_layers=[64, 64, 32, 16],
    learning_rate=LEARNING_RATE
)

# Step 3: Train and evaluate
regression_module.train_and_evaluate_model(
    model_name=MODEL_NAME,
    epochs=EPOCHS,
    patience=5
)

# Step 4: Export for inference
regression_module.create_export_model(model_name=MODEL_NAME)
```

**Generated Artifacts:**
- Processed data: `models/{MODEL_NAME}/datasets/`
- Training plots: `models/{MODEL_NAME}/training/plots/`
- Trained model: `models/{MODEL_NAME}/models/trained/`
- Export model: `models/{MODEL_NAME}/export/saved_model/`

### Inference Pipeline

```python
import tensorflow as tf
from test_model import WaveAnalysis, display_model_results

# Load trained models
classification_model = tf.saved_model.load(
    "models/classification_model/export/saved_model"
)
regression_model = tf.keras.models.load_model(
    "models/regression_model/models/trained/model.keras"
)

# Initialize inference pipeline
analysis = WaveAnalysis(classification_model, regression_model)

# Process new sensor data
file_path = "test/sample_data.csv"
deformation, load_type = analysis.process_file_and_predict_deformation(file_path)

# Display results
display_model_results(deformation, load_type, font_size=10)
print(f"Predicted Deformation: {deformation:.3f} mm")
print(f"Load Type: {load_type}")
```

---

## Model Architecture

### Load Classification (CNN)

**Input:** STFT spectrogram (124 x 129 x 1)
**Output:** Load type classification (point or uniform)

```
┌─────────────────────────────────────┐
│  Input: Voltage-Time Signal (CSV)  │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Convert to WAV (Sampling: 1600Hz) │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STFT (Short-Time Fourier Transform)│
│  Window: Hann, Overlap: 50%         │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  CNN Architecture:                  │
│  - Resize (32x32)                   │
│  - Normalization                    │
│  - Conv2D (32 filters, 3x3)         │
│  - Conv2D (64 filters, 3x3)         │
│  - MaxPooling2D (2x2)               │
│  - Dropout (0.25)                   │
│  - Flatten                          │
│  - Dense (128, ReLU)                │
│  - Dropout (0.5)                    │
│  - Dense (2, Softmax)               │
└─────────────────────────────────────┘
                 │
                 ▼
         Load Type (Point/Uniform)
```

**Key Parameters:**
- Total parameters: 1,624,837
- Optimizer: Adam (lr=0.001)
- Loss: SparseCategoricalCrossentropy
- Metrics: Accuracy

**Why STFT?**
- Captures non-stationary signal characteristics during cyclic loading
- Reveals time-localized frequency signatures specific to load types
- Point loads generate sharper transients vs. uniform loads' gradual evolution
- Robust to phase variations in compression cycles

### Deformation Prediction (Wide & Deep)

**Input:** Peak-to-peak voltage + Load type
**Output:** Deformation magnitude (mm)

```
         Wide Component              Deep Component
         (Memorization)              (Generalization)
                │                           │
                │                           │
                ▼                           ▼
       ┌────────────────┐        ┌─────────────────────┐
       │  Load Type     │        │ Peak-to-Peak Voltage│
       │  (Categorical) │        │   (Continuous)      │
       └────────────────┘        └─────────────────────┘
                │                           │
                │                           ▼
                │                 ┌─────────────────────┐
                │                 │   Normalization     │
                │                 └─────────────────────┘
                │                           │
                │                           ▼
                │                 ┌─────────────────────┐
                │                 │   Dense (64, ReLU)  │
                │                 └─────────────────────┘
                │                           │
                │                           ▼
                │                 ┌─────────────────────┐
                │                 │   Dense (64, ReLU)  │
                │                 └─────────────────────┘
                │                           │
                │                           ▼
                │                 ┌─────────────────────┐
                │                 │   Dense (32, ReLU)  │
                │                 └─────────────────────┘
                │                           │
                │                           ▼
                │                 ┌─────────────────────┐
                │                 │   Dense (16, ReLU)  │
                │                 └─────────────────────┘
                │                           │
                └───────────────┬───────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │    Concatenate     │
                     └────────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  Dense (1, Linear) │
                     └────────────────────┘
                                │
                                ▼
                    Deformation Magnitude (mm)
```

**Key Parameters:**
- Total parameters: 6,920
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Metrics: Root Mean Squared Error (RMSE)

**Why Wide & Deep?**
- **Wide component**: Directly connects load type to output, preserving discriminative power
- **Deep component**: Processes voltage through non-linear transformations for continuous deformation space
- Combines memorization (load-specific patterns) with generalization (voltage-deformation relationship)

---

## Results

### Classification Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 100% | 100% | 100% |
| **Loss** | 0.0021 | 0.0018 | - |


### Regression Performance

| Metric | Value |
|--------|-------|
| **RMSE** | 0.0409 mm |
| **MAE** | 0.0321 mm |
| **R²** | 0.9987 |

**Deformation Prediction Accuracy:**
- 0.5 mm target: Avg. error ±0.035 mm
- 0.7 mm target: Avg. error ±0.041 mm
- 1.0 mm target: Avg. error ±0.046 mm

---

## Testing

The project includes a comprehensive test suite covering all modules.

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Modules

```bash
# Test classification module
pytest tests/test_classification_module.py -v

# Test regression module
pytest tests/test_regression_module.py -v

# Test inference utilities
pytest tests/test_test_model.py -v

# Test shared tools
pytest tests/test_tools.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

For detailed testing documentation, see [RUN_TESTS.md](RUN_TESTS.md).

---

## Directory Structure

Detailed directory structure documentation is available in [MODELS_STRUCTURE.md](MODELS_STRUCTURE.md).

### Model Artifacts

```
models/<model_name>/
├── data/raw/              # Original CSV files
├── datasets/              # Preprocessed datasets
├── metadata/              # Configuration and dataset info
├── models/
│   ├── untrained/         # Initial compiled model
│   └── trained/           # Trained model (.keras)
├── training/plots/        # Loss curves, confusion matrices
├── export/saved_model/    # TensorFlow SavedModel format
└── wave_files/            # Generated WAV files (classification only)
```

---

## Design Philosophy

This project employs an **agent-tool architecture** where functions are designed as stateless tools for autonomous ML agents rather than traditional Python functions.

### Key Principles

1. **No Return Values**: Functions save artifacts to disk instead of returning objects
2. **Disk-Based State**: All models, plots, and data persist to filesystem
3. **Agent Compatibility**: Designed for orchestration by autonomous agents (LangGraph, etc.)
4. **Multi-User Isolation**: Directory-based organization prevents conflicts

### Why This Design?

**Traditional Python:**
```python
model = train_model(data)  # ✗ Agent cannot capture return value
predictions = model.predict(X_test)
```

**Agent-Compatible:**
```python
train_model(data_path="data/train.csv", model_dir="models/run1")  # ✓ Saves to disk
# Agent tracks: {"model_path": "models/run1/model.keras"}
evaluate_model(model_path="models/run1/model.keras", output_dir="results/")
```

This architecture enables:
- **Workflow resumption** after interruptions
- **Multi-session continuity** across different execution contexts
- **Agent memory** through file path tracking
- **Reproducibility** through versioned artifacts

For detailed explanation, see the "Design Philosophy" section in `src/onion_sensor.ipynb`.

---

## Contributing

This is a research project under active development. Contributions are welcome from lab members and collaborators.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies: `pip install -e .`
4. Make changes and add tests
5. Run test suite: `pytest tests/ -v`
6. Commit changes: `git commit -m "Add your feature"`
7. Push to branch: `git push origin feature/your-feature`
8. Submit a pull request

### Coding Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new functionality
- Update documentation for user-facing changes

---

## License

**This project is licensed under a Proprietary Research License.**

Copyright (c) 2024-2025 University of Oklahoma. All Rights Reserved.

This research work and all associated materials are the property of the University of Oklahoma and were conducted under the supervision of Dr. Mrinal Saha at the School of Aerospace and Mechanical Engineering.

### Key Restrictions

- **No reproduction** without prior written permission
- **No distribution** to third parties
- **No commercial use** or commercial advantage
- **No modification** or derivative works
- Academic use and citation permitted with proper attribution

### Permitted Use

This software is made available for:
- Academic review and educational purposes
- Collaboration with authorized research partners
- Citation and reference in academic publications

For full license terms, see the [LICENSE](LICENSE) file.

### Licensing Inquiries

For permissions, collaboration requests, or licensing inquiries, please contact:

**Dr. Mrinal Saha**
School of Aerospace and Mechanical Engineering
University of Oklahoma
Norman, OK 73019
Email: msaha@ou.edu

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{senthil2024onionsensor,
  author = {Senthil, Aryan and Saha, Mrinal},
  title = {Deep Learning-Based Load Characterization and Deformation Prediction
           for Bio-Sourced Onion Peel Piezoelectric Sensors},
  year = {2024},
  institution = {University of Oklahoma},
  howpublished = {\url{https://github.com/AryanSenthil/Onion_Sensor}}
}
```

---

## Contact

**Aryan Senthil**
University of Oklahoma
GitHub: [@AryanSenthil](https://github.com/AryanSenthil)

**Principal Investigator: Dr. Mrinal Saha**
School of Aerospace and Mechanical Engineering
University of Oklahoma

---

## Acknowledgments

- University of Oklahoma for research facilities and support
- Dr. Mrinal Saha for guidance and supervision
- Machine Learning research community for open-source tools (TensorFlow, Keras, scikit-learn)
- Agricultural waste management initiatives for inspiration

---

**Last Updated:** November 2025
**Project Status:** Active Research
