# Models Folder Structure

This document describes the folder structure created by the classification and regression modules when all functions are executed.

## Registry File

- `models.json` - Located at the project root, contains metadata about all trained models (both classification and regression)

## Classification Module Structure

Each model created by the classification module will have the following structure:

```
models/
└── <model_name>/
    ├── .gitkeep
    ├── data/
    │   ├── .gitkeep
    │   └── raw/
    │       └── .gitkeep
    ├── datasets/
    │   ├── .gitkeep
    │   ├── train/
    │   │   └── .gitkeep
    │   ├── val/
    │   │   └── .gitkeep
    │   └── test/
    │       └── .gitkeep
    ├── export/
    │   ├── .gitkeep
    │   └── saved_model/
    │       └── .gitkeep
    ├── metadata/
    │   └── .gitkeep
    ├── models/
    │   ├── .gitkeep
    │   ├── trained/
    │   │   └── .gitkeep
    │   └── untrained/
    │       └── .gitkeep
    ├── training/
    │   ├── .gitkeep
    │   └── plots/
    │       └── .gitkeep
    └── wave_files/
        ├── .gitkeep
        └── wave_files/
            └── .gitkeep
```

### Folder Descriptions (Classification)

- **data/raw/** - Original CSV files copied here
- **wave_files/wave_files/** - Generated WAV files organized by class
- **datasets/** - TensorFlow datasets (train/val/test splits)
- **metadata/** - JSON files with preprocessing and dataset information
- **models/untrained/** - Initial compiled model before training
- **models/trained/** - Trained model after fit
- **training/plots/** - Training visualizations (loss, accuracy, confusion matrix)
- **export/saved_model/** - Final exported TensorFlow SavedModel

## Regression Module Structure

Each model created by the regression module (Wide & Deep) will have the following structure:

```
models/
└── <model_name>/
    ├── .gitkeep
    ├── data/
    │   ├── .gitkeep
    │   └── raw/
    │       └── .gitkeep
    ├── datasets/
    │   └── .gitkeep
    ├── export/
    │   ├── .gitkeep
    │   └── saved_model/
    │       └── .gitkeep
    ├── metadata/
    │   └── .gitkeep
    ├── models/
    │   ├── .gitkeep
    │   ├── trained/
    │   │   └── .gitkeep
    │   └── untrained/
    │       └── .gitkeep
    └── training/
        ├── .gitkeep
        └── plots/
            └── .gitkeep
```

### Folder Descriptions (Regression)

- **data/raw/** - Original CSV files copied here
- **datasets/** - NumPy .npz files containing train/val/test splits with wide and deep features
- **metadata/** - JSON files with dataset information and feature descriptions
- **models/untrained/** - Initial compiled Wide & Deep model before training
  - Includes `model_architecture.json` and `model_summary.txt`
- **models/trained/** - Trained model after fit
- **training/plots/** - Training visualizations (loss, RMSE, predictions vs actual, residuals analysis)
- **export/saved_model/** - Final exported TensorFlow SavedModel

## Git Configuration

The `.gitignore` file in the `models/` folder ignores all data files but preserves the folder structure using `.gitkeep` files.

## Adding .gitkeep to New Models

After running either the classification or regression pipeline, you can add `.gitkeep` files to preserve the structure:

```bash
./add_gitkeep_to_model.sh <model_name>
```

Or manually:

```bash
find models/<model_name> -type d -exec touch {}/.gitkeep \;
```
