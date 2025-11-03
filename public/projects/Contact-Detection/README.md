# Automated Contact Detection for Instron Compression Testing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

**Author:** Aryan Senthil

An automated slope-based transition detection algorithm that precisely identifies the initial contact point during Instron compression testing of piezoelectric pressure sensors. Achieves sub-millimeter accuracy with quantifiable uncertainty metrics.

## Demo

![Application Demo](Screencast%20from%2011-01-2025%2009_34_15%20PM.gif)

## Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI Application](#gui-application)
  - [Jupyter Notebook](#jupyter-notebook)
- [Data Format](#data-format)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [License](#license)

## Background

This project addresses a critical challenge in the mechanical characterization of pressure-sensitive sensors using Instron tensile testing machines configured for compression testing. The experimental setup positions the sensor on a base platform with a metallic load cell applying controlled compressive forces from above.

The primary objective is to characterize the sensor's mechanical and electrical response under controlled loading conditions, with particular focus on piezoelectric response characteristics.

## Problem Statement

**Critical Issue:** Manual identification of the load cell sensor contact point introduces positioning errors that distort force-displacement data, reduce repeatability, and compromise piezoelectric sensor calibration.

Current manual detection methods introduce sub-millimeter positioning errors that cause:

1. **Inter-sample variability** - Spurious response variations between nominally identical sensors
2. **Temporal drift** - Significant measurement variations between testing sessions
3. **Piezoelectric response inconsistency** - Corrupted electrical output measurements due to ambiguous loading history

The contact point defines the zero-load reference state, the baseline for force-displacement relationships, and the starting point for piezoelectric response characterization. The piezoelectric response shows particularly high sensitivity to contact detection errors due to coupling between mechanical preload conditions and charge generation dynamics.

## Solution

This project implements a robust, automated **slope-based transition detection algorithm** that:

- Detects initial contact with sub-millimeter precision
- Operates consistently across test sessions
- Works with mechanical signals from the tensile testing machine
- Requires minimal manual intervention
- Provides quantitative confidence metrics for each detection event

The algorithm analyzes the shift in slope of stress-strain data to identify the precise moment of initial contact. Before contact, the stress-strain curve remains flat (zero slope) because the load cell experiences no load. After contact, the slope becomes positive as the material begins to resist deformation.

## Features

- **Automated Contact Detection**: Eliminates manual positioning errors
- **Sub-millimeter Accuracy**: Typical uncertainty < 0.013 mm
- **Statistical Validation**: Provides mean extension and uncertainty metrics
- **GUI Application**: User-friendly interface for data import and analysis
- **Real-time Visualization**: Interactive stress-extension plots with detected contact point
- **CSV Data Import**: Supports standard Instron CSV output format
- **Jupyter Notebook**: Detailed technical documentation and implementation

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AryanSenthil/Contact-Detection.git
cd Contact-Detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### GUI Application

Run the graphical interface for interactive analysis:

```bash
python contact_detection.py
```

**Workflow:**
1. Click "Import CSV File" to load your Instron data
2. Review the data preview in the left panel
3. Click "Find Extension" to perform automated contact detection
4. View results (mean extension and uncertainty) in the top panel
5. Examine the stress-extension plot with detected contact point marked

### Jupyter Notebook

For detailed technical documentation and step-by-step implementation:

```bash
jupyter notebook instron_extension.ipynb
```

The notebook includes:
- Comprehensive problem description and motivation
- Mathematical formulation of the detection algorithm
- Step-by-step implementation with explanations
- Visualization of results
- Sample data analysis

## Data Format

The application expects CSV files in standard Instron format:
- First 12 lines: Metadata (skipped during import)
- Line 13: Column headers
- Data columns:
  - Column 1: Extension (mm)
  - Column 3: Compressive strain (mm/mm)
  - Column 4: Compressive stress (MPa)

Example data file: `Specimen_RawData_1.csv`

## Results

**Typical Output:**
```
Extension: -0.4496 mm
Uncertainty: ±0.0125 mm
```

Note: The extension value is negative since the load cell is compressing the sensor.

The algorithm provides:
- Precise contact point identification
- Reproducible measurements across test sessions
- Quantified uncertainty for confidence assessment
- Visual confirmation through plotted data

## Repository Structure

```
Contact-Detection/
├── contact_detection.py          # GUI application for contact detection
├── instron_extension.ipynb       # Jupyter notebook with technical details
├── instron_extension.html        # HTML export of notebook
├── requirements.txt              # Python dependencies
├── Specimen_RawData_1.csv       # Example data file
├── Screenshot from 2025-10-31...png  # Setup image
├── Screencast from 11-01-2025...gif  # App demonstration
└── README.md                     # This file
```

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- tkinter (usually included with Python)
- tabulate

See `requirements.txt` for specific versions.

## License

This project is open source. Please contact the author for licensing information.

## Contact

**Aryan Senthil**

For questions, issues, or contributions, please open an issue on the GitHub repository.

## Citation

If you use this code in your research, please cite:

```
Senthil, A. (2025). Automated Contact Detection for Instron Compression Testing. GitHub repository:
https://github.com/AryanSenthil/Contact-Detection
```

## Acknowledgments

This work addresses a critical measurement challenge in piezoelectric sensor characterization, enabling reproducible and reliable mechanical testing with sub-millimeter precision.
