# Automating the Immeasurable: Computer Vision for Gravitational Constant Determination

[![View Notebook](https://img.shields.io/badge/View-Jupyter%20Notebook-orange?logo=jupyter)](cavendish.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Automated laser spot tracking for the Cavendish experiment using computer vision**

**Author:** Aryan Senthil

---

## Overview

This project implements an automated computer vision system for tracking laser projection coordinates during the Cavendish experiment, eliminating manual observation errors and dramatically improving measurement precision. The solution uses centroid-based position tracking to analyze video frames, providing sub-pixel accuracy with high temporal resolution.

**Originally developed in Wolfram Mathematica, now documented in Jupyter Notebook format.**

## The Problem

Traditional Cavendish experiment measurements required:
- **Two-person observation teams** working continuously
- **Manual coordinate recording** every 15 seconds
- **Verbal communication** between observer and recorder
- Result: Human fatigue, transcription errors, and limited sampling frequency

## The Solution

An automated algorithm that:
- ✓ Processes video of laser projection to extract precise coordinates
- ✓ Uses image binarization and centroid calculation for sub-pixel accuracy
- ✓ Provides continuous high-frequency sampling (30-60 Hz)
- ✓ Eliminates human observation errors
- ✓ Produces reproducible, objective measurements

## Key Features

- **Automated Video Processing** - Analyzes recorded footage frame-by-frame
- **Image Binarization** - Isolates laser spot from background using adaptive thresholding
- **Centroid Extraction** - Calculates precise (x, y) coordinates for each frame
- **Calibration System** - Converts pixel coordinates to metric measurements using physical scale reference
- **High Temporal Resolution** - Limited only by camera frame rate, not human reaction time

## Results

The algorithm has been **widely implemented across the laboratory** and is now the standard methodology for Cavendish experiment measurements. It provides:

- Continuous sampling at video frame rate (vs. manual 15-second intervals)
- Objective, reproducible measurements free from observer bias
- Significant reduction in personnel requirements
- Enhanced data quality for gravitational constant calculations

## Repository Contents

- `cavendish.ipynb` - Main Jupyter notebook with full documentation, methodology, and results
- `Cavendish.nb` - Original Wolfram Mathematica notebook implementation (Git LFS)
- `images/` - Experimental setup photos, sample data visualizations, and GIF demonstrations
- `README.md` - This file
- `LICENSE` - MIT License

## Viewing the Project

The complete methodology, algorithm explanation, and results are documented in the [Jupyter notebook](cavendish.ipynb).

### Sample Visualizations

The notebook includes:
- Experimental setup photographs
- Real-time laser tracking demonstrations (GIF)
- Binarized frame examples showing image processing
- Calibration methodology with scale references
- Position vs. time plots showing characteristic oscillations

## Technical Approach

1. **Video Preprocessing** - Import footage and apply region-of-interest cropping
2. **Image Binarization** - Convert frames to binary using adaptive thresholding
3. **Centroid Calculation** - Determine laser spot center from white pixel distribution
4. **Coordinate Calibration** - Transform pixel measurements to physical units
5. **Data Export** - Generate time-series position data for analysis

## About the Cavendish Experiment

The Cavendish experiment, first performed by Henry Cavendish in 1798, measures the gravitational constant (G) using a torsion balance. Small gravitational attractions between masses cause measurable angular deflections, which are amplified by reflecting a laser beam onto a screen. Precise tracking of this laser spot position over time is critical for accurate determination of G.

## Impact

This automation solution has:
- Become the **standard laboratory methodology** for Cavendish measurements
- Freed researchers from tedious manual observation tasks
- Improved measurement reliability and reproducibility
- Enabled higher-quality gravitational constant determinations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed for precision measurement in gravitational physics experiments. The automated tracking methodology has been adopted laboratory-wide due to its superior accuracy and efficiency.

---

**Questions or collaboration inquiries?** Feel free to open an issue or reach out!
