# Multi_LSCI_Cabstone

Multispectral camera video and pulse-oximeter data recording, along with LSCI (Laser Speckle Contrast Imaging) and SPG/iPPG analysis pipelines.

## Project Overview

This repository provides a complete end-to-end toolkit for capturing, synchronizing, and analyzing multimodal biomedical signals (Camera + Pulse Oximeter). It includes hardware trigger code for Arduino, recording scripts, and advanced signal processing pipelines using Speckle Plethysmography (SPG) and Imaging Photoplethysmography (iPPG).

## Directory Structure

Here is a summary of what each folder contains. **(Click on the folders to read their specific README files for detailed usage instructions and examples.)**

- **[`Arduino_pulse_control/`](Arduino_pulse_control/README.md)**: Arduino sketch (`.ino`) and timing logic to trigger the camera and synchronize it with the LED/Pulse Oximeter.
- **[`Camera_SDK/`](Camera_SDK/)**: Thorlabs TSI SDK and DLLs required for operating the Thorlabs scientific camera.
- **[`Recording_camera/`](Recording_camera/README.md)**: Python scripts and configurations (`config.yaml`) used for real-time multimodal recording.
- **[`Speckle_Mapping/`](Speckle_Mapping/README.md)**: Generates 2D LSCI temporal APU (Arbitrary Perfusion Units) maps from the recorded videos.
- **[`Split_video/`](Split_video/README.md)**: Utilities for splitting multiplexed video channels based on user-selected ROI.
- **[`SPG_NIR-iPPG/`](SPG_NIR-iPPG/README.md)**: The core signal processing pipelines containing two main approaches:
  - `Baseline/`: Standard bandpass filtering analysis.
  - `SQI/`: Advanced analysis utilizing Signal Quality Index (SQI) to filter out noisy pulses.

## Quick Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Camera SDK:** Ensure that the DLLs in `Camera_SDK/dll/` are accessible in your system PATH, as required by the Thorlabs SDK.
3. **Data Folders:** Place your input recordings (e.g., `recording_CH3_1064nm.mkv`, `pulse_data.csv`, `dark_frame_CH3.npy`) into the `data/input/` directory before running the analysis pipelines. All outputs will be automatically saved in the `results/` folder.
