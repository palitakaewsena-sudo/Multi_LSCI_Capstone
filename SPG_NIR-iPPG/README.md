# SPG & NIR-iPPG Analysis

This is the core signal processing directory. It extracts cardiovascular signals—Speckle Plethysmography (SPG) and Imaging Photoplethysmography (iPPG)—from the videos and compares them against the reference pulse oximeter (cPPG).

## Folder Structure

We offer two distinct pipelines depending on your data quality and needs:

### 1. `Baseline/`
Contains `baseline_pipeline.py`.
- **What it does:** Uses standard bandpass filtering (e.g., 0.5 - 5.0 Hz) to extract the pulse signal.
- **Best for:** Very clean data without much motion artifact, or for establishing a baseline comparison.

### 2. `SQI/` (Signal Quality Index)
Contains advanced pipelines for both SPG (`NIR-SPG_Advanced`) and iPPG (`NIR-iPPG_Advanced`).
- **What it does:** Runs a sweeping array of filters (Butterworth, Savitzky-Golay). It then evaluates every single heartbeat using a **Signal Quality Index (SQI)**. It discards noisy or corrupted beats before calculating the final Heart Rate and SNR.
- **Best for:** Real-world data where motion artifacts or noise are present.

## How to use
Make sure your data is in `../../data/input/`. Then run the desired pipeline:
```bash
# For SPG Advanced Pipeline
python SQI/NIR-SPG_Advanced/spg_advanced_pipeline.py

# For iPPG Advanced Pipeline
python SQI/NIR-iPPG_Advanced/ippg_advanced_pipeline.py
```

## Expected Output
All results are saved into `results/spg/` or `results/ippg/`. The outputs include:
- **Time-Series Plots:** Showing the filtered SPG/iPPG signal overlaid with the reference cPPG.
- **Bland-Altman Plots:** To evaluate the agreement between the camera-derived heart rate and the reference device.
- **FFT Spectrums:** Frequency domain plots to visualize the dominant heart rate peak.
- **Morphology:** Averaged heartbeat waveform shapes.
