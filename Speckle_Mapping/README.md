# Speckle Mapping (LSCI Analysis)

This folder contains tools for Laser Speckle Contrast Imaging (LSCI) spatial and temporal analysis.

## What it does
The `lsci_temporal_apu.py` script calculates the Speckle Contrast ($K$) from raw speckle images. It computes Temporal Speckle Contrast (over time) and converts it into Arbitrary Perfusion Units (APU), which represent blood flow velocity.

It also supports **Wavelength Differencing** to compare perfusion at different skin depths (e.g., 520nm vs 1064nm).

## How to use
1. Place your video and dark frame files in the `data/input/` directory.
2. Run the script:
   ```bash
   python lsci_temporal_apu.py
   ```

## Expected Output
The script will save images and arrays into `results/temporal_apu/`. You will get:
- `APU_map.png`: A heatmap showing blood perfusion (Red = high flow, Blue = low flow).
- `Wavelength_diff.png`: A comparison heatmap if multiple wavelengths are used.
