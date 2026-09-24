# Multi_LSCI_Cabstone

Multispectral camera video and pulse-oximeter data recording, along with LSCI (Laser Speckle Contrast Imaging) and SPG/iPPG analysis pipelines.

## Project Structure

- `Arduino_pulse_control/`: Arduino sketch (`.ino`) and timing diagrams for triggering and synchronizing the camera.
- `Camera_SDK/`: Thorlabs TSI SDK and DLLs for camera acquisition and control.
- `Recording_camera/`: Scripts for recording multimodal data (e.g., `record_multimodal.py`) and configuration files.
- `SPG_NIR-iPPG/`: Contains two analysis pipelines for SPG and iPPG:
  - `Baseline/`: Baseline pipeline (standard analysis).
  - `SQI/`: Advanced pipelines using Signal Quality Index (SQI).
- `Split_video/`: Tools for splitting video channels.
- `Speckle_Mapping/`: LSCI temporal APU (Arbitrary Perfusion Units) mapping and analysis.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. For the Thorlabs camera SDK, ensure that the DLLs in `Camera_SDK/dll/` are accessible or properly installed in your system PATH as required by the `thorlabs_tsi_sdk`.

## Usage

Example for recording multimodal data:
```bash
python Recording_camera/record_multimodal.py --config Recording_camera/config.yaml
```
