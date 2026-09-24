# Recording Camera

This folder contains scripts to interface with the Thorlabs camera and record synchronized multimodal data.

## What it does
The `record_multimodal.py` script captures frames from the camera while simultaneously reading PPG data from the pulse oximeter (via serial port). Everything is synchronized using the Arduino trigger.

## How to use

1. Open `config.yaml` and adjust your parameters:
   - Camera settings (Exposure time, Gain, ROI)
   - Capture settings (Duration, Framerate)
2. Run the recording script:
   ```bash
   python record_multimodal.py --config config.yaml
   ```

## Expected Output
Once the recording finishes, the script will generate:
- A high-speed video file (e.g., `.mkv` or `.avi`).
- A `pulse_data.csv` file containing the reference heart rate, SpO2, and raw PPG signals synchronized with the video frames.
