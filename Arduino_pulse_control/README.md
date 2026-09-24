# Arduino Pulse Control

This folder contains the firmware and documentation needed to synchronize the hardware (LEDs, Pulse Oximeter, and Camera).

## What it does
The Arduino acts as the master clock. It sends highly precise pulse signals to:
1. Turn on the Near-Infrared (NIR) LEDs.
2. Trigger the Thorlabs camera to start exposure.
3. Turn off the LEDs exactly when the exposure ends.
4. Synchronize time with the pulse oximeter data collection.

## Files
- `sync_120hz_8ms_exp/sync_120hz_8ms_exp.ino`: The main Arduino sketch. Upload this to an Arduino Uno.
- `Timing_Calculations.md`: Explains the math behind the timer intervals (e.g., how we get 120Hz framerate).
- `timing_diagram.pdf`: A visual schematic of the hardware trigger pulses.

## How to use
1. Open the `.ino` file in the Arduino IDE.
2. Connect your Arduino Uno.
3. Verify and Upload the sketch.
4. The Arduino will immediately begin generating TTL trigger pulses on the specified output pins.
