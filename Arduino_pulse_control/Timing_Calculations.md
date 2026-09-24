# Arduino Timer1 Timing Calculations

This document explains how the timer values in `sync_120hz_8ms_exp.ino` are calculated. 

## The Formula

The Arduino Uno runs on a **16 MHz** clock. Timer1 is configured with a **prescaler of 8**. 
This means the timer ticks at a frequency of:
`16,000,000 Hz / 8 = 2,000,000 Hz` (or 2,000,000 ticks per second)

This simplifies to exactly **2,000 ticks per 1 millisecond (ms)**.
Since the timer starts counting from 0, we must subtract 1 from the final tick count.

**General Formula:**
```
Timer Value = (Target Time in ms * 2000) - 1
```

---

## Step-by-Step Examples

### 1. `TIMER1_TOP` (Frame Period at 120 Hz)
We want the camera to capture at 120 Hz (120 frames per second).
* **Target Time:** `1 second / 120 = 0.008333... seconds = 8.333 ms`
* **Calculation:** `(8.333 ms * 2000) - 1 = 16666 - 1 = 16665`
* **Result:** `#define TIMER1_TOP 16665`

### 2. `TIMER1_TRIGGER_LOW` (Pull Trigger LOW)
We want to trigger the camera at 0.200 ms after the LED turns on (to give the LED time to stabilize).
* **Target Time:** `0.200 ms`
* **Calculation:** `(0.200 * 2000) - 1 = 400 - 1 = 399`
* **Result:** `#define TIMER1_TRIGGER_LOW 399`

### 3. `TIMER1_TRIGGER_HIGH` (Release Trigger HIGH)
The camera requires a trigger pulse of a certain width. We want a 150 µs (0.150 ms) pulse.
* **Target Time:** `0.200 ms (start) + 0.150 ms (width) = 0.350 ms`
* **Calculation:** `(0.350 * 2000) - 1 = 700 - 1 = 699`
* **Result:** `#define TIMER1_TRIGGER_HIGH 699`

### 4. `TIMER1_LED_OFF` (Turn LED OFF)
We want the LED to stay on during the entire camera exposure time. If the camera exposure is set to ~7.2 ms, we turn the LED off safely at 7.500 ms.
* **Target Time:** `7.500 ms`
* **Calculation:** `(7.500 * 2000) - 1 = 15000 - 1 = 14999`
* **Result:** `#define TIMER1_LED_OFF 14999`

---

## How to calculate new values?
If you want to change the frequency to **100 Hz**:
1. Period = `1 / 100 = 10 ms`
2. `TIMER1_TOP = (10 * 2000) - 1 = 19999`

If you want the LED to turn off at **5.0 ms**:
1. `TIMER1_LED_OFF = (5.0 * 2000) - 1 = 9999`
