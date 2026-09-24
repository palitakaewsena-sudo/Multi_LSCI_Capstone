# Split Video

This folder provides a utility to crop or split a video based on a Region of Interest (ROI).

## What it does
Sometimes the camera captures a very large field of view, or multiple wavelengths are optically split onto different halves of the camera sensor. `split_channels_video.py` allows you to interactively draw a box over the specific area you want to analyze and extracts only that part of the video.

## How to use
Run the script:
```bash
python split_channels_video.py
```

## Expected Output
1. A GUI window will pop up showing the first frame of the video.
2. Click and drag to draw a box around your ROI.
3. Press `ENTER` or `SPACE` to confirm.
4. The script will process the entire video and save a new cropped video file containing only your selected region.
