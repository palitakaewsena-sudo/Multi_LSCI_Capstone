"""
split_channels_video.py
=======================
Splits the multispectral recording into 4 separate video files,
one for each channel (wavelength), to verify correct channel switching.

Use ``--auto-correct`` to automatically detect and fix channel phase
shifts using intensity-based fingerprinting.

Usage:
    python scripts/split_channels_video.py [--config config.yaml] [--input recording.avi]
    python scripts/split_channels_video.py --auto-correct
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Resolve workspace root
current_dir = Path(__file__).resolve().parent
workspace_root = None
for parent in [current_dir] + list(current_dir.parents):
    if (parent / "config.yaml").is_file():
        workspace_root = parent
        break

if not workspace_root:
    workspace_root = Path("C:/Scientific Camera Interfaces/SDK/Python Toolkit_Project")

if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from multispectral.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Split recording into 4 channel videos.")
    parser.add_argument("--config", type=str, default=str(workspace_root / "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to recording.avi (defaults to config path)")
    parser.add_argument("--compressed", action="store_true",
                        help="Save output videos in lossy compressed format (.avi with MJPG) instead of the default lossless format")
    parser.add_argument("--auto-correct", action="store_true",
                        help="Scan intensity patterns to auto-detect and correct channel phase shifts")
    parser.add_argument("--normalize", action="store_true",
                        help="Auto-contrast (normalize) each frame to 0-255 brightness (useful if a channel is very dark)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Disable interactive ROI cropping (cropping is enabled by default)")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    except Exception as e:
        print(f"Warning: Could not load config. Using standard defaults. Error: {e}")
        config = None

    # Resolve paths
    if args.input:
        video_path = Path(args.input).resolve()
    elif config:
        video_path = config.acquisition.output_dir / config.acquisition.video_filename
    else:
        video_path = workspace_root / "recording.avi"

    video_path = video_path.resolve()
    metadata_path = video_path.parent / (config.acquisition.metadata_filename if config else "metadata.csv")

    print(f"Input Video   : {video_path}")
    print(f"Metadata CSV  : {metadata_path}")

    if not video_path.is_file():
        print(f"Error: Video file not found at {video_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Frame-to-Channel mapping from metadata.csv
    frame_channel_map = {}
    channel_wavelengths = {0: "CH0", 1: "CH1", 2: "CH2", 3: "CH3"}
    
    if config:
        for ch in config.channels:
            channel_wavelengths[ch.channel_id] = f"CH{ch.channel_id}_{ch.wavelength_nm}nm"

    if metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Check column indices
                frame_idx = header.index("FrameID") if "FrameID" in header else 0
                channel_idx = header.index("Channel") if "Channel" in header else 2
                
                for row in reader:
                    if len(row) > max(frame_idx, channel_idx):
                        try:
                            f_id = int(row[frame_idx])
                            ch_id = int(row[channel_idx])
                            frame_channel_map[f_id] = ch_id
                        except ValueError:
                            continue
            print(f"Loaded {len(frame_channel_map)} frame mappings from metadata.")
        except Exception as e:
            print(f"Warning: Could not parse metadata.csv ({e}). Falling back to modulo-4 mapping.")

    # 2. Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Read video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    crop_rect = None
    if not args.no_crop:
        print("\n--- ROI Cropping ---")
        ret, first_frame = cap.read()
        if ret:
            print("Please select the ROI in the popup window, then press SPACE or ENTER.")
            # Let the user drag a box to select ROI
            crop_rect = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            
            x, y, w, h = crop_rect
            if w > 0 and h > 0:
                width, height = w, h
                print(f"  Crop selected: x={x}, y={y}, width={width}, height={height}")
            else:
                print("  No crop selected. Using full frame.")
                crop_rect = None
                
            # Reset frame position to 0 to prepare for video splitting
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("  Warning: Could not read first frame for cropping.")

    if args.compressed:
        codec = "MJPG"
        ext = ".avi"
    else:
        codec = config.acquisition.video_codec if config else "HFYU"
        ext = Path(config.acquisition.video_filename).suffix if config else ".avi"

    print(f"Properties    : {width}x{height} @ {orig_fps} FPS, {frame_count} frames total")

    # --- Dark Frame Handling ---
    print("\n--- Preparing 6ms Dark Frames ---")
    dark_dir = workspace_root / "dark_frame" / "processed_6ms"
    out_dir = video_path.parent
    for ch_id in range(4):
        df_path = dark_dir / f"dark_frame_CH{ch_id}.npy"
        if df_path.is_file():
            df = np.load(str(df_path))
            if crop_rect is not None:
                x, y, w, h = crop_rect
                df = df[y:y+h, x:x+w]
            out_df_path = out_dir / f"dark_frame_CH{ch_id}.npy"
            np.save(str(out_df_path), df)
            print(f"  Saved {out_df_path.name} with shape {df.shape}")
        else:
            print(f"  Warning: Master dark frame not found at {df_path}")


    # Output videos properties
    out_fps = orig_fps / 4.0
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # Initialize 4 VideoWriters
    writers = {}
    output_filenames = {}
    
    for ch_id in range(4):
        ch_name = channel_wavelengths.get(ch_id, f"CH{ch_id}")
        out_name = f"recording_{ch_name}{ext}"
        out_path = video_path.parent / out_name
        
        # Open VideoWriter (isColor = False because the recording is grayscale)
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height), False)
        if not writer.isOpened():
            print(f"Error: Could not open VideoWriter for {out_path}", file=sys.stderr)
            sys.exit(1)
            
        writers[ch_id] = writer
        output_filenames[ch_id] = out_name
        print(f"Created output: {out_name} (effective rate {out_fps:.2f} FPS)")

    # 3. Auto-correct: scan intensity patterns to detect phase shifts
    cycle_offsets = None
    if args.auto_correct:
        print("\n--- Auto-Correct: Scanning intensity patterns ---")
        cycle_offsets = _scan_intensity_offsets(cap, frame_count, crop_rect=crop_rect)
        if cycle_offsets is not None:
            from collections import Counter
            offset_counts = Counter(cycle_offsets)
            has_shifts = any(o != 0 for o in cycle_offsets)
            if has_shifts:
                print("  Phase shifts detected! Offset distribution:")
                for offset, count in sorted(offset_counts.items()):
                    pct = 100.0 * count / len(cycle_offsets)
                    print(f"    offset={offset}: {count} cycles ({pct:.1f}%)")
            else:
                print("  No phase shifts detected — recording is clean.")
            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print()

    # 4. Read and distribute frames
    processed_count = 0
    channel_frame_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for i in range(1, frame_count + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame if the user has selected an ROI
        if crop_rect is not None:
            x, y, w, h = crop_rect
            frame = frame[y:y+h, x:x+w]

        # Convert frame to grayscale if it's BGR
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame

        # Resolve channel for this frame ID
        if cycle_offsets is not None:
            # Auto-correct mode: use intensity-based offset
            cycle_idx = (i - 1) // 4
            position_in_cycle = (i - 1) % 4
            offset = cycle_offsets[cycle_idx] if cycle_idx < len(cycle_offsets) else 0
            channel = (position_in_cycle + offset) % 4
        elif i in frame_channel_map:
            channel = frame_channel_map[i]
        else:
            # Fallback to modulo 4 pattern
            channel = (i - 1) % 4

        # Apply normalization if requested
        if args.normalize:
            gray_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

        # Write to corresponding video writer
        if channel in writers:
            writers[channel].write(gray_frame)
            channel_frame_counts[channel] += 1
            processed_count += 1

    # Cleanup
    cap.release()
    for writer in writers.values():
        writer.release()

    print("=" * 60)
    print("Channel Splitting Completed Successfully!")
    if args.auto_correct and cycle_offsets is not None and any(o != 0 for o in cycle_offsets):
        print("  (Auto-corrected channel mapping applied)")
    print(f"Processed {processed_count}/{frame_count} frames.")
    for ch_id, count in channel_frame_counts.items():
        ch_name = channel_wavelengths.get(ch_id, f"CH{ch_id}")
        print(f" - {output_filenames[ch_id]} ({ch_name}): {count} frames")
    print("=" * 60)


def _scan_intensity_offsets(cap, total_frames, calibration_cycles=20, crop_rect=None):
    """Scan the video and return per-cycle channel offsets using intensity fingerprinting."""
    num_channels = 4
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Build fingerprint from first calibration_cycles
    cal_totals = np.zeros(num_channels, dtype=np.float64)
    cal_counts = np.zeros(num_channels, dtype=np.int64)

    for _ in range(calibration_cycles):
        for ch in range(num_channels):
            ret, frame = cap.read()
            if not ret:
                print("  Warning: Video too short for calibration.")
                return None
            if crop_rect is not None:
                x, y, w, h = crop_rect
                frame = frame[y:y+h, x:x+w]
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cal_totals[ch] += float(np.mean(frame))
            cal_counts[ch] += 1

    fingerprint = cal_totals / np.maximum(cal_counts, 1)
    print(f"  Fingerprint: {', '.join(f'CH{i}={fingerprint[i]:.1f}' for i in range(num_channels))}")

    # Scan all cycles
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    num_cycles = total_frames // num_channels
    offsets = []

    for _ in range(num_cycles):
        observed = np.zeros(num_channels, dtype=np.float64)
        valid = True
        for ch in range(num_channels):
            ret, frame = cap.read()
            if not ret:
                valid = False
                break
            if crop_rect is not None:
                x, y, w, h = crop_rect
                frame = frame[y:y+h, x:x+w]
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            observed[ch] = float(np.mean(frame))

        if not valid:
            break

        best_offset = 0
        best_error = np.inf
        for offset in range(num_channels):
            rotated = np.roll(fingerprint, -offset)
            error = float(np.sum((observed - rotated) ** 2))
            if error < best_error:
                best_error = error
                best_offset = offset

        offsets.append(best_offset)

    return offsets

if __name__ == "__main__":
    main()
