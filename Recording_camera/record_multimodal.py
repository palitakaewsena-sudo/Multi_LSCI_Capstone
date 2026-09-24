#!/usr/bin/env python3
"""Record multispectral camera video and pulse-oximeter data simultaneously.

Connects to a single Arduino Uno trigger controller, captures both
multispectral video frames and synchronized cPPG samples (IR, HR, SpO2)
at 240 Hz, and runs the cPPG signal analysis pipeline upon completion.

Usage::

    python scripts/record_multimodal.py
    python scripts/record_multimodal.py --config config.yaml
    python scripts/record_multimodal.py --no-plot
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# Ensure project root is on sys.path for package imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multispectral.acquisition import AcquisitionRunner, AcquisitionStats
from multispectral.config import ConfigError, load_config
from multispectral.logging_setup import configure_logging
from pulse_oximeter.config import PulseConfigError, load_pulse_config
from pulse_oximeter.analysis import CppgResults, analyze_cppg


LOGGER = logging.getLogger(__name__)


def _observed_harmonics(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    fundamental_hz: float,
) -> tuple[float | None, float | None]:
    """Return the observed 2nd and 3rd harmonic peaks nearest to 2f0 and 3f0."""
    if len(magnitude) == 0 or len(frequencies) == 0:
        return None, None

    peaks, _ = find_peaks(magnitude, distance=30)
    if len(peaks) == 0:
        return None, None

    peak_freqs = frequencies[peaks]
    harmonics: list[float | None] = []
    for multiplier in (2, 3):
        target_hz = multiplier * fundamental_hz
        valid = peak_freqs[np.abs(peak_freqs - target_hz) <= 0.35]
        if len(valid) == 0:
            harmonics.append(None)
        else:
            harmonics.append(float(valid[np.argmin(np.abs(valid - target_hz))]))

    return harmonics[0], harmonics[1]


def _plot_cppg_time_domain_spg_style(
    results: CppgResults,
    save_path: Path,
) -> None:
    """Save cPPG time-domain graph using the same visual style as spg_ppg.py."""
    import matplotlib.pyplot as plt

    time_axis = results.timestamps - results.timestamps[0]
    bpm = results.hr_time_domain_bpm or results.hr_frequency_domain_bpm
    bpm_text = f"Heart Rate: {bpm:.1f} BPM" if bpm is not None else "Heart Rate: N/A"

    plt.rcParams["figure.facecolor"] = "white"
    plt.figure(figsize=(10, 4))
    plt.plot(
        time_axis,
        results.filtered_signal,
        color="tab:red",
        linewidth=1.2,
        label="Filtered cPPG",
    )
    if len(results.peak_indices) > 0:
        plt.scatter(
            time_axis[results.peak_indices],
            results.filtered_signal[results.peak_indices],
            s=28,
            color="crimson",
            zorder=3,
            label=f"Peaks (n={len(results.peak_indices)})",
        )

    plt.title(
        f"cPPG Time Domain Signal ({bpm_text})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("Normalized Amplitude", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    if len(results.peak_indices) > 0:
        plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    LOGGER.info("Saved cPPG time-domain plot: %s", save_path)


def _plot_cppg_fft_spg_style(
    results: CppgResults,
    save_path: Path,
) -> None:
    """Save cPPG FFT graph using the same visual style as spg_ppg.py."""
    import matplotlib.pyplot as plt

    plt.rcParams["figure.facecolor"] = "white"
    plt.figure(figsize=(10, 5))
    plt.plot(
        results.fft_freqs,
        results.fft_magnitude,
        color="tab:red",
        linewidth=1.2,
        alpha=0.7,
        label="Magnitude Spectrum",
    )

    if results.fundamental_freq_hz is not None:
        bpm = results.fundamental_freq_hz * 60.0

        
        h2_obs, h3_obs = _observed_harmonics(
            results.fft_magnitude,
            results.fft_freqs,
            results.fundamental_freq_hz,
        )

        plt.axvline(
            results.fundamental_freq_hz,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Fundamental: {results.fundamental_freq_hz:.2f} Hz "
                f"({bpm:.1f} BPM)"
            ),
        )

        if h2_obs is not None:
            plt.axvline(
                h2_obs,
                color="darkorange",
                linestyle="--",
                linewidth=1.5,
                label=f"2nd Harmonic: {h2_obs:.2f} Hz (Obs)",
            )
        if h3_obs is not None:
            plt.axvline(
                h3_obs,
                color="forestgreen",
                linestyle="--",
                linewidth=1.5,
                label=f"3rd Harmonic: {h3_obs:.2f} Hz (Obs)",
            )

    plt.title(
        "cPPG FFT Frequency Spectrum Analysis",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Frequency (Hz)", fontsize=11)
    plt.ylabel("Magnitude", fontsize=11)
    plt.xlim(0, 4)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    LOGGER.info("Saved cPPG FFT spectrum plot: %s", save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record multispectral video and pulse-oximeter data simultaneously "
            "using a single Arduino trigger controller."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "config.yaml"),
        help="Path to config.yaml (default: project root config.yaml).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating cPPG analysis plots.",
    )
    parser.add_argument(
        "--mkv",
        action="store_true",
        help="Save video as lossless MKV (.mkv with FFV1 codec) to preserve real image data without loss.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Save video in lossy compressed format (.avi with MJPG) instead of the default lossless format.",
    )
    parser.add_argument(
        "--tiff",
        action="store_true",
        help="Save video as a sequence of lossless 16-bit TIFF images (highest quality, preserves bit depth).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save video as a raw binary memory dump (.raw) (fastest write speed, preserves bit depth).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal log output.",
    )
    return parser.parse_args()


def main() -> int:
    if sys.version_info < (3, 11):
        print("Python 3.11 or newer is required.", file=sys.stderr)
        return 2

    args = parse_args()

    # --- Load configs ---
    try:
        camera_config = load_config(args.config)
        pulse_config = load_pulse_config(args.config)

        if args.mkv:
            new_acq = replace(
                camera_config.acquisition,
                video_filename="recording.mkv",
                video_codec="FFV1",
            )
            camera_config = replace(camera_config, acquisition=new_acq)
        elif args.compressed:
            new_acq = replace(
                camera_config.acquisition,
                video_filename="recording.avi",
                video_codec="MJPG",
            )
            camera_config = replace(camera_config, acquisition=new_acq)
        elif args.tiff:
            new_acq = replace(
                camera_config.acquisition,
                video_filename="recording_tiff_sequence",
                video_codec="TIFF",
            )
            camera_config = replace(camera_config, acquisition=new_acq)
        elif args.raw:
            new_acq = replace(
                camera_config.acquisition,
                video_filename="recording.raw",
                video_codec="RAW",
            )
            camera_config = replace(camera_config, acquisition=new_acq)

    except (ConfigError, PulseConfigError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # --- Setup logging & Run acquisition ---
    try:
        runner = AcquisitionRunner(camera_config)
        configure_logging(runner.log_path, verbose=not args.quiet)

        LOGGER.info("=" * 60)
        LOGGER.info("MULTIMODAL RECORDING — Camera + Pulse Oximeter (Single-Board)")
        LOGGER.info("=" * 60)
        LOGGER.info("Duration: %g s", camera_config.acquisition.duration_s)
        LOGGER.info("Output:   %s", camera_config.acquisition.output_dir)

        stats: AcquisitionStats = runner.run()
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Acquisition failed: {exc}", file=sys.stderr)
        return 1

    # --- Report camera results ---
    print(f"\n{'=' * 60}")
    print("  MULTIMODAL RECORDING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Camera:")
    print(f"    Frames   : {stats.frames_captured}/{stats.expected_frames}")
    print(f"    FPS      : {stats.real_fps:.3f}")
    print(f"    Exposure : {stats.actual_exposure_us:.1f} \u00b5s")
    print(f"    Video    : {stats.video_path}")
    print(f"    CSV      : {stats.metadata_path}")

    if stats.channel_counts:
        print(f"    Channels :", end="")
        for ch_id in sorted(stats.channel_counts):
            print(f"  CH{ch_id}={stats.channel_counts[ch_id]}", end="")
        print()

    if stats.dropped_camera_frames > 0:
        print(f"  \u26a0 Dropped camera frames : {stats.dropped_camera_frames}")
    if stats.missing_serial_events > 0:
        print(f"  \u26a0 Missing serial events : {stats.missing_serial_events}")
    if stats.sync_mismatches > 0:
        print(f"  \u26a0 Sync mismatches       : {stats.sync_mismatches}")

    # --- Process and analyze pulse data ---
    # Use the CSV filename from pulse_config to stay in sync with the capture pipeline
    pulse_csv_path = camera_config.acquisition.output_dir / pulse_config.capture.csv_filename
    if pulse_csv_path.is_file():
        print(f"\n  Pulse Oximeter:")
        print(f"    CSV     : {pulse_csv_path}")

        # Read CSV data back for analysis
        timestamps = []
        ir_signal = []
        hr_values = []
        spo2_values = []

        try:
            with open(pulse_csv_path, "r", encoding="utf-8") as f:
                csv_reader = csv.reader(f)
                next(csv_reader, None)  # skip header
                for row in csv_reader:
                    if len(row) >= 5:
                        try:
                            try:
                                ts = datetime.fromisoformat(row[0]).timestamp()
                            except ValueError:
                                ts = float(row[0])
                            
                            ir_val = int(row[2])
                            hr_val = int(row[3])
                            spo2_val = int(row[4])
                            
                            timestamps.append(ts)
                            ir_signal.append(ir_val)
                            hr_values.append(hr_val)
                            spo2_values.append(spo2_val)
                        except (ValueError, TypeError):
                            continue  # Skip malformed rows

            if len(ir_signal) < 2:
                print("    Not enough samples for cPPG analysis.")
                return 0

            print(f"    Samples : {len(ir_signal)}")
            duration = timestamps[-1] - timestamps[0]
            if duration > 0:
                print(f"    Rate    : {(len(ir_signal)-1)/duration:.2f} Hz")
            else:
                print(f"    Rate    : N/A (Duration is 0)")

            # Use actual measured FPS from acquisition stats as the nominal rate;
            # fall back to config target_fps if real_fps is unavailable.
            nominal_rate = stats.real_fps if stats.real_fps > 0 else camera_config.acquisition.target_fps

            # Run analysis
            analysis = analyze_cppg(
                ir_signal,
                timestamps,
                pulse_config.analysis,
                nominal_rate=nominal_rate,
            )

            print(f"\n  cPPG Analysis Results:")
            if analysis.hr_time_domain_bpm is not None:
                print(f"    HR (time domain): {analysis.hr_time_domain_bpm:.2f} BPM")
            if analysis.hr_frequency_domain_bpm is not None:
                print(f"    HR (FFT domain) : {analysis.hr_frequency_domain_bpm:.2f} BPM")
            if hr_values:
                print(f"    Sensor HR Mean  : {sum(hr_values)/len(hr_values):.1f} BPM")
            if spo2_values:
                print(f"    Sensor SpO2 Mean: {sum(spo2_values)/len(spo2_values):.1f} %")

            # Save and display plots
            if not args.no_plot:
                plot_dir = camera_config.acquisition.output_dir
                plot_path = plot_dir / pulse_config.capture.plot_filename
                _plot_cppg_time_domain_spg_style(analysis, save_path=plot_path)

                fft_path = plot_path.with_name(plot_path.stem + "_fft" + plot_path.suffix)
                _plot_cppg_fft_spg_style(analysis, save_path=fft_path)
                print(f"\n    Plots saved to: {plot_dir}")

        except Exception as exc:
            print(f"  ⚠ Failed to analyze cPPG data: {exc}", file=sys.stderr)
            return 1
    else:
        print("\n  ⚠ No pulse data CSV file was generated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
