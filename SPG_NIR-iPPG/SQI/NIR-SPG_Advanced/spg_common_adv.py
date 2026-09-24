"""
SPG Pipeline — Shared Configuration & Utilities
===================================================
config file, helper functions, and data loading shared across all Steps for SPG
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter

# ==========================================
# CONFIGURATION — Change paths to match data
# ==========================================
VIDEO_PATH = r"../../data/input/recording_CH3_1064nm.mkv"
CPPG_PATH = r"../../data/input/pulse_data.csv"
DARK_FRAME_PATH = r"../../data/input/dark_frame_CH3.npy"

DATASET_NAME = os.path.basename(os.path.dirname(VIDEO_PATH))
OUT_DIR = os.path.join(r"../../results/spg", DATASET_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

# SPG / Signal processing parameters
SPATIAL_WINDOW = 7  # 7x7 sliding window for finding K
BP_LOW  = 0.5           # Hz
BP_HIGH = 5.0           # Hz
FILTER_ORDER = 4
N_FFT = 8192

# Time window for display (seconds)
TIME_START = 5.0
TIME_END   = 15.0

# Extract wavelength from filename
_match = re.search(r'(\d+nm)', os.path.basename(VIDEO_PATH), re.IGNORECASE)
WAVELENGTH = _match.group(1) if _match else "unknown"

# ==========================================
# COLORS
# ==========================================
COLORS = {
    'spg':    '#2ca02c',   # Green
    'cppg':   '#1f77b4',   # Blue
    'accent': '#ff7f0e',   # Orange
}

# ==========================================
# STYLE — Publication-quality
# ==========================================
def apply_style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif'],
        'font.size':         10,
        'axes.labelsize':    11,
        'axes.titlesize':    11,
        'legend.fontsize':   9,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'axes.linewidth':    0.8,
        'lines.linewidth':   1.2,
        'grid.linewidth':    0.4,
        'grid.alpha':        0.35,
        'legend.framealpha': 0.95,
        'legend.edgecolor':  '0.8',
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
    })


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def bandpass_filter(data: np.ndarray, fs: float,
                    low: float = BP_LOW, high: float = BP_HIGH,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)


def z_normalize(sig: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-12)


def compute_fft(signal: np.ndarray, fs: float, n_fft: int = N_FFT):
    """Compute FFT magnitude and frequency axis."""
    windowed = signal * np.hanning(len(signal))
    mag = np.abs(np.fft.rfft(windowed, n=n_fft))
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return freq, mag


def find_hr_peak(mag: np.ndarray, freq: np.ndarray,
                 f_low: float = BP_LOW, f_high: float = BP_HIGH):
    """Find the dominant frequency in the cardiac range → HR in BPM."""
    mask = (freq >= f_low) & (freq <= f_high)
    f_masked = freq[mask]
    m_masked = mag[mask]
    if len(m_masked) == 0:
        return 0.0, 0.0
    idx = np.argmax(m_masked)
    f_peak = f_masked[idx]
    return f_peak * 60.0, f_peak  # (BPM, Hz)


def compute_pearson_aligned(sig_a, t_a, sig_b, t_b, label: str = ""):
    """
    Align two signals on common time axis via interpolation,
    then compute Pearson correlation.
    """
    t_start = max(t_a[0], t_b[0])
    t_end = min(t_a[-1], t_b[-1])

    fs_common = max(len(t_a) / (t_a[-1] - t_a[0]),
                    len(t_b) / (t_b[-1] - t_b[0]))
    t_common = np.arange(t_start, t_end, 1.0 / fs_common)

    if len(t_common) < 10:
        print(f"  [{label}] ⚠️  Not enough overlap for correlation")
        return 0.0, 1.0

    a_interp = np.interp(t_common, t_a, sig_a)
    b_interp = np.interp(t_common, t_b, sig_b)

    a_z = z_normalize(a_interp)
    b_z = z_normalize(b_interp)

    r, p = pearsonr(a_z, b_z)
    print(f"  [{label}] Pearson r = {r:.4f}  (p = {p:.2e})")
    return r, p


# ==========================================
# DATA LOADING
# ==========================================

def load_video_spatial_contrast(video_path: str = VIDEO_PATH, w_size: int = SPATIAL_WINDOW, dark_frame_path: str = DARK_FRAME_PATH):
    """
    Load video and calculate Raw SPG Signal (1 / <K^2>)
    where K is Spatial Speckle Contrast (σ / μ) of window w_size
    Returns: (raw_spg, time_axis, fps)
    """
    print(f"  Video: {video_path}")
    dark_frame = None
    if dark_frame_path and os.path.exists(dark_frame_path):
        print(f"  Applying Dark Frame: {dark_frame_path}")
        dark_frame = np.load(dark_frame_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.1f}, Total frames: {total}")
    print(f"  Computing SPG Spatial Contrast (Window: {w_size}x{w_size})...")

    raw_spg_list = []
    frame_idx = 0
    start_frame = int(TIME_START * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < start_frame:
            frame_idx += 1
            continue
            
        # Use the entire frame as requested by the user
        frame_roi = frame
            
        if len(frame_roi.shape) == 3:
            pixel_data = frame_roi[:, :, 2].astype(float)
        else:
            pixel_data = frame_roi.astype(float)
            
        if dark_frame is not None:
            fh, fw = pixel_data.shape
            df_matched = dark_frame[:fh, :fw]
            pixel_data = np.maximum(pixel_data - df_matched, 0.0)
            
            
        # Fast local mean and variance using uniform_filter
        mean_val = uniform_filter(pixel_data, size=w_size)
        sq_mean_val = uniform_filter(pixel_data**2, size=w_size)
        var_val = np.maximum(sq_mean_val - mean_val**2, 0)
        
        # Spatial Speckle Contrast K = std / mean
        # K^2 = var / (mean^2)
        # SPG = 1 / <K^2>
        k_sq = var_val / (mean_val**2 + 1e-9)
        avg_k_sq = np.mean(k_sq)
        val = 1.0 / (avg_k_sq + 1e-9)
        
        raw_spg_list.append(val)
        frame_idx += 1
        
    cap.release()

    raw_spg = np.array(raw_spg_list)
    time_axis = np.arange(len(raw_spg)) / fps + TIME_START
    print(f"  Extracted {len(raw_spg)} frames (Central 20% ROI, starting {TIME_START}s) → {time_axis[-1]:.1f} s")

    return raw_spg, time_axis, fps

def load_cppg(csv_path: str = CPPG_PATH):
    """
    Load contact PPG data from the pulse oximeter CSV.
    Returns: (ir_raw, t_cppg, fs_cppg)
    """
    print(f"  cPPG: {csv_path}")
    df = pd.read_csv(csv_path)
    timestamps = pd.to_datetime(df['Timestamp'])
    t_cppg = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
    ir_raw = df['IR'].values.astype(float)
    fs_cppg = (len(t_cppg) - 1) / (t_cppg[-1] - t_cppg[0])
    print(f"  Samples: {len(ir_raw)}, Duration: {t_cppg[-1]:.1f} s, "
          f"Fs: {fs_cppg:.1f} Hz")
    return ir_raw, t_cppg, fs_cppg

# ==========================================
# EVALUATION METRICS
# ==========================================
from scipy.signal import find_peaks

def compute_snr(signal: np.ndarray, fs: float, f_low: float = BP_LOW, f_high: float = BP_HIGH) -> float:
    n_fft = 8192
    windowed = signal * np.hanning(len(signal))
    mag = np.abs(np.fft.rfft(windowed, n=n_fft))
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    
    mask = (freq >= f_low) & (freq <= f_high)
    f_masked = freq[mask]
    m_masked = mag[mask]
    
    if len(m_masked) == 0:
        return 0.0
        
    peak_idx = np.argmax(m_masked)
    f_peak = f_masked[peak_idx]
    
    # Signal band: f_peak ± 0.2 Hz
    sig_mask = (f_masked >= f_peak - 0.2) & (f_masked <= f_peak + 0.2)
    
    power_total = np.sum(m_masked**2)
    power_sig = np.sum(m_masked[sig_mask]**2)
    power_noise = power_total - power_sig
    
    if power_noise <= 0:
        return 100.0
        
    snr_linear = power_sig / power_noise
    snr_db = 10 * np.log10(snr_linear + 1e-12)
    return snr_db

def compute_hr_error(signal: np.ndarray, fs: float, ref_hr: float, f_low: float = BP_LOW, f_high: float = BP_HIGH) -> float:
    n_fft = 8192
    windowed = signal * np.hanning(len(signal))
    mag = np.abs(np.fft.rfft(windowed, n=n_fft))
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    
    mask = (freq >= f_low) & (freq <= f_high)
    f_masked = freq[mask]
    m_masked = mag[mask]
    
    if len(m_masked) == 0:
        return 999.0
        
    f_peak = f_masked[np.argmax(m_masked)]
    hr_bpm = f_peak * 60.0
    return abs(hr_bpm - ref_hr)

def compute_peak_consistency(signal: np.ndarray, fs: float) -> float:
    distance = int(0.25 * fs) # HR max 240 BPM
    peaks, _ = find_peaks(signal, distance=distance, prominence=np.std(signal)*0.3)
    
    if len(peaks) < 2:
        return 999.0
        
    intervals = np.diff(peaks) / fs
    return np.std(intervals)

def compute_waveform_smoothness(signal: np.ndarray) -> float:
    d2 = np.diff(signal, n=2)
    var_d2 = np.var(d2)
    if var_d2 == 0:
        return 0.0
    return 1.0 / (var_d2 + 1e-12)

def evaluate_candidates(results_list: list) -> list:
    valid_results = [r for r in results_list if r['ok']]
    if not valid_results:
        return results_list
        
    r_arr = np.array([res['r'] for res in valid_results])
    snr_arr = np.array([res['snr'] for res in valid_results])
    hr_err_arr = np.array([res['hr_error'] for res in valid_results])
    pk_cons_arr = np.array([res['peak_consistency'] for res in valid_results])
    smooth_arr = np.array([res['smoothness'] for res in valid_results])
    
    def min_max_scale(arr, invert=False):
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v - min_v == 0:
            return np.ones_like(arr) if not invert else np.zeros_like(arr)
        scaled = (arr - min_v) / (max_v - min_v)
        return 1.0 - scaled if invert else scaled
        
    r_norm = min_max_scale(r_arr)
    snr_norm = min_max_scale(snr_arr)
    smooth_norm = min_max_scale(smooth_arr)
    hr_err_norm = min_max_scale(hr_err_arr, invert=True)
    pk_cons_norm = min_max_scale(pk_cons_arr, invert=True)
    
    # Combine scores (Max possible = 5.0)
    scores = r_norm + snr_norm + smooth_norm + hr_err_norm + pk_cons_norm
    
    for i, res in enumerate(valid_results):
        res['score'] = scores[i]
        
    return results_list

from scipy.signal import find_peaks
def detect_valid_peaks(sig, fs, ref_hr):
    """
    Peak refinement pipeline.

    Produces ONE validated peak set for:
        HR
        IBI
        beat segmentation
        morphology

    Troughs are detected BETWEEN consecutive valid peaks
    using a physiologically constrained search region.
    """

    log = {
        'steps': [],
        'n_peaks': 0,
        'n_troughs': 0,
        'expected_peaks': 0,
        'peak_ratio': 0.0,
        'ibi_mean': 0.0,
        'ibi_std': 0.0,
        'ibi_cv': 0.0,
        'ibi': np.array([]),
        'valid': False
    }

    n_samples = len(sig)
    duration = n_samples / fs

    # ============================================================
    # [1] INITIAL PEAK DETECTION
    # ============================================================

    min_distance_initial = int(0.20 * fs)
    base_prom = np.std(sig) * 0.10

    peaks_raw, _ = find_peaks(
        sig,
        distance=min_distance_initial,
        prominence=base_prom
    )

    log['steps'].append(
        f"[1] Initial peaks: {len(peaks_raw)}"
    )

    if len(peaks_raw) < 3:
        log['valid'] = False
        log['reason'] = 'Too few initial peaks'
        return (
            np.array(peaks_raw, dtype=int),
            np.array([], dtype=int),
            log
        )

    # ============================================================
    # [2] MINIMUM DISTANCE
    # ============================================================

    expected_ibi = 60.0 / ref_hr if ref_hr > 0 else 0.8

    # Do not allow unrealistically short cardiac intervals
    min_ibi = max(
        0.30,
        expected_ibi * 0.40
    )

    peaks_dist = [int(peaks_raw[0])]

    for pk in peaks_raw[1:]:

        pk = int(pk)

        gap = (pk - peaks_dist[-1]) / fs

        if gap >= min_ibi:

            peaks_dist.append(pk)

        else:

            # Keep stronger peak
            if sig[pk] > sig[peaks_dist[-1]]:
                peaks_dist[-1] = pk

    peaks_dist = np.asarray(peaks_dist, dtype=int)

    log['steps'].append(
        f"[2] After min distance "
        f"({min_ibi * 1000:.0f}ms): "
        f"{len(peaks_dist)}"
    )

    # ============================================================
    # [3] ADAPTIVE PROMINENCE
    # ============================================================

    if len(peaks_dist) > 0:

        window = max(1, int(fs))

        peak_proms = []

        for pk in peaks_dist:

            left = max(0, pk - window)
            right = min(n_samples, pk + window)

            local_min = np.min(sig[left:right])

            prom = sig[pk] - local_min

            peak_proms.append(prom)

        peak_proms = np.asarray(peak_proms)

        median_prom = np.median(peak_proms)

        prom_threshold = max(
            np.std(sig) * 0.15,
            median_prom * 0.35
        )

        peaks_prom = peaks_dist[
            peak_proms >= prom_threshold
        ]

    else:

        peaks_prom = peaks_dist

    log['steps'].append(
        f"[3] After adaptive prominence: "
        f"{len(peaks_prom)}"
    )

    # ============================================================
    # [4] IBI VALIDATION
    # ============================================================

    if len(peaks_prom) > 3:

        ibi = np.diff(peaks_prom) / fs

        ibi_median = np.median(ibi)

        ibi_mad = np.median(
            np.abs(ibi - ibi_median)
        )

        # Avoid zero MAD
        if ibi_mad < 1e-6:

            ibi_lo = ibi_median * 0.50
            ibi_hi = ibi_median * 1.50

        else:

            robust_sigma = 1.4826 * ibi_mad

            ibi_lo = max(
                ibi_median * 0.50,
                ibi_median - 3.0 * robust_sigma
            )

            ibi_hi = min(
                ibi_median * 1.50,
                ibi_median + 3.0 * robust_sigma
            )

        keep = [0]

        for i, interval in enumerate(ibi):

            if ibi_lo <= interval <= ibi_hi:
                keep.append(i + 1)

        peaks_ibi = peaks_prom[keep]

    else:

        peaks_ibi = peaks_prom

    log['steps'].append(
        f"[4] After IBI validation: "
        f"{len(peaks_ibi)}"
    )

    # ============================================================
    # [5] SECONDARY PEAK REMOVAL
    # ============================================================

    if len(peaks_ibi) > 2:

        cleaned = [int(peaks_ibi[0])]

        for pk in peaks_ibi[1:]:

            pk = int(pk)

            gap = (pk - cleaned[-1]) / fs

            # Secondary peak inside same cardiac cycle
            if gap < expected_ibi * 0.60:

                if sig[pk] > sig[cleaned[-1]]:
                    cleaned[-1] = pk

            else:

                cleaned.append(pk)

        peaks_clean = np.asarray(
            cleaned,
            dtype=int
        )

    else:

        peaks_clean = np.asarray(
            peaks_ibi,
            dtype=int
        )

    log['steps'].append(
        f"[5] After secondary removal: "
        f"{len(peaks_clean)}"
    )

    # ============================================================
    # [6] PEAK COUNT VALIDATION
    # ============================================================

    expected_n = int(
        duration * ref_hr / 60.0
    ) if ref_hr > 0 else 0

    ratio = (
        len(peaks_clean) / expected_n
        if expected_n > 0 else 0
    )

    log['expected_peaks'] = expected_n
    log['peak_ratio'] = ratio

    log['steps'].append(
        f"[6] Count: {len(peaks_clean)} "
        f"(expected ~{expected_n}, "
        f"ratio={ratio:.2f})"
    )

    # ============================================================
    # HARD PEAK COUNT GATE
    # ============================================================

    if expected_n > 0:

        if ratio < 0.85 or ratio > 1.15:

            log['valid'] = False
            log['reason'] = (
                f'Peak count ratio out of range: {ratio:.2f}'
            )

            return (
                peaks_clean,
                np.array([], dtype=int),
                log
            )

    # ============================================================
    # [7] VALID PEAKS
    # ============================================================

    valid_peaks = peaks_clean.copy()

    # ============================================================
    # [8] PHYSIOLOGICALLY CONSTRAINED TROUGHS
    # ============================================================
    #
    # IMPORTANT:
    # Do NOT search the entire peak-to-peak interval.
    #
    # Search around the middle of the cardiac cycle,
    # avoiding the immediate neighborhood of both peaks.
    #
    # This prevents small local dips from becoming troughs.
    # ============================================================

    valid_troughs = []

    for i in range(len(valid_peaks) - 1):

        p1 = valid_peaks[i]
        p2 = valid_peaks[i + 1]

        interval = p2 - p1

        if interval <= 0:
            continue

        # Ignore edges near peaks
        margin = int(
            0.20 * interval
        )

        left = p1 + margin
        right = p2 - margin

        if right <= left:
            continue

        segment = sig[left:right]

        if len(segment) == 0:
            continue

        # Lowest point inside constrained region
        trough = left + int(
            np.argmin(segment)
        )

        valid_troughs.append(trough)

    valid_troughs = np.asarray(
        valid_troughs,
        dtype=int
    )

    # ============================================================
    # MATCH PEAKS AND TROUGHS
    # ============================================================
    #
    # One morphology beat requires:
    #
    # trough[i]
    #      ↓
    # peak[i+1]
    #      ↓
    # trough[i+1]
    #
    # Therefore use ONLY complete cycles.
    # ============================================================

    if len(valid_peaks) >= 2:

        n_beats = min(
            len(valid_troughs),
            len(valid_peaks) - 1
        )

        valid_troughs = valid_troughs[:n_beats]

    # ============================================================
    # FINAL IBI
    # ============================================================

    ibi_final = (
        np.diff(valid_peaks) / fs
        if len(valid_peaks) > 1
        else np.array([])
    )

    # ============================================================
    # LOG
    # ============================================================

    log['n_peaks'] = len(valid_peaks)

    log['n_troughs'] = len(valid_troughs)

    log['ibi'] = ibi_final

    log['ibi_mean'] = (
        float(np.mean(ibi_final))
        if len(ibi_final) > 0
        else 0
    )

    log['ibi_std'] = (
        float(np.std(ibi_final))
        if len(ibi_final) > 0
        else 0
    )

    log['ibi_cv'] = _cv(ibi_final)

    log['valid'] = (
        len(valid_peaks) >= 3
        and len(valid_troughs) >= 2
    )

    log['steps'].append(
        f"[7] VALID PEAKS: "
        f"{len(valid_peaks)} peaks, "
        f"{len(valid_troughs)} troughs"
    )

    return (
        valid_peaks,
        valid_troughs,
        log
    )


def physiological_check(sig, fs, duration, ref_hr, ref_fpeak):
    """
    Run 7-step peak refinement + HR/frequency gating.
    Returns (passed, info, reject_reasons)
    """
    reasons = []

    peaks, troughs, plog = detect_valid_peaks(sig, fs, ref_hr)

    info = {
        'peaks':          peaks,
        'troughs':        troughs,
        'n_peaks':        plog['n_peaks'],
        'n_troughs':      plog['n_troughs'],
        'expected_peaks': plog.get('expected_peaks', 0),
        'peak_ratio':     plog.get('peak_ratio', 0),
        'ibi':            plog['ibi'],
        'ibi_mean':       plog['ibi_mean'],
        'ibi_std':        plog['ibi_std'],
        'ibi_cv':         plog['ibi_cv'],
        'peak_log':       plog,
    }

    if not plog['valid']:
        reasons.append(plog.get('reason', 'Peak detection failed'))

    # FFT-based HR & frequency
    freq, mag = compute_fft(sig, fs)
    hr_fft, fpeak_hz = find_hr_peak(mag, freq)
    info['hr_fft']    = hr_fft
    info['fpeak_hz']  = fpeak_hz
    info['hr_diff']   = abs(hr_fft - ref_hr)
    info['freq_diff'] = abs(fpeak_hz - ref_fpeak)

    if hr_fft < HR_MIN or hr_fft > HR_MAX:
        reasons.append(f"HR {hr_fft:.1f} outside [{HR_MIN}-{HR_MAX}]")
    if info['hr_diff'] > HR_AGREE_MAX:
        reasons.append(f"HR diff {info['hr_diff']:.1f} > {HR_AGREE_MAX}")
    if info['freq_diff'] > FREQ_AGREE_MAX:
        reasons.append(f"Freq diff {info['freq_diff']:.3f} > {FREQ_AGREE_MAX}")
    if info['ibi_cv'] > IBI_CV_MAX:
        reasons.append(f"IBI CV {info['ibi_cv']:.1f}% > {IBI_CV_MAX}%")

    # Prominence stats from valid peaks
    if len(peaks) > 0:
        window = int(fs)
        prom_vals = np.array([
            sig[pk] - np.min(sig[max(0, pk - window):pk + window])
            for pk in peaks
        ])
        info['prom_mean'] = float(np.mean(prom_vals))
        info['prom_cv']   = _cv(prom_vals)
    else:
        info['prom_mean'] = 0
        info['prom_cv']   = 999

    passed = (len(reasons) == 0)
    return passed, info, reasons


# ╔══════════════════════════════════════════════════════════════════╗
# ║  QUALITY EVALUATION  (9 metrics, weighted composite)           ║
# ╚══════════════════════════════════════════════════════════════════╝
def evaluate_quality(cand, t_sig, cppg_ref, t_cppg, ref_hr, ref_fpeak):
    """Compute 9 quality metrics for a candidate that PASSED the gate."""
    sig  = cand['signal']
    bl   = cand['baseline']
    fs   = cand['fs']
    pi   = cand['phys']

    # 1. HR Agreement (lower error = better)
    cand['q_hr_agree'] = pi['hr_diff']

    # 2. Peak Quality (prominence consistency: 1 - CV/100)
    cand['q_peak_quality'] = max(0.0, 1.0 - pi['prom_cv'] / 100.0)

    # 3. IBI Consistency (1 - CV/100)
    cand['q_ibi_consist'] = max(0.0, 1.0 - pi['ibi_cv'] / 100.0)

    # 4. Beat Consistency (std of IBI, lower = more regular)
    cand['q_beat_consist'] = pi['ibi_std']

    # 5. Morphology Preservation (corr with BPF baseline)
    r_morph, _ = pearsonr(z_normalize(sig), z_normalize(bl))
    cand['q_morphology'] = max(0.0, r_morph)

    # 6. Pearson r vs cPPG
    cand['q_pearson'] = pearson_aligned(z_normalize(sig), t_sig,
                                        z_normalize(cppg_ref), t_cppg)

    # 7. Frequency Quality (1 / (1 + freq_diff))
    cand['q_freq_quality'] = 1.0 / (1.0 + pi['freq_diff'])

    # 8. SNR
    freq, mag = compute_fft(sig, fs)
    mask = (freq >= BP_LOW) & (freq <= BP_HIGH)
    if np.any(mask):
        sp = np.max(mag[mask]) ** 2
        np_ = np.mean(mag[~mask] ** 2) + 1e-12
        cand['q_snr'] = 10 * np.log10(sp / np_)
    else:
        cand['q_snr'] = -99.0

    # 9. Smoothness (second-derivative energy, lower = smoother)
    d2 = np.diff(sig, n=2)
    cand['q_smoothness'] = np.mean(d2 ** 2)


def compute_composite_scores(candidates):
    """Weighted composite score using min-max normalization."""
    valid = [c for c in candidates if c.get('gate_pass')]
    if not valid:
        return

    # Gather raw metric arrays
    hr_arr    = np.array([c['q_hr_agree']     for c in valid])
    pq_arr    = np.array([c['q_peak_quality'] for c in valid])
    ibi_arr   = np.array([c['q_ibi_consist']  for c in valid])
    bc_arr    = np.array([c['q_beat_consist']  for c in valid])
    mo_arr    = np.array([c['q_morphology']    for c in valid])
    pr_arr    = np.array([c['q_pearson']       for c in valid])
    fq_arr    = np.array([c['q_freq_quality']  for c in valid])
    snr_arr   = np.array([c['q_snr']           for c in valid])
    sm_arr    = np.array([c['q_smoothness']    for c in valid])

    # Normalize 0-1  (invert where lower is better)
    scores = (
        W_HR_AGREE     * min_max_scale(hr_arr,  invert=True)   # lower error = better
      + W_PEAK_QUALITY * min_max_scale(pq_arr)                  # higher = better
      + W_IBI_CONSIST  * min_max_scale(ibi_arr)                 # higher = better
      + W_BEAT_CONSIST * min_max_scale(bc_arr,  invert=True)   # lower std = better
      + W_MORPHOLOGY   * min_max_scale(mo_arr)                  # higher = better
      + W_PEARSON      * min_max_scale(pr_arr)                  # higher = better
      + W_FREQ_QUALITY * min_max_scale(fq_arr)                  # higher = better
      + W_SNR          * min_max_scale(snr_arr)                 # higher = better
      + W_SMOOTHNESS   * min_max_scale(sm_arr,  invert=True)   # lower = better
    )

    for i, c in enumerate(valid):
        c['score'] = scores[i]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  BEAT SEGMENTATION & ANALYSIS                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def segment_beats(signal, t, peaks, troughs, fs):
    """
    Segment signal into individual cardiac cycles (Trough → Trough).
    Calculate Rise Time (Trough → Peak) and Pulse Width (Trough → Trough).
    Normalize each beat to 0-100% and baseline/amplitude 0-1.
    """
    beats = []

    for i in range(len(troughs) - 1):
        tr_start = troughs[i]
        tr_end   = troughs[i + 1]

        beat_sig = signal[tr_start:tr_end]
        beat_t   = t[tr_start:tr_end] - t[tr_start]

        if len(beat_sig) < 4:
            continue

        # Find the peak inside this beat
        pk_candidates = peaks[(peaks > tr_start) & (peaks < tr_end)]
        if len(pk_candidates) == 0:
            # Fallback if no valid peak inside: find max
            pk_rel = np.argmin(beat_sig)
            pk_abs = tr_start + pk_rel
        else:
            pk_abs = pk_candidates[0]
            pk_rel = pk_abs - tr_start

        # Normalize time to 0-100%
        beat_pct = np.linspace(0, 100, len(beat_sig))

        # Normalize amplitude to 0-1
        mn, mx = beat_sig.min(), beat_sig.max()
        beat_norm = (beat_sig - mn) / (mx - mn + 1e-12)

        # 3. Morphology Features
        amplitude   = mx - mn
        rise_time   = beat_t[pk_rel] # From trough to peak
        pulse_width = beat_t[-1]     # Full beat duration

        # Area above baseline (straight line from tr_start to tr_end)
        baseline = np.linspace(beat_sig[0], beat_sig[-1], len(beat_sig))
        area = np.trapezoid(np.maximum(0, beat_sig - baseline), beat_t)

        beats.append({
            'signal':     beat_sig,
            'time':       beat_t,
            'pct':        beat_pct,
            'normalized': beat_norm,
            'amplitude':  amplitude,
            'rise_time':  rise_time,
            'pulse_width': pulse_width,
            'area':       area,
            'tr_start':   tr_start,
            'tr_end':     tr_end,
            'pk_rel':     pk_rel,
            'valid':      True
        })

    return beats


def beat_level_analysis(beats):
    """
    1. Compute mean beat.
    2. Compute correlation of each beat to mean beat.
    3. Filter abnormal beats (corr < 0.8).
    4. Recalculate morphology stats.
    """
    if not beats:
        return {}

    # Resample all beats to same length (100 points) to compare shape
    n_resample = 100
    resampled = []
    for b in beats:
        x = np.interp(np.linspace(0, 100, n_resample), b['pct'], b['normalized'])
        resampled.append(x)
    resampled = np.array(resampled)

    # Initial mean beat
    mean_beat_initial = np.mean(resampled, axis=0)

    # Calculate beat consistency (Pearson r vs mean beat)
    correlations = []
    for i, r in enumerate(resampled):
        c, _ = pearsonr(r, mean_beat_initial)
        correlations.append(c)
        beats[i]['correlation'] = c
    correlations = np.array(correlations)

    # Robust outlier rejection using MAD
    amps  = np.array([b['amplitude']   for b in beats])
    rts   = np.array([b['rise_time']   for b in beats])
    pws   = np.array([b['pulse_width'] for b in beats])
    areas = np.array([b['area']        for b in beats])

    def mad_mask(arr, threshold=3.5):
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-6:
            return np.ones(len(arr), dtype=bool)
        z = np.abs(arr - median) / (mad * 1.4826)
        return z < threshold

    mask = (
        mad_mask(amps, 3.5) &
        mad_mask(rts, 3.5) &
        mad_mask(pws, 3.5) &
        mad_mask(areas, 3.5) &
        (correlations > 0.6)
    )

    valid_indices = np.where(mask)[0]

    # Do not drop more than 15% of beats
    if len(valid_indices) < len(beats) * 0.85 or len(valid_indices) < 3:
        n_keep = max(3, int(len(beats) * 0.85))
        valid_indices = np.argsort(correlations)[-n_keep:]

    valid_beats = [beats[i] for i in valid_indices]

    # Re-extract features from VALID beats only
    amps_v  = np.array([b['amplitude']   for b in valid_beats])
    rts_v   = np.array([b['rise_time']   for b in valid_beats])
    pws_v   = np.array([b['pulse_width'] for b in valid_beats])
    areas_v = np.array([b['area']        for b in valid_beats])

    # Final resampled shapes for valid beats
    valid_resampled = resampled[valid_indices]
    final_mean_beat = np.mean(valid_resampled, axis=0) if len(valid_resampled) > 0 else mean_beat_initial

    return {
        'n_beats_total':    len(beats),
        'n_beats_valid':    len(valid_beats),
        'beat_consistency': np.mean([b['correlation'] for b in valid_beats]) if valid_beats else 0,
        'amp_mean':         np.mean(amps_v) if len(amps_v) > 0 else 0,
        'amp_std':          np.std(amps_v) if len(amps_v) > 0 else 0,
        'amp_cv':           _cv(amps_v),
        'rt_mean':          np.mean(rts_v) if len(rts_v) > 0 else 0,
        'rt_std':           np.std(rts_v) if len(rts_v) > 0 else 0,
        'rt_cv':            _cv(rts_v),
        'pw_mean':          np.mean(pws_v) if len(pws_v) > 0 else 0,
        'pw_std':           np.std(pws_v) if len(pws_v) > 0 else 0,
        'pw_cv':            _cv(pws_v),
        'area_mean':        np.mean(areas_v) if len(areas_v) > 0 else 0,
        'area_std':         np.std(areas_v) if len(areas_v) > 0 else 0,
        'area_cv':          _cv(areas_v),
        'mean_beat':        final_mean_beat,
        'resampled_beats':  valid_resampled,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

HR_MIN = 40
HR_MAX = 200

def _cv(arr):
    if len(arr) == 0: return 0.0
    m = np.mean(arr)
    if m == 0: return 0.0
    return float(np.std(arr) / m * 100.0)


HR_AGREE_MAX = 5.0
PEAK_RATIO_MIN = 0.85
PEAK_RATIO_MAX = 1.15



BP_LOW    = 0.5
BP_HIGH   = 5.0
N_FFT     = 8192
TIME_START = 5.0
TIME_END   = 15.0

BW_ORDERS    = [1, 2, 3, 4, 5, 6]
SG_WINDOWS   = [5, 7, 9, 11]
SG_POLYORDER = 3

# Physiological Gate thresholds
HR_MIN       = 40.0     # BPM
HR_MAX       = 200.0    # BPM
IBI_CV_MAX   = 60.0     # %
PEAK_RATIO_MIN = 0.3    # min ratio of actual vs expected peaks
PEAK_RATIO_MAX = 4.0    # max ratio
HR_AGREE_MAX   = 30.0   # BPM — max allowable HR diff vs cPPG
FREQ_AGREE_MAX = 0.5    # Hz  — max allowable dominant freq diff vs cPPG

# Composite score weights
#   HIGH = 2.0,  LOW = 1.0
W_HR_AGREE     = 2.0
W_PEAK_QUALITY = 2.0
W_IBI_CONSIST  = 2.0
W_BEAT_CONSIST = 2.0
W_MORPHOLOGY   = 2.0
W_PEARSON      = 2.0
W_FREQ_QUALITY = 2.0
W_SNR          = 1.0
W_SMOOTHNESS   = 1.0
WEIGHT_SUM = (W_HR_AGREE + W_PEAK_QUALITY + W_IBI_CONSIST + W_BEAT_CONSIST
              + W_MORPHOLOGY + W_PEARSON + W_FREQ_QUALITY + W_SNR + W_SMOOTHNESS)

_match = re.search(r'(\d+nm)', os.path.basename(VIDEO_PATH), re.IGNORECASE)
WAVELENGTH = _match.group(1) if _match else "unknown"

COLORS = {
    'ippg':   '#d62728',
    'cppg':   '#1f77b4',
    'accent': '#ff7f0e',
    'best':   '#2ca02c',
}


def evaluate_quality(cand, t_sig, cppg_ref, t_cppg, ref_hr, ref_fpeak):
    """Compute 9 quality metrics for a candidate that PASSED the gate."""
    sig  = cand['signal']
    bl   = cand['baseline']
    fs   = cand['fs']
    pi   = cand['phys']

    # 1. HR Agreement (lower error = better)
    cand['q_hr_agree'] = pi['hr_diff']

    # 2. Peak Quality (prominence consistency: 1 - CV/100)
    cand['q_peak_quality'] = max(0.0, 1.0 - pi['prom_cv'] / 100.0)

    # 3. IBI Consistency (1 - CV/100)
    cand['q_ibi_consist'] = max(0.0, 1.0 - pi['ibi_cv'] / 100.0)

    # 4. Beat Consistency (std of IBI, lower = more regular)
    cand['q_beat_consist'] = pi['ibi_std']

    # 5. Morphology Preservation (corr with BPF baseline)
    r_morph, _ = pearsonr(z_normalize(sig), z_normalize(bl))
    cand['q_morphology'] = max(0.0, r_morph)

    # 6. Pearson r vs cPPG
    cand['q_pearson'], _ = compute_pearson_aligned(z_normalize(sig), t_sig, z_normalize(cppg_ref), t_cppg, label='')

    # 7. Frequency Quality (1 / (1 + freq_diff))
    cand['q_freq_quality'] = 1.0 / (1.0 + pi['freq_diff'])

    # 8. SNR
    freq, mag = compute_fft(sig, fs)
    mask = (freq >= BP_LOW) & (freq <= BP_HIGH)
    if np.any(mask):
        sp = np.max(mag[mask]) ** 2
        np_ = np.mean(mag[~mask] ** 2) + 1e-12
        cand['q_snr'] = 10 * np.log10(sp / np_)
    else:
        cand['q_snr'] = -99.0

    # 9. Smoothness (second-derivative energy, lower = smoother)
    d2 = np.diff(sig, n=2)
    cand['q_smoothness'] = np.mean(d2 ** 2)


def compute_composite_scores(candidates):
    """Weighted composite score using min-max normalization."""
    valid = [c for c in candidates if c.get('gate_pass')]
    if not valid:
        return

    # Gather raw metric arrays
    hr_arr    = np.array([c['q_hr_agree']     for c in valid])
    pq_arr    = np.array([c['q_peak_quality'] for c in valid])
    ibi_arr   = np.array([c['q_ibi_consist']  for c in valid])
    bc_arr    = np.array([c['q_beat_consist']  for c in valid])
    mo_arr    = np.array([c['q_morphology']    for c in valid])
    pr_arr    = np.array([c['q_pearson']       for c in valid])
    fq_arr    = np.array([c['q_freq_quality']  for c in valid])
    snr_arr   = np.array([c['q_snr']           for c in valid])
    sm_arr    = np.array([c['q_smoothness']    for c in valid])

    # Normalize 0-1  (invert where lower is better)
    scores = (
        W_HR_AGREE     * min_max_scale(hr_arr,  invert=True)   # lower error = better
      + W_PEAK_QUALITY * min_max_scale(pq_arr)                  # higher = better
      + W_IBI_CONSIST  * min_max_scale(ibi_arr)                 # higher = better
      + W_BEAT_CONSIST * min_max_scale(bc_arr,  invert=True)   # lower std = better
      + W_MORPHOLOGY   * min_max_scale(mo_arr)                  # higher = better
      + W_PEARSON      * min_max_scale(pr_arr)                  # higher = better
      + W_FREQ_QUALITY * min_max_scale(fq_arr)                  # higher = better
      + W_SNR          * min_max_scale(snr_arr)                 # higher = better
      + W_SMOOTHNESS   * min_max_scale(sm_arr,  invert=True)   # lower = better
    )

    for i, c in enumerate(valid):
        c['score'] = scores[i]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  BEAT SEGMENTATION & ANALYSIS                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
def segment_beats(signal, t, peaks, troughs, fs):
    """
    Segment signal into individual cardiac cycles (Trough → Trough).
    Calculate Rise Time (Trough → Peak) and Pulse Width (Trough → Trough).
    Normalize each beat to 0-100% and baseline/amplitude 0-1.
    """
    beats = []

    for i in range(len(troughs) - 1):
        tr_start = troughs[i]
        tr_end   = troughs[i + 1]

        beat_sig = signal[tr_start:tr_end]
        beat_t   = t[tr_start:tr_end] - t[tr_start]

        if len(beat_sig) < 4:
            continue

        # Find the peak inside this beat
        pk_candidates = peaks[(peaks > tr_start) & (peaks < tr_end)]
        if len(pk_candidates) == 0:
            # Fallback if no valid peak inside: find max
            pk_rel = np.argmin(beat_sig)
            pk_abs = tr_start + pk_rel
        else:
            pk_abs = pk_candidates[0]
            pk_rel = pk_abs - tr_start

        # Normalize time to 0-100%
        beat_pct = np.linspace(0, 100, len(beat_sig))

        # Normalize amplitude to 0-1
        mn, mx = beat_sig.min(), beat_sig.max()
        beat_norm = (beat_sig - mn) / (mx - mn + 1e-12)

        # 3. Morphology Features
        amplitude   = mx - mn
        rise_time   = beat_t[pk_rel] # From trough to peak
        pulse_width = beat_t[-1]     # Full beat duration

        # Area above baseline (straight line from tr_start to tr_end)
        baseline = np.linspace(beat_sig[0], beat_sig[-1], len(beat_sig))
        area = np.trapezoid(np.maximum(0, beat_sig - baseline), beat_t)

        beats.append({
            'signal':     beat_sig,
            'time':       beat_t,
            'pct':        beat_pct,
            'normalized': beat_norm,
            'amplitude':  amplitude,
            'rise_time':  rise_time,
            'pulse_width': pulse_width,
            'area':       area,
            'tr_start':   tr_start,
            'tr_end':     tr_end,
            'pk_rel':     pk_rel,
            'valid':      True
        })

    return beats


def beat_level_analysis(beats):
    """
    1. Compute mean beat.
    2. Compute correlation of each beat to mean beat.
    3. Filter abnormal beats (corr < 0.8).
    4. Recalculate morphology stats.
    """
    if not beats:
        return {}

    # Resample all beats to same length (100 points) to compare shape
    n_resample = 100
    resampled = []
    for b in beats:
        x = np.interp(np.linspace(0, 100, n_resample), b['pct'], b['normalized'])
        resampled.append(x)
    resampled = np.array(resampled)

    # Initial mean beat
    mean_beat_initial = np.mean(resampled, axis=0)

    # Calculate beat consistency (Pearson r vs mean beat)
    correlations = []
    for i, r in enumerate(resampled):
        c, _ = pearsonr(r, mean_beat_initial)
        correlations.append(c)
        beats[i]['correlation'] = c
    correlations = np.array(correlations)

    # Robust outlier rejection using MAD
    amps  = np.array([b['amplitude']   for b in beats])
    rts   = np.array([b['rise_time']   for b in beats])
    pws   = np.array([b['pulse_width'] for b in beats])
    areas = np.array([b['area']        for b in beats])

    def mad_mask(arr, threshold=3.5):
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-6:
            return np.ones(len(arr), dtype=bool)
        z = np.abs(arr - median) / (mad * 1.4826)
        return z < threshold

    mask = (
        mad_mask(amps, 3.5) &
        mad_mask(rts, 3.5) &
        mad_mask(pws, 3.5) &
        mad_mask(areas, 3.5) &
        (correlations > 0.6)
    )

    valid_indices = np.where(mask)[0]

    # Do not drop more than 15% of beats
    if len(valid_indices) < len(beats) * 0.85 or len(valid_indices) < 3:
        n_keep = max(3, int(len(beats) * 0.85))
        valid_indices = np.argsort(correlations)[-n_keep:]

    valid_beats = [beats[i] for i in valid_indices]

    # Re-extract features from VALID beats only
    amps_v  = np.array([b['amplitude']   for b in valid_beats])
    rts_v   = np.array([b['rise_time']   for b in valid_beats])
    pws_v   = np.array([b['pulse_width'] for b in valid_beats])
    areas_v = np.array([b['area']        for b in valid_beats])

    # Final resampled shapes for valid beats
    valid_resampled = resampled[valid_indices]
    final_mean_beat = np.mean(valid_resampled, axis=0) if len(valid_resampled) > 0 else mean_beat_initial

    return {
        'n_beats_total':    len(beats),
        'n_beats_valid':    len(valid_beats),
        'beat_consistency': np.mean([b['correlation'] for b in valid_beats]) if valid_beats else 0,
        'amp_mean':         np.mean(amps_v) if len(amps_v) > 0 else 0,
        'amp_std':          np.std(amps_v) if len(amps_v) > 0 else 0,
        'amp_cv':           _cv(amps_v),
        'rt_mean':          np.mean(rts_v) if len(rts_v) > 0 else 0,
        'rt_std':           np.std(rts_v) if len(rts_v) > 0 else 0,
        'rt_cv':            _cv(rts_v),
        'pw_mean':          np.mean(pws_v) if len(pws_v) > 0 else 0,
        'pw_std':           np.std(pws_v) if len(pws_v) > 0 else 0,
        'pw_cv':            _cv(pws_v),
        'area_mean':        np.mean(areas_v) if len(areas_v) > 0 else 0,
        'area_std':         np.std(areas_v) if len(areas_v) > 0 else 0,
        'area_cv':          _cv(areas_v),
        'mean_beat':        final_mean_beat,
        'resampled_beats':  valid_resampled,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
def main():
    apply_style()

    # ═════════════════════════════════════════════════════════════
    #  RAW DATA
    # ═════════════════════════════════════════════════════════════
    raw_ippg, t_ippg, fps = load_video(VIDEO_PATH)
    raw_cppg, t_cppg, fs_cppg = load_cppg(CPPG_PATH)
    duration = t_ippg[-1] - t_ippg[0]

    mask_i = (t_ippg >= TIME_START) & (t_ippg <= TIME_END)
    mask_c = (t_cppg >= TIME_START) & (t_cppg <= TIME_END)

    # ═════════════════════════════════════════════════════════════
    #  OD TRANSFORM
    # ═════════════════════════════════════════════════════════════
    I0 = np.mean(raw_ippg)
    od = -np.log(raw_ippg / I0 + 1e-9)

    # ═════════════════════════════════════════════════════════════
    #  DETREND
    # ═════════════════════════════════════════════════════════════
    od_dt   = detrend(od)
    cppg_dt = detrend(raw_cppg)

    # Reference from cPPG
    cppg_ref = bandpass_filter(cppg_dt, fs_cppg, order=2)
    freq_c, mag_c = compute_fft(cppg_ref, fs_cppg)
    ref_hr, ref_fpeak = find_hr_peak(mag_c, freq_c)

    # ═════════════════════════════════════════════════════════════
    #  FILTER SWEEP  →  24 CANDIDATES
    # ═════════════════════════════════════════════════════════════
    candidates = []
    for order in BW_ORDERS:
        bpf = bandpass_filter(od_dt, fps, order=order)
        for w in SG_WINDOWS:
            try:
                sg = savgol_filter(bpf, window_length=w, polyorder=SG_POLYORDER)
                candidates.append({
                    'bw': order, 'sg': w,
                    'signal': sg, 'baseline': bpf,
                    'fs': fps, 'ok': True,
                })
            except Exception:
                candidates.append({'bw': order, 'sg': w, 'ok': False})

    # ═════════════════════════════════════════════════════════════
    #  PHYSIOLOGICAL CHECK  →  PASS / REJECT
    # ═════════════════════════════════════════════════════════════
    n_pass = 0
    n_reject = 0
    for c in candidates:
        if not c['ok']:
            c['gate_pass'] = False
            c['gate_reasons'] = ['Processing failed']
            n_reject += 1
            continue
        passed, info, reasons = physiological_check(
            c['signal'], fps, duration, ref_hr, ref_fpeak
        )
        c['phys'] = info
        c['gate_pass'] = passed
        c['gate_reasons'] = reasons
        if passed:
            n_pass += 1
        else:
            n_reject += 1

    # ═════════════════════════════════════════════════════════════
    #  QUALITY EVALUATION  →  COMPOSITE SCORE
    # ═════════════════════════════════════════════════════════════
    for c in candidates:
        if c.get('gate_pass'):
            evaluate_quality(c, t_ippg, cppg_ref, t_cppg, ref_hr, ref_fpeak)

    compute_composite_scores(candidates)

    # ═════════════════════════════════════════════════════════════
    #  BEST PIPELINE
    # ═════════════════════════════════════════════════════════════
    passed_list = [c for c in candidates if c.get('gate_pass')]
    if not passed_list:
        print("  ❌ No candidate passed the Physiological Gate!")
        return

    best = max(passed_list, key=lambda x: x.get('score', 0))
    final_signal = best['signal']
    pi = best['phys']

    # ═════════════════════════════════════════════════════════════
    #  FINAL PEAK DETECTION  (same parameters as validation)
    # ═════════════════════════════════════════════════════════════
    peaks   = pi['peaks']
    troughs = pi['troughs']

    # ═════════════════════════════════════════════════════════════
    #  BEAT SEGMENTATION  →  BEAT-LEVEL ANALYSIS
    # ═════════════════════════════════════════════════════════════
    beats = segment_beats(final_signal, t_ippg, peaks, troughs, fps)
    ba = beat_level_analysis(beats)

    # HR from IBI
    ibi_arr = pi['ibi']
    hr_from_ibi = 60.0 / np.mean(ibi_arr) if len(ibi_arr) > 0 else 0
    hr_std = np.std(60.0 / ibi_arr) if len(ibi_arr) > 0 and np.all(ibi_arr > 0) else 0

    # ═════════════════════════════════════════════════════════════
    #  OUTPUT
    # ═════════════════════════════════════════════════════════════
    print()
    print("═" * 44)
    print(f"  FINAL iPPG PIPELINE  ({WAVELENGTH})")
    print("═" * 44)
    print()
    print(f"  OD Transform         ✓")
    print(f"  Detrend              ✓")
    print()
    print(f"  Best Butterworth     : Order {best['bw']}")
    print(f"  Best SG Window       : {best['sg']}")
    print()
    print(f"  Candidates           : {len(candidates)}")
    print(f"  Passed Gate          : {n_pass}")
    print(f"  Rejected             : {n_reject}")

    # Print peak refinement steps for best pipeline
    print()
    print("─" * 44)
    print("  PEAK REFINEMENT (Best Pipeline)")
    print("─" * 44)
    print()
    for step in pi.get('peak_log', {}).get('steps', []):
        print(f"  {step}")

    print()
    print("─" * 40)
    print("FINAL PHYSIOLOGY")
    print("─" * 40)
    print()
    print(f"Heart Rate       : {hr_from_ibi:.1f} ± {hr_std:.1f} BPM")
    print(f"IBI              : {pi['ibi_mean']*1000:.1f} ± {pi['ibi_std']*1000:.1f} ms     CV={pi['ibi_cv']:.1f}%")
    print()
    print(f"Amplitude        : {ba['amp_mean']:.4f} ± {ba['amp_std']:.4f} OD CV={ba['amp_cv']:.1f}%")
    print(f"Rise Time        : {ba['rt_mean']*1000:.1f} ± {ba['rt_std']*1000:.1f} ms     CV={ba['rt_cv']:.1f}%")
    print(f"Pulse Width      : {ba['pw_mean']*1000:.1f} ± {ba['pw_std']*1000:.1f} ms     CV={ba['pw_cv']:.1f}%")
    print(f"Area             : {ba['area_mean']:.2e} ± {ba['area_std']:.2e} CV={ba['area_cv']:.1f}%")
    print()
    print(f"Beat Consistency : {ba.get('beat_consistency', 0):.2f}")
    print(f"Beats Analyzed   : {ba.get('n_beats_valid', 0)} / {ba.get('n_beats_total', 0)}")
    removed_beats = ba.get('n_beats_total', 0) - ba.get('n_beats_valid', 0)
    print(f"Beats Removed    : {removed_beats}")
    print()
    print("─" * 40)
    print("QUALITY")
    print("─" * 40)
    print()
    print(f"HR Error         : {pi['hr_diff']:.1f} BPM")
    print(f"Peak Ratio       : {pi['peak_ratio']:.2f}")
    print(f"SNR              : {best['q_snr']:.1f} dB")
    print(f"Dominant Freq    : {pi['fpeak_hz']:.3f} Hz")
    print(f"Pearson r        : {best['q_pearson']:.3f}")
    print()

    # ═════════════════════════════════════════════════════════════
    #  FIGURES
    # ═════════════════════════════════════════════════════════════

    # ── Fig 1: Heatmap ───────────────────────────────────────────
    score_grid = np.full((len(BW_ORDERS), len(SG_WINDOWS)), np.nan)
    for c in candidates:
        if c.get('gate_pass') and 'score' in c:
            r = BW_ORDERS.index(c['bw'])
            col = SG_WINDOWS.index(c['sg'])
            score_grid[r, col] = c['score']

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    im = ax1.imshow(score_grid, aspect='auto', cmap='YlOrRd', origin='lower')
    ax1.set_xticks(range(len(SG_WINDOWS)))
    ax1.set_xticklabels([str(w) for w in SG_WINDOWS])
    ax1.set_yticks(range(len(BW_ORDERS)))
    ax1.set_yticklabels([str(o) for o in BW_ORDERS])
    ax1.set_xlabel('Savitzky-Golay Window')
    ax1.set_ylabel('Butterworth Order')
    ax1.set_title(
        f'Pipeline Score Heatmap  |  {WAVELENGTH}\n'
        f'Best: BW={best["bw"]} SG={best["sg"]} (Score={best["score"]:.2f}/{WEIGHT_SUM:.0f})  '
        f'| Passed {n_pass}/{len(candidates)}',
        fontweight='bold',
    )
    for ri in range(len(BW_ORDERS)):
        for ci in range(len(SG_WINDOWS)):
            val = score_grid[ri, ci]
            if np.isnan(val):
                ax1.text(ci, ri, 'REJECT', ha='center', va='center',
                         fontsize=7, color='gray', style='italic')
            else:
                ax1.text(ci, ri, f'{val:.1f}', ha='center', va='center',
                         fontsize=9, fontweight='bold',
                         color='white' if val > np.nanmax(score_grid)*0.7 else 'black')
    plt.colorbar(im, ax=ax1, label=f'Composite Score (0-{WEIGHT_SUM:.0f})')
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, f"pipeline_heatmap_{WAVELENGTH}.png")
    fig1.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ── Fig 2: Final Signal + Peaks/Troughs + cPPG + FFT ────────
    fig2, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]},
    )

    sig_disp = final_signal[mask_i]
    t_disp   = t_ippg[mask_i]
    ax_top.plot(t_disp, sig_disp, color=COLORS['best'], lw=1.5,
                label=f'Best (BW{best["bw"]} SG{best["sg"]})')

    pk_in = peaks[(peaks >= np.where(mask_i)[0][0]) & (peaks <= np.where(mask_i)[0][-1])]
    tr_in = troughs[(troughs >= np.where(mask_i)[0][0]) & (troughs <= np.where(mask_i)[0][-1])]
    if len(pk_in):
        ax_top.plot(t_ippg[pk_in], final_signal[pk_in], 'r^', ms=8, zorder=5, label='Peak')
    if len(tr_in):
        ax_top.plot(t_ippg[tr_in], final_signal[tr_in], 'bv', ms=8, zorder=5, label='Trough')

    ax_tw = ax_top.twinx()
    ax_tw.plot(t_cppg[mask_c], cppg_ref[mask_c], 'k-', alpha=0.3, lw=1.2, label='cPPG')
    ax_tw.set_ylabel('cPPG', color='gray')
    ax_tw.tick_params(axis='y', labelcolor='gray')
    l1, lb1 = ax_top.get_legend_handles_labels()
    l2, lb2 = ax_tw.get_legend_handles_labels()
    ax_top.legend(l1+l2, lb1+lb2, loc='upper right')
    ax_top.set_title(
        f"Final Signal  |  HR={pi['hr_fft']:.1f} BPM  |  "
        f"r={best['q_pearson']:.4f}  |  Score={best['score']:.2f}/{WEIGHT_SUM:.0f}",
        fontweight='bold', fontsize=11,
    )
    ax_top.set_ylabel('Optical Density')
    ax_top.set_xlabel('Time (s)')
    ax_top.set_xlim(TIME_START, TIME_END)
    ax_top.grid(True, ls=':', alpha=0.4)

    # FFT panel
    freq_f, mag_f = compute_fft(final_signal, fps)
    fft_m = (freq_f >= BP_LOW) & (freq_f <= BP_HIGH)
    mag_n = mag_f / (np.max(mag_f[fft_m]) + 1e-12)
    ax_bot.plot(freq_f[fft_m], mag_n[fft_m], color=COLORS['ippg'], lw=1.5)
    ax_bot.axvline(pi['fpeak_hz'], color='red', ls='--', lw=1,
                   label=f"f={pi['fpeak_hz']:.3f} Hz → {pi['hr_fft']:.1f} BPM")
    ax_bot.set_xlabel('Frequency (Hz)')
    ax_bot.set_ylabel('Normalized Magnitude')
    ax_bot.set_title('FFT Spectrum', fontweight='bold')
    ax_bot.legend()
    ax_bot.grid(True, ls=':', alpha=0.4)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f"morphology_final_{WAVELENGTH}.png")
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # ── Fig 3: Beat Overlay (all beats normalized 0-100%) ────────
    if ba.get('resampled_beats') is not None and len(ba['resampled_beats']) > 0:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        pct_axis = np.linspace(0, 100, len(ba['mean_beat']))
        for idx, rb in enumerate(ba['resampled_beats']):
            ax3.plot(pct_axis, rb, color='gray', alpha=0.2, lw=0.8)
        ax3.plot(pct_axis, ba['mean_beat'], color=COLORS['best'], lw=2.5,
                 label=f'Mean Beat (n={ba["n_beats_valid"]})')
        ax3.fill_between(
            pct_axis,
            ba['mean_beat'] - np.std(ba['resampled_beats'], axis=0),
            ba['mean_beat'] + np.std(ba['resampled_beats'], axis=0),
            color=COLORS['best'], alpha=0.15, label='±1 SD',
        )
        ax3.set_xlabel('Cardiac Cycle (%)')
        ax3.set_ylabel('Normalized Amplitude')
        ax3.set_title(
            f'Beat-Level Overlay  |  {ba["n_beats_valid"]} valid beats (of {ba["n_beats_total"]})  |  '
            f'Beat Consistency = {ba["beat_consistency"]:.3f}',
            fontweight='bold',
        )
        ax3.legend()
        ax3.grid(True, ls=':', alpha=0.4)
        plt.tight_layout()
        out3 = os.path.join(OUT_DIR, f"beat_overlay_{WAVELENGTH}.png")
        fig3.savefig(out3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
    else:
        out3 = None

    # ── Export CSV ────────────────────────────────────────────────
    df = pd.DataFrame([{
        'Wavelength':        WAVELENGTH,
        'BW_Order':          best['bw'],
        'SG_Window':         best['sg'],
        'Score':             best['score'],
        'Score_Max':         WEIGHT_SUM,
        'Pearson_r':         best['q_pearson'],
        'SNR_dB':            best['q_snr'],
        'HR_iPPG_BPM':       pi['hr_fft'],
        'HR_cPPG_BPM':       ref_hr,
        'HR_Error_BPM':      pi['hr_diff'],
        'HR_IBI_mean':       hr_from_ibi,
        'HR_IBI_std':        hr_std,
        'Peak_Count':        pi['n_peaks'],
        'Expected_Peaks':    pi['expected_peaks'],
        'IBI_CV_pct':        pi['ibi_cv'],
        'Beat_Consistency':  ba.get('beat_consistency', 0),
        'N_Beats_Total':     ba.get('n_beats_total', 0),
        'N_Beats_Valid':     ba.get('n_beats_valid', 0),
        'Dominant_Freq_Hz':  pi['fpeak_hz'],
        'Amplitude_mean':    ba.get('amp_mean', 0),
        'Amplitude_std':     ba.get('amp_std', 0),
        'Amplitude_CV':      ba.get('amp_cv', 0),
        'Rise_Time_ms':      ba.get('rt_mean', 0) * 1000,
        'Rise_Time_std_ms':  ba.get('rt_std', 0) * 1000,
        'Rise_Time_CV':      ba.get('rt_cv', 0),
        'Pulse_Width_ms':    ba.get('pw_mean', 0) * 1000,
        'Pulse_Width_std_ms': ba.get('pw_std', 0) * 1000,
        'Pulse_Width_CV':    ba.get('pw_cv', 0),
        'Area_mean':         ba.get('area_mean', 0),
        'Area_std':          ba.get('area_std', 0),
        'Area_CV':           ba.get('area_cv', 0),
    }])
    csv_path = os.path.join(OUT_DIR, f"morphology_results_{WAVELENGTH}.csv")
    df.to_csv(csv_path, index=False)

    print()
    print(f"  Saved to: {OUT_DIR}")
    print(f"     • {os.path.basename(out1)}")
    print(f"     • {os.path.basename(out2)}")
    if out3:
        print(f"     • {os.path.basename(out3)}")
    print(f"     • {os.path.basename(csv_path)}")
    print("═" * 44)


if __name__ == "__main__":
    main()


W_HR_AGREE     = 0.10
W_PEAK_QUALITY = 0.15
W_IBI_CONSIST  = 0.10
W_BEAT_CONSIST = 0.10
W_MORPHOLOGY   = 0.15
W_PEARSON      = 0.15
W_FREQ_QUALITY = 0.05
W_SNR          = 0.15
W_SMOOTHNESS   = 0.05

def min_max_scale(arr, invert=False):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    val = (arr - mn) / (mx - mn)
    if invert:
        return 1.0 - val
    return val

