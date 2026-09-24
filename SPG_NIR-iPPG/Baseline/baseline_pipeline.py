import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, correlate, correlation_lags
from scipy.ndimage import uniform_filter

# Define styling
def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

def z_normalize(sig):
    # Ensure no NaNs and standard deviation is not exactly zero
    sig = np.nan_to_num(sig)
    std = np.std(sig)
    if std == 0:
        std = 1.0
    return (sig - np.mean(sig)) / std

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_hr_fft(sig, fps):
    L = len(sig)
    freqs = np.fft.rfftfreq(L, d=1.0/fps)
    fft_mag = np.abs(np.fft.rfft(sig))
    
    valid_idx = np.where((freqs >= 0.7) & (freqs <= 3.0))[0]
    if len(valid_idx) == 0:
        return 0, freqs, fft_mag
        
    best_idx = valid_idx[np.argmax(fft_mag[valid_idx])]
    best_freq = freqs[best_idx]
    return best_freq * 60.0, freqs, fft_mag

def extract_signals_from_video(video_path, roi=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    ippg_raw = []
    spg_raw = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if roi is None:
            roi_y = int(gray.shape[0] * 0.3)
            roi_h = int(gray.shape[0] * 0.4)
            roi_x = int(gray.shape[1] * 0.3)
            roi_w = int(gray.shape[1] * 0.4)
        else:
            roi_x, roi_y, roi_w, roi_h = roi
            
        roi_img = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # iPPG (Mean intensity)
        mean_val = np.mean(roi_img)
        ippg_raw.append(mean_val)
        
        # SPG (Spatial Speckle Contrast: 1 / <K^2>)
        roi_float = roi_img.astype(np.float32)
        local_mean = uniform_filter(roi_float, size=7)
        local_sq_mean = uniform_filter(roi_float**2, size=7)
        local_var = local_sq_mean - local_mean**2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            K = np.sqrt(np.maximum(local_var, 0)) / (local_mean + 1e-6)
            K = np.nan_to_num(K, nan=1.0, posinf=1.0, neginf=1.0)
            
        K_mean = np.mean(K)
        if K_mean > 0:
            spg_val = 1.0 / (K_mean**2)
        else:
            spg_val = 0.0
            
        spg_raw.append(spg_val)
        
    cap.release()
    return np.array(ippg_raw), np.array(spg_raw), fps

def process_cppg(csv_file):
    if not os.path.exists(csv_file): return None, 0.0
    try:
        df = pd.read_csv(csv_file)
        if 'IR' in df.columns and 'Timestamp' in df.columns:
            # Calculate fs
            t = pd.to_datetime(df['Timestamp'])
            dt = (t.iloc[-1] - t.iloc[0]).total_seconds() / (len(t) - 1)
            fs = 1.0 / dt if dt > 0 else 120.0
            
            # cPPG is usually inverted IR
            return -df['IR'].values.astype(float), fs
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    return None, 0.0

def main():
    apply_style()
    
    base_path = r"C:\Scientific Camera Interfaces\SDK\Python Toolkit_Project\9202026"
    datasets = ["1", "1_2", "1_3", "2", "2_2", "2_3", "3", "3_2", "3_3", "4", "4_2", "4_3"]
    dataset_dirs = [os.path.join(base_path, d) for d in datasets]
    out_dir = r"C:\Scientific Camera Interfaces\SDK\Python Toolkit_Project\results_unified"
    os.makedirs(out_dir, exist_ok=True)
    
    # Store summary stats
    summary_data = []
    
    for ds_path in dataset_dirs:
        ds = os.path.basename(ds_path)
        cppg_file = os.path.join(ds_path, "pulse_data.csv")
        
        # Process cPPG once per dataset
        cppg_raw, fs_ref_raw = process_cppg(cppg_file)
        
        video_files = glob.glob(os.path.join(ds_path, "*.mkv"))
        for vid in video_files:
            fname = os.path.basename(vid)
            if not fname.startswith("recording_CH"):
                continue
                
            ch_wl = fname.replace("recording_", "").replace(".mkv", "")
            
            print(f"Processing {ds} / {ch_wl} ...")
            out_ds_ch = os.path.join(out_dir, ds, ch_wl)
            os.makedirs(out_ds_ch, exist_ok=True)
            
            ippg_raw, spg_raw, fps = extract_signals_from_video(vid)
            if ippg_raw is None:
                continue
                
            # Filter and process
            TIME_START = 5.0
            frame_start = int(TIME_START * fps)
            
            t_ippg = np.arange(len(ippg_raw)) / fps
            t_spg = t_ippg.copy()
            
            # iPPG OD transform
            od_ippg = -np.log(ippg_raw / (np.mean(ippg_raw) + 1e-6))
            
            # Smooth
            od_ippg_s = savgol_filter(od_ippg, 15, 3)
            spg_s = savgol_filter(spg_raw, 15, 3)
            
            clean_ippg = bandpass_filter(od_ippg_s, 0.7, 3.0, fps)
            clean_spg = bandpass_filter(spg_s, 0.7, 3.0, fps)
            
            clean_ippg = clean_ippg[frame_start:]
            t_ippg = t_ippg[frame_start:]
            clean_spg = clean_spg[frame_start:]
            t_spg = t_spg[frame_start:]
            
            # Calculate HR from 0.7-3.0 Hz signals
            hr_ippg, _, _ = calculate_hr_fft(clean_ippg, fps)
            hr_spg, _, _ = calculate_hr_fft(clean_spg, fps)
            
            # Create wider 0.5-5.0 Hz signals for FFT plotting
            fft_ippg = bandpass_filter(od_ippg_s, 0.5, 5.0, fps)[frame_start:]
            fft_spg = bandpass_filter(spg_s, 0.5, 5.0, fps)[frame_start:]
            _, freqs_i, fft_mag_i = calculate_hr_fft(fft_ippg, fps)
            _, freqs_s, fft_mag_s = calculate_hr_fft(fft_spg, fps)
            
            # Reference alignment
            hr_ref = 0.0
            clean_cppg = None
            t_cppg = None
            freqs_c, fft_mag_c = None, None
            
            if cppg_raw is not None:
                fs_ref = fs_ref_raw
                frame_start_ref = int(TIME_START * fs_ref)
                if len(cppg_raw) > frame_start_ref:
                    t_cppg_raw = np.arange(len(cppg_raw)) / fs_ref
                    cppg_s = savgol_filter(cppg_raw, 15, 3)
                    clean_cppg = bandpass_filter(cppg_s, 0.7, 3.0, fs_ref)
                    clean_cppg = clean_cppg[frame_start_ref:]
                    t_cppg = t_cppg_raw[frame_start_ref:]
                    hr_ref, _, _ = calculate_hr_fft(clean_cppg, fs_ref)
                    
                    fft_cppg = bandpass_filter(cppg_s, 0.5, 5.0, fs_ref)[frame_start_ref:]
                    _, freqs_c, fft_mag_c = calculate_hr_fft(fft_cppg, fs_ref)
                    
            error_ippg = abs(hr_ippg - hr_ref) if hr_ref > 0 else 0
            error_spg = abs(hr_spg - hr_ref) if hr_ref > 0 else 0
            
            # Normalization
            norm_ippg = z_normalize(clean_ippg)
            norm_spg = z_normalize(clean_spg)
            if clean_cppg is not None:
                norm_cppg = z_normalize(clean_cppg)
            
            # --- 01. Time Domain NIR-iPPG ---
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(t_ippg, norm_ippg, color='#2980b9', lw=1.5, label='NIR-iPPG')
            
            peaks_i, _ = find_peaks(norm_ippg, distance=int(fps/3.0))
            ax1.plot(t_ippg[peaks_i], norm_ippg[peaks_i], 'o', color='#2980b9', markersize=6)
            ax1.set_xlim(10.0, 20.0)
            ax1.set_title('Time Domain Analysis: NIR-iPPG Signal')
            ax1.legend(loc='upper right')
            plt.tight_layout()
            fig1.savefig(os.path.join(out_ds_ch, "01_time_domain_ippg.png"))
            plt.close(fig1)
            
            # --- 02. Time Domain SPG ---
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(t_spg, norm_spg, color='#ff7f0e', lw=1.5, label='SPG')
            
            peaks_s, _ = find_peaks(norm_spg, distance=int(fps/3.0))
            ax2.plot(t_spg[peaks_s], norm_spg[peaks_s], 'o', color='#ff7f0e', markersize=6)
            ax2.set_xlim(10.0, 20.0)
            ax2.set_title('Time Domain Analysis: SPG Signal')
            ax2.legend(loc='upper right')
            plt.tight_layout()
            fig2.savefig(os.path.join(out_ds_ch, "02_time_domain_spg.png"))
            plt.close(fig2)
            
            # --- 02b. Time Domain Combined (cPPG, SPG, NIR-iPPG) ---
            fig2b, ax2b = plt.subplots(figsize=(12, 5))
            if clean_cppg is not None:
                ax2b.plot(t_cppg, norm_cppg, color='#7f8c8d', lw=2.0, ls='--', alpha=0.8, label='cPPG Ref')
                peaks_c, _ = find_peaks(norm_cppg, distance=int(fs_ref/3.0))
                ax2b.plot(t_cppg[peaks_c], norm_cppg[peaks_c], 'o', color='#7f8c8d', markersize=6)
                
            ax2b.plot(t_spg, norm_spg, color='#e67e22', lw=2.0, alpha=0.8, label='SPG')
            ax2b.plot(t_spg[peaks_s], norm_spg[peaks_s], 'o', color='#e67e22', markersize=6)
            
            ax2b.plot(t_ippg, norm_ippg, color='#2980b9', lw=2.0, alpha=0.8, label='NIR-iPPG')
            ax2b.plot(t_ippg[peaks_i], norm_ippg[peaks_i], 'o', color='#2980b9', markersize=6)
            
            ax2b.set_xlim(10.0, 20.0)
            ax2b.set_title('Comparative Time Domain Analysis: cPPG, SPG, and NIR-iPPG')
            ax2b.set_xlabel('Time (s)', fontweight='bold')
            ax2b.set_ylabel('Normalized Amplitude', fontweight='bold')
            ax2b.legend(loc='upper right')
            ax2b.grid(True, ls=':', alpha=0.6)
            fig2b.tight_layout()
            fig2b.savefig(os.path.join(out_ds_ch, "02b_time_domain_combined.png"), dpi=150)
            plt.close(fig2b)

            # --- 03. Compare Peaks (15-20s window) ---
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            ax3.plot(t_ippg, norm_ippg, color='#2980b9', lw=2.0, alpha=0.8, label='NIR-iPPG')
            ax3.plot(t_spg, norm_spg, color='#ff7f0e', lw=2.0, alpha=0.8, label='SPG')
            
            ax3.plot(t_ippg[peaks_i], norm_ippg[peaks_i], 'o', color='#2980b9', markersize=6)
            ax3.plot(t_spg[peaks_s], norm_spg[peaks_s], 'o', color='#ff7f0e', markersize=6)
            
            t_peaks_s = t_spg[peaks_s]
            y_peaks_s = norm_spg[peaks_s]
            t_peaks_i = t_ippg[peaks_i]
            y_peaks_i = norm_ippg[peaks_i]
            
            for ts, ys in zip(t_peaks_s, y_peaks_s):
                if len(t_peaks_i) == 0: break
                diffs = t_peaks_i - ts
                idx_closest = np.argmin(np.abs(diffs))
                ti = t_peaks_i[idx_closest]
                yi = y_peaks_i[idx_closest]
                
                # if SPG leads by < 0.3s
                if 0 < (ti - ts) < 0.3:
                    pad_x, pad_y = 0.05, 0.5
                    rect = patches.Rectangle((ts - pad_x, min(ys, yi) - pad_y), 
                                             (ti - ts + 2*pad_x), abs(ys - yi) + 2*pad_y, 
                                             linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')
                    ax3.add_patch(rect)
                    
            ax3.set_xlim(15.0, 20.0)
            ax3.set_title('Comparative Time Domain Analysis: NIR-iPPG vs SPG')
            ax3.legend(loc='upper right')
            ax3.grid(True, ls=':')
            plt.tight_layout()
            fig3.savefig(os.path.join(out_ds_ch, "03_compare_peaks.png"))
            plt.close(fig3)
            
            # --- 04. Cross Correlation ---
            mask_corr = (t_ippg >= 10.0) & (t_ippg <= 25.0)
            best_lag_ms = 0
            leader = "None"
            
            if np.any(mask_corr):
                norm_ippg_corr = z_normalize(clean_ippg[mask_corr])
                norm_spg_corr = z_normalize(clean_spg[mask_corr])
                corr = correlate(norm_ippg_corr, norm_spg_corr, mode='full')
                lags = correlation_lags(len(norm_ippg_corr), len(norm_spg_corr))
                lags_time = lags / fps
                
                valid_mask = (lags_time >= -1.0) & (lags_time <= 1.0)
                valid_lags = lags_time[valid_mask]
                valid_corr = corr[valid_mask]
                
                if len(valid_corr) > 0:
                    max_idx = np.argmax(valid_corr)
                    best_lag_ms = valid_lags[max_idx] * 1000.0
                    
                    if best_lag_ms > 15: leader = "SPG"
                    elif best_lag_ms < -15: leader = "NIR-iPPG"
                    else: leader = "Synchronized"
                    
                    fig4, (ax4_1, ax4_2) = plt.subplots(2, 1, figsize=(10, 8))
                    ax4_1.plot(t_ippg[mask_corr], norm_ippg_corr, color='#2980b9', lw=2.0, alpha=0.8, label='NIR-iPPG')
                    ax4_1.plot(t_ippg[mask_corr], norm_spg_corr, color='#ff7f0e', lw=2.0, alpha=0.8, label='SPG')
                    ax4_1.set_title('Time Domain Signals for Cross-Correlation (10-25s)')
                    ax4_1.legend(loc='upper right')
                    ax4_1.grid(True, ls=':')
                    
                    ax4_2.plot(valid_lags * 1000.0, valid_corr, color='purple', lw=2)
                    ax4_2.axvline(best_lag_ms, color='green', ls='--', lw=2, label=f'Max Corr at {best_lag_ms:.1f} ms')
                    ax4_2.axvline(0, color='gray', ls='-', lw=1, alpha=0.5)
                    ax4_2.set_xlabel('Time Lag (ms) -> Positive means SPG leads')
                    ax4_2.legend(loc='upper right')
                    ax4_2.grid(True, ls=':')
                    
                    plt.tight_layout()
                    fig4.savefig(os.path.join(out_ds_ch, "04_cross_correlation.png"))
                    plt.close(fig4)
            
            # --- 05. FFT Spectrum ---
            def get_peak(freq_arr, mag_arr, f_target):
                if f_target > 5.0 or f_target < 0.5: return 0.0, 0.0
                search_idx = np.where(np.abs(freq_arr - f_target) < 0.2)[0]
                if len(search_idx) == 0: 
                    idx = np.argmin(np.abs(freq_arr - f_target))
                    return freq_arr[idx], mag_arr[idx]
                best_idx = search_idx[np.argmax(mag_arr[search_idx])]
                return freq_arr[best_idx], mag_arr[best_idx]

            f0_c = hr_ref / 60.0 if hr_ref > 0 else 0
            f0_s = hr_spg / 60.0 if hr_spg > 0 else 0
            f0_i = hr_ippg / 60.0 if hr_ippg > 0 else 0

            mag_i_norm = fft_mag_i / (np.max(fft_mag_i) + 1e-12)
            mag_s_norm = fft_mag_s / (np.max(fft_mag_s) + 1e-12)
            
            mag_c_norm = None
            if clean_cppg is not None and 'fft_mag_c' in locals() and fft_mag_c is not None:
                mag_c_norm = fft_mag_c / (np.max(fft_mag_c) + 1e-12)

            # Easy on the eyes colors
            color_c = '#7f8c8d' # Soft gray
            color_s = '#e67e22' # Soft orange
            color_i = '#2980b9' # Soft blue

            def plot_fft_with_markers(ax, freq, mag, f0, hr_bpm, color, label_prefix, is_combined, is_ref=False):
                # Plot the main signal line
                ls = '--' if is_ref else '-'
                if is_ref:
                    ax.plot(freq, mag, color=color, lw=2.0, ls=ls, alpha=0.8, label=f"{label_prefix}")
                else:
                    ax.plot(freq, mag, color=color, lw=2.0, ls=ls, alpha=0.8, label=label_prefix)

                if f0 > 0:
                    if is_combined:
                        fh, mc = get_peak(freq, mag, f0)
                        if mc > 0:
                            ax.plot(fh, mc, marker='o', color=color, markersize=8, alpha=0.9)
                            lbl = f"{fh:.2f}Hz ({hr_bpm:.1f} BPM)"
                            ax.axvline(fh, color=color, linestyle=':', alpha=0.8, lw=1.5, label=lbl)
                    else:
                        for h in [1, 2, 3]:
                            fh_target = f0 * h
                            if fh_target > 5.0: continue
                            fh, mh = get_peak(freq, mag, fh_target)
                            if mh > 0.02:
                                ax.plot(fh, mh, marker='o', color=color, markersize=7, alpha=0.9)
                                if h == 1:
                                    lbl = f"$f_0$: {fh:.2f}Hz ({hr_bpm:.1f} BPM)"
                                else:
                                    lbl = f"${h}f_0$: {fh:.2f}Hz"
                                ax.axvline(fh, color=color, linestyle=':', alpha=0.8, lw=1.5, label=lbl)

            # 1) Individual Plot: SPG
            fig_s, ax_s = plt.subplots(figsize=(10, 5))
            plot_fft_with_markers(ax_s, freqs_s, mag_s_norm, f0_s, hr_spg, color_s, 'SPG', False, False)
            
            ax_s.set_xlim(0.5, 5.0); ax_s.set_ylim(0, 1.15)
            ax_s.set_xlabel('Frequency (Hz)', fontweight='bold'); ax_s.set_ylabel('Normalized Mag.', fontweight='bold')
            ax_s.set_title('Frequency Domain Analysis: SPG Signal')
            ax_s.legend(loc='upper right'); ax_s.grid(True, ls=':', alpha=0.6)
            fig_s.tight_layout()
            fig_s.savefig(os.path.join(out_ds_ch, "05_fft_spg.png"), dpi=150)
            plt.close(fig_s)

            # 2) Individual Plot: NIR-iPPG
            fig_i, ax_i = plt.subplots(figsize=(10, 5))
            plot_fft_with_markers(ax_i, freqs_i, mag_i_norm, f0_i, hr_ippg, color_i, 'NIR-iPPG', False, False)
            
            ax_i.set_xlim(0.5, 5.0); ax_i.set_ylim(0, 1.15)
            ax_i.set_xlabel('Frequency (Hz)', fontweight='bold'); ax_i.set_ylabel('Normalized Mag.', fontweight='bold')
            ax_i.set_title('Frequency Domain Analysis: NIR-iPPG Signal')
            ax_i.legend(loc='upper right'); ax_i.grid(True, ls=':', alpha=0.6)
            fig_i.tight_layout()
            fig_i.savefig(os.path.join(out_ds_ch, "05_fft_ippg.png"), dpi=150)
            plt.close(fig_i)

            # 3) Combined Plot: cPPG, SPG, NIR-iPPG
            fig_c, ax_c = plt.subplots(figsize=(10, 5))
            if mag_c_norm is not None:
                plot_fft_with_markers(ax_c, freqs_c, mag_c_norm, f0_c, hr_ref, color_c, 'cPPG Ref', True, True)
            plot_fft_with_markers(ax_c, freqs_s, mag_s_norm, f0_s, hr_spg, color_s, 'SPG', True, False)
            plot_fft_with_markers(ax_c, freqs_i, mag_i_norm, f0_i, hr_ippg, color_i, 'NIR-iPPG', True, False)
            
            ax_c.set_xlim(0.5, 5.0); ax_c.set_ylim(0, 1.15)
            ax_c.set_xlabel('Frequency (Hz)', fontweight='bold'); ax_c.set_ylabel('Normalized Mag.', fontweight='bold')
            ax_c.set_title('Comparative Frequency Domain Analysis: cPPG, SPG, and NIR-iPPG')
            ax_c.legend(loc='upper right'); ax_c.grid(True, ls=':', alpha=0.6)
            fig_c.tight_layout()
            fig_c.savefig(os.path.join(out_ds_ch, "05_fft_combined.png"), dpi=150)
            plt.close(fig_c)
            
            # --- Save npz ---
            if clean_cppg is not None:
                np.savez_compressed(os.path.join(out_ds_ch, "data_signals.npz"), 
                                    t_ippg=t_ippg, ippg=norm_ippg, 
                                    t_spg=t_spg, spg=norm_spg,
                                    t_cppg=t_cppg, cppg=norm_cppg)
                                    
            # Accumulate stats
            summary_data.append({
                "Dataset": ds,
                "Channel": ch_wl,
                "HR_Ref": round(hr_ref, 2),
                "HR_iPPG": round(hr_ippg, 2),
                "Error_iPPG": round(error_ippg, 2),
                "HR_SPG": round(hr_spg, 2),
                "Error_SPG": round(error_spg, 2),
                "Time_Lag_ms": round(best_lag_ms, 2),
                "Leading": leader
            })
            
    # Generate Summary Report
    print("\nGenerating Summary Reports...")
    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(summary_dir, "master_summary.csv"), index=False)
    
    # Generate bar chart of average errors
    avg_errors = df.groupby("Channel").agg({
        "Error_iPPG": ["mean", "std"],
        "Error_SPG": ["mean", "std"]
    })
    avg_errors.columns = ['_'.join(col).strip() for col in avg_errors.columns.values]
    
    with open(os.path.join(summary_dir, "mae_sd_table.txt"), "w") as f:
        f.write(avg_errors.to_string())
    
    # Sort the index based on the wavelength numerical value
    def extract_wl(ch):
        s = ch.split('_')[-1].replace('nm', '')
        return int(s) if s.isdigit() else 0
    sorted_idx = sorted(avg_errors.index, key=extract_wl)
    avg_errors = avg_errors.reindex(sorted_idx)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(avg_errors))
    width = 0.35
    
    ax.bar(x - width/2, avg_errors["Error_iPPG_mean"], width, yerr=avg_errors["Error_iPPG_std"], label='NIR-iPPG', color='#2980b9', capsize=4)
    ax.bar(x + width/2, avg_errors["Error_SPG_mean"], width, yerr=avg_errors["Error_SPG_std"], label='SPG', color='#ff7f0e', capsize=4)
    
    ax.set_ylabel('Mean Absolute Error (BPM)')
    ax.set_title('Average Heart Rate Error by Channel and Method')
    ax.set_xticks(x)
    clean_labels = [lbl.split('_', 1)[-1] if '_' in lbl else lbl for lbl in avg_errors.index]
    ax.set_xticklabels(clean_labels, rotation=15)
    ax.legend(loc='upper right')
    ax.grid(True, ls=':', axis='y')
    
    plt.tight_layout()
    fig.savefig(os.path.join(summary_dir, "summary_hr_error_chart.png"))
    plt.close(fig)
    
    # Write Conclusion Text
    best_method = avg_errors[["Error_iPPG_mean", "Error_SPG_mean"]].mean().idxmin()
    best_channel = avg_errors[["Error_iPPG_mean", "Error_SPG_mean"]].min(axis=1).idxmin()
    with open(os.path.join(summary_dir, "overall_conclusion.txt"), "w", encoding='utf-8') as f:
        f.write("=== OVERALL CONCLUSION ===\n")
        f.write(f"Best Method Overall: {best_method}\n")
        f.write(f"Best Performing Channel: {best_channel}\n\n")
        f.write("Averages per channel (BPM Error):\n")
        f.write(avg_errors.to_string())
        
    print(f"All done! Check {out_dir} folder.")

if __name__ == "__main__":
    main()
