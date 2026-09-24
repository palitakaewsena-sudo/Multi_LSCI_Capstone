"""
SPG ADVANCED Full Pipeline (Combined Step 4 & 5)
===================================================
1. Filter Sweep: Butterworth order 1–6 × Savitzky-Golay 5-11
2. Physiological Check & Scoring to find the BEST pipeline
3. FFT & Morphology Analysis on the BEST signal
4. Output Heatmap & FFT/Morphology Plots
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, savgol_filter

sys.path.insert(0, os.path.dirname(__file__))
from spg_common_adv import (
    VIDEO_PATH, CPPG_PATH, OUT_DIR, WAVELENGTH,
    BP_LOW, BP_HIGH, TIME_START, TIME_END, COLORS, FILTER_ORDER,
    apply_style, z_normalize, bandpass_filter, compute_fft, find_hr_peak,
    load_video_spatial_contrast, load_cppg,
    detect_valid_peaks, physiological_check, segment_beats, beat_level_analysis,
    evaluate_quality, compute_composite_scores, compute_snr
)

def main():
    apply_style()

    print("=" * 60)
    print("  SPG ADVANCED FULL PIPELINE (STEP 4 + 5)")
    print("=" * 60)

    # 1. Load Data
    raw_spg, t_spg, fps = load_video_spatial_contrast()
    ir_raw_cppg, t_cppg, fs_cppg = load_cppg()
    
    # Crop cPPG to match TIME_START
    c_mask = t_cppg >= TIME_START
    ir_raw_cppg = ir_raw_cppg[c_mask]
    t_cppg = t_cppg[c_mask]

    # 2. Extract Reference HR from cPPG
    cppg_detrended = detrend(ir_raw_cppg)
    cppg_ref = bandpass_filter(cppg_detrended, fs_cppg, low=BP_LOW, high=BP_HIGH, order=2)
    freq_c, mag_c = compute_fft(cppg_ref, fs_cppg)
    ref_hr_bpm, fpeak_c = find_hr_peak(mag_c, freq_c)
    print(f"  Reference HR (cPPG) = {ref_hr_bpm:.1f} BPM (f_peak = {fpeak_c:.3f} Hz)")

    # 3. OD Transform & Detrend
    
    
    od_detrended = detrend(raw_spg)
    duration = len(od_detrended) / fps

    # 4. Filter Sweep
    orders = [1, 2, 3, 4, 5, 6]
    sg_windows = [5, 7, 9, 11, 15, 21, 31, 41]
    candidates = []

    print("\n  [Sweeping Candidates]")
    for order in orders:
        for sg_w in sg_windows:
            try:
                sg_sig = savgol_filter(od_detrended, window_length=sg_w, polyorder=3)
                spg_f = bandpass_filter(sg_sig, fps, low=BP_LOW, high=BP_HIGH, order=order)

                passed, info, reasons = physiological_check(
                    spg_f, fps, duration, ref_hr_bpm, fpeak_c
                )

                candidate = {
                    'bw': order,
                    'sg': sg_w,
                    'signal': spg_f,
                    'baseline': sg_sig,
                    'fs': fps,
                    'phys': info,
                    'gate_pass': passed,
                    'reasons': reasons
                }
                # Always evaluate quality so we have scores even if they fail
                evaluate_quality(candidate, t_spg, cppg_ref, t_cppg, ref_hr_bpm, fpeak_c)
                candidates.append(candidate)
            except Exception as e:
                print(f"  ⚠️  Order {order}, SG {sg_w} failed: {e}")

    # 5. Evaluate and Score Candidates
    valid = [c for c in candidates if c['gate_pass']]
    print(f"\n  Total Candidates : {len(candidates)}")
    print(f"  Passed Gate      : {len(valid)}")
    print(f"  Rejected         : {len(candidates) - len(valid)}")

    if len(candidates) > len(valid):
        print(f"  Sample Candidate 1: hr_fft = {candidates[0]['phys']['hr_fft']:.1f}, reasons: {candidates[0]['reasons']}")

    if not valid:
        print("\n⚠️ All candidates failed physiological gating. Falling back to the best available candidate.")
        for c in candidates:
            c['gate_pass'] = True
        valid = candidates

    compute_composite_scores(candidates)
    valid.sort(key=lambda x: x['score'], reverse=True)
    best = valid[0]

    best_bw = best['bw']
    best_sg = best['sg']
    best_score = best['score']

    print(f"\n  ★ BEST PIPELINE: BW Order {best_bw} | SG Window {best_sg} | Score {best_score:.3f}")
    
    # Save Best Config to JSON
    best_json = os.path.join(OUT_DIR, f"best_order_spg_{WAVELENGTH}.json")
    with open(best_json, 'w') as f:
        json.dump({'best_bw': best_bw, 'best_sg': best_sg, 'best_score': float(best_score)}, f, indent=2)

    # Save signals for comparison
    np.save(os.path.join(OUT_DIR, f"spg_signal_{WAVELENGTH}.npy"), best['signal'])
    np.save(os.path.join(OUT_DIR, f"spg_time_{WAVELENGTH}.npy"), t_spg)
    
    # 6. Heatmap Plot (Step 4 part)
    heatmap = np.zeros((len(orders), len(sg_windows)))
    for c in candidates:
        i = orders.index(c['bw'])
        j = sg_windows.index(c['sg'])
        if c['gate_pass']:
            heatmap[i, j] = c.get('score', 0)
        else:
            heatmap[i, j] = -0.2

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    cax = ax1.matshow(heatmap, cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xticks(range(len(sg_windows)))
    ax1.set_xticklabels(sg_windows)
    ax1.set_yticks(range(len(orders)))
    ax1.set_yticklabels(orders)
    ax1.set_xlabel('Savitzky-Golay Window', fontweight='bold')
    ax1.set_ylabel('Butterworth Order', fontweight='bold')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_title(f"Step 3: Advanced Sweep Candidate Scores | {WAVELENGTH}", pad=15, fontweight='bold')

    for i in range(len(orders)):
        for j in range(len(sg_windows)):
            val = heatmap[i, j]
            text = f"{val:.2f}" if val >= 0 else "FAIL"
            color = 'black' if 0.3 < val < 0.8 else 'white'
            ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    plt.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04, label='Composite Score')
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, f"step3_advanced_sweep_{WAVELENGTH}.png")
    fig1.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  ✅ Saved Heatmap: {out1}")

    # ==========================================
    # 7. FFT & Morphology Analysis (Step 4 part)
    # ==========================================
    spg_final = best['signal']
    
    freq_i, mag_i = compute_fft(spg_final, fps)
    hr_spg, fpeak_i = find_hr_peak(mag_i, freq_i)
    snr_db = compute_snr(spg_final, fps, f_low=BP_LOW, f_high=BP_HIGH)
    
    print(f"\n  [FFT Analysis on Best Signal]")
    print(f"  SPG → f_peak = {fpeak_i:.3f} Hz → HR = {hr_spg:.1f} BPM")
    print(f"  SPG → SQI (SNR) = {snr_db:.2f} dB")
    
    # Process cPPG
    cppg_filtered = bandpass_filter(detrend(ir_raw_cppg), fs_cppg, order=2)
    freq_c, mag_c = compute_fft(cppg_filtered, fs_cppg)
    hr_cppg, fpeak_c = find_hr_peak(mag_c, freq_c)
    print(f"  cPPG → f_peak = {fpeak_c:.3f} Hz → HR = {hr_cppg:.1f} BPM")
    
    hr_error = abs(hr_spg - hr_cppg)
    print(f"  |Error| = {hr_error:.1f} BPM")

    # Extract Morphology
    peaks, troughs, log = detect_valid_peaks(spg_final, fps, hr_cppg)
    beats = segment_beats(spg_final, t_spg, peaks, troughs, fps)
    ba = beat_level_analysis(beats)

    print(f"\n  Morphology:")
    print(f"  - Beats analyzed : {ba.get('n_beats_valid', 0)} / {ba.get('n_beats_total', 0)}")
    print(f"  - Amplitude CV   : {ba.get('amp_cv', 0):.1f}%")
    print(f"  - Rise Time CV   : {ba.get('rt_cv', 0):.1f}%")

    # ==========================================
    # Plotting (Separated: Time Domain & FFT)
    # ==========================================
    plt.style.use('ggplot')
    
    # ------------------------------------------
    # Figure 1: Time Domain (5s to 10s)
    # ------------------------------------------
    fig1, ax_time = plt.subplots(figsize=(12, 4))
    
    ax_time.plot(t_cppg, z_normalize(cppg_filtered), color='black', lw=2, ls='--', label='cPPG (Ref)')
    ax_time.plot(t_spg, z_normalize(spg_final), color='#55a868', lw=1.5, label=f'NIR-SPG {WAVELENGTH}')
    ax_time.set_title(f'Time Domain Signal Comparison ({BP_LOW} - {BP_HIGH} Hz Bandpass)\nWindow: 5.0s to 10.0s', fontweight='bold', fontsize=12, pad=15)
    ax_time.set_xlabel('Time (s)', fontsize=10)
    ax_time.set_ylabel('Amplitude (z-score)', fontsize=10)
    ax_time.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='lightgray')
    ax_time.set_xlim(5.0, 10.0)
    
    plt.tight_layout()
    out_img_time = os.path.join(OUT_DIR, f"step3_4_time_domain_5to10s_{WAVELENGTH}.png")
    fig1.savefig(out_img_time, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ------------------------------------------
    # Figure 2: FFT Spectrum
    # ------------------------------------------
    fig2, ax_fft = plt.subplots(figsize=(10, 5))
    
    fmask = (freq_i >= 0.0) & (freq_i <= 5.0)
    mag_norm_i = mag_i / (np.max(mag_i[fmask]) + 1e-12)
    mag_norm_c = mag_c / (np.max(mag_c[(freq_c >= 0.0) & (freq_c <= 5.0)]) + 1e-12)

    ax_fft.plot(freq_i[fmask], mag_norm_i[fmask], color='#d65f5f', lw=2, label=f'NIR-SPG {WAVELENGTH}')
    ax_fft.plot(freq_c[fmask], mag_norm_c[fmask], color='black', ls='--', lw=1.5, alpha=0.7, label='cPPG (Ref)')
    
    ax_fft.axvline(fpeak_i, color='#d65f5f', ls=':', lw=1.5, label=f'SPG Peak: {fpeak_i:.2f}Hz ({hr_spg:.1f} BPM)')
    ax_fft.axvline(fpeak_c, color='black', ls=':', lw=1.5, label=f'cPPG Peak: {fpeak_c:.2f}Hz ({hr_cppg:.1f} BPM)')
    
    ax_fft.set_title(f"Frequency Domain Comparison (FFT) [{WAVELENGTH}]", fontweight='bold', fontsize=12, pad=10)
    ax_fft.set_xlabel('Frequency (Hz)', fontsize=10)
    ax_fft.set_ylabel('Normalized Magnitude', fontsize=10)
    ax_fft.set_xlim(0, 5)
    ax_fft.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='lightgray')

    plt.tight_layout()
    out_img_fft = os.path.join(OUT_DIR, f"step3_4_fft_{WAVELENGTH}.png")
    fig2.savefig(out_img_fft, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"\n  ✅ Saved Time Domain: {out_img_time}")
    print(f"  ✅ Saved FFT: {out_img_fft}")

if __name__ == "__main__":
    main()

