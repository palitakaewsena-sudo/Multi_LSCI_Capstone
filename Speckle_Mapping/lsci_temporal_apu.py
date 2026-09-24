import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import argparse
# ==============================================================================
# Configuration (Set Folder Paths here)
# ==============================================================================
DEFAULT_DATA_DIR = r"./data/input"
DEFAULT_OUT_DIR = r"./results/temporal_apu"
# ==============================================================================


def process_lsci_apu_map(video_or_tif_stack, out_path, title_prefix="", dark_std_map=None):
    """Calculate Temporal Speckle Contrast (K) image in Arbitrary Perfusion Units (APU)"""
    img_stack = video_or_tif_stack.astype(np.float32)
    mean_intensity = np.mean(img_stack, axis=0)
    std_intensity = np.std(img_stack, axis=0)
    
    if dark_std_map is not None:
        min_h = min(std_intensity.shape[0], dark_std_map.shape[0])
        min_w = min(std_intensity.shape[1], dark_std_map.shape[1])
        std_intensity = std_intensity[:min_h, :min_w]
        mean_intensity = mean_intensity[:min_h, :min_w]
        dark_std_cropped = dark_std_map[:min_h, :min_w]
        
        var_measured = std_intensity ** 2
        var_dark = dark_std_cropped ** 2
        var_corrected = np.maximum(var_measured - var_dark, 0)
        std_intensity = np.sqrt(var_corrected)
        title_prefix += " (FPN Corrected)"

    eps = 1e-8
    K_map = std_intensity / (mean_intensity + eps)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(K_map, cmap='gray', vmin=0.0, vmax=0.35)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Speckle Contrast (K) in Arbitrary Perfusion Units (APU)', fontsize=11, fontweight='bold')

    plt.title(f'MS-LSCI Flow Map - {title_prefix}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # --- Plot Mean Intensity Map ---
    mean_out_path = out_path.replace("Temporal_APU", "Mean_Intensity")
    if mean_out_path == out_path:
        mean_out_path = out_path.replace(".png", "_Mean_Intensity.png")
        
    fig_mean, ax_mean = plt.subplots(figsize=(7, 6))
    im_mean = ax_mean.imshow(mean_intensity, cmap='gray')
    ax_mean.axis('off')

    cbar_mean = fig_mean.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)
    cbar_mean.set_label('Mean Intensity', fontsize=11, fontweight='bold')

    plt.title(f'Mean Intensity Map - {title_prefix}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(mean_out_path, dpi=300)
    plt.close()
    # -------------------------------

    # --- Plot BFI Map (1 / K^2) ---
    BFI_map = np.zeros_like(K_map)
    valid_mask = mean_intensity > 5.0  # Intensity mask to prevent salt&pepper background noise
    BFI_map[valid_mask] = 1.0 / (K_map[valid_mask]**2 + 1e-8)
    bfi_out_path = out_path.replace("Temporal_APU", "BFI_Map")
    if bfi_out_path == out_path:
        bfi_out_path = out_path.replace(".png", "_BFI_Map.png")
        
    fig_bfi, ax_bfi = plt.subplots(figsize=(7, 6))
    # Use nanpercentile to avoid Errno 22 if NaNs exist
    vmax_bfi = np.nanpercentile(BFI_map, 95)
    if np.isnan(vmax_bfi) or vmax_bfi <= 0:
        vmax_bfi = 1.0
        
    im_bfi = ax_bfi.imshow(BFI_map, cmap='jet', vmin=0.0, vmax=vmax_bfi)
    ax_bfi.axis('off')

    cbar_bfi = fig_bfi.colorbar(im_bfi, ax=ax_bfi, fraction=0.046, pad=0.04)
    cbar_bfi.set_label('Blood Flow Index (1/K²)', fontsize=11, fontweight='bold')

    plt.title(f'BFI Map - {title_prefix}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(bfi_out_path, dpi=300)
    plt.close()
    # -------------------------------

    # --- Plot K Map ---
    k_out_path = out_path.replace("Temporal_APU", "K_Map")
    if k_out_path == out_path:
        k_out_path = out_path.replace(".png", "_K_Map.png")
        
    fig_k, ax_k = plt.subplots(figsize=(7, 6))
    im_k = ax_k.imshow(K_map, cmap='gray', vmin=0.0, vmax=0.35)
    ax_k.axis('off')

    cbar_k = fig_k.colorbar(im_k, ax=ax_k, fraction=0.046, pad=0.04)
    cbar_k.set_label('Speckle Contrast (K)', fontsize=11, fontweight='bold')

    plt.title(f'K Map - {title_prefix}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(k_out_path, dpi=300)
    plt.close()
    # -------------------------------

    return K_map

def plot_spatial_profiles(K_maps, out_dir):
    import re
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Sort K_maps by wavelength
    def get_wl(k):
        match = re.search(r'(\d+)nm', k)
        return int(match.group(1)) if match else 0
        
    keys_sorted = sorted(K_maps.keys(), key=get_wl)
    colors = ['tab:green', 'tab:red', 'tab:orange', 'k'] # Colors mapping to 520, 786, 787, 1064
    
    for i, (key, color) in enumerate(zip(keys_sorted, colors)):
        if i >= 4: break
        
        K_map = K_maps[key]
        
        # BFI = 1 / K^2
        BFI = 1.0 / (K_map**2 + 1e-8)
        
        # 1D Spatial Profile
        profile = np.mean(BFI, axis=1)
        
        # Normalize roughly matching the user's graph (-1 to 1 with data mostly positive)
        median_val = np.median(profile)
        norm_prof = profile - median_val
        max_val = np.max(np.abs(norm_prof))
        if max_val > 0:
            norm_prof = norm_prof / max_val
            norm_prof = norm_prof * 0.35 # scale peak to ~0.35 for visualization
            
        y_vals = np.arange(len(norm_prof))
        
        ax = axes[i]
        ax.plot(norm_prof, y_vals, color=color, linewidth=2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(norm_prof)])
        
        # Setup grid and labels to match user's image
        ax.set_ylabel('Spatial\nLocation', fontweight='bold', fontsize=11)
        ax.set_xlabel('Normalized Intensity', fontweight='bold', fontsize=11)
        wl = f"{get_wl(key)} nm" if get_wl(key) > 0 else key
        
        # Adding (a), (b), (c), (d)
        letters = ['(a)', '(b)', '(c)', '(d)']
        ax.set_title(f'1/VR² Plot Layer 1 Flow: {wl}', fontweight='bold', fontsize=11)
        ax.text(-0.15, 1.05, letters[i], transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        ax.grid(True, which='major', color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(np.arange(-1, 1.1, 0.2))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "BFI_Spatial_Profiles.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved BFI Spatial Profiles: {out_path}")

def plot_k_spatial_profiles(K_maps, out_dir):
    import re
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    def get_wl(k):
        match = re.search(r'(\d+)nm', k)
        return int(match.group(1)) if match else 0
        
    keys_sorted = sorted(K_maps.keys(), key=get_wl)
    colors = ['tab:green', 'tab:red', 'tab:orange', 'k']
    
    for i, (key, color) in enumerate(zip(keys_sorted, colors)):
        if i >= 4: break
        
        K_map = K_maps[key]
        
        # 1D Spatial Profile of K directly
        profile = np.mean(K_map, axis=1)
        
        # Normalize roughly matching the user's graph (-1 to 1)
        median_val = np.median(profile)
        norm_prof = profile - median_val
        max_val = np.max(np.abs(norm_prof))
        if max_val > 0:
            norm_prof = norm_prof / max_val
            norm_prof = norm_prof * 0.35
            
        y_vals = np.arange(len(norm_prof))
        
        ax = axes[i]
        ax.plot(norm_prof, y_vals, color=color, linewidth=2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(norm_prof)])
        
        ax.set_ylabel('Spatial\nLocation', fontweight='bold', fontsize=11)
        ax.set_xlabel('Normalized K Value', fontweight='bold', fontsize=11)
        wl = f"{get_wl(key)} nm" if get_wl(key) > 0 else key
        
        letters = ['(a)', '(b)', '(c)', '(d)']
        ax.set_title(f'K Value Spatial Profile: {wl}', fontweight='bold', fontsize=11)
        ax.text(-0.15, 1.05, letters[i], transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        ax.grid(True, which='major', color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(np.arange(-1, 1.1, 0.2))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "K_Spatial_Profiles.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved K Spatial Profiles: {out_path}")

def compute_wavelength_differencing(K_shallow, K_deep, method='absolute', normalize=True):
    """Calculate Wavelength Differencing between 2 wavelength bands"""
    K_s = K_shallow.astype(np.float32)
    K_d = K_deep.astype(np.float32)

    if method == 'absolute':
        diff_map = K_d - K_s
        diff_map = np.clip(diff_map, 0, None)
    elif method == 'normalized':
        eps = 1e-8
        diff_map = (K_d - K_s) / (K_d + K_s + eps)
        diff_map = np.clip(diff_map, -1.0, 1.0)
    else:
        raise ValueError("method must be 'absolute' or 'normalized'")

    if normalize and np.max(diff_map) > np.min(diff_map):
        diff_map = (diff_map - np.min(diff_map)) / (np.max(diff_map) - np.min(diff_map))

    return diff_map

def run_temporal_analysis(data_dir, out_dir, dark_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(data_dir, "*.mkv"))
    video_files = [v for v in video_files if os.path.basename(v).lower() not in ["recording.mkv", "record.mkv"]]
    
    if dark_dir and not os.path.exists(dark_dir):
        print(f"⚠️ Dark dir {dark_dir} not found. Proceeding without FPN correction.")
        dark_dir = None
        
    print(f"\nFound {len(video_files)} target videos in {data_dir}")
    
    K_maps = {}
    
    for v_path in video_files:
        basename = os.path.basename(v_path)
        print(f"\nProcessing {basename}...")
        
        # Load matching dark frame for this channel
        dark_std_map = None
        if dark_dir:
            dark_path = os.path.join(dark_dir, basename)
            if os.path.exists(dark_path):
                print(f"[INFO] Using matching dark frame: {dark_path}")
                cap = cv2.VideoCapture(dark_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame)
                cap.release()
                if len(frames) > 0:
                    dark_std_map = np.std(np.array(frames, dtype=np.float64), axis=0)
            else:
                print(f"⚠️ No matching dark frame found for {basename} in {dark_dir}")
        
        cap = cv2.VideoCapture(v_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray)
        cap.release()
        
        if len(frames) == 0:
            print(f"No frames found for {basename}")
            continue
            
        video_stack = np.array(frames)
        print(f"Loaded {len(frames)} frames. Computing APU map...")
        out_file = os.path.join(out_dir, basename.replace(".mkv", "_Temporal_APU.png"))
        
        try:
            K_result = process_lsci_apu_map(video_stack, out_file, title_prefix=basename, dark_std_map=dark_std_map)
            K_maps[basename] = K_result
            print(f"✅ Saved: {out_file}")
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            
    # Wavelength Differencing Logic
    print("\n=== Computing Wavelength Differencing ===")
    
    # Identify the shallow map (520nm)
    shallow_key = next((k for k in K_maps.keys() if '520nm' in k), None)
    
    if not shallow_key:
        print("⚠️ Could not find a 520nm video to use as shallow layer.")
        return
        
    K_shallow = K_maps[shallow_key]
    
    for deep_key, K_deep in K_maps.items():
        if deep_key == shallow_key:
            continue
            
        print(f"Differencing: {deep_key} (Deep) - {shallow_key} (Shallow)")
        diff_result = compute_wavelength_differencing(K_shallow, K_deep, method='absolute', normalize=False)
        
        # 3-channel visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(K_shallow, cmap='gray', vmin=0, vmax=0.35)
        axes[0].set_title(f'Shallow Layer: {shallow_key}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(K_deep, cmap='gray', vmin=0, vmax=0.35)
        axes[1].set_title(f'Deep Layer: {deep_key}', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        im3 = axes[2].imshow(diff_result, cmap='jet')
        axes[2].set_title(f'Wavelength Difference\n(Deep - Shallow)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label('Relative Difference (APU)', fontsize=10)

        plt.tight_layout()
        out_diff_path = os.path.join(out_dir, f"Diff_{deep_key.replace('.mkv', '')}_vs_520nm.png")
        plt.savefig(out_diff_path, dpi=300)
        plt.close()
        
        print(f"✅ Saved Difference Map: {out_diff_path}")

    # Plot the 1D spatial profiles (1/VR^2)
    print("\n=== Generating Spatial BFI Profiles ===")
    plot_spatial_profiles(K_maps, out_dir)
    print("\n=== Generating Spatial K Profiles ===")
    plot_k_spatial_profiles(K_maps, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run temporal APU analysis with Channel-Specific Dark Frame (FPN) Correction.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to input data directory")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Path to output directory")
    parser.add_argument("--dark-dir", type=str, default=None, help="Path to directory containing matching dark frame videos")
    args = parser.parse_args()
    run_temporal_analysis(args.data_dir, args.out_dir, dark_dir=args.dark_dir)
