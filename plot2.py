import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from fileopener import load as load_npz
import os

def load_mat_velocity(mat_file_path, frame_idx=0):
    """
    Loads u and v arrays from a MATLAB .mat file.
    Note: MATLAB structures can vary. This assumes standard 3D arrays 
    or cell arrays commonly exported by PIVlab.
    """
    mat_data = sio.loadmat(mat_file_path)
    
    # Print keys to help you debug if your variable names are different
    # print("Available keys in .mat:", [k for k in mat_data.keys() if not k.startswith('__')])
    
    # Try common PIVlab export keys ('u' or 'u_original')
    u_key = 'u' if 'u' in mat_data else 'u_original'
    v_key = 'v' if 'v' in mat_data else 'v_original'
    
    if u_key not in mat_data or v_key not in mat_data:
        raise KeyError(f"Could not find velocity keys 'u' or 'v' in {mat_file_path}. Check your MATLAB variable names.")
        
    u_mat = mat_data[u_key]
    v_mat = mat_data[v_key]
    
    # If MATLAB saved it as a cell array (object array of arrays)
    if u_mat.dtype == object:
        u_frame = u_mat[0, frame_idx]
        v_frame = v_mat[0, frame_idx]
    # If it's a standard 3D array (rows, cols, frames)
    elif len(u_mat.shape) == 3:
        u_frame = u_mat[:, :, frame_idx]
        v_frame = v_mat[:, :, frame_idx]
    else:
        u_frame = u_mat
        v_frame = v_mat
        
    return u_frame, v_frame

def compare_mat_vs_npz(mat_path, npz_path, frame_idx=0, flip_mat_v=False):
    """
    Compares PIV velocity fields between a .mat file and a .npz file for a given frame.
    """
    # 1. Load Python Data via your fileopener
    print("Loading Python .npz file...")
    npz_data = load_npz(npz_path)
    u_npz = npz_data.u_original[frame_idx]
    v_npz = npz_data.v_original[frame_idx]
    
    # 2. Load MATLAB Data
    print("Loading MATLAB .mat file...")
    u_mat, v_mat = load_mat_velocity(mat_path, frame_idx)
    
    # Account for MATLAB y-axis sign flips if necessary
    if flip_mat_v:
        v_mat = -v_mat
        
    # 3. Verify and match grid shapes
    print(f"MATLAB Frame Shape: {u_mat.shape}")
    print(f"NumPy Frame Shape:  {u_npz.shape}")
    
    if u_mat.shape != u_npz.shape:
        print("⚠️ Warning: Shapes do not match! Attempting to transpose MATLAB matrix...")
        u_mat = u_mat.T
        v_mat = v_mat.T
        if u_mat.shape != u_npz.shape:
            raise ValueError(f"Shape mismatch cannot be resolved automatically: MAT {u_mat.shape} vs NPZ {u_npz.shape}")

    # 4. Compute Velocity Magnitudes
    mag_mat = np.hypot(u_mat, v_mat)
    mag_npz = np.hypot(u_npz, v_npz)
    
    # 5. Quantify Errors (ignoring NaN masked areas)
    u_diff = u_npz - u_mat
    v_diff = v_npz - v_mat
    mag_diff = mag_npz - mag_mat
    
    mae_u = np.nanmean(np.abs(u_diff))
    rmse_u = np.sqrt(np.nanmean(u_diff**2))
    
    mae_v = np.nanmean(np.abs(v_diff))
    rmse_v = np.sqrt(np.nanmean(v_diff**2))
    
    print("\n================ STATISTICAL COMPARISON ================")
    print(f"U-component: MAE = {mae_u:.4f} px/frame | RMSE = {rmse_u:.4f} px/frame")
    print(f"V-component: MAE = {mae_v:.4f} px/frame | RMSE = {rmse_v:.4f} px/frame")
    print("========================================================")

    # 6. Plot side-by-side maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Determine uniform color bounds for the velocity profiles
    vmax = max(np.nanmax(mag_mat), np.nanmax(mag_npz))
    vmin = min(np.nanmin(mag_mat), np.nanmin(mag_npz))
    
    # Plot 1: MATLAB Velocity Magnitude
    im0 = axes[0].imshow(mag_mat, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    axes[0].set_title(f"MATLAB Magnitude (Frame {frame_idx})")
    fig.colorbar(im0, ax=axes[0], label="Velocity (px/frame)")
    
    # Plot 2: NumPy Velocity Magnitude
    im1 = axes[1].imshow(mag_npz, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
    axes[1].set_title(f"NPZ Magnitude (Frame {frame_idx})")
    fig.colorbar(im1, ax=axes[1], label="Velocity (px/frame)")
    
    # Plot 3: Absolute Difference Map (Python minus MATLAB)
    # Using 'bwr' (blue-white-red) showing under/overestimation balances
    diff_bound = np.nanmax(np.abs(mag_diff))
    im2 = axes[2].imshow(mag_diff, cmap='bwr', vmin=-diff_bound, vmax=diff_bound, origin='upper')
    axes[2].set_title("Difference Map (NPZ - MAT)")
    fig.colorbar(im2, ax=axes[2], label="Difference Delta")
    
    for ax in axes:
        ax.set_xlabel("Columns (X)")
        ax.set_ylabel("Rows (Y)")
        
    plt.tight_layout()
    plt.show()
    # --- NEW: SAVE THE FILE TO THE HARD DRIVE ---
    output_dir = os.path.dirname(npz_path)
    save_path = os.path.join(output_dir, f"velocity_comparison_frame_{frame_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visual plot saved directly to: {save_path}")

if __name__ == "__main__":
    # --- Define paths to your test datasets ---
    # Update the .mat path to match your target file
    mat_file = r"C:\Users\JackHu\Downloads\synthetic_primary_export.mat"
    npz_file = r"C:\Users\JackHu\Downloads\primary_export.npz"
    
    try:
        # If your V plots look completely inverted or errors are high, 
        # toggle flip_mat_v=True to correct for MATLAB's native Y-inversion coordinate setup.
        compare_mat_vs_npz(mat_file, npz_file, frame_idx=0, flip_mat_v=False)
        
    except FileNotFoundError as e:
        print(f"Error: Could not locate one of your files. Check paths.\nDetails: {e}")