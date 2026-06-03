import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter
from tqdm import tqdm

def _load_secondary_export(path: Path) -> dict:
    """Load secondary export from .npz or .mat (v7.3 HDF5 or legacy)."""
    suffix = path.suffix.lower()

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as sec:
            return dict(
                S33=np.asarray(sec["S33"]),
                vorticity=np.asarray(sec["vorticity"]),
                velmag=np.asarray(sec["velmag"]),
                avgu=np.asarray(sec["avgu"]),
                avgv=np.asarray(sec["avgv"]),
                typevector=sec["typevector"].astype(bool),
                X=sec["X"].astype(np.float64),
                Y=sec["Y"].astype(np.float64),
                AA=int(sec["AA"]),
                t=np.asarray(sec["t"]),
                condition=sec["condition"].item(),
            )

    if suffix == ".mat":
        # Try v7.3 HDF5 first
        try:
            with h5py.File(path, "r") as f:
                def _3d(key):
                    # h5py reads MATLAB 3-D arrays as (frames, cols, rows) → transpose to (rows, cols, frames)
                    return np.transpose(np.array(f[key]), (2, 1, 0)).astype(np.float32)
                def _2d(key):
                    return np.array(f[key]).T
                def _1d(key):
                    return np.array(f[key]).squeeze()

                # condition is stored as a uint16 char array in MATLAB HDF5
                cond_raw = np.array(f["condition"]).squeeze()
                condition = "".join(chr(int(c)) for c in np.atleast_1d(cond_raw))

                return dict(
                    S33=_3d("S33"),
                    vorticity=_3d("vorticity"),
                    velmag=_3d("velmag"),
                    avgu=_3d("avgu"),
                    avgv=_3d("avgv"),
                    typevector=_2d("typevector").astype(bool),
                    X=_1d("X").astype(np.float64),
                    Y=_1d("Y").astype(np.float64),
                    AA=int(_1d("AA")),
                    t=_1d("t").astype(np.float64),
                    condition=condition,
                )
        except OSError:
            pass  # not HDF5, fall through to scipy

        # Legacy .mat (v4/v5/v6)
        from scipy.io import loadmat
        mat = loadmat(str(path))
        return dict(
            S33=np.asarray(mat["S33"]),
            vorticity=np.asarray(mat["vorticity"]),
            velmag=np.asarray(mat["velmag"]),
            avgu=np.asarray(mat["avgu"]),
            avgv=np.asarray(mat["avgv"]),
            typevector=np.asarray(mat["typevector"]).astype(bool),
            X=np.asarray(mat["X"]).squeeze().astype(np.float64),
            Y=np.asarray(mat["Y"]).squeeze().astype(np.float64),
            AA=int(np.asarray(mat["AA"]).squeeze()),
            t=np.asarray(mat["t"]).squeeze().astype(np.float64),
            condition=str(np.asarray(mat["condition"]).squeeze()),
        )

    raise ValueError(f"Unsupported secondary export format '{suffix}'. Expected .npz or .mat.")


def run_tertiary_export_with_mat_mask(secondary_file, mask_mat_file, R, out_dir):
    """
    Translates a secondary export .npz/.mat file into a tertiary export .npz file
    using a MATLAB .mat mask file (v7.3 HDF5 format).
    """
    secondary_file = Path(secondary_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not secondary_file.exists():
        raise FileNotFoundError(f"Secondary export not found: {secondary_file}")
    if not Path(mask_mat_file).exists():
        raise FileNotFoundError(f"Mask file not found: {mask_mat_file}")

    # 1. Load the secondary export data (.npz or .mat)
    print(f"Loading secondary export: {secondary_file.name}...")
    sec = _load_secondary_export(secondary_file)
    S33       = sec["S33"]
    vorticity = sec["vorticity"]
    velmag    = sec["velmag"]
    avgu      = sec["avgu"]
    avgv      = sec["avgv"]
    typevector = sec["typevector"]
    X         = sec["X"]
    Y         = sec["Y"]
    AA        = sec["AA"]
    t         = sec["t"]
    condition = sec["condition"]

    # 2. Load the MATLAB .mat dynamic mask using h5py
    print(f"Loading MATLAB v7.3 mask file: {Path(mask_mat_file).name}...")
    with h5py.File(mask_mat_file, 'r') as f:
        if 'dynamic_masking' not in f:
            raise KeyError("Could not find variable 'dynamic_masking' inside the .mat file.")
        # MATLAB (rows, cols, frames) maps to (frames, cols, rows) in h5py C-order
        mask_data = np.array(f['dynamic_masking'])
        # Transpose axes back to (rows, cols, frames) to align with secondary arrays
        dynamic_mask = np.transpose(mask_data, (2, 1, 0)).astype(bool)

    n_frames = int(min(S33.shape[2], dynamic_mask.shape[2]))
    print(f"Applying static typevector + dynamic mask across {n_frames} frames...")

    X_norm = X / R
    Y_norm = Y / R
    rows, cols = S33.shape[:2]

    # Preallocate tertiary fields with NaN
    masked_shear = np.full((rows, cols, n_frames), np.nan, dtype=np.float32)
    masked_vort = np.full_like(masked_shear, np.nan)
    masked_velmag = np.full_like(masked_shear, np.nan)
    masked_u = np.full_like(masked_shear, np.nan)
    masked_v = np.full_like(masked_shear, np.nan)

    # 3. Replicate Stage 3 Uniform Filtering (5x5 box filter) and Masking
    for k in tqdm(range(n_frames), desc="Stage 3 — Processing Tertiary Fields"):
        mask_combined = typevector & dynamic_mask[:, :, k]

        # Apply 5x5 box filter — matches MATLAB's imfilter method
        shear_frame = uniform_filter(S33[:, :, k].astype(np.float64), size=5, mode="constant", cval=0.0)
        vort_frame = uniform_filter(vorticity[:, :, k].astype(np.float64), size=5, mode="constant", cval=0.0)
        vel_frame = uniform_filter(velmag[:, :, k].astype(np.float64), size=5, mode="constant", cval=0.0)

        # Blank out non-fluid pixels to NaN
        shear_frame[~mask_combined] = np.nan
        vort_frame[~mask_combined] = np.nan
        vel_frame[~mask_combined] = np.nan

        uc = avgu[:, :, k].astype(np.float64)
        vc = avgv[:, :, k].astype(np.float64)
        uc[~mask_combined] = np.nan
        vc[~mask_combined] = np.nan

        masked_shear[:, :, k] = shear_frame
        masked_vort[:, :, k] = vort_frame
        masked_velmag[:, :, k] = vel_frame
        masked_u[:, :, k] = uc
        masked_v[:, :, k] = vc

    t_trimmed = t[:n_frames]

    # Save output arguments matching fn_tertiary_export specification layout
    save_kwargs = dict(
        X_norm=X_norm, Y_norm=Y_norm,
        X=X, Y=Y,
        masked_shear=masked_shear,
        masked_vort=masked_vort,
        masked_velmag=masked_velmag,
        masked_u=masked_u,
        masked_v=masked_v,
        AA=np.int64(AA),
        t=t_trimmed,
        condition=np.array(condition),
        dynamic_masking=dynamic_mask[:, :, :n_frames]
    )

    out_path = out_dir / f"{condition}_Tertiary_Export.npz"
    np.savez_compressed(out_path, **save_kwargs)
    print(f"✅ Stage 3 Complete! Saved Tertiary export to:\n    {out_path}")
    return out_path


def plot_tertiary_contours(tertiary_file_path, start_frame=50, end_frame=200, step=5):
    """
    Loads a tertiary export file and saves high-resolution contour plots 
    of the smoothed, masked flow fields.
    """
    print(f"\nLoading tertiary export for plotting: {Path(tertiary_file_path).name}")
    with np.load(tertiary_file_path, allow_pickle=True) as data:
        X = data["X"]
        Y = data["Y"]
        masked_shear = data["masked_shear"]
        masked_velmag = data["masked_velmag"]
        condition = str(data["condition"])
        
    n_frames = masked_shear.shape[2]
    output_dir = Path(tertiary_file_path).parent / f"tertiary_contours_{condition}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating side-by-side contour sequence in:\n    {output_dir}")
    generated_count = 0
    
    for f_idx in range(start_frame, end_frame + 1, step):
        if f_idx >= n_frames:
            print(f" Stopping early: Frame {f_idx} exceeds available frames ({n_frames}).")
            break
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Panel 1: Max Shear Rate (S33)
        shear_frame = masked_shear[:, :, f_idx]
        if not np.all(np.isnan(shear_frame)):
            levels_shear = np.linspace(np.nanmin(shear_frame), np.nanmax(shear_frame), 30)
            cf0 = axes[0].contourf(X * 1000, Y * 1000, shear_frame, levels=levels_shear, cmap='jet')
            fig.colorbar(cf0, ax=axes[0], label='Max Shear Rate (1/s)')
        axes[0].set_title(f"Max Shear Rate ($S_{{33}}$) - Frame {f_idx}")
        axes[0].set_xlabel("X coordinate (mm)")
        axes[0].set_ylabel("Y coordinate (mm)")
        axes[0].set_aspect('equal')
        axes[0].invert_yaxis() # Match top-left image origin
        
        # Panel 2: Velocity Magnitude (velmag)
        vel_frame = masked_velmag[:, :, f_idx]
        if not np.all(np.isnan(vel_frame)):
            levels_vel = np.linspace(np.nanmin(vel_frame), np.nanmax(vel_frame), 30)
            cf1 = axes[1].contourf(X * 1000, Y * 1000, vel_frame, levels=levels_vel, cmap='viridis')
            fig.colorbar(cf1, ax=axes[1], label='Velocity Magnitude (m/s)')
        axes[1].set_title(f"Velocity Magnitude - Frame {f_idx}")
        axes[1].set_xlabel("X coordinate (mm)")
        axes[1].set_ylabel("Y coordinate (mm)")
        axes[1].set_aspect('equal')
        axes[1].invert_yaxis()
        
        plt.suptitle(f"Condition: {condition} | Frame {f_idx}", fontsize=13, y=0.98)
        plt.tight_layout()
        
        # Save plots
        save_path = output_dir / f"tertiary_contour_frame_{f_idx:03d}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        generated_count += 1
        
    print(f"✅ Successfully exported {generated_count} contour plots!")
def _load_tertiary(path):
    """Load tertiary export fields from .npz or .mat (v7.3 HDF5 or legacy)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return dict(
                X=np.asarray(data["X"]),
                Y=np.asarray(data["Y"]),
                masked_v=np.asarray(data["masked_v"]),
                masked_shear=np.asarray(data["masked_shear"]),
                masked_velmag=np.asarray(data["masked_velmag"]),
                condition=str(data["condition"]),
            )

    if suffix == ".mat":
        try:
            with h5py.File(path, "r") as f:
                def _3d(key):
                    # PIVlab stores (nx, ny, nframes) in MATLAB → h5py reads as (nframes, ny, nx)
                    # transpose(1,2,0) → (ny, nx, nframes) which is (rows, cols, frames) for contourf
                    return np.transpose(np.array(f[key]), (1, 2, 0)).astype(np.float32)
                def _1d(key):
                    return np.array(f[key]).squeeze()
                cond_raw = np.array(f["condition"]).squeeze()
                condition = "".join(chr(int(c)) for c in np.atleast_1d(cond_raw))
                return dict(
                    X=_1d("X").astype(np.float64),
                    Y=_1d("Y").astype(np.float64),
                    masked_v=_3d("masked_v"),
                    masked_shear=_3d("masked_shear"),
                    masked_velmag=_3d("masked_velmag"),
                    condition=condition,
                )
        except OSError:
            pass
        from scipy.io import loadmat
        mat = loadmat(str(path))
        return dict(
            X=np.asarray(mat["X"]).squeeze().astype(np.float64),
            Y=np.asarray(mat["Y"]).squeeze().astype(np.float64),
            masked_v=np.asarray(mat["masked_v"]),
            masked_shear=np.asarray(mat["masked_shear"]),
            masked_velmag=np.asarray(mat["masked_velmag"]),
            condition=str(np.asarray(mat["condition"]).squeeze()),
        )

    raise ValueError(f"Unsupported tertiary file format '{suffix}'. Expected .npz or .mat.")


def plot_tertiary_contours(tertiary_file_path, start_frame=50, end_frame=200, step=10):
    print(f"\n[DEBUG] Loading tertiary export: {tertiary_file_path}")

    data = _load_tertiary(tertiary_file_path)
    X           = data["X"]
    Y           = data["Y"]
    masked_v    = data["masked_v"]
    masked_shear = data["masked_shear"]
    masked_velmag = data["masked_velmag"]
    condition   = data["condition"]
        
    print(f"[DEBUG] X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"[DEBUG] masked_v shape: {masked_v.shape}  (expect rows x cols x frames)")
    print(f"[DEBUG] X len={len(X.ravel())} should match cols={masked_v.shape[1]}, Y len={len(Y.ravel())} should match rows={masked_v.shape[0]}")

    output_dir = Path.home() / "Downloads" / f"tertiary_contours_{condition}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    generated_count = 0
    
    for f_idx in range(start_frame, end_frame + 1, step):
        if f_idx >= masked_v.shape[2]:
            print(f"[DEBUG] Stopping: Frame {f_idx} is out of bounds (max: {masked_v.shape[2]-1})")
            break
            
        # --- NEW: Plotting v_component along with the others ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # 3 panels now
        
        # 1. Shear Rate
        cf0 = axes[0].contourf(X * 1000, Y * 1000, masked_shear[:, :, f_idx], cmap='jet')
        fig.colorbar(cf0, ax=axes[0], label='Shear (1/s)')
        axes[0].set_title(f"Shear Rate - Frame {f_idx}")

        # 2. Velocity Magnitude
        cf1 = axes[1].contourf(X * 1000, Y * 1000, masked_velmag[:, :, f_idx], cmap='viridis')
        fig.colorbar(cf1, ax=axes[1], label='Vel Mag (m/s)')
        axes[1].set_title(f"Vel Mag - Frame {f_idx}")
        
        # 3. V-Component (The one you were missing)
        v_frame = masked_v[:, :, f_idx]
        # Define your range (levels) here
        levels = np.linspace(-0.25, 0.25, 30) 

        if not np.all(np.isnan(v_frame)):
            # 1. Pass the levels to contourf using the argument 'levels'
            # 2. Added 'extend="both"' to capture values outside your -0.25 to 0.25 range
            cf2 = axes[2].contourf(X * 1000, Y * 1000, v_frame, 
                           levels=levels, 
                           cmap='coolwarm', 
                           extend='both')
    
            # 3. Colorbar only needs the mappable (cf2) and the axis (ax)
            fig.colorbar(cf2, ax=axes[2], label='V-Component (m/s)')

        axes[2].set_title(f"V-Component - Frame {f_idx}")

        for ax in axes:
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xlabel("mm")
            
        save_path = output_dir / f"tertiary_contour_frame_{f_idx:03d}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        generated_count += 1
        print(f"Saved: {save_path.name}")
        
    print(f"✅ Total plots saved: {generated_count}")

if __name__ == "__main__":
    # --- Set this to skip the export step and plot directly from an existing tertiary file ---
    #tertiary_input = r"C:\Users\JackHu\Downloads\100mL-20deg-35cpm_Tertiary_Export.mat"
    tertiary_input = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\PipelineTest_new\Tertiary Exports\firstsave_Tertiary_Export.npz"
    # --- Only needed when tertiary_input is None ---
    secondary_npz    = r"C:\Users\JackHu\Downloads\100mL-20deg-35cpm_Secondary_Export.mat"
    mask_mat         = r"Z:\EXPERIMENTAL TECHNIQUES\PIV\Rock - Reflective Flakes\100mL-20deg-35cpm\Matlab analysis\Primary Export\Dynamic Masks\dynamic_masking_100mL-20deg-35cpm.mat"
    tertiary_out_dir = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data"
    vessel_radius_meters = 0.005

    try:
        if tertiary_input is not None:
            tertiary_path = tertiary_input
            print(f"Using existing tertiary export: {tertiary_path}")
        else:
            tertiary_path = run_tertiary_export_with_mat_mask(
                secondary_file=secondary_npz,
                mask_mat_file=mask_mat,
                R=vessel_radius_meters,
                out_dir=tertiary_out_dir
            )

        plot_tertiary_contours(
            tertiary_file_path=tertiary_path,
            start_frame=50,
            end_frame=200,
            step=10
        )

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")