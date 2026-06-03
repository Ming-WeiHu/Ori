"""plot_primary_export.py

Loads a PIVlab primary export (.mat v6/v7/v7.3 or piv_simple .npz) and produces:
  1. Side-by-side u / v contour plots for a chosen frame range.
  2. A time-series plot of spatially-averaged |u| and |v| over up to 1000 frames.

Dynamic masking is applied per-frame when a mask file is supplied (same logic as
the tertiary pipeline).  Pass mask_file=None to use the static typevector only.

All plots are saved to ~/Downloads/primary_comparison/{matlab|python}/.
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from piv_pipeline.io_loaders import load_primary, parse_filename
from piv_pipeline.fn_tertiary_export import _load_dynamic_mask


# ─────────────────────── mask loading / alignment ─────────────────────────

def load_dynamic_mask_for_grid(mask_path, grid_shape) -> np.ndarray:
    """Load a dynamic mask and align it to the PIV velocity grid.

    Handles three cases automatically:
      * mask already at grid resolution+orientation → used as-is
      * mask transposed relative to grid             → axes swapped
      * mask at a different resolution (e.g. full     → nearest-resize per frame
        image resolution) to the PIV vector grid

    grid_shape : (rows, cols) of the velocity field.
    Returns (rows, cols, frames) bool.
    """
    raw = _load_dynamic_mask(Path(mask_path))   # (a, b, frames) bool
    rows, cols = grid_shape
    a, b, n_frames = raw.shape
    print(f"  [mask] raw shape {raw.shape}, target grid {(rows, cols)}")

    if (a, b) == (rows, cols):
        aligned = raw
    elif (a, b) == (cols, rows):
        print("  [mask] orientation transposed vs grid — swapping axes (1,0,2)")
        aligned = np.transpose(raw, (1, 0, 2))
    else:
        print(f"  [mask] resolution mismatch — nearest-resizing each frame "
              f"{(a, b)} → {(rows, cols)}")
        aligned = np.zeros((rows, cols, n_frames), dtype=bool)
        for k in range(n_frames):
            r = cv2.resize(raw[:, :, k].astype(np.float32), (cols, rows),
                           interpolation=cv2.INTER_NEAREST)
            aligned[:, :, k] = r >= 0.5

    frac = aligned.mean()
    print(f"  [mask] aligned shape {aligned.shape}, "
          f"mean kept fraction = {frac:.3f} "
          f"({'⚠ mask is ~all-True, little visible effect' if frac > 0.98 else 'ok'})")
    return aligned


# ─────────────────────── unit conversion (→ m/s) ──────────────────────────

def convert_to_ms(data: dict, fps: float, howstupid: float = 1.0) -> dict:
    """Convert u_list / v_list from PIXELS/FRAME to METRES/SECOND in place.

    Mirrors fn_secondary_export:
      * if calu/calv present → multiply by them (already m/s per px/frame)
      * else fallback factor  → calxy * fps

    NOTE: fn_secondary_export also negates v (image-y points down). That sign
    flip is NOT applied here so the raw PIVlab sign convention is preserved —
    flip manually if your MATLAB m/s file uses the negated convention.
    """
    calxy = data["calxy"]
    if data.get("has_calu") and data.get("has_calv"):
        u_factor = howstupid * data["calu"]
        v_factor = howstupid * data["calv"]
        print(f"  [units] px/frame → m/s via calu={data['calu']:.6g}, "
              f"calv={data['calv']:.6g}")
    else:
        fallback = calxy * fps
        u_factor = v_factor = howstupid * fallback
        print(f"  [units] px/frame → m/s via fallback calxy*fps = {fallback:.6g}")

    data["u_list"] = [u * u_factor for u in data["u_list"]]
    data["v_list"] = [v * v_factor for v in data["v_list"]]
    return data


# ─────────────────────────── contour plots ────────────────────────────────

def plot_uv_contours(primary_data: dict, condition: str,
                     start_frame: int = 50, end_frame: int = 200, step: int = 10,
                     dynamic_mask: np.ndarray | None = None,
                     clim: float = 0.25,
                     out_dir: Path | None = None) -> None:
    """Save side-by-side u / v filled contour plots for every `step`-th frame.

    dynamic_mask : (rows, cols, frames) bool array, or None for static mask only.
    clim         : contour scale limit; levels span [-clim, +clim].
    """
    u_list     = primary_data["u_list"]
    v_list     = primary_data["v_list"]
    x          = primary_data["x"]
    y          = primary_data["y"]
    typevector = primary_data["typevector"].astype(bool)
    n_pairs    = primary_data["n_pairs"]

    if out_dir is None:
        out_dir = Path.home() / "Downloads" / f"primary_plots_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving contour plots to: {out_dir}")

    end_frame = min(end_frame, n_pairs - 1)
    if dynamic_mask is not None:
        end_frame = min(end_frame, dynamic_mask.shape[2] - 1)

    count = 0
    for f_idx in tqdm(range(start_frame, end_frame + 1, step),
                      desc="Contour frames", unit="frame"):
        u_frame = u_list[f_idx].copy()
        v_frame = v_list[f_idx].copy()

        if dynamic_mask is not None:
            mask = typevector & dynamic_mask[:, :, f_idx]
        else:
            mask = typevector

        u_frame[~mask] = np.nan
        v_frame[~mask] = np.nan

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

        for ax, field, label, cmap in [
            (axes[0], u_frame, "U  (m/s)", "coolwarm"),
            (axes[1], v_frame, "V  (m/s)", "coolwarm"),
        ]:
            if not np.all(np.isnan(field)):
                levels = np.linspace(-clim, clim, 30)
                cf = ax.contourf(
                    x * 1000, y * 1000, field,
                    levels=levels,
                    cmap=cmap, extend="both",
                )
                fig.colorbar(cf, ax=ax, label=label)
            ax.set_title(f"{label.split()[0]} — Frame {f_idx}")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_aspect("equal")
            ax.invert_yaxis()

        mask_note = "typevector + dynamic mask" if dynamic_mask is not None else "typevector only"
        fig.suptitle(f"{condition} | Frame {f_idx}  [{mask_note}]", fontsize=13)

        save_path = out_dir / f"uv_contour_frame_{f_idx:04d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1

    print(f"  Saved {count} contour plots.")


# ─────────────────────────── time-series plot ─────────────────────────────

def plot_time_series(primary_data: dict, condition: str,
                     n_frames: int = 1000,
                     dynamic_mask: np.ndarray | None = None,
                     out_dir: Path | None = None) -> None:
    """Plot spatially-averaged |u| and |v| vs frame index over `n_frames` frames."""
    u_list     = primary_data["u_list"]
    v_list     = primary_data["v_list"]
    typevector = primary_data["typevector"].astype(bool)
    n_pairs    = primary_data["n_pairs"]

    if out_dir is None:
        out_dir = Path.home() / "Downloads" / f"primary_plots_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = min(n_frames, n_pairs)
    if dynamic_mask is not None:
        n_frames = min(n_frames, dynamic_mask.shape[2])

    frames = np.arange(n_frames)
    mean_u = np.full(n_frames, np.nan)
    mean_v = np.full(n_frames, np.nan)

    print(f"Computing spatial mean over {n_frames} frames...")
    for i in tqdm(range(n_frames), desc="Time series", unit="frame"):
        u = u_list[i]
        v = v_list[i]

        if dynamic_mask is not None:
            mask = typevector & dynamic_mask[:, :, i]
        else:
            mask = typevector

        valid = mask & np.isfinite(u) & np.isfinite(v)
        if valid.any():
            mean_u[i] = np.mean(np.abs(u[valid]))
            mean_v[i] = np.mean(np.abs(v[valid]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(frames, mean_u, linewidth=0.8, color="steelblue")
    axes[0].set_ylabel("Mean |U|  (m/s)")
    axes[0].set_title(f"{condition} — Spatial mean velocity magnitude over time")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(frames, mean_v, linewidth=0.8, color="darkorange")
    axes[1].set_ylabel("Mean |V|  (m/s)")
    axes[1].set_xlabel("Frame index")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = out_dir / f"time_series_{condition}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved time-series plot: {save_path}")


# ──────────────────────────────── main ────────────────────────────────────

if __name__ == "__main__":
    # ── Input files ─────────────────────────────────────────────────────────
    primary_mat = r"C:\Users\JackHu\Downloads\100mL-20deg-35cpm_Primary Export.mat"
    primary_npz = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\Test1\firstsave.npz"

    # Dynamic mask for the MATLAB source (v7.3 .mat).
    # Set to None for a source that has no dynamic mask.
    mask_mat = r"Z:\EXPERIMENTAL TECHNIQUES\PIV\Rock - Reflective Flakes\100mL-20deg-35cpm\Matlab analysis\Primary Export\Dynamic Masks\dynamic_masking_100mL-20deg-35cpm.mat"

    # ── Contour plot range (frames 50–200, every 10) ────────────────────────
    CONTOUR_START = 50
    CONTOUR_END   = 200
    CONTOUR_STEP  = 10
    # Contour colorscale limit (m/s). ±0.25 saturates instantaneous primary
    # velocities (p99 ≈ 0.47 MATLAB / 1.5 Python) — raise so detail is visible.
    CONTOUR_CLIM  = .4

    # ── Time series ─────────────────────────────────────────────────────────
    TIME_SERIES_FRAMES = 1000

    # ── Units ───────────────────────────────────────────────────────────────
    FPS = 500          # used for px/frame → m/s fallback (calxy * fps)
    HOWSTUPID = 1      # PIVlab scaling correction (1 normally, 10 for legacy bug)

    comparison_root = Path.home() / "Downloads" / "primary_comparison"

    # (primary_file, folder_label, dynamic_mask_file_or_None, already_in_ms)
    # MATLAB primary is already m/s; Python primary is px/frame and gets converted.
    sources = [
        (primary_mat, "matlab", mask_mat, True),
        (primary_npz, "python", mask_mat, False),
    ]

    for file_path, label, mask_path, already_ms in sources:
        print(f"\n{'='*60}")
        print(f"  {label.upper()} — {Path(file_path).name}")
        print(f"{'='*60}")
        try:
            data = load_primary(file_path)
            print(f"  Source : {data['source']}")
            print(f"  Frames : {data['n_pairs']}")
            print(f"  Grid   : {data['x'].shape}  (rows x cols)")
            print(f"  calxy  : {data['calxy']:.6f} m/px")

            condition = parse_filename(Path(file_path).stem)["condition"]
            print(f"  Condition: {condition}")

            # Convert to m/s if the source isn't already
            if already_ms:
                print("  [units] source already in m/s — no conversion.")
            else:
                convert_to_ms(data, fps=FPS, howstupid=HOWSTUPID)

            # Fingerprint — proves the two sources hold DIFFERENT data and shows
            # whether the ±0.25 contour levels actually exercise the colorscale.
            # If matlab & python print identical numbers → same input (bug/aliasing).
            # If |u|/|v| p99 ≪ 0.25 → everything sits near white, plots look "identical".
            tv = data["typevector"].astype(bool)
            mid = min(100, data["n_pairs"] - 1)
            um = data["u_list"][mid][tv]; um = um[np.isfinite(um)]
            vm = data["v_list"][mid][tv]; vm = vm[np.isfinite(vm)]
            if um.size and vm.size:
                print(f"  [fingerprint] grid={data['x'].shape} n_frames={data['n_pairs']} "
                      f"frame{mid}: "
                      f"|u| mean={np.mean(np.abs(um)):.4g} std={np.std(um):.4g} "
                      f"p99={np.percentile(np.abs(um),99):.4g} | "
                      f"|v| mean={np.mean(np.abs(vm)):.4g} "
                      f"p99={np.percentile(np.abs(vm),99):.4g}  (levels ±0.25)")

            dynamic_mask = None
            if mask_path is not None:
                print(f"  Loading dynamic mask: {Path(mask_path).name}")
                grid_shape = data["typevector"].shape  # (rows, cols)
                dynamic_mask = load_dynamic_mask_for_grid(mask_path, grid_shape)
            else:
                print("  No dynamic mask — using typevector only.")

            out_dir = comparison_root / label

            plot_uv_contours(
                data, condition,
                start_frame=CONTOUR_START,
                end_frame=CONTOUR_END,
                step=CONTOUR_STEP,
                dynamic_mask=dynamic_mask,
                clim=CONTOUR_CLIM,
                out_dir=out_dir,
            )

            plot_time_series(
                data, condition,
                n_frames=TIME_SERIES_FRAMES,
                dynamic_mask=dynamic_mask,
                out_dir=out_dir,
            )

        except Exception as e:
            print(f"\n❌ Failed for {label}: {e}")
            raise

    print(f"\nDone. Comparison output in:\n  {comparison_root}")
    print(f"  ├── matlab/   (typevector + MATLAB dynamic mask)")
    print(f"  └── python/   (typevector + MATLAB dynamic mask)")
