"""v_threshold_sweep.py

Find the upper |v| threshold that makes the Python mean-|V| time series match
the MATLAB one. Both sources get the SAME dynamic mask applied first; then we
sweep an extra exclusion on Python's |v| (drop any vector with |v| > T from the
per-frame spatial mean) and report the T that best matches MATLAB.

Outputs:
  * console table of T vs Python overall mean|V| vs RMS-diff-to-MATLAB
  * best T (and what %ile / fraction-excluded it corresponds to)
  * a time-series comparison PNG in ~/Downloads/
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from piv_pipeline.io_loaders import load_primary, parse_filename
from plot_primary_export import convert_to_ms, load_dynamic_mask_for_grid

# ── inputs (same as plot_primary_export.py) ────────────────────────────────
PRIMARY_MAT = r"C:\Users\JackHu\Downloads\100mL-20deg-35cpm_Primary Export.mat"
PRIMARY_NPZ = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\Test1\firstsave.npz"
MASK_MAT    = r"Z:\EXPERIMENTAL TECHNIQUES\PIV\Rock - Reflective Flakes\100mL-20deg-35cpm\Matlab analysis\Primary Export\Dynamic Masks\dynamic_masking_100mL-20deg-35cpm.mat"
FPS = 500
N_FRAMES = 1000


def stack_abs_v(data, typevector, dyn_mask, n):
    """Return per-frame list of masked |v| arrays (m/s) and the boolean mask cube."""
    v_list = data["v_list"]
    n = min(n, data["n_pairs"], dyn_mask.shape[2])
    return v_list, n


def mean_v_series(v_list, typevector, dyn_mask, n, v_thresh=np.inf):
    """Per-frame spatial mean of |v| over valid cells, excluding |v| > v_thresh."""
    out = np.full(n, np.nan)
    for i in range(n):
        v = v_list[i]
        mask = typevector & dyn_mask[:, :, i] & np.isfinite(v)
        av = np.abs(v[mask])
        av = av[av <= v_thresh]
        if av.size:
            out[i] = av.mean()
    return out


def main():
    # ── load MATLAB (already m/s) ──────────────────────────────────────────
    print("Loading MATLAB primary...")
    mat = load_primary(PRIMARY_MAT)
    tv_m = mat["typevector"].astype(bool)
    dyn_m = load_dynamic_mask_for_grid(MASK_MAT, tv_m.shape)

    # ── load Python (px/frame → m/s) ───────────────────────────────────────
    print("Loading Python primary...")
    py = load_primary(PRIMARY_NPZ)
    convert_to_ms(py, fps=FPS, howstupid=1)
    tv_p = py["typevector"].astype(bool)
    dyn_p = load_dynamic_mask_for_grid(MASK_MAT, tv_p.shape)

    n = min(N_FRAMES, mat["n_pairs"], py["n_pairs"],
            dyn_m.shape[2], dyn_p.shape[2])
    print(f"\nComparing over {n} frames.\n")

    # ── target: MATLAB mean|V| series ──────────────────────────────────────
    mv_mat = mean_v_series(mat["v_list"], tv_m, dyn_m, n)
    mat_overall = np.nanmean(mv_mat)

    # ── Python, unthresholded ──────────────────────────────────────────────
    mv_py_raw = mean_v_series(py["v_list"], tv_p, dyn_p, n)
    py_overall = np.nanmean(mv_py_raw)

    print(f"MATLAB  overall mean|V| = {mat_overall*1000:8.3f} mm/s")
    print(f"Python  overall mean|V| = {py_overall*1000:8.3f} mm/s   "
          f"(ratio {py_overall/mat_overall:.2f}x)\n")

    # ── distribution of Python |v| (m/s) to choose sweep range ─────────────
    allv = []
    for i in range(n):
        v = py["v_list"][i]
        m = tv_p & dyn_p[:, :, i] & np.isfinite(v)
        allv.append(np.abs(v[m]))
    allv = np.concatenate(allv)
    print("Python |v| distribution (m/s):")
    for q in (50, 75, 90, 95, 97.5, 99, 99.5, 99.9):
        print(f"  p{q:<5} = {np.percentile(allv, q)*1000:8.3f} mm/s")
    print()

    # ── sweep upper threshold T ────────────────────────────────────────────
    # span from the median up past p99.9
    t_candidates = np.percentile(allv, np.linspace(80, 100, 41))
    print(f"{'T (mm/s)':>10} {'%ile':>6} {'%excl':>7} "
          f"{'Py mean|V|':>11} {'RMSdiff':>9}")
    best = None
    for T in t_candidates:
        mv = mean_v_series(py["v_list"], tv_p, dyn_p, n, v_thresh=T)
        py_o = np.nanmean(mv)
        rms = np.sqrt(np.nanmean((mv - mv_mat) ** 2)) * 1000
        pct_excl = float((allv > T).mean() * 100)
        pctile = float((allv <= T).mean() * 100)
        print(f"{T*1000:10.3f} {pctile:6.1f} {pct_excl:7.2f} "
              f"{py_o*1000:11.3f} {rms:9.3f}")
        if best is None or rms < best[1]:
            best = (T, rms, py_o, pct_excl, pctile)

    T, rms, py_o, pct_excl, pctile = best
    print(f"\n→ Best match: exclude |v| > {T*1000:.3f} mm/s "
          f"({T:.5f} m/s, px/frame≈{T/ (py['calv'] if py.get('has_calv') else py['calxy']*FPS):.3f})")
    print(f"   = drop the top {pct_excl:.2f}% of vertical vectors (above p{pctile:.1f})")
    print(f"   Python mean|V| {py_o*1000:.3f} mm/s vs MATLAB {mat_overall*1000:.3f} mm/s "
          f"(RMS diff {rms:.3f} mm/s)")

    # ── plot ───────────────────────────────────────────────────────────────
    mv_py_best = mean_v_series(py["v_list"], tv_p, dyn_p, n, v_thresh=T)
    fig, ax = plt.subplots(figsize=(14, 6))
    fr = np.arange(n)
    ax.plot(fr, mv_mat * 1000, lw=0.9, color="black", label="MATLAB mean|V|")
    ax.plot(fr, mv_py_raw * 1000, lw=0.8, color="tab:orange", alpha=0.7,
            label="Python (no threshold)")
    ax.plot(fr, mv_py_best * 1000, lw=0.9, color="tab:blue",
            label=f"Python (exclude |v|>{T*1000:.0f} mm/s)")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean |V|  (mm/s)")
    ax.set_title("Vertical velocity time series — MATLAB vs Python (thresholded)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    out = Path.home() / "Downloads" / "v_threshold_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
