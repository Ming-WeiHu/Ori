"""
fileopener.py — open and inspect a PIV results .npz produced by piv_gui.

Loads the 7 fields (calxy, calu, calv, x, y, u_original, v_original), prints a
human-readable summary, and exposes a small API for programmatic use.

Usage from a shell:
    python fileopener.py path/to/results.npz
    python fileopener.py path/to/results.npz --pair 0          # plot pair 0
    python fileopener.py path/to/results.npz --mean            # plot time-avg

Usage from Python:
    from fileopener import load
    r = load("results.npz")
    r.u_original.shape    # (n_pairs, n_rows, n_cols)
    r.u_ms(0)             # pair-0 u in m/s, NaN over mask
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from piv_pipeline.io_loaders import load_primary
except ImportError:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent / "piv_pipeline"))
    from io_loaders import load_primary


@dataclass
class PIVResults:
    """The 7 fields from a piv_gui save, with helpers for unit conversion."""

    calxy: float          # m / px
    calu: float           # (m/s) / (px/frame), signed by x-axis dropdown
    calv: float           # (m/s) / (px/frame), signed by y-axis dropdown
    x: np.ndarray         # (n_rows, n_cols) pixels
    y: np.ndarray         # (n_rows, n_cols) pixels
    u_original: np.ndarray  # (n_pairs, n_rows, n_cols) px/frame
    v_original: np.ndarray  # (n_pairs, n_rows, n_cols) px/frame
    path: Path

    @property
    def n_pairs(self) -> int:
        return self.u_original.shape[0]

    @property
    def grid_shape(self) -> tuple:
        return self.u_original.shape[1:]

    def u_ms(self, k: Optional[int] = None) -> np.ndarray:
        """u in m/s for pair k (or all pairs if k is None)."""
        arr = self.u_original if k is None else self.u_original[k]
        return arr * self.calu

    def v_ms(self, k: Optional[int] = None) -> np.ndarray:
        arr = self.v_original if k is None else self.v_original[k]
        return arr * self.calv

    def speed_ms(self, k: Optional[int] = None) -> np.ndarray:
        return np.hypot(self.u_ms(k), self.v_ms(k))

    def x_m(self) -> np.ndarray:
        return self.x * self.calxy

    def y_m(self) -> np.ndarray:
        return self.y * self.calxy

    def time_average(self) -> tuple:
        """Return (<u>_t, <v>_t) in px/frame, ignoring NaN. Columns that are
        entirely NaN across pairs (fully-masked) come back as NaN."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (np.nanmean(self.u_original, axis=0),
                    np.nanmean(self.v_original, axis=0))


# ───────────────────────────── loader ────────────────────────────────────────

REQUIRED_KEYS = ("calxy", "calu", "calv", "x", "y", "u_original", "v_original")


def load(path: str | Path) -> PIVResults:
    """Open a PIV results `.npz` or `.mat` file and return a `PIVResults`."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        d = np.load(p, allow_pickle=False)
        missing = [k for k in REQUIRED_KEYS if k not in d.files]
        if missing:
            raise KeyError(
                f"{p} is missing required keys: {missing}\n"
                f"got: {sorted(d.files)}")
        return PIVResults(
            calxy=float(d["calxy"]),
            calu=float(d["calu"]),
            calv=float(d["calv"]),
            x=d["x"],
            y=d["y"],
            u_original=d["u_original"],
            v_original=d["v_original"],
            path=p,
        )
    if suffix == ".mat":
        primary = load_primary(p)
        return PIVResults(
            calxy=float(primary["calxy"]),
            calu=float(primary.get("calu", 1.0)),
            calv=float(primary.get("calv", 1.0)),
            x=primary["x"],
            y=primary["y"],
            u_original=np.stack(primary["u_list"], axis=0),
            v_original=np.stack(primary["v_list"], axis=0),
            path=p,
        )
    raise ValueError(f"Unsupported file type: {p.suffix}. Use .npz or .mat.")


# ─────────────────────────── reporting ───────────────────────────────────────

def summarize(r: PIVResults) -> None:
    """Print a concise human-readable report."""
    print("=" * 64)
    print(r.path)
    print("=" * 64)
    print()
    print("--- scalars (first 3) ---")
    print(f"  calxy = {r.calxy:.6e} m/px        "
          f"(1 px = {r.calxy * 1000:.4f} mm)")
    print(f"  calu  = {r.calu:+.4f} (m/s)/(px/frame)")
    print(f"  calv  = {r.calv:+.4f} (m/s)/(px/frame)")
    print()
    print("--- matrices (last 4) ---")
    print(f"  x          shape {r.x.shape}      "
          f"pixels {float(r.x.min()):g} ... {float(r.x.max()):g}")
    print(f"  y          shape {r.y.shape}      "
          f"pixels {float(r.y.min()):g} ... {float(r.y.max()):g}")
    print(f"  u_original shape {r.u_original.shape}   px/frame, NaN over mask")
    print(f"  v_original shape {r.v_original.shape}")
    print()

    u, v = r.u_original, r.v_original
    valid_per_pair = int(np.isfinite(u[0]).sum())
    total = u[0].size
    speed_px = np.hypot(u, v)
    speed_ms = speed_px * abs(r.calu)
    print("--- field stats across all pairs ---")
    print(f"  valid vectors per pair : {valid_per_pair} / {total}  "
          f"(mask keeps {100 * valid_per_pair / total:.1f}% of grid)")
    print(f"  mean  |V|              : {np.nanmean(speed_px):.3f} px/frame   "
          f"({np.nanmean(speed_ms) * 1000:.2f} mm/s)")
    print(f"  max   |V|              : {np.nanmax(speed_px):.3f} px/frame   "
          f"({np.nanmax(speed_ms) * 1000:.2f} mm/s)")
    print(f"  median u, v            : {float(np.nanmedian(u)):+.3f}, "
          f"{float(np.nanmedian(v)):+.3f} px/frame")
    print(f"  NaN fraction (masked)  : {np.isnan(u).mean():.3f}")
    print()

    # First / middle / last 5 pairs
    print("--- per-pair time series (first / middle / last 5) ---")
    print("  pair    med_u_px   med_v_px    mean|V| (mm/s)")
    def line(k):
        mu = float(np.nanmedian(u[k])); mv = float(np.nanmedian(v[k]))
        mm = float(np.nanmean(np.hypot(u[k], v[k])) * abs(r.calu) * 1000)
        print(f"   {k:4d}   {mu:+7.3f}    {mv:+7.3f}     {mm:7.2f}")
    n = r.n_pairs
    head = list(range(min(5, n)))
    mid = list(range(max(0, n // 2 - 2), min(n, n // 2 + 3))) if n > 10 else []
    tail = list(range(max(0, n - 5), n)) if n > 10 else []
    seen = set()
    for k in head:
        line(k); seen.add(k)
    if mid and min(mid) > max(head) + 1:
        print("   ...")
    for k in mid:
        if k not in seen:
            line(k); seen.add(k)
    if tail and min(tail) > max(seen) + 1:
        print("   ...")
    for k in tail:
        if k not in seen:
            line(k); seen.add(k)
    print()

    # Time-averaged field
    u_mean, v_mean = r.time_average()
    print("--- time-averaged field ---")
    print(f"  <u>_t  range {float(np.nanmin(u_mean)):+.3f} ... "
          f"{float(np.nanmax(u_mean)):+.3f} px/frame")
    print(f"  <v>_t  range {float(np.nanmin(v_mean)):+.3f} ... "
          f"{float(np.nanmax(v_mean)):+.3f} px/frame")
    mean_mag = float(
        np.nanmean(np.hypot(u_mean, v_mean)) * abs(r.calu) * 1000)
    print(f"  <|V|>_t mean magnitude : {mean_mag:.2f} mm/s")


# ─────────────────────────── optional plot ───────────────────────────────────

def plot_pair(r: PIVResults, k: int = 0,
              ax=None, downsample: int = 2) -> None:
    """Quiver-plot a single pair. Requires matplotlib."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    s = downsample
    u = r.u_ms(k)[::s, ::s]
    v = r.v_ms(k)[::s, ::s]
    x = r.x[::s, ::s]
    y = r.y[::s, ::s]
    speed = np.hypot(u, v) * 1000  # mm/s
    q = ax.quiver(x, y, u, v, speed, cmap="viridis", angles="xy",
                  scale_units="xy")
    ax.set_aspect("equal")
    ax.invert_yaxis()           # image-y points down
    ax.set_title(f"{r.path.name} — pair {k}")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    plt.colorbar(q, ax=ax, label="|V| (mm/s)")


def plot_mean(r: PIVResults, ax=None, downsample: int = 2) -> None:
    """Quiver-plot the time-averaged field. Requires matplotlib."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    u_mean, v_mean = r.time_average()
    s = downsample
    u = u_mean[::s, ::s] * r.calu
    v = v_mean[::s, ::s] * r.calv
    x = r.x[::s, ::s]
    y = r.y[::s, ::s]
    speed = np.hypot(u, v) * 1000
    q = ax.quiver(x, y, u, v, speed, cmap="viridis", angles="xy",
                  scale_units="xy")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{r.path.name} — time average ({r.n_pairs} pairs)")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    plt.colorbar(q, ax=ax, label="|V| (mm/s)")


def plot_uv_curve(r: PIVResults, ax=None) -> None:
    """Plot the time series of average u and average v across all frames in m/s."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    u_avg = np.nanmean(r.u_original, axis=(1, 2)) * r.calu
    v_avg = np.nanmean(r.v_original, axis=(1, 2)) * r.calv
    frames = np.arange(r.n_pairs)

    ax.plot(frames, u_avg, label="mean u", color="tab:blue", linewidth=1.5)
    ax.plot(frames, v_avg, label="mean v", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("frame index")
    ax.set_ylabel("mean velocity (m/s)")
    ax.set_title(f"{r.path.name} — mean u/v time series")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()


def plot_uv_curve_raw(r: PIVResults, ax=None) -> None:
    """Plot the time series of average u and average v across all frames in raw px/frame."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    u_avg = np.nanmean(r.u_original, axis=(1, 2))
    v_avg = np.nanmean(r.v_original, axis=(1, 2))
    frames = np.arange(r.n_pairs)

    ax.plot(frames, u_avg, label="mean u", color="tab:blue", linewidth=1.5)
    ax.plot(frames, v_avg, label="mean v", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("frame index")
    ax.set_ylabel("mean velocity (px/frame)")
    ax.set_title(f"{r.path.name} — mean u/v time series (raw px/frame)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

# ─────────────────────────────── CLI ─────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Open a PIV results .npz or .mat file."
    )
    p.add_argument("npz", help="path to results.npz or results.mat")
    p.add_argument("--pair", type=int, default=None,
                   help="plot a specific pair index (requires matplotlib)")
    p.add_argument("--mean", action="store_true",
                   help="plot the time-averaged field")
    p.add_argument("--uv-curve", action="store_true",
                   help="plot the mean u and mean v time series in m/s")
    p.add_argument("--uv-curve-raw", action="store_true",
                   help="plot the mean u and mean v time series in raw px/frame")
    p.add_argument("--downsample", type=int, default=2,
                   help="quiver downsample factor (default 2)")
    p.add_argument("--output-dir", default=None,
                   help="folder to save generated plots (default: input file folder)")
    p.add_argument("--no-summary", action="store_true",
                   help="skip the text summary")
    args = p.parse_args()

    print(f"Running fileopener on: {args.npz}")
    r = load(args.npz)
    if not args.no_summary:
        summarize(r)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else r.path.parent
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pair is not None or args.mean or args.uv_curve or args.uv_curve_raw:
        import matplotlib.pyplot as plt
        if args.pair is not None:
            plot_pair(r, args.pair, downsample=args.downsample)
            output_image = output_dir / f"flow_field_pair_{args.pair}.png"
            plt.gcf().savefig(output_image, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✅ Pair plot saved successfully to: {output_image}")
        if args.mean:
            plot_mean(r, downsample=args.downsample)
            output_image = output_dir / "flow_field_mean.png"
            plt.gcf().savefig(output_image, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✅ Mean plot saved successfully to: {output_image}")
        if args.uv_curve:
            plot_uv_curve(r)
            output_image = output_dir / "flow_field_uv_curve.png"
            plt.gcf().savefig(output_image, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✅ UV curve saved successfully to: {output_image}")
        if args.uv_curve_raw:
            plot_uv_curve_raw(r)
            output_image = output_dir / "flow_field_uv_curve_raw.png"
            plt.gcf().savefig(output_image, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✅ Raw UV curve saved successfully to: {output_image}")



if __name__ == "__main__":
    _cli()
