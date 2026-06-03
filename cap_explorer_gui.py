"""cap_explorer_gui.py — interactive |v| (and |u|) cap explorer.

Built on plot_primary_export.py's loaders. Loads a Python primary export
(.npz) and, optionally, a MATLAB primary (.mat) as a reference, applies the
same dynamic mask, then lets you drag a velocity-cap slider and watch the
spatial-mean |V| time series respond live.

The cap excludes any vector whose |v| exceeds the slider value from each
frame's spatial mean — exactly the operation explored in v_threshold_sweep.py,
but interactive. A second slider caps |u| the same way (optional).

Run:
    python cap_explorer_gui.py
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent))
from piv_pipeline.io_loaders import load_primary, parse_filename
from plot_primary_export import convert_to_ms, load_dynamic_mask_for_grid


# Default paths (same as plot_primary_export.py) — editable in the UI.
DEF_NPZ  = r"C:\Users\JackHu\OneDrive - Oribiotech Ltd\Desktop\sample data\Test1\firstsave.npz"
DEF_MAT  = r"C:\Users\JackHu\Downloads\100mL-20deg-35cpm_Primary Export.mat"
DEF_MASK = r"Z:\EXPERIMENTAL TECHNIQUES\PIV\Rock - Reflective Flakes\100mL-20deg-35cpm\Matlab analysis\Primary Export\Dynamic Masks\dynamic_masking_100mL-20deg-35cpm.mat"
DEF_FPS  = 500
N_FRAMES = 1000


def _precompute_absvel(data, dyn_mask, n):
    """Per-frame flat arrays of |u| and |v| (m/s) over valid cells.

    Returns (absu_list, absv_list) — each a list of 1-D float arrays, one per
    frame, holding the magnitudes that survive typevector ∩ dynamic mask.
    Slider updates then just threshold + mean these, which is fast.
    """
    tv = data["typevector"].astype(bool)
    u_list, v_list = data["u_list"], data["v_list"]
    absu, absv = [], []
    for i in range(n):
        u, v = u_list[i], v_list[i]
        if dyn_mask is not None:
            m = tv & dyn_mask[:, :, i]
        else:
            m = tv
        m = m & np.isfinite(u) & np.isfinite(v)
        absu.append(np.abs(u[m]))
        absv.append(np.abs(v[m]))
    return absu, absv


def _series_with_cap(abs_list, cap):
    """Per-frame mean of values <= cap. NaN if a frame is fully excluded."""
    out = np.full(len(abs_list), np.nan)
    for i, a in enumerate(abs_list):
        if a.size:
            kept = a[a <= cap]
            if kept.size:
                out[i] = kept.mean()
    return out


def _shear_series(u2d, v2d, tv, dyn, dx, dy, n, vcap, ucap,
                  AA=10, stride=10):
    """Spatial-mean shear-rate (1/s) per AA-window, with the cap applied.

    Mirrors the pipeline's Stage 1: cap → AA-frame nanmean (NaN→0) → velocity
    gradients over metric spacing → S33 (max shear) → spatial mean over the
    mask. Subsampled by `stride` windows for a responsive preview. Returns
    (frame_centres, shear_values).
    """
    centres, vals = [], []
    n_windows = n - AA + 1
    for g in range(0, n_windows, stride):
        us, vs = [], []
        for k in range(g, g + AA):
            u = u2d[k].astype(np.float64).copy()
            v = v2d[k].astype(np.float64).copy()
            bad = (np.abs(v) > vcap) | (np.abs(u) > ucap)
            u[bad] = np.nan
            v[bad] = np.nan
            us.append(u)
            vs.append(-v)            # pipeline sign-flips v
        with np.errstate(all="ignore"):
            u_avg = np.nanmean(np.stack(us, -1), axis=2)
            v_avg = np.nanmean(np.stack(vs, -1), axis=2)
        u_avg = np.where(np.isnan(u_avg), 0.0, u_avg)
        v_avg = np.where(np.isnan(v_avg), 0.0, v_avg)
        dudy, dudx = np.gradient(u_avg, dy, dx)
        dvdy, dvdx = np.gradient(v_avg, dy, dx)
        Sxx, Syy, Sxy = dudx, dvdy, 0.5 * (dudy + dvdx)
        rad = np.sqrt((0.5 * (Sxx - Syy)) ** 2 + Sxy ** 2)
        S33 = rad                    # 0.5*(lam1-lam2) == rad
        mask = tv & dyn[:, :, g] if dyn is not None else tv
        m = mask & np.isfinite(S33)
        centres.append(g)
        vals.append(float(np.nanmean(S33[m])) if m.any() else np.nan)
    return np.array(centres), np.array(vals)


class CapExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PIV velocity-cap explorer")
        self.geometry("1300x860")
        self.minsize(1000, 680)
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        # Data state
        self.py_absv = self.py_absu = None
        self.mat_mv = self.mat_mu = None       # MATLAB reference series (m/s)
        self.mat_absv_max = None               # max |v| in MATLAB field (m/s)
        self.mat_absv_p999 = None              # p99.9 |v| in MATLAB field (m/s)
        self.py_mv_raw = self.py_mu_raw = None  # Python uncapped series
        self.all_absv = None                    # concatenated for percentile calc
        self.all_absu = None
        self.calv = 1.0                         # m/s per px/frame (for px readout)
        self.n = 0
        self.condition = ""
        # 2D fields for the optional shear preview
        self.py_u2d = self.py_v2d = None        # lists of 2D m/s frames
        self.tv2d = None                        # typevector (bool)
        self.dyn2d = None                       # dynamic mask cube or None
        self.dx = self.dy = 1.0                 # grid spacing (metres)
        self.mat_shear = None                   # MATLAB shear series (1/s)

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # File row(s)
        files = ttk.LabelFrame(self, text="Inputs", padding=8)
        files.pack(fill="x", padx=8, pady=6)
        self.npz_var  = tk.StringVar(value=DEF_NPZ)
        self.mat_var  = tk.StringVar(value=DEF_MAT)
        self.mask_var = tk.StringVar(value=DEF_MASK)
        self.fps_var  = tk.StringVar(value=str(DEF_FPS))
        self._file_row(files, "Python .npz:", self.npz_var, 0, ("npz",))
        self._file_row(files, "MATLAB .mat (ref, optional):", self.mat_var, 1, ("mat",))
        self._file_row(files, "Dynamic mask (optional):", self.mask_var, 2, ("mat", "npz"))
        r = ttk.Frame(files); r.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(r, text="FPS:").pack(side="left")
        ttk.Entry(r, textvariable=self.fps_var, width=8).pack(side="left", padx=6)
        self.load_btn = ttk.Button(r, text="Load", command=self._load_async)
        self.load_btn.pack(side="left", padx=12)
        self.status = ttk.Label(r, text="Set paths and click Load.", foreground="gray")
        self.status.pack(side="left", padx=8)

        # Cap sliders
        caps = ttk.LabelFrame(self, text="Velocity caps (mm/s) — exclude vectors above", padding=8)
        caps.pack(fill="x", padx=8, pady=4)
        self.vcap = tk.DoubleVar(value=400.0)
        self.ucap = tk.DoubleVar(value=2000.0)
        self.vcap_str = tk.StringVar(value="400")   # type-in box (synced)
        self.ucap_str = tk.StringVar(value="2000")
        self.vcap_lbl = ttk.Label(caps, text="", width=24, font=("Consolas", 9))
        self.ucap_lbl = ttk.Label(caps, text="", width=24, font=("Consolas", 9))
        self.vscale = ttk.Scale(caps, from_=10, to=2000, orient="horizontal",
                                variable=self.vcap, command=lambda *_: self._update())
        self.uscale = ttk.Scale(caps, from_=10, to=4000, orient="horizontal",
                                variable=self.ucap, command=lambda *_: self._update())
        # Row 0: |v| cap  →  label | slider | type-in | readout
        ttk.Label(caps, text="|v| cap:", width=9).grid(row=0, column=0, sticky="w")
        self.vscale.grid(row=0, column=1, sticky="ew", padx=6)
        vent = ttk.Entry(caps, textvariable=self.vcap_str, width=9)
        vent.grid(row=0, column=2, padx=4)
        vent.bind("<Return>", lambda e: self._commit_entry("v"))
        vent.bind("<FocusOut>", lambda e: self._commit_entry("v"))
        self.vcap_lbl.grid(row=0, column=3, sticky="w")
        # Row 1: |u| cap
        ttk.Label(caps, text="|u| cap:", width=9).grid(row=1, column=0, sticky="w")
        self.uscale.grid(row=1, column=1, sticky="ew", padx=6)
        uent = ttk.Entry(caps, textvariable=self.ucap_str, width=9)
        uent.grid(row=1, column=2, padx=4)
        uent.bind("<Return>", lambda e: self._commit_entry("u"))
        uent.bind("<FocusOut>", lambda e: self._commit_entry("u"))
        self.ucap_lbl.grid(row=1, column=3, sticky="w")
        caps.columnconfigure(1, weight=1)
        ttk.Label(caps, text="(type a value + Enter, or drag)", foreground="gray"
                  ).grid(row=2, column=1, sticky="w", padx=6)
        # Quick-set buttons for v percentiles
        qf = ttk.Frame(caps); qf.grid(row=3, column=0, columnspan=4, sticky="w", pady=4)
        ttk.Label(qf, text="Set |v| cap to:").pack(side="left")
        for lbl in ("p95", "p99", "p99.5", "p99.9", "MATLAB p99.9", "MATLAB max"):
            ttk.Button(qf, text=lbl, width=9,
                       command=lambda l=lbl: self._quickset(l)).pack(side="left", padx=2)
        # Shear preview toggle
        sf = ttk.Frame(caps); sf.grid(row=4, column=0, columnspan=4, sticky="w", pady=2)
        self.show_shear = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, text="Show shear preview (recomputes on slider release — slower)",
                        variable=self.show_shear,
                        command=self._update_shear).pack(side="left")

        # Export row
        exp = ttk.Frame(self, padding=(8, 2))
        exp.pack(fill="x", padx=8)
        self.export_btn = ttk.Button(exp, text="Export capped .npz",
                                     command=self._export_npz, state="disabled")
        self.export_btn.pack(side="left")
        ttk.Label(exp, text="(NaNs any vector with |v|>cap or |u|>cap; "
                            "7-key piv format, pipeline-ready)",
                  foreground="gray").pack(side="left", padx=8)
        self.export_status = ttk.Label(exp, text="", foreground="gray")
        self.export_status.pack(side="left", padx=8)

        # Stats readout
        self.stats = ttk.Label(self, text="", font=("Consolas", 10), justify="left")
        self.stats.pack(fill="x", padx=12, pady=2)

        # Plot — V, U, shear (3 rows)
        self.fig = Figure(figsize=(12, 9), dpi=90)
        self.ax_v = self.fig.add_subplot(311)
        self.ax_u = self.fig.add_subplot(312, sharex=self.ax_v)
        self.ax_shear = self.fig.add_subplot(313, sharex=self.ax_v)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)
        # Shear preview computes on slider RELEASE (not live drag) to stay snappy
        self.vscale.bind("<ButtonRelease-1>", lambda e: self._update_shear())
        self.uscale.bind("<ButtonRelease-1>", lambda e: self._update_shear())
        self._shear_uncapped = None   # cached uncapped shear series

    def _file_row(self, parent, label, var, row, exts):
        ttk.Label(parent, text=label, width=26).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=80).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="…", width=3,
                   command=lambda: self._browse(var, exts)).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)

    def _browse(self, var, exts):
        ft = [(f"*.{e}", f"*.{e}") for e in exts] + [("All", "*.*")]
        p = filedialog.askopenfilename(filetypes=ft)
        if p:
            var.set(p)

    # ---------------------------------------------------------------- load
    def _load_async(self):
        self.load_btn.config(state="disabled")
        self.status.config(text="Loading… (large files take ~10-30s)", foreground="black")
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            fps = float(self.fps_var.get())
            npz_path = self.npz_var.get().strip()
            mat_path = self.mat_var.get().strip()
            mask_path = self.mask_var.get().strip()

            py = load_primary(npz_path)
            convert_to_ms(py, fps=fps, howstupid=1)
            self.calv = float(py["calv"]) if py.get("has_calv") else py["calxy"] * fps
            self.condition = parse_filename(Path(npz_path).stem)["condition"]

            dyn_py = None
            if mask_path and Path(mask_path).exists():
                dyn_py = load_dynamic_mask_for_grid(mask_path, py["typevector"].shape)

            n = min(N_FRAMES, py["n_pairs"])
            if dyn_py is not None:
                n = min(n, dyn_py.shape[2])

            absu, absv = _precompute_absvel(py, dyn_py, n)
            self.py_absu, self.py_absv, self.n = absu, absv, n
            self.all_absv = np.concatenate([a for a in absv if a.size])
            self.all_absu = np.concatenate([a for a in absu if a.size])
            self.py_mv_raw = _series_with_cap(absv, np.inf)
            self.py_mu_raw = _series_with_cap(absu, np.inf)

            # Stash 2D fields + grid for the optional shear preview
            self.py_u2d = py["u_list"]
            self.py_v2d = py["v_list"]
            self.tv2d = py["typevector"].astype(bool)
            self.dyn2d = dyn_py
            xg = np.asarray(py["x"]); yg = np.asarray(py["y"])
            self.dx = float(xg[0, 1] - xg[0, 0]) if xg.shape[1] > 1 else 1.0
            self.dy = float(yg[1, 0] - yg[0, 0]) if yg.shape[0] > 1 else 1.0

            # MATLAB reference (already m/s)
            self.mat_mv = self.mat_mu = None
            if mat_path and Path(mat_path).exists():
                mat = load_primary(mat_path)
                dyn_m = (load_dynamic_mask_for_grid(mask_path, mat["typevector"].shape)
                         if (mask_path and Path(mask_path).exists()) else None)
                nm = min(n, mat["n_pairs"])
                mabsu, mabsv = _precompute_absvel(mat, dyn_m, nm)
                mv = _series_with_cap(mabsv, np.inf)
                mu = _series_with_cap(mabsu, np.inf)
                # pad/truncate to n
                self.mat_mv = np.full(n, np.nan); self.mat_mv[:len(mv)] = mv[:n]
                self.mat_mu = np.full(n, np.nan); self.mat_mu[:len(mu)] = mu[:n]
                # MATLAB |v| field tail — for the data-driven cap buttons
                mat_all = np.concatenate([a for a in mabsv if a.size])
                self.mat_absv_max = float(mat_all.max())
                self.mat_absv_p999 = float(np.percentile(mat_all, 99.9))

            self.after(0, self._on_loaded)
        except Exception as e:
            self.after(0, lambda: self.status.config(
                text=f"Load failed: {e}", foreground="red"))
            self.after(0, lambda: self.load_btn.config(state="normal"))

    def _on_loaded(self):
        # Set slider ranges from the data
        vmax = float(self.all_absv.max()) * 1000
        self.vscale.config(to=vmax)
        self.vcap.set(float(np.percentile(self.all_absv, 99)) * 1000)
        umax = float(self.all_absu.max()) * 1000
        self.uscale.config(to=umax)
        self.ucap.set(float(np.percentile(self.all_absu, 99)) * 1000)
        self.status.config(
            text=f"Loaded {self.condition}: {self.n} frames"
                 + ("  | MATLAB ref ✓" if self.mat_mv is not None else "  | no MATLAB ref"),
            foreground="green")
        self.load_btn.config(state="normal")
        self.export_btn.config(state="normal")
        self._shear_uncapped = None   # invalidate cache for the new dataset
        self._update()
        self._update_shear()

    def _quickset(self, label):
        if self.all_absv is None:
            return
        if label == "MATLAB max":
            if self.mat_absv_max is not None:
                self.vcap.set(self.mat_absv_max * 1000)
            else:
                self.status.config(text="No MATLAB file loaded — can't derive max.",
                                   foreground="red")
                return
        elif label == "MATLAB p99.9":
            if self.mat_absv_p999 is not None:
                self.vcap.set(self.mat_absv_p999 * 1000)
            else:
                self.status.config(text="No MATLAB file loaded — can't derive p99.9.",
                                   foreground="red")
                return
        else:
            q = {"p95": 95, "p99": 99, "p99.5": 99.5, "p99.9": 99.9}[label]
            self.vcap.set(float(np.percentile(self.all_absv, q)) * 1000)
        self._update()
        self._update_shear()

    def _commit_entry(self, which: str):
        """Apply a typed cap value: parse, clamp to the slider range, update."""
        sv = self.vcap_str if which == "v" else self.ucap_str
        var = self.vcap if which == "v" else self.ucap
        scale = self.vscale if which == "v" else self.uscale
        try:
            val = float(sv.get())
        except (ValueError, tk.TclError):
            sv.set(f"{var.get():.1f}")   # revert to current on bad input
            return
        lo = float(scale.cget("from")); hi = float(scale.cget("to"))
        val = max(lo, min(hi, val))      # clamp into the valid range
        var.set(val)
        self._update()                   # refreshes the entry text + plot
        self._update_shear()             # shear (cheap no-op if disabled)

    # ----------------------------------------------------------- shear preview
    def _update_shear(self):
        ax = self.ax_shear
        ax.clear()
        ax.set_xlabel("Frame index"); ax.set_ylabel("Mean shear [1/s]")
        ax.grid(True, alpha=0.4)
        if self.py_u2d is None or not self.show_shear.get():
            ax.text(0.5, 0.5, "shear preview off — tick the box (slower)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            self.canvas.draw_idle()
            return
        self.status.config(text="Computing shear preview…", foreground="black")
        self.update_idletasks()
        vcap = self.vcap.get() / 1000.0
        ucap = self.ucap.get() / 1000.0
        try:
            if self._shear_uncapped is None:
                self._shear_uncapped = _shear_series(
                    self.py_u2d, self.py_v2d, self.tv2d, self.dyn2d,
                    self.dx, self.dy, self.n, np.inf, np.inf)
            c0, s0 = self._shear_uncapped
            c1, s1 = _shear_series(
                self.py_u2d, self.py_v2d, self.tv2d, self.dyn2d,
                self.dx, self.dy, self.n, vcap, ucap)
        except Exception as e:
            self.status.config(text=f"Shear preview failed: {e}", foreground="red")
            return
        ax.plot(c0, s0, color="tab:orange", alpha=0.6, lw=1.0, label="uncapped")
        ax.plot(c1, s1, color="tab:blue", lw=1.2,
                label=f"|v|≤{self.vcap.get():.0f} mm/s")
        ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()
        self.status.config(
            text=f"shear (subsampled): uncapped {np.nanmean(s0):.3f} "
                 f"→ capped {np.nanmean(s1):.3f} 1/s",
            foreground="green")

    # -------------------------------------------------------------- export
    def _export_npz(self):
        """Write a new 7-key .npz with over-cap vectors NaN'd.

        Reloads the ORIGINAL npz (raw px-space x,y and px/frame u/v) so the
        output stays in the canonical piv format. A whole vector is removed
        (both u and v → NaN) if EITHER |v|·|calv| > vcap OR |u|·|calu| > ucap.
        """
        src = self.npz_var.get().strip()
        if not src or not Path(src).exists():
            self.export_status.config(text="Source .npz missing.", foreground="red")
            return
        vcap = self.vcap.get() / 1000.0   # mm/s → m/s
        ucap = self.ucap.get() / 1000.0

        default_name = Path(src).stem + f"_vcap{self.vcap.get():.0f}.npz"
        out = filedialog.asksaveasfilename(
            defaultextension=".npz", initialfile=default_name,
            filetypes=[("NumPy npz", "*.npz")])
        if not out:
            return

        try:
            d = np.load(src, allow_pickle=False)
            calxy = np.float64(d["calxy"])
            calu = np.float64(d["calu"]); calv = np.float64(d["calv"])
            x = d["x"]; y = d["y"]
            u = d["u_original"].astype(np.float32).copy()
            v = d["v_original"].astype(np.float32).copy()

            # cap test in m/s; exclude whole vector if either component exceeds
            absu_ms = np.abs(u) * abs(float(calu))
            absv_ms = np.abs(v) * abs(float(calv))
            exclude = (absv_ms > vcap) | (absu_ms > ucap)
            n_before = np.isfinite(v).sum()
            u[exclude] = np.nan
            v[exclude] = np.nan
            n_removed = int(exclude.sum())
            pct = 100.0 * n_removed / max(int(np.isfinite(absv_ms).sum()), 1)

            np.savez(out,
                     calxy=calxy, calu=calu, calv=calv,
                     x=x.astype(np.float32), y=y.astype(np.float32),
                     u_original=u, v_original=v)
            self.export_status.config(
                text=f"Saved {Path(out).name}  (removed {n_removed} vectors, {pct:.2f}%)",
                foreground="green")
        except Exception as e:
            self.export_status.config(text=f"Export failed: {e}", foreground="red")

    # -------------------------------------------------------------- update
    def _update(self):
        if self.py_absv is None:
            return
        vcap = self.vcap.get() / 1000.0   # mm/s → m/s
        ucap = self.ucap.get() / 1000.0

        mv_capped = _series_with_cap(self.py_absv, vcap)
        mu_capped = _series_with_cap(self.py_absu, ucap)
        v_raw_o, v_cap_o = np.nanmean(self.py_mv_raw) * 1000, np.nanmean(mv_capped) * 1000
        u_raw_o, u_cap_o = np.nanmean(self.py_mu_raw) * 1000, np.nanmean(mu_capped) * 1000
        v_excl = float((self.all_absv > vcap).mean() * 100)
        u_excl = float((self.all_absu > ucap).mean() * 100)
        v_pctile = float((self.all_absv <= vcap).mean() * 100)
        u_pctile = float((self.all_absu <= ucap).mean() * 100)
        v_px = vcap / self.calv if self.calv else float("nan")
        u_px = ucap / self.calv if self.calv else float("nan")

        self.vcap_lbl.config(text=f"≈ {v_px:5.2f} px/fr")
        self.ucap_lbl.config(text=f"≈ {u_px:5.2f} px/fr")
        # Keep the type-in boxes in sync with slider/quick-set/percentile changes
        self.vcap_str.set(f"{self.vcap.get():.1f}")
        self.ucap_str.set(f"{self.ucap.get():.1f}")

        lines = [
            f"|v| cap {self.vcap.get():7.1f} mm/s (p{v_pctile:.1f}, drop {v_excl:.2f}%)  "
            f"→ mean|V| {v_raw_o:.2f} → {v_cap_o:.2f} mm/s",
            f"|u| cap {self.ucap.get():7.1f} mm/s (p{u_pctile:.1f}, drop {u_excl:.2f}%)  "
            f"→ mean|U| {u_raw_o:.2f} → {u_cap_o:.2f} mm/s",
        ]
        if self.mat_mv is not None:
            v_rms = np.sqrt(np.nanmean((mv_capped - self.mat_mv) ** 2)) * 1000
            u_rms = np.sqrt(np.nanmean((mu_capped - self.mat_mu) ** 2)) * 1000
            lines.append(
                f"MATLAB mean|V| {np.nanmean(self.mat_mv)*1000:.2f}  (RMS {v_rms:.2f})   |   "
                f"MATLAB mean|U| {np.nanmean(self.mat_mu)*1000:.2f}  (RMS {u_rms:.2f})  mm/s")
        self.stats.config(text="\n".join(lines))

        fr = np.arange(self.n)

        # ── V chart ──
        self.ax_v.clear()
        if self.mat_mv is not None:
            self.ax_v.plot(fr, self.mat_mv * 1000, lw=0.9, color="black", label="MATLAB")
        self.ax_v.plot(fr, self.py_mv_raw * 1000, lw=0.8, color="tab:orange",
                       alpha=0.55, label="Python uncapped")
        self.ax_v.plot(fr, mv_capped * 1000, lw=1.0, color="tab:blue",
                       label=f"Python |v|≤{self.vcap.get():.0f}")
        self.ax_v.set_ylabel("Mean |V|  (mm/s)")
        self.ax_v.set_title(f"{self.condition} — velocity-cap effect")
        self.ax_v.legend(loc="upper right", fontsize=8)
        self.ax_v.grid(True, alpha=0.4)

        # ── U chart ──
        self.ax_u.clear()
        if self.mat_mu is not None:
            self.ax_u.plot(fr, self.mat_mu * 1000, lw=0.9, color="black", label="MATLAB")
        self.ax_u.plot(fr, self.py_mu_raw * 1000, lw=0.8, color="tab:orange",
                       alpha=0.55, label="Python uncapped")
        self.ax_u.plot(fr, mu_capped * 1000, lw=1.0, color="tab:green",
                       label=f"Python |u|≤{self.ucap.get():.0f}")
        self.ax_u.set_xlabel("Frame index")
        self.ax_u.set_ylabel("Mean |U|  (mm/s)")
        self.ax_u.legend(loc="upper right", fontsize=8)
        self.ax_u.grid(True, alpha=0.4)

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    CapExplorer().mainloop()
