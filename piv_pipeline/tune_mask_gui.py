"""tune_mask_gui.py — Tkinter GUI for dialing in dynamic-mask parameters.

Pick a TIF folder, scrub through frames, switch between mask methods, drag
sliders. The raw image, the intermediate (smoothed / std / top-hat), and the
resulting binary mask all update live. The CLI fragment at the bottom shows
the flags to feed run_pipeline.bat once you find a setting you like.

Usage
-----
    python -m piv_pipeline.tune_mask_gui
    # — or —
    python piv_pipeline/tune_mask_gui.py
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
import skimage.filters
import skimage.io

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ──────────────────────────────── mask backends ───────────────────────────

def _ensure_odd(k: int) -> int:
    """Normalize kernel size. 0 = no smoothing; otherwise force odd, min 3."""
    k = int(k)
    if k <= 0:
        return 0
    if k < 3:
        return 3
    if k % 2 == 0:
        k += 1
    return k


def _box_or_raw(img_f: np.ndarray, ksize: int) -> np.ndarray:
    """Box smooth with the given kernel, or return the image unchanged if k=0."""
    if ksize == 0:
        return img_f
    return cv2.boxFilter(img_f, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)


def _img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _apply_morph(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """Apply morph open then close. 0 disables either step."""
    m = mask.astype(np.uint8)
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def _resize_to_grid_match_pipeline(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Exact same logic as fn_dynamic_masking._resize_to_grid."""
    resized = cv2.resize(mask.astype(np.float32), (cols, rows),
                          interpolation=cv2.INTER_LINEAR)
    out = resized >= 0.5
    # Fill holes smaller than 200 px
    inv = (~out).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep_inv = np.zeros(n, dtype=bool)
    keep_inv[1:] = stats[1:, cv2.CC_STAT_AREA] >= 200
    inv_clean = keep_inv[labels].astype(bool)
    return ~inv_clean


def _compute_simple(img, ksize, threshold, direction, open_ksize=0, close_ksize=0):
    ksize = _ensure_odd(ksize)
    open_k = _ensure_odd(open_ksize)
    close_k = _ensure_odd(close_ksize)
    img_f = img.astype(np.float32)
    inter = _box_or_raw(img_f, ksize)
    if direction == "below":
        mask = inter <= threshold
    else:
        mask = inter >= threshold
    mask = _apply_morph(mask, open_k, close_k)
    label_k = f"raw (k=0)" if ksize == 0 else f"smoothed (k={ksize})"
    morph_suffix = ""
    if open_k or close_k:
        morph_suffix = f" | morph open={open_k} close={close_k}"
    return inter, mask, f"{label_k} — thr={threshold:.0f} {direction}{morph_suffix}"


def _compute_brightness(img, ksize, keep_pct=None, uniform_floor=0.0):
    ksize = _ensure_odd(ksize)
    img_f = img.astype(np.float32)
    inter = _box_or_raw(img_f, ksize)

    # Uniform-frame safety net (applies before Otsu)
    frame_std = float(img_f.std())
    if uniform_floor > 0 and frame_std < uniform_floor:
        mask = np.ones_like(inter, dtype=bool)
        label_k = f"raw (k=0)" if ksize == 0 else f"smoothed (k={ksize})"
        return inter, mask, (
            f"{label_k} — UNIFORM (std={frame_std:.1f} < {uniform_floor:.0f}), "
            f"skip mask"
        )

    if keep_pct is None or keep_pct < 0:
        try:
            thr = float(skimage.filters.threshold_otsu(inter))
            mode = "Otsu"
        except Exception:
            thr = float(np.median(inter))
            mode = "median"
    else:
        thr = float(np.percentile(inter, 100.0 - float(keep_pct)))
        mode = f"keep top {keep_pct}%"
    mask = inter >= thr
    label_k = f"raw (k=0)" if ksize == 0 else f"smoothed (k={ksize})"
    return inter, mask, f"{label_k} — {mode}={thr:.0f} (frame std={frame_std:.1f})"


def _compute_texture(img, ksize):
    ksize = _ensure_odd(ksize)
    img_f = img.astype(np.float32)
    mean = cv2.boxFilter(img_f, -1, (ksize, ksize), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(img_f * img_f, -1, (ksize, ksize),
                        borderType=cv2.BORDER_REFLECT)
    inter = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    try:
        thr = float(skimage.filters.threshold_otsu(inter))
    except Exception:
        thr = float(np.percentile(inter, 25))
    mask = inter >= thr
    return inter, mask, f"local std (k={ksize}) — Otsu={thr:.1f}"


def _compute_tophat(img, particle_ksize, region_ksize, keep_pct):
    pk = _ensure_odd(particle_ksize)
    rk = _ensure_odd(region_ksize)
    img_u8 = _img_to_uint8(img)
    pkern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
    tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, pkern)
    inter = cv2.boxFilter(tophat.astype(np.float32), -1, (rk, rk),
                           borderType=cv2.BORDER_REFLECT)
    thr = float(np.percentile(inter, 100.0 - float(keep_pct)))
    mask = inter >= thr
    return inter, mask, f"smoothed tophat (pk={pk}, rk={rk}) — keep top {keep_pct}%"


def _compute_deviation(img, ksize, threshold):
    ksize = _ensure_odd(ksize)
    img_f = img.astype(np.float32)
    smoothed = _box_or_raw(img_f, ksize)
    med = float(np.median(smoothed))
    inter = np.abs(smoothed - med)   # the deviation map
    if threshold is None or threshold < 0:
        try:
            thr = float(skimage.filters.threshold_otsu(inter))
        except Exception:
            thr = float(np.percentile(inter, 50))
    else:
        thr = float(threshold)
    mask = inter >= thr
    label_k = f"|raw - median| (k=0" if ksize == 0 else f"|smoothed - median| (k={ksize}"
    return inter, mask, f"{label_k}, med={med:.0f}, thr={thr:.1f})"


# ──────────────────────────────── GUI ─────────────────────────────────────

class MaskTunerGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PIV Mask Tuner")
        self.geometry("1500x950")
        self.minsize(1100, 700)

        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        # State
        self.tif_folder: Path | None = None
        self.tif_files: list[Path] = []
        self.current_img: np.ndarray | None = None

        # Tk variables shared across method panels
        self.frame_var = tk.IntVar(value=0)
        self.frame_label_var = tk.StringVar(value="—")
        self.method_var = tk.StringVar(value="simple")
        self.stats_var = tk.StringVar(value="Pick a TIF folder to start.")
        self.cmd_var = tk.StringVar(value="")

        # Per-method variables
        self.simple_ksize = tk.IntVar(value=15)
        self.simple_threshold = tk.IntVar(value=110)
        self.simple_direction = tk.StringVar(value="above")
        self.bright_ksize = tk.IntVar(value=15)
        self.tex_ksize = tk.IntVar(value=15)
        self.tophat_pk = tk.IntVar(value=15)
        self.tophat_rk = tk.IntVar(value=31)
        self.tophat_kp = tk.IntVar(value=70)
        self.dev_ksize = tk.IntVar(value=15)
        self.dev_threshold = tk.IntVar(value=-1)   # -1 = auto Otsu
        self.bright_keep_pct = tk.IntVar(value=-1)  # -1 = auto Otsu
        self.bright_uniform_floor = tk.IntVar(value=0)  # 0 = disabled
        # Morphology (currently exposed for the 'simple' method)
        self.simple_open_ksize = tk.IntVar(value=0)
        self.simple_close_ksize = tk.IntVar(value=0)

        self._build_ui()
        # Re-render whenever any tunable changes
        for v in (self.simple_ksize, self.simple_threshold,
                  self.bright_ksize, self.tex_ksize,
                  self.tophat_pk, self.tophat_rk, self.tophat_kp,
                  self.dev_ksize, self.dev_threshold,
                  self.bright_keep_pct, self.bright_uniform_floor,
                  self.simple_open_ksize, self.simple_close_ksize):
            v.trace_add("write", lambda *a: self._render())
        for v in (self.simple_direction, self.method_var):
            v.trace_add("write", lambda *a: self._on_method_change())

    # ----------------------------------------------------------- UI -------

    def _build_ui(self) -> None:
        # Row 1: folder picker
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="TIF folder:").pack(side="left")
        self.folder_label = ttk.Label(top, text="(none — click Browse…)",
                                       foreground="gray")
        self.folder_label.pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(top, text="Browse…", command=self._pick_folder).pack(side="right")

        # Row 2: frame slider
        row = ttk.Frame(self, padding=(8, 2))
        row.pack(fill="x")
        ttk.Label(row, text="Frame:", width=10).pack(side="left")
        self.frame_slider = ttk.Scale(row, from_=0, to=999, orient="horizontal",
                                        variable=self.frame_var,
                                        command=lambda *a: self._on_frame_change())
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(row, textvariable=self.frame_label_var,
                   width=20, font=("Consolas", 9)).pack(side="left")

        # Row 3: method
        row = ttk.Frame(self, padding=(8, 2))
        row.pack(fill="x")
        ttk.Label(row, text="Method:", width=10).pack(side="left")
        ttk.Combobox(row, textvariable=self.method_var, width=14,
                      values=("simple", "tophat", "brightness", "texture",
                              "deviation"),
                      state="readonly").pack(side="left", padx=8)
        ttk.Label(row, foreground="gray",
                   text="simple = manual threshold • tophat = sparse bright • "
                        "brightness = dense bright • texture = polarity-agnostic • "
                        "deviation = mixed dense+sparse"
                   ).pack(side="left", padx=8)

        # Parameters area — rebuilt when method changes
        self.params_frame = ttk.LabelFrame(self, text="Parameters", padding=8)
        self.params_frame.pack(fill="x", padx=8, pady=4)
        self._build_method_params()

        # Image preview — 3 panels: raw / intermediate / mask
        preview = ttk.Frame(self)
        preview.pack(fill="both", expand=True, padx=8, pady=4)
        self.fig = Figure(figsize=(14, 5), dpi=80)
        self.ax_raw = self.fig.add_subplot(1, 3, 1)
        self.ax_inter = self.fig.add_subplot(1, 3, 2)
        self.ax_mask = self.fig.add_subplot(1, 3, 3)
        for ax in (self.ax_raw, self.ax_inter, self.ax_mask):
            ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=preview)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Footer
        footer = ttk.Frame(self, padding=(8, 4))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.stats_var,
                   font=("Consolas", 10)).pack(anchor="w")
        cmd_row = ttk.Frame(footer)
        cmd_row.pack(fill="x", pady=4)
        ttk.Label(cmd_row, text="CLI:").pack(side="left")
        cmd_entry = ttk.Entry(cmd_row, textvariable=self.cmd_var,
                                font=("Consolas", 9), state="readonly")
        cmd_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(cmd_row, text="Copy", command=self._copy_cmd).pack(side="left")

    def _build_method_params(self) -> None:
        for w in self.params_frame.winfo_children():
            w.destroy()

        method = self.method_var.get()
        if method == "simple":
            self._slider("Smoothing kernel (0 = none, else odd)",
                          self.simple_ksize, 0, 51)
            self._slider("Threshold (0–255)",
                          self.simple_threshold, 0, 255)
            r = ttk.Frame(self.params_frame); r.pack(fill="x", pady=2)
            ttk.Label(r, text="Direction:", width=28).pack(side="left")
            ttk.Radiobutton(r, text="above (keep brighter)", value="above",
                             variable=self.simple_direction).pack(side="left")
            ttk.Radiobutton(r, text="below (keep darker)", value="below",
                             variable=self.simple_direction).pack(side="left", padx=8)

        elif method == "tophat":
            self._slider("Particle kernel (px, odd)", self.tophat_pk, 3, 51)
            self._slider("Region smooth kernel (px, odd)", self.tophat_rk, 3, 101)
            self._slider("Keep top % (lower = stricter)", self.tophat_kp, 5, 95)

        elif method == "brightness":
            self._slider("Smoothing kernel (0 = none, else odd)",
                          self.bright_ksize, 0, 51)
            self._slider("Keep top % (-1 = Otsu auto)",
                          self.bright_keep_pct, -1, 100)
            ttk.Separator(self.params_frame, orient="horizontal").pack(fill="x", pady=4)
            ttk.Label(self.params_frame, foreground="gray",
                       text="Uniform-frame safety net: if frame std < floor, skip mask "
                            "(keep everything — particles are uniformly distributed). "
                            "0 = disabled. Try 20-30 for bioreactor data."
                       ).pack(anchor="w", pady=2)
            self._slider("Uniform floor (frame std, 0 = disabled)",
                          self.bright_uniform_floor, 0, 65535)

        elif method == "texture":
            self._slider("Local-std kernel (px, odd)", self.tex_ksize, 3, 51)
            ttk.Label(self.params_frame, foreground="gray",
                       text="Threshold is auto (Otsu). Switch method to 'simple' "
                            "for a manual threshold.").pack(anchor="w", pady=2)

        elif method == "deviation":
            self._slider("Smoothing kernel (0 = none, else odd)",
                          self.dev_ksize, 0, 51)
            self._slider("Deviation threshold (-1 = Otsu auto)",
                          self.dev_threshold, -1, 100)
            ttk.Label(self.params_frame, foreground="gray",
                       text="Catches BOTH dense (brighter than median) and "
                            "sparse (darker than median) regions. Leave threshold "
                            "at -1 to auto-tune via Otsu."
                       ).pack(anchor="w", pady=2)

        # Morph + resize — applied to the binary mask regardless of method
        # so the GUI mask matches what the pipeline writes.
        ttk.Separator(self.params_frame, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(self.params_frame, foreground="gray",
                   text="Pipeline post-processing (applies after threshold for ALL methods). "
                        "Open removes specks; close fills gaps. 0 disables either step."
                   ).pack(anchor="w")
        self._slider("Morph open kernel (px, 0 disables)",
                      self.simple_open_ksize, 0, 51)
        self._slider("Morph close kernel (px, 0 disables)",
                      self.simple_close_ksize, 0, 101)

    def _slider(self, label: str, var: tk.IntVar, lo: int, hi: int) -> None:
        row = ttk.Frame(self.params_frame); row.pack(fill="x", pady=2)
        ttk.Label(row, text=label + ":", width=28).pack(side="left")
        s = ttk.Scale(row, from_=lo, to=hi, orient="horizontal",
                       variable=var, command=lambda *a: self._render())
        s.pack(side="left", fill="x", expand=True)
        ttk.Label(row, textvariable=var, width=6,
                   font=("Consolas", 10)).pack(side="left")

    # ----------------------------------------------------- callbacks ------

    def _pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder of TIF frames")
        if not folder:
            return
        self.tif_folder = Path(folder)
        # Dedup TIF list — match the pipeline's _list_tif_files. Without
        # this, Windows case-insensitive glob duplicates each file and
        # the frame slider points at the wrong image.
        _all = list(self.tif_folder.glob("*.tif")) + list(self.tif_folder.glob("*.TIF"))
        _seen: set[str] = set()
        deduped: list = []
        for f in _all:
            k = str(f).lower()
            if k in _seen:
                continue
            _seen.add(k)
            deduped.append(f)
        self.tif_files = sorted(deduped)
        if not self.tif_files:
            self.folder_label.configure(
                text=f"{folder} — no .tif files", foreground="red")
            return
        self.folder_label.configure(text=folder, foreground="black")
        n = len(self.tif_files)
        self.frame_slider.configure(to=n - 1)
        self.frame_var.set(min(n // 2, n - 1))
        self._on_frame_change()

    def _on_frame_change(self) -> None:
        if not self.tif_files:
            return
        idx = int(self.frame_var.get())
        idx = max(0, min(idx, len(self.tif_files) - 1))
        self.frame_label_var.set(
            f"{idx} / {len(self.tif_files) - 1}  {self.tif_files[idx].name}"
        )
        img = skimage.io.imread(str(self.tif_files[idx]))
        if img.ndim == 3:
            img = img.mean(axis=2)
        self.current_img = img
        self._render()

    def _on_method_change(self) -> None:
        self._build_method_params()
        self._render()

    # ------------------------------------------------------ rendering ----

    def _render(self) -> None:
        if self.current_img is None:
            return
        img = self.current_img
        method = self.method_var.get()
        # Morph + resize parameters (shared across all methods so the GUI
        # mask matches what the pipeline writes).
        open_k = _ensure_odd(int(self.simple_open_ksize.get()))
        close_k = _ensure_odd(int(self.simple_close_ksize.get()))

        try:
            if method == "simple":
                k = int(self.simple_ksize.get())
                t = int(self.simple_threshold.get())
                d = self.simple_direction.get()
                ok = int(self.simple_open_ksize.get())
                ck = int(self.simple_close_ksize.get())
                inter, mask, label = _compute_simple(img, k, t, d, ok, ck)
                cmd = (f"--mask-method simple "
                       f"--mask-particle-ksize {_ensure_odd(k)} "
                       f"--mask-threshold {t} --mask-direction {d}")
            elif method == "tophat":
                pk = int(self.tophat_pk.get())
                rk = int(self.tophat_rk.get())
                kp = int(self.tophat_kp.get())
                inter, mask, label = _compute_tophat(img, pk, rk, kp)
                cmd = (f"--mask-method tophat "
                       f"--mask-particle-ksize {_ensure_odd(pk)} "
                       f"--mask-region-ksize {_ensure_odd(rk)} "
                       f"--mask-keep-pct {kp}")
            elif method == "brightness":
                k = int(self.bright_ksize.get())
                kp = int(self.bright_keep_pct.get())
                uf = int(self.bright_uniform_floor.get())
                kp_arg = None if kp < 0 else float(kp)
                inter, mask, label = _compute_brightness(img, k, kp_arg, float(uf))
                cmd = (f"--mask-method brightness "
                       f"--mask-particle-ksize {_ensure_odd(k)}")
                if kp_arg is not None:
                    cmd += f" --mask-keep-pct {kp}"
                if uf > 0:
                    cmd += f" --mask-uniform-floor {uf}"
            elif method == "texture":
                k = int(self.tex_ksize.get())
                inter, mask, label = _compute_texture(img, k)
                cmd = (f"--mask-method texture "
                       f"--mask-particle-ksize {_ensure_odd(k)}")
            else:  # deviation
                k = int(self.dev_ksize.get())
                t = int(self.dev_threshold.get())
                t_arg = None if t < 0 else float(t)
                inter, mask, label = _compute_deviation(img, k, t_arg)
                cmd = (f"--mask-method deviation "
                       f"--mask-particle-ksize {_ensure_odd(k)}")
                if t_arg is not None:
                    cmd += f" --mask-deviation-threshold {t}"
        except Exception as e:
            self.stats_var.set(f"Render error: {e}")
            return

        # Append morph flags to the CLI string regardless of method
        if open_k > 0:
            cmd += f" --mask-open-ksize {open_k}"
        if close_k > 0:
            cmd += f" --mask-close-ksize {close_k}"

        # Apply morph + resize (same as the pipeline) so the GUI matches verify_mask
        if open_k > 0 or close_k > 0:
            mask = _apply_morph(mask, open_k, close_k)
        # Match the pipeline's _resize_to_grid exactly: float32 + 0.5 threshold + hole fill
        mask_grid = _resize_to_grid_match_pipeline(mask, 127, 127)
        # Append " | morph + grid" to label so the user sees what's applied
        if open_k or close_k:
            label = f"{label} | morph open={open_k} close={close_k} | 127x127 grid"
        else:
            label = f"{label} | no morph | 127x127 grid"

        # Draw
        for ax in (self.ax_raw, self.ax_inter, self.ax_mask):
            ax.clear()
            ax.axis("off")
        self.ax_raw.imshow(img, cmap="gray")
        self.ax_raw.set_title("raw TIF", fontsize=10)
        self.ax_inter.imshow(inter, cmap="gray")
        self.ax_inter.set_title(label, fontsize=10)
        self.ax_mask.imshow(mask_grid, cmap="gray", vmin=0, vmax=1)
        self.ax_mask.set_title(
            f"final mask (127x127) — {mask_grid.mean() * 100:.1f}% kept",
            fontsize=10,
        )
        self.fig.tight_layout()
        self.canvas.draw_idle()

        frame_std = float(img.astype(np.float32).std())
        self.stats_var.set(
            f"frame std = {frame_std:6.1f}   |   "
            f"kept {mask.mean() * 100:5.1f}%   |   "
            f"intermediate range [{float(inter.min()):.1f}, "
            f"{float(inter.max()):.1f}]   median {float(np.median(inter)):.1f}"
        )
        self.cmd_var.set(cmd)

    # -------------------------------------------------------- copy --------

    def _copy_cmd(self) -> None:
        cmd = self.cmd_var.get()
        if not cmd:
            return
        self.clipboard_clear()
        self.clipboard_append(cmd)
        self.update()
        self.stats_var.set(f"Copied to clipboard: {cmd}")


# ────────────────────────────────── entry ─────────────────────────────────

def main() -> None:
    app = MaskTunerGUI()
    app.mainloop()


if __name__ == "__main__":
    # Allow direct `python tune_mask_gui.py` invocation
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
