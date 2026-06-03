"""verify_mask.py — visual sanity check for dynamic masking.

Loads a pipeline output and produces a side-by-side PNG showing, for 3 sample
frames (start / middle / end):
    1. The raw TIF
    2. The static typevector (PIVlab/piv_simple mask, same every frame)
    3. The per-frame dynamic mask (Stage 2 output)
    4. The combined mask (typevector ∩ dynamic)
    5. The masked shear field

Plus prints kept-fraction stats so you can sanity-check the mask isn't
over- or under-excluding.

Usage
-----
    python -m piv_pipeline.verify_mask \
        --output-root "C:\\...\\PipelineTest_withmask" \
        --tif-folder  "C:\\...\\100mL-20deg-35cpm"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import skimage.io


def _find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}")
    return matches[0]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Visual sanity check for dynamic masking.")
    p.add_argument("--output-root", required=True,
                   help="Pipeline output folder containing "
                        "'Secondary Exports/', 'Dynamic Masks/', 'Tertiary Exports/'.")
    p.add_argument("--tif-folder", required=True,
                   help="Folder of source TIF frames that fed Stage 2.")
    p.add_argument("--out", default="verify_mask.png",
                   help="Output PNG path (default: verify_mask.png in CWD).")
    p.add_argument("--frames", default="0,500,1000",
                   help="Comma-separated frame indices to inspect.")
    args = p.parse_args(argv)

    out_root = Path(args.output_root)
    tif_folder = Path(args.tif_folder)
    out_png = Path(args.out)

    sec_path = _find_one(out_root / "Secondary Exports", "*_Secondary_Export.npz")
    mask_path = _find_one(out_root / "Dynamic Masks", "dynamic_masking_*.npz")
    tert_path = _find_one(out_root / "Tertiary Exports", "*_Tertiary_Export.npz")

    print(f"Secondary: {sec_path.name}")
    print(f"Mask:      {mask_path.name}")
    print(f"Tertiary:  {tert_path.name}")

    # Dedup TIF list — must match the pipeline's _list_tif_files, otherwise
    # on Windows tifs[k] points at the wrong file (case-insensitive glob
    # returns each TIF twice and shifts indices by 2x).
    _all = list(tif_folder.glob("*.tif")) + list(tif_folder.glob("*.TIF"))
    _seen: set[str] = set()
    tifs: list = []
    for f in _all:
        k = str(f).lower()
        if k in _seen:
            continue
        _seen.add(k)
        tifs.append(f)
    tifs = sorted(tifs)
    if not tifs:
        raise FileNotFoundError(f"No .tif files in {tif_folder}")

    frame_idx = [int(x) for x in args.frames.split(",")]

    with np.load(sec_path, allow_pickle=False) as d:
        typevector = d["typevector"].astype(bool)
        S33_shape = d["S33"].shape

    with np.load(mask_path, allow_pickle=False) as d:
        dyn_masks = d["dynamic_masking"]   # (rows, cols, n_imgs)

    with np.load(tert_path, allow_pickle=False) as d:
        masked_shear = d["masked_shear"]   # (rows, cols, n_frames)

    print(f"\nGrid: {typevector.shape}   n_dyn_frames: {dyn_masks.shape[2]}   "
          f"n_shear_frames: {masked_shear.shape[2]}")

    # ----- kept-fraction stats ---------------------------------------------
    static_keep = typevector.mean()
    dyn_keep_per_frame = dyn_masks.reshape(-1, dyn_masks.shape[2]).mean(axis=0)
    dyn_keep_mean = float(dyn_keep_per_frame.mean())
    combined_per_frame = (typevector[:, :, None] & dyn_masks).reshape(-1, dyn_masks.shape[2]).mean(axis=0)
    combined_keep_mean = float(combined_per_frame.mean())

    print("\nKept fraction (1.0 = whole grid, 0.0 = nothing):")
    print(f"  typevector (static):           {static_keep*100:6.2f}%")
    print(f"  dynamic mean across frames:    {dyn_keep_mean*100:6.2f}%   "
          f"(min {dyn_keep_per_frame.min()*100:.1f}%, max {dyn_keep_per_frame.max()*100:.1f}%)")
    print(f"  combined mean across frames:   {combined_keep_mean*100:6.2f}%")

    if combined_keep_mean < 0.05:
        print("\n⚠ WARNING: combined mask keeps <5% of grid — dynamic mask may be over-excluding.")
    elif combined_keep_mean > 0.95 * static_keep:
        print("\n⚠ NOTE: dynamic mask is barely tighter than typevector — Stage 2 may not be helping much.")
    else:
        print("\n✓ Kept fraction looks reasonable.")

    # ----- visualization ----------------------------------------------------
    n_rows = len(frame_idx)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, k in enumerate(frame_idx):
        k_clamped = min(k, len(tifs) - 1)
        img = skimage.io.imread(str(tifs[k_clamped]))
        if img.ndim == 3:
            img = img.mean(axis=2)

        dyn = dyn_masks[:, :, min(k, dyn_masks.shape[2] - 1)]
        combined = typevector & dyn

        k_shear = min(k, masked_shear.shape[2] - 1)
        shear = masked_shear[:, :, k_shear]
        vmax = np.nanpercentile(shear, 95) if np.isfinite(shear).any() else 1.0

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"Frame {k_clamped}: raw TIF")

        axes[row, 1].imshow(typevector, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"typevector ({static_keep*100:.0f}% kept)")

        axes[row, 2].imshow(dyn, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"dynamic mask ({dyn.mean()*100:.0f}% kept)")

        axes[row, 3].imshow(combined, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"combined ({combined.mean()*100:.0f}% kept)")

        im = axes[row, 4].imshow(shear, cmap="hot", vmin=0, vmax=vmax)
        axes[row, 4].set_title(f"masked_shear (k={k_shear})")
        plt.colorbar(im, ax=axes[row, 4], fraction=0.046)

        for col in range(5):
            axes[row, col].axis("off")

    fig.suptitle(f"Mask verification — {sec_path.stem.replace('_Secondary_Export','')}",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nSaved: {out_png.resolve()}")


if __name__ == "__main__":
    main()
