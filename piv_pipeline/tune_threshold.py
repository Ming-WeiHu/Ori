"""tune_threshold.py — visual tuner for the 'simple' mask method.

Loads ONE TIF frame, computes the box-smoothed image, and shows masks at
several threshold values for both directions side by side. Plus a histogram
of the smoothed intensities so you can see exactly where to put the cut.

Usage
-----
    python -m piv_pipeline.tune_threshold \
        --tif-folder "C:\\...\\100mL-20deg-35cpm" \
        --frame 500 \
        --ksize 15 \
        --out  "tune.png"

Pick the best mask, then run the pipeline with:
    --mask-method simple --mask-particle-ksize <K> \
        --mask-threshold <V> --mask-direction <above|below>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import skimage.io


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Visual tuner for mask_method=simple.")
    p.add_argument("--tif-folder", required=True,
                   help="Folder of source TIFs.")
    p.add_argument("--frame", type=int, default=500,
                   help="Which frame index to load (default 500).")
    p.add_argument("--ksize", type=int, default=15,
                   help="Box smooth kernel size (default 15).")
    p.add_argument("--thresholds", default="50,80,110,140,170",
                   help="Comma-separated thresholds to try (default 50,80,110,140,170).")
    p.add_argument("--out", default="tune_threshold.png",
                   help="Output PNG path.")
    args = p.parse_args(argv)

    tif_folder = Path(args.tif_folder)
    # Dedup TIF list (Windows case-insensitive glob otherwise duplicates each file).
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
        raise SystemExit(f"No TIFs in {tif_folder}")

    idx = min(max(0, args.frame), len(tifs) - 1)
    img = skimage.io.imread(str(tifs[idx]))
    if img.ndim == 3:
        img = img.mean(axis=2)

    print(f"Loaded frame {idx}: {tifs[idx].name}, shape={img.shape}, "
          f"dtype={img.dtype}, range=[{img.min():.0f}, {img.max():.0f}]")

    img_f = img.astype(np.float32)
    smoothed = cv2.boxFilter(img_f, ddepth=-1, ksize=(args.ksize, args.ksize),
                              borderType=cv2.BORDER_REFLECT)

    print(f"\nSmoothed intensity stats:")
    print(f"  min:    {smoothed.min():.1f}")
    print(f"  25th:   {np.percentile(smoothed, 25):.1f}")
    print(f"  median: {np.percentile(smoothed, 50):.1f}")
    print(f"  75th:   {np.percentile(smoothed, 75):.1f}")
    print(f"  max:    {smoothed.max():.1f}")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    n_thr = len(thresholds)

    # Grid: 3 rows (raw+smoothed+hist on row 0; above masks row 1; below masks row 2)
    fig = plt.figure(figsize=(4 * max(3, n_thr), 12))

    # Row 0: raw, smoothed, histogram
    ax_raw = fig.add_subplot(3, 3, 1)
    ax_raw.imshow(img, cmap="gray")
    ax_raw.set_title(f"Frame {idx}: raw TIF")
    ax_raw.axis("off")

    ax_sm = fig.add_subplot(3, 3, 2)
    ax_sm.imshow(smoothed, cmap="gray")
    ax_sm.set_title(f"Smoothed (ksize={args.ksize})")
    ax_sm.axis("off")

    ax_h = fig.add_subplot(3, 3, 3)
    ax_h.hist(smoothed.ravel(), bins=100, color="steelblue")
    for t in thresholds:
        ax_h.axvline(t, color="red", lw=1, alpha=0.6)
        ax_h.text(t, ax_h.get_ylim()[1] * 0.9, str(int(t)),
                  rotation=90, ha="right", color="red", fontsize=8)
    ax_h.set_title("Smoothed-intensity histogram (red = test thresholds)")
    ax_h.set_xlabel("smoothed intensity")
    ax_h.set_ylabel("pixel count")

    # Rows 1 & 2: masks at each threshold for above / below
    for col, t in enumerate(thresholds):
        ax_a = fig.add_subplot(3, n_thr, n_thr + col + 1)
        mask_above = smoothed >= t
        ax_a.imshow(mask_above, cmap="gray", vmin=0, vmax=1)
        ax_a.set_title(f"above {int(t)} — {mask_above.mean()*100:.0f}% kept")
        ax_a.axis("off")

        ax_b = fig.add_subplot(3, n_thr, 2 * n_thr + col + 1)
        mask_below = smoothed <= t
        ax_b.imshow(mask_below, cmap="gray", vmin=0, vmax=1)
        ax_b.set_title(f"below {int(t)} — {mask_below.mean()*100:.0f}% kept")
        ax_b.axis("off")

    fig.suptitle(
        f"Threshold tuner — frame {idx}, ksize={args.ksize}\n"
        f"top: raw | smoothed | histogram      "
        f"middle: above-threshold masks      "
        f"bottom: below-threshold masks",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"\nSaved: {Path(args.out).resolve()}")
    print("\nPick the mask that best matches the particle regions in the raw TIF,")
    print("then run the pipeline with:")
    print("  --mask-method simple --mask-particle-ksize <K> "
          "--mask-threshold <V> --mask-direction <above|below>")


if __name__ == "__main__":
    main()
