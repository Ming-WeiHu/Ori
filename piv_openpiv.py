"""piv_openpiv.py — OpenPIV-backed alternative to piv_simple.

Same public interface (`PreprocSettings`, `PIVSettings`, `CalibrationSettings`,
`preprocess`, `run_piv`, `apply_calibration`) so callers can swap engines by
changing one import. Output is identical in shape and units, so the .npz
the GUI / pipeline consumes works either way.

How it differs from piv_simple internally:
  * PIV core uses openpiv-python (`pyprocess.extended_search_area_piv`) for
    the final pass, with our own iterative window-deformation loop wrapping
    it for multi-pass. Saves rewriting the FFT correlation by hand.
  * Vector validation + outlier replacement use openpiv's `validation` and
    `filters` modules instead of our NaN-aware median filter.
  * Pre-processing is reused as-is from piv_simple — same CLAHE, highpass,
    capping, Wiener etc. — so the pixels feeding into PIV are identical
    between the two engines.

Settings that don't map cleanly to OpenPIV are documented at their usage site
and either fall through to OpenPIV defaults or are emulated approximately.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple
import cv2
import numpy as np

# Re-export piv_simple's dataclasses + preprocess so the public surface matches.
from piv_simple import (  # noqa: F401  (re-exported for callers)
    PreprocSettings,
    PIVSettings,
    CalibrationSettings,
    preprocess,
    apply_calibration,
    _ensure_gray,
    _load_image,
    _replace_outliers,          # reuse the reference's NaN-aware median test
    _sample_mask_at_centers,    # reuse the reference's grid-centre sampler
)


# ────────────────────────── small helpers ────────────────────────────────

def _ensure_openpiv():
    try:
        import openpiv  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "openpiv-python is not installed in this environment. "
            "Install with:  pip install openpiv"
        ) from e


def _sample_dense_at_centers(dense: np.ndarray,
                             x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a full-resolution float field at PIV grid centres (nearest)."""
    h, w = dense.shape
    xi = np.clip(np.round(x).astype(np.int64), 0, w - 1)
    yi = np.clip(np.round(y).astype(np.int64), 0, h - 1)
    return dense[yi, xi]


def _subpixel_name(piv_settings: PIVSettings) -> str:
    """Map piv_simple's subpixel choices to OpenPIV's."""
    mapping = {
        "gauss2x3": "gaussian",
        "gaussian": "gaussian",
        "centroid": "centroid",
        "parabolic": "parabolic",
    }
    return mapping.get(piv_settings.subpixel_method, "gaussian")


def _search_area_size(window: int, robustness: str) -> int:
    """Robustness='extreme' → 2x search area (≈ zero-padded linear correlation).
    'standard' → equal to window size (cyclic FFT)."""
    return window * 2 if robustness == "extreme" else window


def _deform_image(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Bilinear remap of `img` by per-pixel displacement (dx, dy).

    Used between multi-pass iterations to deform frame B toward frame A.
    """
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(img.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _upsample_vector_field(x: np.ndarray, y: np.ndarray,
                            u: np.ndarray, v: np.ndarray,
                            shape: Tuple[int, int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinearly interpolate a sparse PIV grid (x,y,u,v) up to full image
    resolution (`shape` = (h, w)). NaNs are replaced with 0 before interp."""
    h, w = shape
    # cv2.remap expects float32 displacement fields at full res; we build them
    # by inverse-warping from the grid using cv2.resize.
    u_clean = np.where(np.isnan(u), 0.0, u).astype(np.float32)
    v_clean = np.where(np.isnan(v), 0.0, v).astype(np.float32)
    u_full = cv2.resize(u_clean, (w, h), interpolation=cv2.INTER_LINEAR)
    v_full = cv2.resize(v_clean, (w, h), interpolation=cv2.INTER_LINEAR)
    return u_full, v_full


# ───────────────────────────── PIV core ──────────────────────────────────

def _single_pass(frame_a: np.ndarray, frame_b: np.ndarray,
                 window: int, step: int,
                 piv_settings: PIVSettings):
    """One pass via openpiv.pyprocess.extended_search_area_piv.

    IMPORTANT — coordinate convention:
        OpenPIV returns `v` with POSITIVE = UPWARD (it negates the row shift
        internally). piv_simple — and our `_deform_image` here — use POSITIVE
        = DOWNWARD (image-row direction). We negate v on the way out so the
        whole pipeline (deformation feedback, calibration, .npz output) is in
        the SAME convention as piv_simple. Without this, the vertical
        deformation pushes frame B the wrong way each pass and v inflates
        (u stays correct because its sign already matches).
    """
    from openpiv import pyprocess
    overlap = max(0, window - step)
    search = _search_area_size(window, piv_settings.correlation_robustness)
    result = pyprocess.extended_search_area_piv(
        frame_a.astype(np.float64),
        frame_b.astype(np.float64),
        window_size=window,
        overlap=overlap,
        search_area_size=search,
        subpixel_method=_subpixel_name(piv_settings),
    )
    # OpenPIV returns (u, v) when sig2noise_method is None, else (u, v, s2n).
    u, v = result[0], result[1]
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=search,
        overlap=overlap,
    )
    v = -v   # OpenPIV v is up-positive → flip to image (down-positive) convention
    return x, y, u, v


def _multipass_openpiv(frame_a: np.ndarray, frame_b: np.ndarray,
                        piv_settings: PIVSettings,
                        mask_keep: Optional[np.ndarray] = None,
                        return_originals: bool = False):
    """Multi-pass loop on top of OpenPIV's single-pass.

    Strategy: at each pass, run OpenPIV at that pass's window/step on the
    deformed frame B (deformation = accumulated u, v from prior passes).
    Add the new vectors to the accumulator. Validate + replace between
    passes so the next deformation has clean data to work with.
    """
    _ensure_openpiv()
    h, w = frame_a.shape
    # Dense full-resolution accumulated displacement (image convention: v down).
    accumulated_u = np.zeros((h, w), dtype=np.float32)
    accumulated_v = np.zeros((h, w), dtype=np.float32)

    final_x = final_y = final_u = final_v = None
    final_u_orig = final_v_orig = None

    n_passes = len(piv_settings.window_sizes)
    for i, (win, st) in enumerate(zip(piv_settings.window_sizes,
                                       piv_settings.steps)):
        # Deform B by the running estimate, then correlate to get the RESIDUAL.
        frame_b_deformed = _deform_image(frame_b, accumulated_u, accumulated_v)
        x, y, du, dv = _single_pass(frame_a, frame_b_deformed, win, st, piv_settings)

        # Total displacement at this grid = previous estimate (sampled to this
        # grid) + this pass's residual.  Mirrors piv_simple's `u2 = u_prev + du`.
        u_prev = _sample_dense_at_centers(accumulated_u, x, y)
        v_prev = _sample_dense_at_centers(accumulated_v, x, y)
        u = u_prev + du
        v = v_prev + dv

        # Optional mask: NaN out vectors whose centre falls outside keep region
        if mask_keep is not None:
            keep_at_centers = _sample_mask_at_centers(mask_keep, x, y)
            u[~keep_at_centers] = np.nan
            v[~keep_at_centers] = np.nan

        if i == n_passes - 1:
            # Snapshot the raw TOTAL vectors before outlier replacement
            # (matches piv_simple's `*_original` session fields).
            final_u_orig = u.copy()
            final_v_orig = v.copy()

        if piv_settings.replace_outliers:
            u, v = _replace_outliers(
                u, v,
                kernel=piv_settings.median_filter_size,
                threshold=piv_settings.outlier_threshold,
            )

        final_x, final_y, final_u, final_v = x, y, u, v

        # Re-build the dense accumulated field from the cleaned TOTAL for the
        # next pass's deformation.
        if i < n_passes - 1:
            accumulated_u, accumulated_v = _upsample_vector_field(x, y, u, v, (h, w))

    if return_originals:
        return final_x, final_y, final_u, final_v, final_u_orig, final_v_orig
    return final_x, final_y, final_u, final_v


# ─────────────────────────── run_piv (public) ────────────────────────────

def run_piv(frame_a: np.ndarray, frame_b: np.ndarray,
            piv_settings: Optional[PIVSettings] = None,
            preproc_settings: Optional[PreprocSettings] = None,
            background: Optional[np.ndarray] = None,
            calibration: Optional[CalibrationSettings] = None,
            mask: Optional[np.ndarray] = None,
            return_originals: bool = False):
    """Drop-in replacement for piv_simple.run_piv backed by OpenPIV.

    Signature, semantics, and output format are identical. See piv_simple
    for the contract — same inputs, same outputs, same units.

    Settings that are silently ignored or approximated:
      - `algorithm` (always FFT via OpenPIV)
      - `disable_autocorrelation` (OpenPIV handles its own preprocessing)
      - `repeat_last_pass` (no equivalent — multi-pass uses fixed passes)
    """
    piv_settings = piv_settings or PIVSettings()
    preproc_settings = preproc_settings or PreprocSettings()

    a = _ensure_gray(frame_a)
    b = _ensure_gray(frame_b)
    bg = _ensure_gray(background) if background is not None else None

    a = preprocess(a, preproc_settings, bg)
    b = preprocess(b, preproc_settings, bg)

    mask_keep: Optional[np.ndarray] = None
    if mask is not None:
        if mask.dtype == bool:
            mask_keep = mask
        else:
            m = mask
            if m.ndim == 3:
                m = m[..., 0]
            mask_keep = m <= 127  # PIVlab: bright = exclude, dark = analyse
        if mask_keep.shape != a.shape:
            mask_keep = cv2.resize(
                mask_keep.astype(np.uint8), (a.shape[1], a.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)

    if return_originals:
        x, y, u, v, u_orig, v_orig = _multipass_openpiv(
            a, b, piv_settings, mask_keep, return_originals=True)
        if calibration is not None:
            x, y, u, v = apply_calibration(x, y, u, v, calibration)
            vs = calibration.m_per_second_per_px_per_frame
            u_orig = u_orig * vs * calibration.x_sign
            v_orig = v_orig * vs * calibration.y_sign
        return x, y, u, v, u_orig, v_orig

    x, y, u, v = _multipass_openpiv(a, b, piv_settings, mask_keep)
    if calibration is not None:
        x, y, u, v = apply_calibration(x, y, u, v, calibration)
    return x, y, u, v


# ────────────────────────────── CLI ──────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="OpenPIV-backed PIV (drop-in for piv_simple)."
    )
    p.add_argument("frame_a")
    p.add_argument("frame_b")
    p.add_argument("--background", default=None)
    p.add_argument("--mask", default=None)
    p.add_argument("--out", default="piv_vectors_openpiv.npz")

    p.add_argument("--windows", default="64,32,16")
    p.add_argument("--steps", default="32,16,8")
    p.add_argument("--subpixel", default="gauss2x3",
                   choices=("gauss2x3", "centroid", "parabolic"))
    p.add_argument("--robustness", default="standard",
                   choices=("standard", "extreme"))
    args = p.parse_args()

    frame_a = _load_image(args.frame_a)
    frame_b = _load_image(args.frame_b)
    background = _load_image(args.background) if args.background else None
    mask = _load_image(args.mask) if args.mask else None

    piv_settings = PIVSettings(
        window_sizes=tuple(int(s) for s in args.windows.split(",")),
        steps=tuple(int(s) for s in args.steps.split(",")),
        subpixel_method=args.subpixel,
        correlation_robustness=args.robustness,
    )

    x, y, u, v = run_piv(
        frame_a, frame_b,
        piv_settings=piv_settings,
        background=background,
        mask=mask,
    )

    np.savez(args.out, x=x, y=y, u=u, v=v)
    print(f"Saved: {args.out}  (engine: openpiv)")


if __name__ == "__main__":
    _cli()
